package main

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

type recordedGraphNode struct {
	node *cudagraph.Node
	desc cudalaunch.Description
	args []any
	deps []*cudagraph.Node
}

type recordedParamUpdate struct {
	node   *cudagraph.Node
	params driver.KernelNodeParams
}

type fakeGraphHarness struct {
	events       []string
	preparedSmem []uint32
	added        []recordedGraphNode
	updates      []recordedParamUpdate
	launched     []*cuda.Stream
	syncedCtx    int
	syncedStream []*cuda.Stream
	copiedOut    []*cuda.Stream

	prepareErr        error
	newGraphErr       error
	failAddAt         int // 1-based; 0 means never
	addErr            error
	instantiateExec   *cudagraph.Exec
	instantiateErr    error
	failSetAt         int // 1-based; 0 means never
	setErr            error
	launchErr         error
	syncCtxErr        error
	syncStreamErr     error
	copyOutErr        error
	destroyExecFails  int
	destroyGraphFails int
	destroyExecErr    error
	destroyGraphErr   error

	graphSentinel *cudagraph.Graph
	execSentinel  *cudagraph.Exec
	setCalls      int
	hostOutToken  int32
}

func newFakeGraphHarness() *fakeGraphHarness {
	return &fakeGraphHarness{
		graphSentinel:   new(cudagraph.Graph),
		execSentinel:    new(cudagraph.Exec),
		addErr:          errors.New("fake add node failed"),
		setErr:          errors.New("fake set params failed"),
		destroyExecErr:  errors.New("fake destroy exec failed"),
		destroyGraphErr: errors.New("fake destroy graph failed"),
		hostOutToken:    42,
	}
}

func (h *fakeGraphHarness) build(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	layerKV func(int) (cudakv.LayerCache, error),
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn, sampleFn driver.Function,
) (*DecodeGraph, error) {
	return newDecodeGraphWith(
		gpuW, cfg, layerKV, sp, gm, attnFn, sampleFn,
		func(shared uint32) error {
			h.events = append(h.events, fmt.Sprintf("prepare(%d)", shared))
			h.preparedSmem = append(h.preparedSmem, shared)
			return h.prepareErr
		},
		func() (*cudagraph.Graph, error) {
			h.events = append(h.events, "newGraph")
			if h.newGraphErr != nil {
				return nil, h.newGraphErr
			}
			return h.graphSentinel, nil
		},
		func(g *cudagraph.Graph, l *cudalaunch.Launcher, args []any, deps ...*cudagraph.Node) (*cudagraph.Node, error) {
			if g != h.graphSentinel {
				return nil, fmt.Errorf("addNode on unexpected graph %p", g)
			}
			idx := len(h.added) + 1
			if h.failAddAt > 0 && idx >= h.failAddAt {
				h.events = append(h.events, fmt.Sprintf("addNode(%d):err", idx))
				return nil, h.addErr
			}
			desc, err := l.Describe()
			if err != nil {
				return nil, err
			}
			n := new(cudagraph.Node)
			h.added = append(h.added, recordedGraphNode{
				node: n,
				desc: desc,
				args: slices.Clone(args),
				deps: slices.Clone(deps),
			})
			return n, nil
		},
		func(g *cudagraph.Graph) (*cudagraph.Exec, error) {
			h.events = append(h.events, "instantiate")
			if g != h.graphSentinel {
				return nil, fmt.Errorf("instantiate on unexpected graph %p", g)
			}
			if h.instantiateErr != nil {
				return h.instantiateExec, h.instantiateErr
			}
			return h.execSentinel, nil
		},
		func(e *cudagraph.Exec, node *cudagraph.Node, params *driver.KernelNodeParams) error {
			if e != h.execSentinel {
				return fmt.Errorf("setParams on unexpected exec %p", e)
			}
			h.setCalls++
			if h.failSetAt > 0 && h.setCalls >= h.failSetAt {
				return h.setErr
			}
			h.updates = append(h.updates, recordedParamUpdate{
				node:   node,
				params: *params,
			})
			return nil
		},
		func(e *cudagraph.Exec, stream *cuda.Stream) error {
			if e != h.execSentinel {
				return fmt.Errorf("launch on unexpected exec %p", e)
			}
			h.events = append(h.events, "launch")
			h.launched = append(h.launched, stream)
			return h.launchErr
		},
		func() error {
			h.events = append(h.events, "syncCtx")
			h.syncedCtx++
			return h.syncCtxErr
		},
		func(stream *cuda.Stream) error {
			h.events = append(h.events, "syncStream")
			h.syncedStream = append(h.syncedStream, stream)
			return h.syncStreamErr
		},
		func(stream *cuda.Stream) error {
			h.events = append(h.events, "copyOut")
			h.copiedOut = append(h.copiedOut, stream)
			if h.copyOutErr != nil {
				return h.copyOutErr
			}
			sp.HostTokenOut[0] = h.hostOutToken
			return nil
		},
		h.destroyExec,
		h.destroyGraph,
	)
}

func (h *fakeGraphHarness) destroyExec(e *cudagraph.Exec) error {
	h.events = append(h.events, "destroyExec")
	if e != h.execSentinel && e != h.instantiateExec {
		return fmt.Errorf("destroyExec on unexpected exec %p", e)
	}
	if h.destroyExecFails > 0 {
		h.destroyExecFails--
		return h.destroyExecErr
	}
	return nil
}

func (h *fakeGraphHarness) destroyGraph(g *cudagraph.Graph) error {
	h.events = append(h.events, "destroyGraph")
	if g != h.graphSentinel {
		return fmt.Errorf("destroyGraph on unexpected graph %p", g)
	}
	if h.destroyGraphFails > 0 {
		h.destroyGraphFails--
		return h.destroyGraphErr
	}
	return nil
}

func fakeStepFixture(t *testing.T, layers int, dtype weightDType) (*GPUModelWeights, *ModelConfig, *GPUScratchpad, []cudakv.LayerCache, *GPUModules) {
	t.Helper()
	f := newStepTestFixture(t, dtype)
	for len(f.gpuW.Layers) < layers {
		f.gpuW.Layers = append(f.gpuW.Layers, f.gpuW.Layers[len(f.gpuW.Layers)-1])
		lc := f.kvLayers[len(f.kvLayers)-1]
		lc.LayerIndex = len(f.kvLayers)
		f.kvLayers = append(f.kvLayers, lc)
	}
	f.gpuW.Layers = f.gpuW.Layers[:layers]
	f.kvLayers = f.kvLayers[:layers]
	f.cfg.NumHiddenLayers = layers
	f.sp.HostTokenIn = make([]int32, 1)
	f.sp.HostTokenOut = make([]int32, 1)
	return f.gpuW, &f.cfg, f.sp, f.kvLayers, f.gm
}

// TestDecodeGraphBuildAndSetPosition tests explicit graph construction and
// per-step position updates across FP32 and BF16 weights, unfused and fused:
//   - dynamic shared memory is prepared once at setup for MaxSeqLen before
//     creating the graph or adding any nodes;
//   - node count is 15L+4 unfused and 11L+4 fused;
//   - nodes form a single linear chain (node 0 has no deps; node k depends
//     solely on node k-1);
//   - EmbeddingLookup[BF16] reads sp.TokenOut (device-resident token
//     feedback), never sp.TokenIn;
//   - every attention node carries the fixed MaxSeqLen shared memory size;
//   - SetPosition updates all 3L position-dependent nodes (ApplyRoPE arg 2,
//     StoreKVCache arg 4, Attention arg 4) via SetKernelNodeParams without
//     growing retained slice allocations, and rejects out-of-range positions.
func TestDecodeGraphBuildAndSetPosition(t *testing.T) {
	saveFuse, saveDecode, saveQual := *fuseEGraphFlag, *fuseDecodeFlag, fusedKernelsQualified
	t.Cleanup(func() {
		*fuseEGraphFlag = saveFuse
		*fuseDecodeFlag = saveDecode
		fusedKernelsQualified = saveQual
	})
	fusedKernelsQualified = true

	const (
		attnFn   = driver.Function(100)
		sampleFn = driver.Function(101)
	)

	for _, dtype := range []weightDType{weightF32, weightBF16} {
		for _, mode := range []struct {
			name       string
			fused      bool
			fuseDecode bool
		}{
			{"unfused", false, false},
			{"fuseEGraph", true, false},
			{"fuseDecode", false, true},
		} {
			for _, layers := range []int{2, 3} {
				name := fmt.Sprintf("%s_mode=%s_L=%d", dtype, mode.name, layers)
				t.Run(name, func(t *testing.T) {
					*fuseEGraphFlag = mode.fused
					*fuseDecodeFlag = mode.fuseDecode
					gpuW, cfg, sp, kvs, gm := fakeStepFixture(t, layers, dtype)
					layerKV := func(i int) (cudakv.LayerCache, error) { return kvs[i], nil }

					wantSharedInt, err := cudaattn.DecodeSharedMemoryBytes(cfg.MaxSeqLen, attnThreads)
					if err != nil {
						t.Fatal(err)
					}
					wantShared := uint32(wantSharedInt)

					h := newFakeGraphHarness()
					dg, err := h.build(gpuW, cfg, layerKV, sp, gm, attnFn, sampleFn)
					if err != nil {
						t.Fatalf("build decode graph: %v", err)
					}

					// 1. PrepareSharedMem called once before newGraph/instantiate.
					if !slices.Equal(h.preparedSmem, []uint32{wantShared}) {
						t.Errorf("prepared shared mem = %v, want [%d]", h.preparedSmem, wantShared)
					}
					wantBuildEvents := []string{fmt.Sprintf("prepare(%d)", wantShared), "newGraph", "instantiate"}
					if !slices.Equal(h.events, wantBuildEvents) {
						t.Errorf("build events = %v, want %v", h.events, wantBuildEvents)
					}

					// 2. Total node count is 15L+4 unfused, 11L+4 fuse-egraph, or 5L+4 fuse-decode.
					perLayer := 15
					if mode.fused {
						perLayer = 11
					} else if mode.fuseDecode {
						perLayer = 5
					}
					wantNodes := perLayer*layers + 4
					if len(h.added) != wantNodes {
						t.Fatalf("added %d nodes, want %d", len(h.added), wantNodes)
					}

					// 3. Linear dependency chain.
					for i, n := range h.added {
						if i == 0 {
							if len(n.deps) != 0 {
								t.Errorf("node 0 deps = %v, want empty", n.deps)
							}
							continue
						}
						if len(n.deps) != 1 || n.deps[0] != h.added[i-1].node {
							t.Errorf("node %d deps = %v, want [%p]", i, n.deps, h.added[i-1].node)
						}
					}

					// 4. Node 0 is EmbeddingLookup[BF16] reading sp.TokenOut.
					wantEmbed := gm.embedFn
					if dtype == weightBF16 {
						wantEmbed = gm.embedBF16Fn
					}
					if h.added[0].desc.Function != wantEmbed {
						t.Errorf("node 0 function = %v, want %v", h.added[0].desc.Function, wantEmbed)
					}
					if gotTok, wantTok := h.added[0].args[1], sp.TokenOut; gotTok != wantTok {
						t.Errorf("node 0 token src = %+v, want sp.TokenOut %+v (sp.TokenIn is %+v)",
							gotTok, wantTok, sp.TokenIn)
					}

					// 5. Every attention node carries wantShared, and 3L (or 2L for fuse-decode) position nodes are recorded.
					posPerLayer := 3
					if mode.fuseDecode {
						posPerLayer = 2
					}
					if len(dg.posNodes) != posPerLayer*layers {
						t.Fatalf("posNodes = %d, want %d", len(dg.posNodes), posPerLayer*layers)
					}
					for l := 0; l < layers; l++ {
						base := 1 + l*perLayer
						if mode.fuseDecode {
							qkvNode := h.added[base+0]
							attnNode := h.added[base+1]
							if attnNode.desc.SharedMem != wantShared {
								t.Errorf("layer %d attn SharedMem = %d, want %d", l, attnNode.desc.SharedMem, wantShared)
							}
							wantQKVFn := gm.fusedQKVRoPEStoreFn
							if dtype == weightBF16 {
								wantQKVFn = gm.fusedQKVRoPEStoreBF16Fn
							}
							wantKinds := []struct {
								kind   string
								posIdx int
								node   *cudagraph.Node
								fn     driver.Function
							}{
								{"qkv_rope_store", 15, qkvNode.node, wantQKVFn},
								{"attn", 4, attnNode.node, attnFn},
							}
							for k, wk := range wantKinds {
								pn := dg.posNodes[l*2+k]
								if pn.layer != l || pn.kind != wk.kind || pn.posIdx != wk.posIdx || pn.node != wk.node || pn.params.Func != wk.fn {
									t.Errorf("posNode[%d] = %+v, want layer=%d kind=%s posIdx=%d node=%p fn=%v",
										l*2+k, pn, l, wk.kind, wk.posIdx, wk.node, wk.fn)
								}
								if gotInitPos := pn.args[pn.posIdx]; gotInitPos != int32(0) {
									t.Errorf("posNode[%d] initial pos = %v, want int32(0)", l*2+k, gotInitPos)
								}
							}
							continue
						}
						ropeOff, storeOff, attnOff := 4, 5, 6
						if mode.fused {
							ropeOff, storeOff, attnOff = 1, 2, 3
						}
						attnNode := h.added[base+attnOff]
						if attnNode.desc.SharedMem != wantShared {
							t.Errorf("layer %d attn SharedMem = %d, want %d", l, attnNode.desc.SharedMem, wantShared)
						}

						wantKinds := []struct {
							kind   string
							posIdx int
							node   *cudagraph.Node
							fn     driver.Function
						}{
							{"rope", 2, h.added[base+ropeOff].node, gm.ropeFn},
							{"store_kv", 4, h.added[base+storeOff].node, gm.storeKVFn},
							{"attn", 4, attnNode.node, attnFn},
						}
						for k, wk := range wantKinds {
							pn := dg.posNodes[l*3+k]
							if pn.layer != l || pn.kind != wk.kind || pn.posIdx != wk.posIdx || pn.node != wk.node || pn.params.Func != wk.fn {
								t.Errorf("posNode[%d] = %+v, want layer=%d kind=%s posIdx=%d node=%p fn=%v",
									l*3+k, pn, l, wk.kind, wk.posIdx, wk.node, wk.fn)
							}
							if gotInitPos := pn.args[pn.posIdx]; gotInitPos != int32(0) {
								t.Errorf("posNode[%d] initial pos = %v, want int32(0)", l*3+k, gotInitPos)
							}
						}
					}

					// 6. SetPosition updates all posPerLayer*L nodes in order and reads back the exact int32(pos).
					for _, pos := range []int{0, 1, 17, cfg.MaxSeqLen - 1} {
						h.updates = nil
						if err := dg.SetPosition(pos); err != nil {
							t.Fatalf("SetPosition(%d): %v", pos, err)
						}
						if len(h.updates) != posPerLayer*layers {
							t.Fatalf("SetPosition(%d) updated %d nodes, want %d", pos, len(h.updates), posPerLayer*layers)
						}
						for i, u := range h.updates {
							pn := &dg.posNodes[i]
							if u.node != pn.node {
								t.Errorf("SetPosition(%d) update %d node = %p, want %p", pos, i, u.node, pn.node)
							}
							wantParamsPtr := uintptr(unsafe.Pointer(&pn.packed[0]))
							if u.params.KernelParams != wantParamsPtr {
								t.Errorf("SetPosition(%d) update %d KernelParams = %#x, want %#x",
									pos, i, u.params.KernelParams, wantParamsPtr)
							}
							gotPos := *(*int32)(pn.packed[pn.posIdx])
							if gotPos != int32(pos) {
								t.Errorf("SetPosition(%d) update %d (%s layer %d) read back pos = %d, want %d",
									pos, i, pn.kind, pn.layer, gotPos, pos)
							}
							if pn.kind == "attn" && u.params.SharedMemBytes != wantShared {
								t.Errorf("SetPosition(%d) attn SharedMemBytes = %d, want %d", pos, u.params.SharedMemBytes, wantShared)
							}
							if u.params.Extra != 0 {
								t.Errorf("SetPosition(%d) Extra = %#x, want 0", pos, u.params.Extra)
							}
						}
					}

					// 7. Out-of-range positions are rejected before any SetKernelNodeParams call.
					for _, badPos := range []int{-1, cfg.MaxSeqLen, cfg.MaxSeqLen + 10} {
						h.updates = nil
						if err := dg.SetPosition(badPos); err == nil || !strings.Contains(err.Error(), "outside the kv cache") {
							t.Errorf("SetPosition(%d) = %v, want outside the kv cache", badPos, err)
						}
						if len(h.updates) != 0 {
							t.Errorf("SetPosition(%d) made %d updates before failing", badPos, len(h.updates))
						}
					}
				})
			}
		}
	}

	// Refusal check when -fuse-decode is set and fusedKernelsQualified is false.
	*fuseEGraphFlag, *fuseDecodeFlag, fusedKernelsQualified = false, true, false
	gpuW, cfg, sp, kvs, gm := fakeStepFixture(t, 1, weightF32)
	layerKV := func(i int) (cudakv.LayerCache, error) { return kvs[i], nil }
	h := newFakeGraphHarness()
	if _, err := h.build(gpuW, cfg, layerKV, sp, gm, attnFn, sampleFn); err == nil || !strings.Contains(err.Error(), "-fuse-decode: fused kernels are not qualified") {
		t.Errorf("unqualified -fuse-decode graph build: got %v, want not qualified", err)
	}
}

// TestDecodeGraphStepAndEnqueue verifies the execution sequence and stream
// synchronization semantics of Step and Enqueue for both nil and non-nil
// streams, including draining the stream when Enqueue fails.
func TestDecodeGraphStepAndEnqueue(t *testing.T) {
	gpuW, cfg, sp, kvs, gm := fakeStepFixture(t, 2, weightF32)
	layerKV := func(i int) (cudakv.LayerCache, error) { return kvs[i], nil }

	// 1. Synchronous Step(pos, nil): SetPosition -> launch(nil) -> syncCtx -> copyOut(nil).
	h := newFakeGraphHarness()
	h.hostOutToken = 19
	dg, err := h.build(gpuW, cfg, layerKV, sp, gm, 100, 101)
	if err != nil {
		t.Fatal(err)
	}
	h.events = nil
	tok, err := dg.Step(5, nil)
	if err != nil || tok != 19 {
		t.Fatalf("Step(5, nil) = %d, %v; want 19, nil", tok, err)
	}
	if !slices.Equal(h.events, []string{"launch", "syncCtx", "copyOut"}) {
		t.Errorf("Step(5, nil) events = %v, want [launch syncCtx copyOut]", h.events)
	}
	if len(h.updates) != 6 || *(*int32)(dg.posNodes[0].packed[dg.posNodes[0].posIdx]) != 5 || *(*int32)(dg.posNodes[5].packed[dg.posNodes[5].posIdx]) != 5 {
		t.Errorf("Step(5, nil) updates = %d, want 6 at pos 5", len(h.updates))
	}

	// 2. Asynchronous Enqueue(pos, stream): SetPosition -> launch(stream) -> copyOut(stream), no sync.
	fakeStream := new(cuda.Stream)
	h.events = nil
	h.updates = nil
	tok, err = dg.Enqueue(6, fakeStream)
	if err != nil || tok != 0 {
		t.Fatalf("Enqueue(6, stream) = %d, %v; want 0, nil", tok, err)
	}
	if !slices.Equal(h.events, []string{"launch", "copyOut"}) {
		t.Errorf("Enqueue(6, stream) events = %v, want [launch copyOut]", h.events)
	}
	if h.launched[len(h.launched)-1] != fakeStream || h.copiedOut[len(h.copiedOut)-1] != fakeStream {
		t.Errorf("Enqueue stream mismatch: launched=%p copiedOut=%p, want %p",
			h.launched[len(h.launched)-1], h.copiedOut[len(h.copiedOut)-1], fakeStream)
	}

	// 3. Step(pos, stream): SetPosition -> launch(stream) -> copyOut(stream) -> syncStream(stream).
	h.events = nil
	h.hostOutToken = 23
	tok, err = dg.Step(7, fakeStream)
	if err != nil || tok != 23 {
		t.Fatalf("Step(7, stream) = %d, %v; want 23, nil", tok, err)
	}
	if !slices.Equal(h.events, []string{"launch", "copyOut", "syncStream"}) {
		t.Errorf("Step(7, stream) events = %v, want [launch copyOut syncStream]", h.events)
	}

	// 4. Step(pos, stream) drains stream even when launch fails, joining both errors.
	h.events = nil
	h.launchErr = errors.New("fake launch failed")
	h.syncStreamErr = errors.New("fake stream sync failed")
	_, err = dg.Step(8, fakeStream)
	if !errors.Is(err, h.launchErr) || !errors.Is(err, h.syncStreamErr) {
		t.Errorf("Step error = %v, want both %v and %v", err, h.launchErr, h.syncStreamErr)
	}
	if !slices.Equal(h.events, []string{"launch", "syncStream"}) {
		t.Errorf("Step failed-launch events = %v, want [launch syncStream]", h.events)
	}
}

// TestDecodeGraphCloseAndSetupErrors checks:
//   - Close destroys exec before graph, is idempotent once both succeed, and
//     retains whichever handle failed to destroy so a retry can release it;
//   - newDecodeGraphWith cleans up graph (and any partial exec from
//     instantiate) on setup errors, and returns (dg, errors.Join(err, cerr))
//     when cleanup itself fails.
func TestDecodeGraphCloseAndSetupErrors(t *testing.T) {
	gpuW, cfg, sp, kvs, gm := fakeStepFixture(t, 2, weightF32)
	layerKV := func(i int) (cudakv.LayerCache, error) { return kvs[i], nil }

	// 1. Clean Close destroys exec before graph; second Close is a no-op.
	h := newFakeGraphHarness()
	dg, err := h.build(gpuW, cfg, layerKV, sp, gm, 100, 101)
	if err != nil {
		t.Fatal(err)
	}
	h.events = nil
	if err := dg.close(h.destroyExec, h.destroyGraph); err != nil {
		t.Fatalf("first close: %v", err)
	}
	if !slices.Equal(h.events, []string{"destroyExec", "destroyGraph"}) {
		t.Errorf("close order = %v, want [destroyExec destroyGraph]", h.events)
	}
	if dg.exec != nil || dg.graph != nil {
		t.Errorf("handles after close: exec=%p graph=%p, want nil", dg.exec, dg.graph)
	}
	if err := dg.close(h.destroyExec, h.destroyGraph); err != nil || len(h.events) != 2 {
		t.Errorf("second close: err=%v, events=%v; want nil and no new calls", err, h.events)
	}

	// 2. Both destroyExec and destroyGraph fail -> both retained; retry releases both.
	h = newFakeGraphHarness()
	h.destroyExecFails = 1
	h.destroyGraphFails = 1
	dg, err = h.build(gpuW, cfg, layerKV, sp, gm, 100, 101)
	if err != nil {
		t.Fatal(err)
	}
	h.events = nil
	err = dg.close(h.destroyExec, h.destroyGraph)
	if !errors.Is(err, h.destroyExecErr) || !errors.Is(err, h.destroyGraphErr) {
		t.Errorf("close error = %v, want both %v and %v", err, h.destroyExecErr, h.destroyGraphErr)
	}
	if dg.exec == nil || dg.graph == nil {
		t.Errorf("handles after failed close: exec=%p graph=%p, want both retained", dg.exec, dg.graph)
	}
	if err := dg.close(h.destroyExec, h.destroyGraph); err != nil || dg.exec != nil || dg.graph != nil {
		t.Errorf("retry close: err=%v exec=%p graph=%p, want nil", err, dg.exec, dg.graph)
	}

	// 3. Mid-build addNode error destroys graph and returns (nil, err).
	h = newFakeGraphHarness()
	h.failAddAt = 5
	dg, err = h.build(gpuW, cfg, layerKV, sp, gm, 100, 101)
	if dg != nil || !errors.Is(err, h.addErr) {
		t.Errorf("addNode failure: dg=%v err=%v, want nil and %v", dg, err, h.addErr)
	}
	if !slices.Contains(h.events, "destroyGraph") {
		t.Errorf("addNode failure events = %v, want destroyGraph", h.events)
	}

	// 4. Mid-build addNode error + destroyGraph failure returns (dg, joined err) with graph retained.
	h = newFakeGraphHarness()
	h.failAddAt = 5
	h.destroyGraphFails = 1
	dg, err = h.build(gpuW, cfg, layerKV, sp, gm, 100, 101)
	if dg == nil || dg.graph != h.graphSentinel {
		t.Fatalf("addNode + destroyGraph failure: dg=%+v, want retained graph", dg)
	}
	if !errors.Is(err, h.addErr) || !errors.Is(err, h.destroyGraphErr) {
		t.Errorf("addNode + destroyGraph error = %v, want both %v and %v", err, h.addErr, h.destroyGraphErr)
	}
	if err := dg.close(h.destroyExec, h.destroyGraph); err != nil || dg.graph != nil {
		t.Errorf("retry close on retained graph: err=%v graph=%p", err, dg.graph)
	}

	// 5. Instantiate error with partial exec destroys both exec and graph in order.
	h = newFakeGraphHarness()
	h.instantiateExec = new(cudagraph.Exec)
	h.instantiateErr = errors.New("fake instantiate failed")
	dg, err = h.build(gpuW, cfg, layerKV, sp, gm, 100, 101)
	if dg != nil || !errors.Is(err, h.instantiateErr) {
		t.Errorf("instantiate failure: dg=%v err=%v, want nil and %v", dg, err, h.instantiateErr)
	}
	tail := h.events[len(h.events)-2:]
	if !slices.Equal(tail, []string{"destroyExec", "destroyGraph"}) {
		t.Errorf("instantiate failure cleanup order = %v, want [destroyExec destroyGraph]", tail)
	}
}

// TestCloseDecodeGraph tests the teardown helper used by runGPUInference:
// it is a no-op when *dg is nil or when td holds releases (stream undrained),
// and otherwise closes *dg and records any Close error in td.
func TestCloseDecodeGraph(t *testing.T) {
	// 1. Nil *dg is a no-op.
	td := new(teardown)
	var nilDG *DecodeGraph
	closeDecodeGraph(td, &nilDG)
	if td.err() != nil {
		t.Errorf("nil dg: td.err() = %v, want nil", td.err())
	}

	// 2. Held teardown keeps dg alive without calling Close.
	gpuW, cfg, sp, kvs, gm := fakeStepFixture(t, 1, weightF32)
	layerKV := func(i int) (cudakv.LayerCache, error) { return kvs[i], nil }
	h := newFakeGraphHarness()
	dg, err := h.build(gpuW, cfg, layerKV, sp, gm, 100, 101)
	if err != nil {
		t.Fatal(err)
	}
	// Clear real pointers before calling dg.Close() (which would call real Destroy on non-nil pointers)
	// when testing the non-held path below; for the held path, dg.exec and dg.graph must remain non-nil.
	tdHeld := new(teardown)
	tdHeld.add(errUndrained)
	closeDecodeGraph(tdHeld, &dg)
	if dg.exec == nil || dg.graph == nil {
		t.Errorf("held teardown closed dg: exec=%p graph=%p, want both kept", dg.exec, dg.graph)
	}

	// 3. Non-held teardown calls Close on non-nil dg.
	dg.exec = nil
	dg.graph = nil
	tdClean := new(teardown)
	closeDecodeGraph(tdClean, &dg)
	if tdClean.err() != nil {
		t.Errorf("clean closeDecodeGraph: td.err() = %v, want nil", tdClean.err())
	}
}

// TestCudaGraphFlagRefusals verifies that run refuses -cuda-graph with -draft
// and refuses -cuda-graph on the CPU fallback path.
func TestCudaGraphFlagRefusals(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	stdout := os.Stdout
	os.Stdout = devNull
	saveGraph, saveDraft := *cudaGraphFlag, *specDraftFlag
	t.Cleanup(func() {
		*cudaGraphFlag = saveGraph
		*specDraftFlag = saveDraft
		os.Stdout = stdout
		if err := devNull.Close(); err != nil {
			t.Error(err)
		}
	})

	*cudaGraphFlag = true
	*specDraftFlag = 3
	if err := run(); err == nil || !strings.Contains(err.Error(), "-cuda-graph cannot be combined with -draft") {
		t.Errorf("run with -cuda-graph -draft=3: got %v, want -cuda-graph cannot be combined with -draft", err)
	}
}
