package main

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// posNode records one position-dependent kernel node in a DecodeGraph
// together with its base launch parameters, a mutable copy of its argument
// list, and the reusable packed-argument buffer for SetPosition.
type posNode struct {
	layer  int
	kind   string // "rope", "store_kv", or "attn"
	posIdx int
	node   *cudagraph.Node
	params driver.KernelNodeParams
	args   []any
	packed []unsafe.Pointer
}

// DecodeGraph is an instantiated single-token CUDA graph for the
// autoregressive decode step. It reads the input token ID directly from
// sp.TokenOut on the device (written by the preceding prefill or decode
// step's ArgmaxLogits kernel) and updates the 3*NumHiddenLayers
// position-dependent kernel nodes in place before each launch.
type DecodeGraph struct {
	graph       *cudagraph.Graph
	exec        *cudagraph.Exec
	sp          *GPUScratchpad
	maxSeq      int
	posNodes    []posNode
	syncArbiter *cudacapture.SyncArbiter
	pingPong    *cudagraph.PingPongTimer
	eventExec   driver.GraphExec
	eventNodes  []cudagraph.EventRecordNode
	setParams   func(node *cudagraph.Node, params *driver.KernelNodeParams) error
	launchExec  func(stream *cuda.Stream) error
	syncCtx     func() error
	syncStream  func(stream *cuda.Stream) error
	copyOut     func(stream *cuda.Stream) error
}

// NewDecodeGraph constructs and instantiates a single-token decode CUDA
// graph for gpuW, cfg, kv, sp, gm, attnFn and sampleFn on dev.
func NewDecodeGraph(
	dev *cuda.Device,
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	kv *cudakv.Cache,
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
) (*DecodeGraph, error) {
	return newDecodeGraphWith(
		gpuW, cfg, kv.Layer, sp, gm, attnFn, sampleFn,
		func(shared uint32) error {
			return cudalaunch.Kernel(attnFn).SharedMem(shared).PrepareSharedMem(dev)
		},
		func() (*cudagraph.Graph, error) { return cudagraph.New(dev) },
		func(g *cudagraph.Graph, l *cudalaunch.Launcher, args []any, deps ...*cudagraph.Node) (*cudagraph.Node, error) {
			return g.AddLauncher(l, args, deps...)
		},
		(*cudagraph.Graph).Instantiate,
		func(e *cudagraph.Exec, node *cudagraph.Node, params *driver.KernelNodeParams) error {
			return e.SetKernelNodeParams(node, params)
		},
		func(e *cudagraph.Exec, stream *cuda.Stream) error {
			return e.Launch(stream)
		},
		func() error { return driver.CtxSynchronize() },
		func(stream *cuda.Stream) error { return stream.Synchronize(context.Background()) },
		func(stream *cuda.Stream) error {
			if stream != nil {
				return sp.TokenOut.CopyToHostAsync(sp.HostTokenOut, stream)
			}
			return sp.TokenOut.CopyToHost(sp.HostTokenOut)
		},
		func(e *cudagraph.Exec) error { return e.Destroy() },
		func(g *cudagraph.Graph) error { return g.Destroy() },
	)
}

func newDecodeGraphWith(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	layerKV func(int) (cudakv.LayerCache, error),
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
	prepareAttn func(shared uint32) error,
	newGraph func() (*cudagraph.Graph, error),
	addNode func(g *cudagraph.Graph, l *cudalaunch.Launcher, args []any, deps ...*cudagraph.Node) (*cudagraph.Node, error),
	instantiate func(g *cudagraph.Graph) (*cudagraph.Exec, error),
	setParams func(e *cudagraph.Exec, node *cudagraph.Node, params *driver.KernelNodeParams) error,
	launchExec func(e *cudagraph.Exec, stream *cuda.Stream) error,
	syncCtx func() error,
	syncStream func(stream *cuda.Stream) error,
	copyOut func(stream *cuda.Stream) error,
	destroyExec func(e *cudagraph.Exec) error,
	destroyGraph func(g *cudagraph.Graph) error,
) (res *DecodeGraph, err error) {
	if *fuseEGraphFlag && !fusedKernelsQualified {
		return nil, fmt.Errorf("-fuse-egraph: fused kernels are not qualified on a device; run without it")
	}
	if *fuseDecodeFlag && !fusedKernelsQualified {
		return nil, fmt.Errorf("-fuse-decode: fused kernels are not qualified on a device; run without it")
	}
	if err := gpuW.checkDTypes(); err != nil {
		return nil, err
	}
	if cfg.MaxSeqLen <= 0 {
		return nil, fmt.Errorf("maxSeqLen %d must be > 0", cfg.MaxSeqLen)
	}
	attnShared, err := cudaattn.DecodeSharedMemoryBytes(cfg.MaxSeqLen, attnThreads)
	if err != nil {
		return nil, fmt.Errorf("attention shared memory for %d tokens: %w", cfg.MaxSeqLen, err)
	}
	if err := prepareAttn(uint32(attnShared)); err != nil {
		return nil, fmt.Errorf("prepare attention shared memory (%d bytes): %w", attnShared, err)
	}
	g, err := newGraph()
	if err != nil {
		return nil, fmt.Errorf("create decode graph: %w", err)
	}
	dg := &DecodeGraph{
		graph:      g,
		sp:         sp,
		maxSeq:     cfg.MaxSeqLen,
		posNodes:   make([]posNode, 0, 3*cfg.NumHiddenLayers),
		syncCtx:    syncCtx,
		syncStream: syncStream,
		copyOut:    copyOut,
	}
	defer func() {
		if err != nil {
			if cerr := dg.close(destroyExec, destroyGraph); cerr != nil {
				err = errors.Join(err, cerr)
				res = dg
			}
		}
	}()

	var prev []*cudagraph.Node
	qkvRoPEStoreLayer, ropeLayer, storeLayer, attnLayer := 0, 0, 0, 0
	if err = emitForwardStep(gpuW, cfg, layerKV, sp, gm, attnFn, sampleFn, sp.TokenOut, 0, uint32(attnShared), func(l *cudalaunch.Launcher, args ...any) error {
		desc, derr := l.Describe()
		if derr != nil {
			return derr
		}
		node, nerr := addNode(g, l, args, prev...)
		if nerr != nil {
			return nerr
		}
		prev = []*cudagraph.Node{node}
		posIdx := -1
		layer := 0
		kind := ""
		switch desc.Function {
		case gm.fusedQKVRoPEStoreFn, gm.fusedQKVRoPEStoreBF16Fn:
			if desc.Function != 0 {
				posIdx = 15
				layer = qkvRoPEStoreLayer
				qkvRoPEStoreLayer++
				kind = "qkv_rope_store"
			}
		case gm.ropeFn:
			posIdx = 2
			layer = ropeLayer
			ropeLayer++
			kind = "rope"
		case gm.ropeScaledFn:
			if gm.ropeScaledFn != 0 {
				posIdx = 2
				layer = ropeLayer
				ropeLayer++
				kind = "rope"
			}
		case gm.storeKVFn:
			posIdx = 4
			layer = storeLayer
			storeLayer++
			kind = "store_kv"
		case attnFn:
			posIdx = 4
			layer = attnLayer
			attnLayer++
			kind = "attn"
		}
		if posIdx >= 0 {
			dg.posNodes = append(dg.posNodes, posNode{
				layer:  layer,
				kind:   kind,
				posIdx: posIdx,
				node:   node,
				params: driver.KernelNodeParams{
					Func:           desc.Function,
					GridDim:        driver.Dim3(desc.Grid),
					BlockDim:       driver.Dim3(desc.Block),
					SharedMemBytes: desc.SharedMem,
				},
				args: slices.Clone(args),
			})
		}
		return nil
	}); err != nil {
		return nil, fmt.Errorf("populate decode graph: %w", err)
	}
	wantPosNodes := 3 * cfg.NumHiddenLayers
	if qkvRoPEStoreLayer == cfg.NumHiddenLayers {
		wantPosNodes = 2 * cfg.NumHiddenLayers
	}
	if len(dg.posNodes) != wantPosNodes {
		err = fmt.Errorf("decode graph recorded %d position nodes, want %d", len(dg.posNodes), wantPosNodes)
		return nil, err
	}
	dg.exec, err = instantiate(g)
	if err != nil {
		return nil, fmt.Errorf("instantiate decode graph: %w", err)
	}
	dg.setParams = func(node *cudagraph.Node, params *driver.KernelNodeParams) error {
		return setParams(dg.exec, node, params)
	}
	dg.launchExec = func(stream *cuda.Stream) error {
		return launchExec(dg.exec, stream)
	}
	return dg, nil
}

// SetPosition updates the position argument on all 3*NumHiddenLayers
// position-dependent kernel nodes (ApplyRoPE, StoreKVCache, and
// CausalFlashAttentionDecode) in the instantiated graph executable.
func (dg *DecodeGraph) SetPosition(pos int) error {
	if err := checkPos(pos, dg.maxSeq); err != nil {
		return err
	}
	for i := range dg.posNodes {
		if err := dg.setNodePosition(&dg.posNodes[i], pos); err != nil {
			return err
		}
	}
	return nil
}

func (dg *DecodeGraph) setNodePosition(p *posNode, pos int) error {
	p.args[p.posIdx] = int32(pos)
	packed, err := cudalaunch.PackArgs(p.args...)
	if err != nil {
		return fmt.Errorf("layer %d %s pack args at position %d: %w", p.layer, p.kind, pos, err)
	}
	p.packed = packed
	if len(packed) > 0 {
		p.params.KernelParams = uintptr(unsafe.Pointer(&p.packed[0]))
	} else {
		p.params.KernelParams = 0
	}
	p.params.Extra = 0
	defer runtime.KeepAlive(p.packed)
	if err := dg.setParams(p.node, &p.params); err != nil {
		return fmt.Errorf("layer %d %s set position %d: %w", p.layer, p.kind, pos, err)
	}
	return nil
}

// Enqueue updates the graph's position-dependent kernel nodes to pos,
// launches the graph executable on stream, and copies the sampled output
// token into sp.HostTokenOut. When stream is non-nil, the launch and D2H
// copy remain pending on stream and the caller must synchronize stream
// before reading sp.HostTokenOut[0]. When stream is nil, Enqueue
// synchronizes the context and returns sp.HostTokenOut[0].
func (dg *DecodeGraph) Enqueue(pos int, stream *cuda.Stream) (int32, error) {
	if stream == nil && dg.syncArbiter != nil {
		if err := dg.syncArbiter.CheckSync("DecodeGraph.Enqueue"); err != nil {
			return 0, err
		}
	}
	if err := dg.SetPosition(pos); err != nil {
		return 0, err
	}
	if dg.pingPong != nil {
		if _, err := dg.pingPong.PrepareReplay(dg.eventExec, dg.eventNodes); err != nil {
			return 0, fmt.Errorf("prepare decode graph replay at position %d: %w", pos, err)
		}
	}
	if err := dg.launchExec(stream); err != nil {
		return 0, fmt.Errorf("launch decode graph at position %d: %w", pos, err)
	}
	if stream != nil {
		if err := dg.copyOut(stream); err != nil {
			return 0, err
		}
		return 0, nil
	}
	if err := dg.syncCtx(); err != nil {
		return 0, err
	}
	if err := dg.copyOut(nil); err != nil {
		return 0, err
	}
	return dg.sp.HostTokenOut[0], nil
}

// Step executes one decode step at position pos on stream and returns the
// sampled next token ID. When stream is non-nil, Step drains stream before
// returning, even if Enqueue returned an error.
func (dg *DecodeGraph) Step(pos int, stream *cuda.Stream) (int32, error) {
	if stream != nil && dg.syncArbiter != nil {
		if err := dg.syncArbiter.CheckSync("DecodeGraph.Step"); err != nil {
			return 0, err
		}
	}
	tok, err := dg.Enqueue(pos, stream)
	if stream == nil {
		return tok, err
	}
	if waitErr := dg.syncStream(stream); waitErr != nil {
		err = errors.Join(err, waitErr)
	}
	if err != nil {
		return 0, err
	}
	return dg.sp.HostTokenOut[0], nil
}

// Close destroys the instantiated graph executable and then the underlying
// graph, retaining either handle if its destruction fails so a retry can
// attempt it again.
func (dg *DecodeGraph) Close() error {
	return dg.close(
		func(e *cudagraph.Exec) error { return e.Destroy() },
		func(g *cudagraph.Graph) error { return g.Destroy() },
	)
}

func (dg *DecodeGraph) close(destroyExec func(*cudagraph.Exec) error, destroyGraph func(*cudagraph.Graph) error) error {
	var err error
	if dg.exec != nil {
		if eerr := destroyExec(dg.exec); eerr != nil {
			err = errors.Join(err, fmt.Errorf("destroy decode graph exec: %w", eerr))
		} else {
			dg.exec = nil
		}
	}
	if dg.graph != nil {
		if gerr := destroyGraph(dg.graph); gerr != nil {
			err = errors.Join(err, fmt.Errorf("destroy decode graph: %w", gerr))
		} else {
			dg.graph = nil
		}
	}
	return err
}

// closeDecodeGraph closes *dg when non-nil and td has not held releases,
// recording any error in td.
func closeDecodeGraph(td *teardown, dg **DecodeGraph) {
	if *dg == nil || td.held() {
		return
	}
	if cerr := (*dg).Close(); cerr != nil {
		td.add(fmt.Errorf("close decode graph: %w", cerr))
	}
}
