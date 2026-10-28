// SPDX-License-Identifier: Apache-2.0

package cudaattn_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var errTreeNativeGateClosed error = treeNativeGateClosedError{}

type treeNativeGateClosedError struct{}

func (treeNativeGateClosedError) Error() string {
	return "native qualification pending: CUDAATTN_TREE_NATIVE=1 required"
}

func (treeNativeGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

func admitTreeNativeGate(getenv func(string) string) (cudalock.Owner, int, error) {
	if getenv == nil || getenv("CUDAATTN_TREE_NATIVE") != "1" {
		return cudalock.Owner{}, 0, errTreeNativeGateClosed
	}
	owner, err := cudalock.Inherited(getenv)
	if err != nil {
		return cudalock.Owner{}, 0, fmt.Errorf("cudaattn: native gate enabled but %w: run under gpu-lock", err)
	}
	ordinal := 0
	if raw := strings.TrimSpace(getenv("CUDAATTN_TREE_DEVICE")); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 0 {
			return cudalock.Owner{}, 0, fmt.Errorf("cudaattn: invalid CUDAATTN_TREE_DEVICE=%q", raw)
		}
		ordinal = parsed
	}
	return owner, ordinal, nil
}

func finiteWithinTol(got, want, tol float32) bool {
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) ||
		math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) ||
		math.IsNaN(float64(tol)) || math.IsInf(float64(tol), 0) || tol < 0 {
		return false
	}
	return math.Abs(float64(got-want)) <= float64(tol)
}

// treeNativeCleanupOwners retains io.Closer cleanup handles returned inside
// nvrtcx.Compile or CompiledModule.Load errors and closes them in LIFO order
// before dependent device resources are torn down.
type treeNativeCleanupOwners struct {
	owners []io.Closer
}

func (r *treeNativeCleanupOwners) adoptError(err error) error {
	var owner io.Closer
	if errors.As(err, &owner) && owner != nil {
		r.owners = append(r.owners, owner)
	}
	return err
}

func (r *treeNativeCleanupOwners) closeBefore(dependents func() error) error {
	for len(r.owners) > 0 {
		last := len(r.owners) - 1
		if err := r.owners[last].Close(); err != nil {
			return fmt.Errorf("cudaattn: close retained compiler/module owner: %w", err)
		}
		r.owners = r.owners[:last]
	}
	if dependents == nil {
		return nil
	}
	return dependents()
}

type treeSyntheticCleanupError struct {
	cause error
	calls int
	fail  bool
}

func (e *treeSyntheticCleanupError) Error() string { return e.cause.Error() }
func (e *treeSyntheticCleanupError) Unwrap() error { return e.cause }
func (e *treeSyntheticCleanupError) Close() error {
	e.calls++
	if e.fail {
		return errors.New("synthetic owner cleanup still failing")
	}
	return nil
}

// treeNativeFixture holds a deterministic 7-node candidate tree with two
// speculative branches (branch 1: nodes 0->1->3->5; branch 2: nodes 0->2->4)
// and one fully masked rejected tail node (node 6, bitmask 0).
type treeNativeFixture struct {
	numNodes    int
	numHeads    int
	headDim     int
	scale       float32
	bitmasks    []uint64
	branchNodes map[int][]int
	q, k, v     []float32
	want        []float32
}

func newTreeNativeFixture(t *testing.T) treeNativeFixture {
	t.Helper()
	const (
		numNodes = 7
		numHeads = 2
		headDim  = 6
	)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	// Node topology:
	//   0: root (attends to {0})
	//   1: branch 1 depth 1 (attends to {0, 1})
	//   2: branch 2 depth 1 (attends to {0, 2})
	//   3: branch 1 depth 2 (attends to {0, 1, 3})
	//   4: branch 2 depth 2 (attends to {0, 2, 4})
	//   5: branch 1 leaf    (attends to {0, 1, 3, 5})
	//   6: rejected/masked  (attends to {})
	bitmasks := []uint64{
		0: (1 << 0),
		1: (1 << 0) | (1 << 1),
		2: (1 << 0) | (1 << 2),
		3: (1 << 0) | (1 << 1) | (1 << 3),
		4: (1 << 0) | (1 << 2) | (1 << 4),
		5: (1 << 0) | (1 << 1) | (1 << 3) | (1 << 5),
		6: 0,
	}
	branchNodes := map[int][]int{
		0: {0},
		1: {0, 1, 3, 5},
		2: {0, 2, 4},
	}

	total := numNodes * numHeads * headDim
	q := make([]float32, total)
	k := make([]float32, total)
	v := make([]float32, total)
	for i := 0; i < total; i++ {
		q[i] = float32((i*17+3)%29-14) / 7.0
		k[i] = float32((i*13+5)%31-15) / 9.0
		v[i] = float32((i*11+7)%23-11) / 5.0
	}

	want, err := cudaattn.TreeAttentionReference(q, k, v, bitmasks, numNodes, numHeads, headDim, scale)
	if err != nil {
		t.Fatalf("TreeAttentionReference: %v", err)
	}
	return treeNativeFixture{
		numNodes:    numNodes,
		numHeads:    numHeads,
		headDim:     headDim,
		scale:       scale,
		bitmasks:    bitmasks,
		branchNodes: branchNodes,
		q:           q,
		k:           k,
		v:           v,
		want:        want,
	}
}

func (f treeNativeFixture) denseOracle() []float32 {
	denseMask := cudaattn.TreeDenseReferenceMask(f.bitmasks, f.numNodes)
	out := make([]float32, len(f.want))
	for node := 0; node < f.numNodes; node++ {
		for head := 0; head < f.numHeads; head++ {
			qBase := (node*f.numHeads + head) * f.headDim
			scores := make([]float64, f.numNodes)
			maxScore := math.Inf(-1)
			valid := false
			for anc := 0; anc < f.numNodes; anc++ {
				if !denseMask[node][anc] {
					scores[anc] = math.Inf(-1)
					continue
				}
				kBase := (anc*f.numHeads + head) * f.headDim
				var dot float64
				for d := 0; d < f.headDim; d++ {
					dot += float64(f.q[qBase+d]) * float64(f.k[kBase+d])
				}
				dot *= float64(f.scale)
				scores[anc] = dot
				if !valid || dot > maxScore {
					maxScore = dot
					valid = true
				}
			}
			if !valid {
				continue
			}
			var sumExp float64
			weights := make([]float64, f.numNodes)
			for anc := 0; anc < f.numNodes; anc++ {
				if denseMask[node][anc] {
					w := math.Exp(scores[anc] - maxScore)
					weights[anc] = w
					sumExp += w
				}
			}
			for d := 0; d < f.headDim; d++ {
				var acc float64
				for anc := 0; anc < f.numNodes; anc++ {
					if denseMask[node][anc] {
						vBase := (anc*f.numHeads + head) * f.headDim
						acc += weights[anc] * float64(f.v[vBase+d])
					}
				}
				out[qBase+d] = float32(acc / sumExp)
			}
		}
	}
	return out
}

func (f treeNativeFixture) verifyOutput(got []float32, tol float32) error {
	if len(got) != len(f.want) {
		return fmt.Errorf("output length = %d, want %d", len(got), len(f.want))
	}
	// Fully masked row (node 6) must be exact finite zero without NaN or Inf.
	maskedStart := 6 * f.numHeads * f.headDim
	maskedEnd := 7 * f.numHeads * f.headDim
	for i := maskedStart; i < maskedEnd; i++ {
		if !finiteWithinTol(got[i], 0, 0) || got[i] != 0 {
			return fmt.Errorf("fully masked node 6 element %d = %v, want exact 0", i-maskedStart, got[i])
		}
	}
	for i := range f.want {
		if !finiteWithinTol(got[i], f.want[i], tol) {
			return fmt.Errorf("output[%d] = %v, want %v (tol %v)", i, got[i], f.want[i], tol)
		}
	}
	return nil
}

// TestTreeNativeFixtureContract verifies the tree-attention native fixture,
// dense-vs-bitmask equivalence, gate admission, cleanup owner retention, and
// negative controls (including NaN/Inf poison rejection) on CPU.
func TestTreeNativeFixtureContract(t *testing.T) {
	checkUnloaded := func() {
		t.Helper()
		if driver.Loaded() || nvrtc.Loaded() {
			t.Fatal("CPU fixture contract test loaded an NVIDIA library")
		}
	}
	checkUnloaded()
	defer checkUnloaded()

	f := newTreeNativeFixture(t)
	oracle := f.denseOracle()
	if err := f.verifyOutput(oracle, 1e-6); err != nil {
		t.Fatalf("dense oracle mismatch against TreeAttentionReference: %v", err)
	}

	// Verify branch isolation in dense reference mask: branch 1 leaf (node 5)
	// never attends to branch 2 nodes (2, 4), and vice versa.
	denseMask := cudaattn.TreeDenseReferenceMask(f.bitmasks, f.numNodes)
	for _, b1Node := range f.branchNodes[1][1:] {
		for _, b2Node := range f.branchNodes[2][1:] {
			if denseMask[b1Node][b2Node] || denseMask[b2Node][b1Node] {
				t.Fatalf("cross-branch attention leaked between node %d and node %d", b1Node, b2Node)
			}
		}
	}

	// Verify shared memory sizing and kernel shader contract.
	for _, blockSize := range []int{1, 32, 64, 128} {
		gotBytes := cudaattn.TreeAttentionSharedBytes(f.numNodes, blockSize)
		wantBytes := 4 * (f.numNodes + blockSize)
		if gotBytes != wantBytes {
			t.Fatalf("TreeAttentionSharedBytes(%d, %d) = %d, want %d", f.numNodes, blockSize, gotBytes, wantBytes)
		}
	}
	shader := cudaattn.GenerateTreeAttentionCUDAShader()
	if !strings.Contains(shader, "TreeFlashAttentionCUDA") || !strings.Contains(shader, "ancestor_mask") {
		t.Fatal("tree attention CUDA shader missing expected entry point or mask logic")
	}

	t.Run("gate_admission", func(t *testing.T) {
		if _, _, err := admitTreeNativeGate(func(string) string { return "" }); !errors.Is(err, errTreeNativeGateClosed) || !errors.Is(err, cudaerrors.ErrHardwarePending) {
			t.Fatalf("closed gate err = %v, want errTreeNativeGateClosed wrapping ErrHardwarePending", err)
		}
		if _, _, err := admitTreeNativeGate(func(k string) string {
			if k == "CUDAATTN_TREE_NATIVE" {
				return "1"
			}
			return ""
		}); !errors.Is(err, cudalock.ErrNotInherited) {
			t.Fatalf("open gate without GPU_LOCK_HELD err = %v, want cudalock.ErrNotInherited", err)
		}
		if _, _, err := admitTreeNativeGate(func(k string) string {
			switch k {
			case "CUDAATTN_TREE_NATIVE":
				return "1"
			case "GPU_LOCK_HELD":
				return "1"
			}
			return ""
		}); !errors.Is(err, cudalock.ErrNotInherited) {
			t.Fatalf("open gate with literal GPU_LOCK_HELD=1 err = %v, want cudalock.ErrNotInherited", err)
		}
	})

	t.Run("cleanup_owner_retention", func(t *testing.T) {
		for _, stage := range []string{"compile-program", "module-load"} {
			cause := errors.New(stage)
			owner := &treeSyntheticCleanupError{cause: cause, fail: true}
			wrapped := fmt.Errorf("tree native %s: %w", stage, owner)
			var retained treeNativeCleanupOwners
			if got := retained.adoptError(wrapped); got != wrapped || !errors.Is(got, cause) {
				t.Fatalf("%s: adoptError replaced original error: %v", stage, got)
			}
			dependentReleased := false
			dependents := func() error {
				dependentReleased = true
				return nil
			}
			if err := retained.closeBefore(dependents); err == nil || owner.calls != 1 || dependentReleased {
				t.Fatalf("%s: failed owner close did not block dependent teardown: err=%v calls=%d released=%v",
					stage, err, owner.calls, dependentReleased)
			}
			owner.fail = false
			if err := retained.closeBefore(dependents); err != nil || owner.calls != 2 || !dependentReleased {
				t.Fatalf("%s: retry owner close failed: err=%v calls=%d released=%v",
					stage, err, owner.calls, dependentReleased)
			}
		}
	})

	t.Run("nan_and_inf_poison_rejected", func(t *testing.T) {
		allNaN := make([]float32, len(f.want))
		for i := range allNaN {
			allNaN[i] = float32(math.NaN())
		}
		if err := f.verifyOutput(allNaN, 1e-5); err == nil {
			t.Fatal("unchanged all-NaN poison buffer was accepted")
		}
		for _, tc := range []struct {
			name string
			idx  int
			val  float32
		}{
			{"active_node_nan", 0, float32(math.NaN())},
			{"leaf_node_nan", 5*f.numHeads*f.headDim + 2, float32(math.NaN())},
			{"masked_node_nan", 6 * f.numHeads * f.headDim, float32(math.NaN())},
			{"active_node_pos_inf", 1, float32(math.Inf(1))},
			{"active_node_neg_inf", 2, float32(math.Inf(-1))},
			{"masked_node_pos_inf", 6*f.numHeads*f.headDim + 1, float32(math.Inf(1))},
			{"masked_node_neg_inf", 6*f.numHeads*f.headDim + 2, float32(math.Inf(-1))},
		} {
			poisoned := slices.Clone(f.want)
			poisoned[tc.idx] = tc.val
			if err := f.verifyOutput(poisoned, 1e-5); err == nil {
				t.Fatalf("%s at index %d (%v) was accepted", tc.name, tc.idx, tc.val)
			}
		}
		badRef := f
		badRef.want = slices.Clone(f.want)
		badRef.want[0] = float32(math.NaN())
		if err := badRef.verifyOutput(f.want, 1e-5); err == nil {
			t.Fatal("NaN reference value in verifyOutput was accepted")
		}
	})

	t.Run("swapped_branch_mask_rejected", func(t *testing.T) {
		mutMasks := slices.Clone(f.bitmasks)
		mutMasks[4], mutMasks[5] = mutMasks[5], mutMasks[4]
		mutOut, err := cudaattn.TreeAttentionReference(f.q, f.k, f.v, mutMasks, f.numNodes, f.numHeads, f.headDim, f.scale)
		if err != nil {
			t.Fatalf("TreeAttentionReference: %v", err)
		}
		if err := f.verifyOutput(mutOut, 1e-5); err == nil {
			t.Fatal("swapping branch 1 and branch 2 leaf bitmasks was accepted")
		}
	})

	t.Run("cross_branch_contamination_rejected", func(t *testing.T) {
		mutMasks := slices.Clone(f.bitmasks)
		mutMasks[3] |= (1 << 2) // Contaminate branch 1 node 3 with sibling branch 2 node 2.
		mutOut, err := cudaattn.TreeAttentionReference(f.q, f.k, f.v, mutMasks, f.numNodes, f.numHeads, f.headDim, f.scale)
		if err != nil {
			t.Fatalf("TreeAttentionReference: %v", err)
		}
		if err := f.verifyOutput(mutOut, 1e-5); err == nil {
			t.Fatal("cross-branch bitmask contamination was accepted")
		}
	})

	t.Run("unmasked_rejected_row_rejected", func(t *testing.T) {
		mutMasks := slices.Clone(f.bitmasks)
		mutMasks[6] = 1 // Make fully masked node 6 attend to root.
		mutOut, err := cudaattn.TreeAttentionReference(f.q, f.k, f.v, mutMasks, f.numNodes, f.numHeads, f.headDim, f.scale)
		if err != nil {
			t.Fatalf("TreeAttentionReference: %v", err)
		}
		if err := f.verifyOutput(mutOut, 1e-5); err == nil {
			t.Fatal("non-zero output on fully masked row was accepted")
		}
	})

	t.Run("negated_query_liveness", func(t *testing.T) {
		negQ := make([]float32, len(f.q))
		for i, v := range f.q {
			negQ[i] = -v
		}
		negOut, err := cudaattn.TreeAttentionReference(negQ, f.k, f.v, f.bitmasks, f.numNodes, f.numHeads, f.headDim, f.scale)
		if err != nil {
			t.Fatalf("TreeAttentionReference: %v", err)
		}
		if err := f.verifyOutput(negOut, 1e-5); err == nil {
			t.Fatal("negating Q left multi-ancestor tree attention output unchanged")
		}
	})

	t.Run("corrupted_output_rejected", func(t *testing.T) {
		corrupt := slices.Clone(f.want)
		corrupt[5*f.numHeads*f.headDim+f.headDim-1] += 0.01
		if err := f.verifyOutput(corrupt, 1e-5); err == nil {
			t.Fatal("corrupted leaf output was accepted")
		}
	})

	t.Run("invalid_shape_refused", func(t *testing.T) {
		if _, err := cudaattn.TreeAttentionReference(f.q[:len(f.q)-1], f.k, f.v, f.bitmasks, f.numNodes, f.numHeads, f.headDim, f.scale); !errors.Is(err, cudaattn.ErrTreeAttentionShape) {
			t.Fatalf("truncated q err = %v, want ErrTreeAttentionShape", err)
		}
		if _, err := cudaattn.TreeAttentionReference(f.q, f.k, f.v, f.bitmasks[:f.numNodes-1], f.numNodes, f.numHeads, f.headDim, f.scale); !errors.Is(err, cudaattn.ErrTreeAttentionShape) {
			t.Fatalf("truncated bitmasks err = %v, want ErrTreeAttentionShape", err)
		}
	})
}

// TestNativeTreeAttentionReference compiles and executes TreeFlashAttentionCUDA
// on an admitted GPU under gpu-lock and verifies elementwise parity with
// TreeAttentionReference across multiple block widths, plus device mutant refusal.
func TestNativeTreeAttentionReference(t *testing.T) {
	owner, ordinal, err := admitTreeNativeGate(os.Getenv)
	if errors.Is(err, errTreeNativeGateClosed) {
		t.Skip(err.Error())
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := driver.Load(""); err != nil {
		t.Fatalf("load CUDA driver: %v", err)
	}
	if err := cuda.Init(); err != nil {
		t.Fatalf("initialize CUDA: %v", err)
	}
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatalf("open CUDA device %d: %v", ordinal, err)
	}
	var (
		retained treeNativeCleanupOwners
		stream   *cuda.Stream
		mod      driver.Module
		floatBuf []cudamem.Slice[float32]
		maskBuf  []cudamem.Slice[uint64]
	)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if stream != nil {
			if err := stream.Synchronize(ctx); err != nil {
				t.Errorf("retain native tree-attention resources after stream drain failure: %v", err)
				return
			}
		}
		if err := retained.closeBefore(func() error {
			if mod != 0 {
				if err := dev.WithContext(func() error { return mod.Close() }); err != nil {
					return fmt.Errorf("close module: %w", err)
				}
				mod = 0
			}
			for _, b := range floatBuf {
				if err := b.Free(); err != nil {
					return fmt.Errorf("free float buffer: %w", err)
				}
			}
			for _, b := range maskBuf {
				if err := b.Free(); err != nil {
					return fmt.Errorf("free mask buffer: %w", err)
				}
			}
			if stream != nil {
				if err := stream.Close(); err != nil {
					return fmt.Errorf("close stream: %w", err)
				}
				stream = nil
			}
			if err := dev.Close(); err != nil {
				return fmt.Errorf("close device: %w", err)
			}
			return nil
		}); err != nil {
			t.Errorf("native tree-attention cleanup failed; retaining remaining resources: %v", err)
		}
	})

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatalf("compute capability: %v", err)
	}
	compiled, err := nvrtcx.Compile(
		cudaattn.GenerateTreeAttentionCUDAShader(),
		nvrtcx.WithName("tree_flash_attn.cu"),
		nvrtcx.WithPhysicalArch(major, minor),
	)
	if err != nil {
		_ = retained.adoptError(err)
		t.Fatalf("compile TreeFlashAttentionCUDA for sm_%d%d: %v", major, minor, err)
	}
	mod, err = compiled.Load(dev)
	if err != nil {
		_ = retained.adoptError(err)
		t.Fatalf("load TreeFlashAttentionCUDA module: %v", err)
	}
	fn, err := mod.Function("TreeFlashAttentionCUDA")
	if err != nil {
		t.Fatalf("lookup TreeFlashAttentionCUDA: %v", err)
	}
	stream, err = dev.NewStream()
	if err != nil {
		t.Fatalf("create stream: %v", err)
	}

	f := newTreeNativeFixture(t)
	allocFloats := func(vals []float32) cudamem.Slice[float32] {
		s, err := cudamem.Alloc[float32](dev, len(vals))
		if s.Ptr() != 0 {
			floatBuf = append(floatBuf, s)
		}
		if err != nil {
			t.Fatalf("alloc float32[%d]: %v", len(vals), err)
		}
		if err := dev.WithContext(func() error { return s.CopyFromHost(vals) }); err != nil {
			t.Fatalf("copy float32[%d] to device: %v", len(vals), err)
		}
		return s
	}
	allocMasks := func(vals []uint64) cudamem.Slice[uint64] {
		s, err := cudamem.Alloc[uint64](dev, len(vals))
		if s.Ptr() != 0 {
			maskBuf = append(maskBuf, s)
		}
		if err != nil {
			t.Fatalf("alloc uint64[%d]: %v", len(vals), err)
		}
		if err := dev.WithContext(func() error { return s.CopyFromHost(vals) }); err != nil {
			t.Fatalf("copy uint64[%d] to device: %v", len(vals), err)
		}
		return s
	}

	dq := allocFloats(f.q)
	dk := allocFloats(f.k)
	dv := allocFloats(f.v)
	poison := make([]float32, len(f.want))
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	dout := allocFloats(poison)
	dmasks := allocMasks(f.bitmasks)

	for _, blockSize := range []uint32{1, 32, 64} {
		if err := dev.WithContext(func() error { return dout.CopyFromHost(poison) }); err != nil {
			t.Fatalf("poison output: %v", err)
		}
		shared := uint32(cudaattn.TreeAttentionSharedBytes(f.numNodes, int(blockSize)))
		if err := cudalaunch.Kernel(fn).
			Grid(uint32(f.numNodes), uint32(f.numHeads), 1).
			Block(blockSize, 1, 1).
			SharedMem(shared).
			On(stream).
			Launch(dq, dk, dv, dout, dmasks, int32(f.numNodes), int32(f.numHeads), int32(f.headDim), f.scale); err != nil {
			t.Fatalf("launch block=%d: %v", blockSize, err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		err := stream.Synchronize(ctx)
		cancel()
		if err != nil {
			t.Fatalf("synchronize block=%d: %v", blockSize, err)
		}
		got := make([]float32, len(f.want))
		if err := dev.WithContext(func() error { return dout.CopyToHost(got) }); err != nil {
			t.Fatalf("copy output block=%d: %v", blockSize, err)
		}
		if err := f.verifyOutput(got, 1e-5); err != nil {
			t.Fatalf("block=%d verification failed: %v", blockSize, err)
		}
	}

	// Device mutant: swap branch 1 and branch 2 leaf masks on the device and
	// confirm the device result diverges from the baseline fixture.
	mutMasks := slices.Clone(f.bitmasks)
	mutMasks[4], mutMasks[5] = mutMasks[5], mutMasks[4]
	dmutMasks := allocMasks(mutMasks)
	if err := dev.WithContext(func() error { return dout.CopyFromHost(poison) }); err != nil {
		t.Fatalf("poison mutant output: %v", err)
	}
	shared := uint32(cudaattn.TreeAttentionSharedBytes(f.numNodes, 32))
	if err := cudalaunch.Kernel(fn).
		Grid(uint32(f.numNodes), uint32(f.numHeads), 1).
		Block(32, 1, 1).
		SharedMem(shared).
		On(stream).
		Launch(dq, dk, dv, dout, dmutMasks, int32(f.numNodes), int32(f.numHeads), int32(f.headDim), f.scale); err != nil {
		t.Fatalf("launch mutant: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	err = stream.Synchronize(ctx)
	cancel()
	if err != nil {
		t.Fatalf("synchronize mutant: %v", err)
	}
	mutGot := make([]float32, len(f.want))
	if err := dev.WithContext(func() error { return dout.CopyToHost(mutGot) }); err != nil {
		t.Fatalf("copy mutant output: %v", err)
	}
	if err := f.verifyOutput(mutGot, 1e-5); err == nil {
		t.Fatal("device mutant with swapped branch masks matched baseline reference")
	}
	t.Logf("native tree attention qualified: device=%d sm_%d%d lease_pid=%d nodes=%d heads=%d dim=%d",
		ordinal, major, minor, owner.PID, f.numNodes, f.numHeads, f.headDim)
}
