// SPDX-License-Identifier: Apache-2.0

package cudaspec_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudamtp"
	"github.com/tmc/lib/nvidia/x/cudaspec"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var errSpecNativeGateClosed error = specNativeGateClosedError{}

type specNativeGateClosedError struct{}

func (specNativeGateClosedError) Error() string {
	return "native qualification pending: CUDASPEC_NATIVE=1 required"
}

func (specNativeGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

func admitSpecNativeGate(getenv func(string) string) (cudalock.Owner, int, error) {
	if getenv == nil || getenv("CUDASPEC_NATIVE") != "1" {
		return cudalock.Owner{}, 0, errSpecNativeGateClosed
	}
	owner, err := cudalock.Inherited(getenv)
	if err != nil {
		return cudalock.Owner{}, 0, fmt.Errorf("cudaspec: native gate enabled but %w: run under gpu-lock", err)
	}
	ordinal := 0
	if raw := strings.TrimSpace(getenv("CUDASPEC_DEVICE")); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 0 {
			return cudalock.Owner{}, 0, fmt.Errorf("cudaspec: invalid CUDASPEC_DEVICE=%q", raw)
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

// specNativeCleanupOwners retains io.Closer cleanup handles returned inside
// nvrtcx.Compile or CompiledModule.Load errors and closes them in LIFO order
// before dependent device resources are torn down.
type specNativeCleanupOwners struct {
	owners []io.Closer
}

func (r *specNativeCleanupOwners) adoptError(err error) error {
	var owner io.Closer
	if errors.As(err, &owner) && owner != nil {
		r.owners = append(r.owners, owner)
	}
	return err
}

func (r *specNativeCleanupOwners) closeBefore(dependents func() error) error {
	for len(r.owners) > 0 {
		last := len(r.owners) - 1
		if err := r.owners[last].Close(); err != nil {
			return fmt.Errorf("cudaspec: close retained compiler/module owner: %w", err)
		}
		r.owners = r.owners[:last]
	}
	if dependents == nil {
		return nil
	}
	return dependents()
}

type specSyntheticCleanupError struct {
	cause error
	calls int
	fail  bool
}

func (e *specSyntheticCleanupError) Error() string { return e.cause.Error() }
func (e *specSyntheticCleanupError) Unwrap() error { return e.cause }
func (e *specSyntheticCleanupError) Close() error {
	e.calls++
	if e.fail {
		return errors.New("synthetic owner cleanup still failing")
	}
	return nil
}

type specCycleEngine struct {
	calls           int
	foreignSeq      string
	stepErr         error
	completed       map[int64]bool
	produceNodeAttn func(*cudabatch.ExecutionStep) ([]float32, error)
	projectNodeAttn func(attnOut []float32, nodeIdx int) (int32, error)
}

func (e *specCycleEngine) ExecuteForward(_ context.Context, step *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
	idx := e.calls
	e.calls++
	var attnOut []float32
	if e.produceNodeAttn != nil {
		var err error
		attnOut, err = e.produceNodeAttn(step)
		if err != nil {
			return nil, nil, err
		}
	}
	if e.stepErr != nil {
		return nil, nil, e.stepErr
	}
	if step == nil || len(step.Metadata.RowBindings) != 1 {
		return nil, nil, errors.New("cudaspec: expected single verification row binding")
	}
	if e.projectNodeAttn == nil {
		return nil, nil, errors.New("cudaspec: missing node attention projection")
	}
	tok, err := e.projectNodeAttn(attnOut, idx)
	if err != nil {
		return nil, nil, err
	}
	seqID := step.Metadata.RowBindings[0].SeqID
	if e.foreignSeq != "" {
		seqID = e.foreignSeq
	}
	if e.completed != nil {
		e.completed[step.StepID] = true
	}
	return map[string]int32{seqID: tok}, nil, nil
}

type specNativeCycleFixture struct {
	allocator     *cudakv.BlockAllocator
	req           cudaspec.BatchCycleRequest
	tree          *cudaspec.DraftTree
	treeMask      [][]bool
	bitmasks      []uint64
	numNodes      int
	numHeads      int
	headDim       int
	scale         float32
	q, k          []float32
	vAccept       []float32
	vReject       []float32
	wantAttn      []float32
	wantAttnRej   []float32
	wantLogits    []float32
	wantDrafts    []int32
	targetReadout []float32
}

func newSpecNativeCycleFixture(t *testing.T) specNativeCycleFixture {
	t.Helper()
	allocator := cudakv.NewBlockAllocator(2, 8)
	const (
		prefixSeqID = int64(10)
		branchSeqID = int64(11)
	)
	if err := allocator.AllocSequence(prefixSeqID, nil, 2); err != nil {
		t.Fatalf("AllocSequence prefix: %v", err)
	}

	mtpCfg := cudamtp.MTPConfig{
		NumDraftHeads: 2,
		Dim:           4,
		VocabSize:     6,
	}
	hidden := []float32{1.0, 0.5, -0.5, 1.5}
	// Identity draft projections per head.
	draftWeights := make([]float32, mtpCfg.NumDraftHeads*mtpCfg.Dim*mtpCfg.Dim)
	for h := 0; h < mtpCfg.NumDraftHeads; h++ {
		for d := 0; d < mtpCfg.Dim; d++ {
			draftWeights[(h*mtpCfg.Dim+d)*mtpCfg.Dim+d] = float32(h + 1)
		}
	}
	// Head 0 peaks at token 2; Head 1 peaks at token 3.
	headWeights := make([]float32, mtpCfg.NumDraftHeads*mtpCfg.Dim*mtpCfg.VocabSize)
	for d := 0; d < mtpCfg.Dim; d++ {
		headWeights[(0*mtpCfg.Dim+d)*mtpCfg.VocabSize+2] = 2.0
		headWeights[(1*mtpCfg.Dim+d)*mtpCfg.VocabSize+3] = 2.5
	}

	wantLogits, err := cudamtp.MTPReference(hidden, draftWeights, headWeights, mtpCfg)
	if err != nil {
		t.Fatalf("MTPReference: %v", err)
	}
	draftDigest, err := cudaspec.MTPDraftDigest(mtpCfg, draftWeights, headWeights)
	if err != nil {
		t.Fatalf("MTPDraftDigest: %v", err)
	}

	prefix := []int32{0, 1}
	modelDigest := [32]byte{0: 0x42, 31: 0x99}
	key := cudaspec.ComputeSpeculationKey(modelDigest[:], prefix, 1, len(prefix), mtpCfg.NumDraftHeads, "fp32", 1)
	req := cudaspec.BatchCycleRequest{
		DraftDigest:    draftDigest,
		Key:            key,
		ExpectedKey:    key,
		Prefix:         prefix,
		PrefixSequence: prefixSeqID,
		Sequence:       branchSeqID,
		Hidden:         hidden,
		DraftWeights:   draftWeights,
		HeadWeights:    headWeights,
		MTP:            mtpCfg,
	}

	// Candidate tree: root (node 0, token 1) -> accepted branch (node 1: token 2 -> node 2: token 3)
	// and competing branch (node 3: token 4 -> node 4: token 5).
	tree := cudaspec.NewDraftTree(1)
	n1 := tree.AddNode(0, 2)
	tree.AddNode(n1.ID, 3)
	n3 := tree.AddNode(0, 4)
	tree.AddNode(n3.ID, 5)
	treeMask := tree.BuildCausalTreeMask()
	bitmasks, err := cudaspec.TreeMaskToBitmasks(treeMask)
	if err != nil {
		t.Fatalf("TreeMaskToBitmasks: %v", err)
	}

	numNodes := len(tree.Nodes)
	numHeads := 2
	headDim := 4
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	total := numNodes * numHeads * headDim
	q := make([]float32, total)
	k := make([]float32, total)
	for i := 0; i < total; i++ {
		q[i] = float32((i*13+3)%19-9) / 7.0
		k[i] = float32((i*11+5)%23-11) / 9.0
	}

	// Construct V tensors for the 5 tree nodes (rowDim = numHeads*headDim = 8):
	// - Node 0 (attends to {0}): dominant on coords d=0, 4 -> projects via targetReadout to token 2.
	// - Node 1 in vAccept (attends to {0, 1}): dominant on coords d=1, 5 -> projects to token 3.
	// - Node 1 in vReject (attends to {0, 1}): dominant on coords d=3, 7 -> projects to token 0.
	//   Because Node 0's causal mask 0b00001 excludes Node 1, changing Node 1's V row leaves
	//   Node 0's attention output identical (still token 2) while flipping Node 1's output to token 0.
	// - Node 2 (attends to {0, 1, 2}): dominant on coords d=2, 6 -> projects to bonus token 1.
	// - Nodes 3, 4 (competing branch): non-trivial values isolated by causal tree masks from branch 1.
	vAccept := []float32{
		// Node 0
		4.0, 0.1, -0.1, 0.05, 4.0, -0.05, 0.1, 0.0,
		// Node 1 (accepted draft 1 -> token 3)
		0.1, 16.0, -0.1, 0.05, -0.1, 16.0, 0.05, 0.0,
		// Node 2 (bonus position -> token 1)
		0.05, -0.1, 42.0, 0.1, 0.0, 0.05, 42.0, -0.1,
		// Node 3 (competing branch depth 1)
		-6.0, 5.0, -4.0, 3.0, -2.0, 1.0, -0.5, 2.5,
		// Node 4 (competing branch depth 2)
		9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0,
	}
	vReject := slices.Clone(vAccept)
	copy(vReject[8:16], []float32{
		0.05, -0.1, 0.1, 16.0, -0.05, 0.05, 0.0, 16.0,
	})

	wantAttn, err := cudaattn.TreeAttentionReference(q, k, vAccept, bitmasks, numNodes, numHeads, headDim, scale)
	if err != nil {
		t.Fatalf("TreeAttentionReference(vAccept): %v", err)
	}
	wantAttnRej, err := cudaattn.TreeAttentionReference(q, k, vReject, bitmasks, numNodes, numHeads, headDim, scale)
	if err != nil {
		t.Fatalf("TreeAttentionReference(vReject): %v", err)
	}

	// Single node-independent linear readout matrix [rowDim=8, vocabSize=6]:
	// Coords {0, 4} -> vocab 2
	// Coords {1, 5} -> vocab 3
	// Coords {2, 6} -> vocab 1
	// Coords {3, 7} -> vocab 0
	rowDim := numHeads * headDim
	targetReadout := make([]float32, rowDim*mtpCfg.VocabSize)
	setReadout := func(d, v int, weight float32) {
		targetReadout[d*mtpCfg.VocabSize+v] = weight
	}
	setReadout(0, 2, 1.0)
	setReadout(4, 2, 1.0)
	setReadout(1, 3, 1.0)
	setReadout(5, 3, 1.0)
	setReadout(2, 1, 1.0)
	setReadout(6, 1, 1.0)
	setReadout(3, 0, 1.0)
	setReadout(7, 0, 1.0)

	f := specNativeCycleFixture{
		allocator:     allocator,
		req:           req,
		tree:          tree,
		treeMask:      treeMask,
		bitmasks:      bitmasks,
		numNodes:      numNodes,
		numHeads:      numHeads,
		headDim:       headDim,
		scale:         scale,
		q:             q,
		k:             k,
		vAccept:       vAccept,
		vReject:       vReject,
		wantAttn:      wantAttn,
		wantAttnRej:   wantAttnRej,
		wantLogits:    wantLogits,
		wantDrafts:    []int32{2, 3},
		targetReadout: targetReadout,
	}

	for node, wantTok := range []int32{2, 3, 1} {
		gotTok, err := f.predictTargetTokenFromNodeAttn(wantAttn, node)
		if err != nil || gotTok != wantTok {
			t.Fatalf("accept readout node %d = (%d, %v), want %d", node, gotTok, err, wantTok)
		}
	}
	for node, wantTok := range []int32{2, 0} {
		gotTok, err := f.predictTargetTokenFromNodeAttn(wantAttnRej, node)
		if err != nil || gotTok != wantTok {
			t.Fatalf("reject readout node %d = (%d, %v), want %d", node, gotTok, err, wantTok)
		}
	}

	return f
}

func (f specNativeCycleFixture) verifyMTPLogits(got []float32, tol float32) error {
	if len(got) != len(f.wantLogits) {
		return fmt.Errorf("cudaspec: MTP logits length = %d, want %d", len(got), len(f.wantLogits))
	}
	for i, w := range f.wantLogits {
		if !finiteWithinTol(got[i], w, tol) {
			return fmt.Errorf("cudaspec: device MTP logit[%d] = %v, want %v (tol %v)", i, got[i], w, tol)
		}
	}
	return nil
}

func verifyTreeAttentionAgainstRef(got, want []float32, tol float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("cudaspec: tree attention length = %d, want %d", len(got), len(want))
	}
	for i, w := range want {
		if !finiteWithinTol(got[i], w, tol) {
			return fmt.Errorf("cudaspec: device tree attention[%d] = %v, want %v (tol %v)", i, got[i], w, tol)
		}
	}
	return nil
}

// predictTargetTokenFromNodeAttn projects a tree node's attention output vector
// [numHeads*headDim] through the fixture's single node-independent readout
// matrix [numHeads*headDim, vocabSize] to select a greedy target token.
func (f specNativeCycleFixture) predictTargetTokenFromNodeAttn(attnOut []float32, nodeIdx int) (int32, error) {
	rowDim := f.numHeads * f.headDim
	vocab := f.req.MTP.VocabSize
	if nodeIdx < 0 || (nodeIdx+1)*rowDim > len(attnOut) {
		return 0, fmt.Errorf("cudaspec: attention output length %d insufficient for node %d (rowDim %d)", len(attnOut), nodeIdx, rowDim)
	}
	if len(f.targetReadout) != rowDim*vocab {
		return 0, fmt.Errorf("cudaspec: readout length = %d, want %d", len(f.targetReadout), rowDim*vocab)
	}
	row := attnOut[nodeIdx*rowDim : (nodeIdx+1)*rowDim]
	for d, val := range row {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			return 0, fmt.Errorf("cudaspec: non-finite attention output at node %d dim %d: %v", nodeIdx, d, val)
		}
	}
	bestV := 0
	var bestScore float32
	for v := 0; v < vocab; v++ {
		var sum float32
		for d := 0; d < rowDim; d++ {
			w := f.targetReadout[d*vocab+v]
			if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
				return 0, fmt.Errorf("cudaspec: non-finite readout weight at dim %d vocab %d: %v", d, v, w)
			}
			sum += row[d] * w
		}
		if math.IsNaN(float64(sum)) || math.IsInf(float64(sum), 0) {
			return 0, fmt.Errorf("cudaspec: non-finite target logit at node %d vocab %d: %v", nodeIdx, v, sum)
		}
		if v == 0 || sum > bestScore {
			bestScore = sum
			bestV = v
		}
	}
	return int32(bestV), nil
}

// TestSpecNativeCycleFixtureContract exercises the speculative cycle fixture,
// tree mask parity, BatchDraftProducer-to-cycle dataflow, node-attention target
// readout parity, commit/rollback KV reference counting, exact-ID/Wait/Done
// completion semantics, cleanup owner retention, and negative controls on CPU.
func TestSpecNativeCycleFixtureContract(t *testing.T) {
	checkUnloaded := func() {
		t.Helper()
		if driver.Loaded() || nvrtc.Loaded() {
			t.Fatal("CPU fixture contract test loaded an NVIDIA library")
		}
	}
	checkUnloaded()
	defer checkUnloaded()

	f := newSpecNativeCycleFixture(t)
	if err := f.verifyMTPLogits(f.wantLogits, 0); err != nil {
		t.Fatalf("verifyMTPLogits: %v", err)
	}
	if err := verifyTreeAttentionAgainstRef(f.wantAttn, f.wantAttn, 0); err != nil {
		t.Fatalf("verifyTreeAttentionAgainstRef: %v", err)
	}
	if err := cudaspec.VerifyTreeAttentionOutputs(f.q, f.k, f.vAccept, f.treeMask, f.numNodes, f.numHeads, f.headDim, f.scale); err != nil {
		t.Fatalf("VerifyTreeAttentionOutputs(vAccept): %v", err)
	}
	if err := cudaspec.VerifyTreeAttentionOutputs(f.q, f.k, f.vReject, f.treeMask, f.numNodes, f.numHeads, f.headDim, f.scale); err != nil {
		t.Fatalf("VerifyTreeAttentionOutputs(vReject): %v", err)
	}
	// Prove Node 0's output in wantAttn and wantAttnRej is identical because
	// Node 0's causal mask excludes Node 1's changed V row.
	rowDim := f.numHeads * f.headDim
	if !slices.Equal(f.wantAttn[:rowDim], f.wantAttnRej[:rowDim]) {
		t.Fatalf("node 0 attention changed when only node 1 V changed: %v vs %v", f.wantAttn[:rowDim], f.wantAttnRej[:rowDim])
	}
	densePrefixTree := cudaspec.DenseReferenceMask(len(f.req.Prefix), f.tree, []int{4})
	for col := range densePrefixTree[len(f.req.Prefix)+4] {
		if densePrefixTree[len(f.req.Prefix)+4][col] {
			t.Fatalf("fully masked tree node row retained unmasked column %d", col)
		}
	}

	t.Run("gate_admission", func(t *testing.T) {
		if _, _, err := admitSpecNativeGate(func(string) string { return "" }); !errors.Is(err, errSpecNativeGateClosed) || !errors.Is(err, cudaerrors.ErrHardwarePending) {
			t.Fatalf("closed gate err = %v, want errSpecNativeGateClosed wrapping ErrHardwarePending", err)
		}
		if _, _, err := admitSpecNativeGate(func(k string) string {
			if k == "CUDASPEC_NATIVE" {
				return "1"
			}
			return ""
		}); !errors.Is(err, cudalock.ErrNotInherited) {
			t.Fatalf("open gate without GPU_LOCK_HELD err = %v, want cudalock.ErrNotInherited", err)
		}
		if _, _, err := admitSpecNativeGate(func(k string) string {
			switch k {
			case "CUDASPEC_NATIVE":
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
			owner := &specSyntheticCleanupError{cause: cause, fail: true}
			wrapped := fmt.Errorf("spec native %s: %w", stage, owner)
			var retained specNativeCleanupOwners
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
		for _, nonFinite := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			if finiteWithinTol(nonFinite, 0, 1e-5) || finiteWithinTol(0, nonFinite, 1e-5) {
				t.Fatalf("finiteWithinTol accepted non-finite value %v", nonFinite)
			}
			badLogits := slices.Clone(f.wantLogits)
			badLogits[0] = nonFinite
			if err := f.verifyMTPLogits(badLogits, 1e-5); err == nil {
				t.Fatalf("verifyMTPLogits accepted non-finite value %v", nonFinite)
			}
			badAttn := slices.Clone(f.wantAttn)
			badAttn[0] = nonFinite
			if err := verifyTreeAttentionAgainstRef(badAttn, f.wantAttn, 1e-5); err == nil {
				t.Fatalf("verifyTreeAttentionAgainstRef accepted non-finite value %v", nonFinite)
			}
			if _, err := f.predictTargetTokenFromNodeAttn(badAttn, 0); err == nil {
				t.Fatalf("predictTargetTokenFromNodeAttn accepted non-finite value %v", nonFinite)
			}
		}
	})

	t.Run("commit_retains_branch_kv_until_owner_close", func(t *testing.T) {
		fx := newSpecNativeCycleFixture(t)
		prefixBlocks := fx.allocator.Table(fx.req.PrefixSequence)
		var draftCompleted bool
		completed := make(map[int64]bool)
		fx.req.DraftProducer = &cudaspec.BatchDraftProducer{
			Engine: cudaspec.DraftLogitsFunc(func(_ context.Context, step *cudaspec.BatchDraftStep) ([]float32, error) {
				if step.DraftDigest != fx.req.DraftDigest {
					return nil, errors.New("draft step digest mismatch")
				}
				return slices.Clone(fx.wantLogits), nil
			}),
			Wait: func(*cudaspec.BatchDraftStep) error {
				draftCompleted = true
				return nil
			},
			Done: func(*cudaspec.BatchDraftStep) bool {
				return draftCompleted
			},
		}
		eng := &specCycleEngine{
			completed:       completed,
			projectNodeAttn: fx.predictTargetTokenFromNodeAttn,
			produceNodeAttn: func(s *cudabatch.ExecutionStep) ([]float32, error) {
				holders, err := fx.allocator.Holders(prefixBlocks[0])
				if err != nil || holders != 2 {
					return nil, fmt.Errorf("prefix block holders = %d, %v; want 2 during verification", holders, err)
				}
				return fx.wantAttn, nil
			},
		}
		verifier := cudaspec.BatchVerifier{
			ReadOnlyPrefix: true,
			Engine:         eng,
			Wait:           func(*cudabatch.ExecutionStep) error { return nil },
			Done:           func(s *cudabatch.ExecutionStep) bool { return completed[s.StepID] },
		}
		res, err := cudaspec.ExecuteBatchCycle(context.Background(), fx.allocator, fx.req, verifier)
		if err != nil {
			t.Fatalf("ExecuteBatchCycle commit: %v", err)
		}
		if !res.Outcome.Committed || !slices.Equal(res.Outcome.AcceptedTokens, fx.wantDrafts) || res.Outcome.BonusToken != 1 || res.Outcome.DiscardedCount != 0 {
			t.Fatalf("unexpected commit outcome: %+v", res.Outcome)
		}
		if res.Key.DraftDigest != fx.req.DraftDigest {
			t.Fatalf("result.Key.DraftDigest = %x, want %x", res.Key.DraftDigest, fx.req.DraftDigest)
		}
		// While committed branch KV is open, both prefix block (holders=2) and private block remain allocated (6 free of 8).
		if fx.allocator.NumFreeBlocks() != 6 {
			t.Fatalf("free blocks while committed branch live = %d, want 6", fx.allocator.NumFreeBlocks())
		}
		if err := res.KV.Release(prefixBlocks); err == nil {
			t.Fatal("releasing borrowed prefix blocks directly succeeded")
		}
		cache := cudaspec.NewSpeculationCache(4)
		if err := cache.Put(&cudaspec.SpeculationCacheEntry{
			Key:        res.Key,
			DraftPath:  res.Outcome.AcceptedTokens,
			BonusToken: res.Outcome.BonusToken,
		}); err != nil {
			t.Fatalf("SpeculationCache.Put: %v", err)
		}
		if got, ok := cache.Get(res.Key); !ok || got.BonusToken != 1 || !slices.Equal(got.DraftPath, fx.wantDrafts) {
			t.Fatalf("SpeculationCache.Get = %+v, %v", got, ok)
		}
		if err := res.KV.Close(); err != nil {
			t.Fatalf("res.KV.Close: %v", err)
		}
		if fx.allocator.NumFreeBlocks() != 7 || !reflect.DeepEqual(fx.allocator.Table(fx.req.PrefixSequence), prefixBlocks) {
			t.Fatalf("free blocks after close = %d, want 7 with prefix intact", fx.allocator.NumFreeBlocks())
		}
	})

	t.Run("rollback_releases_private_kv_and_preserves_prefix", func(t *testing.T) {
		fx := newSpecNativeCycleFixture(t)
		prefixBlocks := fx.allocator.Table(fx.req.PrefixSequence)
		completed := make(map[int64]bool)
		fx.req.DraftProducer = &cudaspec.BatchDraftProducer{
			Engine: cudaspec.DraftLogitsFunc(func(_ context.Context, _ *cudaspec.BatchDraftStep) ([]float32, error) {
				return slices.Clone(fx.wantLogits), nil
			}),
			Wait: func(*cudaspec.BatchDraftStep) error { return nil },
		}
		eng := &specCycleEngine{
			completed:       completed,
			projectNodeAttn: fx.predictTargetTokenFromNodeAttn,
			produceNodeAttn: func(*cudabatch.ExecutionStep) ([]float32, error) {
				// wantAttnRej projects Node 0 -> 2 and Node 1 -> 0 (rejecting draft 3).
				return fx.wantAttnRej, nil
			},
		}
		verifier := cudaspec.BatchVerifier{
			ReadOnlyPrefix: true,
			Engine:         eng,
			Wait:           func(*cudabatch.ExecutionStep) error { return nil },
			Done:           func(s *cudabatch.ExecutionStep) bool { return completed[s.StepID] },
		}
		res, err := cudaspec.ExecuteBatchCycle(context.Background(), fx.allocator, fx.req, verifier)
		if err != nil {
			t.Fatalf("ExecuteBatchCycle rollback: %v", err)
		}
		if res.Outcome.Committed || !slices.Equal(res.Outcome.AcceptedTokens, []int32{2}) || res.Outcome.BonusToken != 0 || res.Outcome.DiscardedCount != 1 {
			t.Fatalf("unexpected rollback outcome: %+v", res.Outcome)
		}
		// Rollback must have already released the private block while retaining the root prefix block.
		if fx.allocator.NumFreeBlocks() != 7 {
			t.Fatalf("free blocks after rollback = %d, want 7", fx.allocator.NumFreeBlocks())
		}
		holders, err := fx.allocator.Holders(prefixBlocks[0])
		if err != nil || holders != 1 {
			t.Fatalf("prefix block holders after rollback = %d, %v; want 1", holders, err)
		}
	})

	t.Run("shifted_producer_logit_changes_proposal_and_triggers_rollback", func(t *testing.T) {
		fx := newSpecNativeCycleFixture(t)
		fx.req.DraftProducer = &cudaspec.BatchDraftProducer{
			Engine: cudaspec.DraftLogitsFunc(func(_ context.Context, _ *cudaspec.BatchDraftStep) ([]float32, error) {
				shifted := slices.Clone(fx.wantLogits)
				// Shift head 1 peak from token 3 to token 4.
				shifted[fx.req.MTP.VocabSize+4] = shifted[fx.req.MTP.VocabSize+3] + 5.0
				return shifted, nil
			}),
			Wait: func(*cudaspec.BatchDraftStep) error { return nil },
		}
		eng := &specCycleEngine{
			projectNodeAttn: fx.predictTargetTokenFromNodeAttn,
			produceNodeAttn: func(*cudabatch.ExecutionStep) ([]float32, error) {
				return fx.wantAttn, nil
			},
		}
		verifier := cudaspec.BatchVerifier{
			ReadOnlyPrefix: true,
			Engine:         eng,
			Wait:           func(*cudabatch.ExecutionStep) error { return nil },
		}
		res, err := cudaspec.ExecuteBatchCycle(context.Background(), fx.allocator, fx.req, verifier)
		if err != nil {
			t.Fatalf("ExecuteBatchCycle shifted producer: %v", err)
		}
		if res.Outcome.Committed || !slices.Equal(res.Outcome.DraftTokens, []int32{2, 4}) || !slices.Equal(res.Outcome.AcceptedTokens, []int32{2}) || res.Outcome.BonusToken != 3 {
			t.Fatalf("unexpected shifted producer outcome: %+v", res.Outcome)
		}
	})

	t.Run("negative_controls", func(t *testing.T) {
		t.Run("wrong_branch_id_refused", func(t *testing.T) {
			fx := newSpecNativeCycleFixture(t)
			fx.req.Key.BranchID = 2
			eng := &specCycleEngine{
				projectNodeAttn: fx.predictTargetTokenFromNodeAttn,
				produceNodeAttn: func(*cudabatch.ExecutionStep) ([]float32, error) { return fx.wantAttn, nil },
			}
			verifier := cudaspec.BatchVerifier{
				ReadOnlyPrefix: true,
				Engine:         eng,
				Wait:           func(*cudabatch.ExecutionStep) error { return nil },
			}
			if _, err := cudaspec.ExecuteBatchCycle(context.Background(), fx.allocator, fx.req, verifier); err == nil {
				t.Fatal("wrong branch ID in request key was accepted")
			}
			if eng.calls != 0 || fx.allocator.NumFreeBlocks() != 7 {
				t.Fatalf("calls=%d free=%d, want 0 calls and 7 free blocks", eng.calls, fx.allocator.NumFreeBlocks())
			}
		})

		t.Run("foreign_seq_id_refused_and_cleaned_up", func(t *testing.T) {
			fx := newSpecNativeCycleFixture(t)
			eng := &specCycleEngine{
				foreignSeq:      "spec-foreign-branch",
				projectNodeAttn: fx.predictTargetTokenFromNodeAttn,
				produceNodeAttn: func(*cudabatch.ExecutionStep) ([]float32, error) { return fx.wantAttn, nil },
			}
			verifier := cudaspec.BatchVerifier{
				ReadOnlyPrefix: true,
				Engine:         eng,
				Wait:           func(*cudabatch.ExecutionStep) error { return nil },
			}
			if _, err := cudaspec.ExecuteBatchCycle(context.Background(), fx.allocator, fx.req, verifier); err == nil {
				t.Fatal("foreign verification row ID was accepted")
			}
			if fx.allocator.NumFreeBlocks() != 7 {
				t.Fatalf("foreign row failure leaked KV blocks: free=%d, want 7", fx.allocator.NumFreeBlocks())
			}
		})

		t.Run("unawaited_step_retains_kv_until_explicit_close", func(t *testing.T) {
			fx := newSpecNativeCycleFixture(t)
			eng := &specCycleEngine{
				projectNodeAttn: fx.predictTargetTokenFromNodeAttn,
				produceNodeAttn: func(*cudabatch.ExecutionStep) ([]float32, error) { return fx.wantAttn, nil },
			}
			verifier := cudaspec.BatchVerifier{
				ReadOnlyPrefix: true,
				Engine:         eng,
				Wait:           func(*cudabatch.ExecutionStep) error { return errors.New("stream drain timed out") },
				Done:           func(*cudabatch.ExecutionStep) bool { return false },
			}
			res, err := cudaspec.ExecuteBatchCycle(context.Background(), fx.allocator, fx.req, verifier)
			if err == nil || res == nil || res.KV == nil {
				t.Fatalf("expected drain failure with retained KV handle, got res=%+v err=%v", res, err)
			}
			// Because Done is false, the private KV block must remain held (6 free of 8).
			if fx.allocator.NumFreeBlocks() != 6 {
				t.Fatalf("undrained step released KV prematurely: free=%d, want 6", fx.allocator.NumFreeBlocks())
			}
			if err := res.KV.Close(); err != nil {
				t.Fatalf("explicit owner recovery close: %v", err)
			}
			if fx.allocator.NumFreeBlocks() != 7 {
				t.Fatalf("free blocks after explicit owner close = %d, want 7", fx.allocator.NumFreeBlocks())
			}
		})

		t.Run("false_done_without_wait_error_retains_kv", func(t *testing.T) {
			fx := newSpecNativeCycleFixture(t)
			eng := &specCycleEngine{
				projectNodeAttn: fx.predictTargetTokenFromNodeAttn,
				produceNodeAttn: func(*cudabatch.ExecutionStep) ([]float32, error) { return fx.wantAttn, nil },
			}
			verifier := cudaspec.BatchVerifier{
				ReadOnlyPrefix: true,
				Engine:         eng,
				Wait:           func(*cudabatch.ExecutionStep) error { return nil },
				Done:           func(*cudabatch.ExecutionStep) bool { return false },
			}
			res, err := cudaspec.ExecuteBatchCycle(context.Background(), fx.allocator, fx.req, verifier)
			if err == nil || res == nil || res.KV == nil {
				t.Fatalf("expected unconfirmed completion error with retained KV, got res=%+v err=%v", res, err)
			}
			if fx.allocator.NumFreeBlocks() != 6 {
				t.Fatalf("unconfirmed step released KV prematurely: free=%d, want 6", fx.allocator.NumFreeBlocks())
			}
			if err := res.KV.Close(); err != nil {
				t.Fatalf("explicit owner recovery close: %v", err)
			}
		})

		t.Run("cache_rejects_missing_or_stale_draft_digest", func(t *testing.T) {
			cache := cudaspec.NewSpeculationCache(4)
			if err := cache.Put(&cudaspec.SpeculationCacheEntry{
				Key:        f.req.Key, // Zero DraftDigest before cycle verification.
				DraftPath:  f.wantDrafts,
				BonusToken: 1,
			}); !errors.Is(err, cudaspec.ErrDraftDigestRequired) {
				t.Fatalf("Put without DraftDigest err = %v, want ErrDraftDigestRequired", err)
			}
		})
	})
}

// TestNativeSpeculativeCycle executes an end-to-end native speculative cycle
// on an admitted CUDA device under gpu-lock, compiling and launching both
// MTPMultiHeadForwardCUDA (inside BatchDraftProducer) and
// TreeFlashAttentionCUDA (inside the verification engine) so completed native
// device outputs directly drive draft proposal, verification, commit, and
// rollback KV ownership through ExecuteBatchCycle.
func TestNativeSpeculativeCycle(t *testing.T) {
	owner, ordinal, err := admitSpecNativeGate(os.Getenv)
	if errors.Is(err, errSpecNativeGateClosed) {
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
		retained specNativeCleanupOwners
		stream   *cuda.Stream
		mtpMod   driver.Module
		treeMod  driver.Module
		floatBuf []cudamem.Slice[float32]
		maskBuf  []cudamem.Slice[uint64]
	)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if stream != nil {
			if err := stream.Synchronize(ctx); err != nil {
				t.Errorf("retain native speculative cycle resources after stream drain failure: %v", err)
				return
			}
		}
		if err := retained.closeBefore(func() error {
			if mtpMod != 0 {
				if err := dev.WithContext(func() error { return mtpMod.Close() }); err != nil {
					return fmt.Errorf("close MTP module: %w", err)
				}
				mtpMod = 0
			}
			if treeMod != 0 {
				if err := dev.WithContext(func() error { return treeMod.Close() }); err != nil {
					return fmt.Errorf("close tree attention module: %w", err)
				}
				treeMod = 0
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
			t.Errorf("native speculative cycle cleanup failed; retaining remaining resources: %v", err)
		}
	})

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatalf("compute capability: %v", err)
	}
	f := newSpecNativeCycleFixture(t)

	compiledMTP, err := nvrtcx.Compile(
		cudamtp.GenerateMTPCudaSource(f.req.MTP),
		nvrtcx.WithName("cudaspec_mtp.cu"),
		nvrtcx.WithPhysicalArch(major, minor),
	)
	if err != nil {
		_ = retained.adoptError(err)
		t.Fatalf("compile MTP kernel: %v", err)
	}
	mtpMod, err = compiledMTP.Load(dev)
	if err != nil {
		_ = retained.adoptError(err)
		t.Fatalf("load MTP module: %v", err)
	}
	mtpFn, err := mtpMod.Function("MTPMultiHeadForwardCUDA")
	if err != nil {
		t.Fatalf("lookup MTPMultiHeadForwardCUDA: %v", err)
	}

	compiledTree, err := nvrtcx.Compile(
		cudaattn.GenerateTreeAttentionCUDAShader(),
		nvrtcx.WithName("cudaspec_tree_attn.cu"),
		nvrtcx.WithPhysicalArch(major, minor),
	)
	if err != nil {
		_ = retained.adoptError(err)
		t.Fatalf("compile TreeFlashAttentionCUDA: %v", err)
	}
	treeMod, err = compiledTree.Load(dev)
	if err != nil {
		_ = retained.adoptError(err)
		t.Fatalf("load TreeFlashAttentionCUDA module: %v", err)
	}
	treeFn, err := treeMod.Function("TreeFlashAttentionCUDA")
	if err != nil {
		t.Fatalf("lookup TreeFlashAttentionCUDA: %v", err)
	}

	stream, err = dev.NewStream()
	if err != nil {
		t.Fatalf("create stream: %v", err)
	}

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

	dHidden := allocFloats(f.req.Hidden)
	dDraftW := allocFloats(f.req.DraftWeights)
	dHeadW := allocFloats(f.req.HeadWeights)
	poisonLogits := make([]float32, len(f.wantLogits))
	for i := range poisonLogits {
		poisonLogits[i] = float32(math.NaN())
	}
	dLogits := allocFloats(poisonLogits)

	dq := allocFloats(f.q)
	dk := allocFloats(f.k)
	dvAccept := allocFloats(f.vAccept)
	dvReject := allocFloats(f.vReject)
	poisonAttn := make([]float32, len(f.wantAttn))
	for i := range poisonAttn {
		poisonAttn[i] = float32(math.NaN())
	}
	dAttnOut := allocFloats(poisonAttn)
	dMasks := allocMasks(f.bitmasks)

	const blockSize = 32
	gridX := uint32((f.req.MTP.VocabSize + blockSize - 1) / blockSize)
	gridY := uint32(f.req.MTP.NumDraftHeads)
	mtpShared := uint32(f.req.MTP.Dim * 4)
	treeShared := uint32(cudaattn.TreeAttentionSharedBytes(f.numNodes, blockSize))

	syncStream := func() error {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		return stream.Synchronize(ctx)
	}

	runDeviceCycle := func(dv cudamem.Slice[float32], hostV, wantAttn []float32) (*cudaspec.BatchCycleResult, error) {
		fx := newSpecNativeCycleFixture(t)
		var draftSynced bool
		fx.req.DraftProducer = &cudaspec.BatchDraftProducer{
			Engine: cudaspec.DraftLogitsFunc(func(_ context.Context, step *cudaspec.BatchDraftStep) ([]float32, error) {
				if step.DraftDigest != fx.req.DraftDigest {
					return nil, errors.New("cudaspec: draft step digest mismatch")
				}
				if err := dev.WithContext(func() error { return dLogits.CopyFromHost(poisonLogits) }); err != nil {
					return nil, fmt.Errorf("poison MTP logits: %w", err)
				}
				if err := cudalaunch.Kernel(mtpFn).
					Grid(gridX, gridY, 1).
					Block(blockSize, 1, 1).
					SharedMem(mtpShared).
					On(stream).
					Launch(dHidden, dDraftW, dHeadW, dLogits, int32(fx.req.MTP.Dim), int32(fx.req.MTP.VocabSize)); err != nil {
					return nil, fmt.Errorf("launch MTPMultiHeadForwardCUDA: %w", err)
				}
				if err := syncStream(); err != nil {
					return nil, fmt.Errorf("synchronize MTPMultiHeadForwardCUDA: %w", err)
				}
				draftSynced = true
				gotLogits := make([]float32, len(fx.wantLogits))
				if err := dev.WithContext(func() error { return dLogits.CopyToHost(gotLogits) }); err != nil {
					return nil, fmt.Errorf("copy MTP logits: %w", err)
				}
				if err := fx.verifyMTPLogits(gotLogits, 1e-5); err != nil {
					return nil, err
				}
				return gotLogits, nil
			}),
			Wait: func(*cudaspec.BatchDraftStep) error {
				if err := syncStream(); err != nil {
					return err
				}
				draftSynced = true
				return nil
			},
			Done: func(*cudaspec.BatchDraftStep) bool {
				return draftSynced
			},
		}

		completedSteps := make(map[int64]bool)
		eng := &specCycleEngine{
			projectNodeAttn: fx.predictTargetTokenFromNodeAttn,
			produceNodeAttn: func(s *cudabatch.ExecutionStep) ([]float32, error) {
				if err := dev.WithContext(func() error { return dAttnOut.CopyFromHost(poisonAttn) }); err != nil {
					return nil, fmt.Errorf("poison tree attention output: %w", err)
				}
				if err := cudalaunch.Kernel(treeFn).
					Grid(uint32(fx.numNodes), uint32(fx.numHeads), 1).
					Block(blockSize, 1, 1).
					SharedMem(treeShared).
					On(stream).
					Launch(dq, dk, dv, dAttnOut, dMasks, int32(fx.numNodes), int32(fx.numHeads), int32(fx.headDim), fx.scale); err != nil {
					return nil, fmt.Errorf("launch TreeFlashAttentionCUDA: %w", err)
				}
				if err := syncStream(); err != nil {
					return nil, fmt.Errorf("synchronize TreeFlashAttentionCUDA: %w", err)
				}
				gotAttn := make([]float32, len(wantAttn))
				if err := dev.WithContext(func() error { return dAttnOut.CopyToHost(gotAttn) }); err != nil {
					return nil, fmt.Errorf("copy tree attention output: %w", err)
				}
				if err := verifyTreeAttentionAgainstRef(gotAttn, wantAttn, 1e-5); err != nil {
					return nil, err
				}
				if err := cudaspec.VerifyTreeAttentionOutputs(fx.q, fx.k, hostV, fx.treeMask, fx.numNodes, fx.numHeads, fx.headDim, fx.scale); err != nil {
					return nil, err
				}
				completedSteps[s.StepID] = true
				return gotAttn, nil
			},
		}
		verifier := cudaspec.BatchVerifier{
			ReadOnlyPrefix: true,
			Engine:         eng,
			Wait: func(s *cudabatch.ExecutionStep) error {
				if err := syncStream(); err != nil {
					return err
				}
				completedSteps[s.StepID] = true
				return nil
			},
			Done: func(s *cudabatch.ExecutionStep) bool {
				return completedSteps[s.StepID]
			},
		}
		res, err := cudaspec.ExecuteBatchCycle(context.Background(), fx.allocator, fx.req, verifier)
		if err != nil {
			return res, err
		}
		if res.Outcome.Committed {
			if fx.allocator.NumFreeBlocks() != 6 {
				t.Fatalf("committed cycle free blocks = %d, want 6 before owner close", fx.allocator.NumFreeBlocks())
			}
			if err := res.KV.Close(); err != nil {
				t.Fatalf("close committed KV: %v", err)
			}
		}
		if fx.allocator.NumFreeBlocks() != 7 {
			t.Fatalf("free blocks after cycle completion = %d, want 7", fx.allocator.NumFreeBlocks())
		}
		return res, nil
	}

	commitRes, err := runDeviceCycle(dvAccept, f.vAccept, f.wantAttn)
	if err != nil {
		t.Fatalf("native commit cycle failed: %v", err)
	}
	if !commitRes.Outcome.Committed || !slices.Equal(commitRes.Outcome.AcceptedTokens, []int32{2, 3}) || commitRes.Outcome.BonusToken != 1 {
		t.Fatalf("unexpected native commit outcome: %+v", commitRes.Outcome)
	}

	rollbackRes, err := runDeviceCycle(dvReject, f.vReject, f.wantAttnRej)
	if err != nil {
		t.Fatalf("native rollback cycle failed: %v", err)
	}
	if rollbackRes.Outcome.Committed || !slices.Equal(rollbackRes.Outcome.AcceptedTokens, []int32{2}) || rollbackRes.Outcome.BonusToken != 0 || rollbackRes.Outcome.DiscardedCount != 1 {
		t.Fatalf("unexpected native rollback outcome: %+v", rollbackRes.Outcome)
	}

	t.Logf("native speculative cycle qualified: device=%d sm_%d%d lease_pid=%d committed_tokens=%v rollback_discarded=%d",
		ordinal, major, minor, owner.PID, commitRes.Outcome.AcceptedTokens, rollbackRes.Outcome.DiscardedCount)
}
