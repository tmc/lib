// SPDX-License-Identifier: Apache-2.0

package cudaspec_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaspec"
)

// mockReleaser records blocks released.
type mockReleaser struct {
	released []int
}

func (m *mockReleaser) Release(blocks []int) error {
	m.released = append(m.released, blocks...)
	return nil
}

// 1. TestTreeMaskDenseReference: Dense reference masks for prefix/tree/glue operations,
// comparing admitted sparse output against them.
func TestTreeMaskDenseReference(t *testing.T) {
	// Build a draft tree with 4 nodes:
	//       Root (0, tok 10)
	//       /             \
	//   N1 (1, tok 20)   N2 (2, tok 30)
	//     |
	//   N3 (3, tok 25)
	dt := cudaspec.NewDraftTree(10)
	n1 := dt.AddNode(0, 20)
	dt.AddNode(0, 30)
	dt.AddNode(n1.ID, 25)

	numPrefix := 3 // 3 prefix prompt tokens
	denseMask := cudaspec.DenseReferenceMask(numPrefix, dt, nil)
	totalRows := numPrefix + 4
	if len(denseMask) != totalRows || len(denseMask[0]) != totalRows {
		t.Fatalf("denseMask dimension = %dx%d, want %dx%d", len(denseMask), len(denseMask[0]), totalRows, totalRows)
	}

	// Verify prefix tokens: prefix 1 (row 1) attends to 0, 1, not 2
	if !denseMask[1][0] || !denseMask[1][1] || denseMask[1][2] {
		t.Errorf("prefix causal mask wrong at row 1: %v", denseMask[1])
	}

	// Verify tree tokens:
	// N1 (row 4): attends to prefix (0, 1, 2) and tree root (3) and self (4)
	if !denseMask[4][0] || !denseMask[4][1] || !denseMask[4][2] || !denseMask[4][3] || !denseMask[4][4] {
		t.Errorf("tree N1 ancestor mask wrong at row 4: %v", denseMask[4])
	}
	// N1 does NOT attend to sibling N2 (5) or child N3 (6)
	if denseMask[4][5] || denseMask[4][6] {
		t.Errorf("tree N1 should not attend to sibling or child: %v", denseMask[4])
	}

	// N3 (row 6, child of N1): attends to prefix (0, 1, 2) + root (3) + N1 (4) + self (6)
	if !denseMask[6][0] || !denseMask[6][1] || !denseMask[6][2] || !denseMask[6][3] || !denseMask[6][4] || !denseMask[6][6] {
		t.Errorf("tree N3 ancestor mask wrong at row 6: %v", denseMask[6])
	}
	// N3 does NOT attend to sibling N2 (5)
	if denseMask[6][5] {
		t.Errorf("tree N3 should not attend to sibling N2")
	}

	// Numerical comparison: run VerifyTreeAttentionOutputs on tree attention
	numNodes := 4
	numHeads := 2
	headDim := 4
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := make([]float32, numNodes*numHeads*headDim)
	k := make([]float32, numNodes*numHeads*headDim)
	v := make([]float32, numNodes*numHeads*headDim)
	for i := range q {
		q[i] = float32(i%7) * 0.1
		k[i] = float32(i%5) * 0.1
		v[i] = float32(i%11) * 0.1
	}

	treeOnlyMask := dt.BuildCausalTreeMask()
	if err := cudaspec.VerifyTreeAttentionOutputs(q, k, v, treeOnlyMask, numNodes, numHeads, headDim, scale); err != nil {
		t.Fatalf("VerifyTreeAttentionOutputs failed: %v", err)
	}
}

// 2. TestFullyMaskedTreeRow: Verify fully masked rows contribute zero without NaN/Inf.
func TestFullyMaskedTreeRow(t *testing.T) {
	numNodes := 3
	numHeads := 2
	headDim := 4
	scale := float32(0.5)

	q := make([]float32, numNodes*numHeads*headDim)
	k := make([]float32, numNodes*numHeads*headDim)
	v := make([]float32, numNodes*numHeads*headDim)
	for i := range q {
		q[i] = 1.0
		k[i] = 1.0
		v[i] = 2.0
	}

	// Bitmask where row 1 is fully masked (0)
	bitmasks := []uint64{
		1, // node 0 attends to 0
		0, // node 1 is fully masked!
		5, // node 2 attends to 0 and 2
	}

	out, err := cudaattn.TreeAttentionReference(q, k, v, bitmasks, numNodes, numHeads, headDim, scale)
	if err != nil {
		t.Fatalf("TreeAttentionReference failed: %v", err)
	}

	// Check node 1 output: must be exact zeros
	for headIdx := 0; headIdx < numHeads; headIdx++ {
		outOff := (1*numHeads + headIdx) * headDim
		for d := 0; d < headDim; d++ {
			val := out[outOff+d]
			if val != 0.0 || math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Fatalf("fully masked node 1 head %d dim %d = %f, want 0.0", headIdx, d, val)
			}
		}
	}

	// Check node 0 output: should be non-zero (v = 2.0)
	for headIdx := 0; headIdx < numHeads; headIdx++ {
		outOff := (0*numHeads + headIdx) * headDim
		if math.Abs(float64(out[outOff]-2.0)) > 1e-4 {
			t.Errorf("unmasked node 0 head %d = %f, want 2.0", headIdx, out[outOff])
		}
	}
}

// 3. TestSpeculationCommit: Commit accepted branch, verify state transitions to committed.
func TestSpeculationCommit(t *testing.T) {
	bt := cudaspec.NewBranchTree(100, []int{10, 11}, 1)
	releaser := &mockReleaser{}

	b1, err := bt.ProposeBranch(0, []int32{2, 3}, []int{20, 21})
	if err != nil {
		t.Fatalf("ProposeBranch 1 failed: %v", err)
	}
	b2, err := bt.ProposeBranch(0, []int32{4, 5}, []int{30, 31})
	if err != nil {
		t.Fatalf("ProposeBranch 2 failed: %v", err)
	}

	// Commit branch b1 with matching tokens
	if err := bt.CommitBranch(b1.ID, []int32{2, 3}, releaser); err != nil {
		t.Fatalf("CommitBranch b1 failed: %v", err)
	}

	branch1, _ := bt.GetBranch(b1.ID)
	if branch1.State != cudaspec.ProposalCommitted {
		t.Errorf("branch 1 state = %v, want ProposalCommitted", branch1.State)
	}

	// Competitor branch b2 should have been rolled back
	branch2, _ := bt.GetBranch(b2.ID)
	if branch2.State != cudaspec.ProposalRolledBack {
		t.Errorf("competing branch 2 state = %v, want ProposalRolledBack", branch2.State)
	}
	// b2's private KV blocks should have been released
	if !reflect.DeepEqual(releaser.released, []int{30, 31}) {
		t.Errorf("releaser released %v, want [30 31]", releaser.released)
	}
}

// 4. TestSpeculationRollback: Roll back branch and release its KV blocks.
func TestSpeculationRollback(t *testing.T) {
	bt := cudaspec.NewBranchTree(100, []int{10, 11}, 1)
	releaser := &mockReleaser{}

	b, err := bt.ProposeBranch(0, []int32{2, 3}, []int{40, 41})
	if err != nil {
		t.Fatalf("ProposeBranch failed: %v", err)
	}

	if err := bt.RollbackBranch(b.ID, releaser); err != nil {
		t.Fatalf("RollbackBranch failed: %v", err)
	}

	br, _ := bt.GetBranch(b.ID)
	if br.State != cudaspec.ProposalRolledBack {
		t.Errorf("branch state = %v, want ProposalRolledBack", br.State)
	}
	if !reflect.DeepEqual(releaser.released, []int{40, 41}) {
		t.Errorf("releaser released %v, want [40 41]", releaser.released)
	}
}

// 5. TestRejectedBranchKVRelease: Explicit integration with real cudakv.BlockAllocator,
// ensuring physical KV blocks are released on reject while prefix blocks maintain reference counts.
func TestRejectedBranchKVRelease(t *testing.T) {
	blockSize := 4
	totalBlocks := 16
	alloc := cudakv.NewBlockAllocator(blockSize, totalBlocks)

	// Allocate a root sequence with 2 blocks (8 tokens)
	rootSeqID := int64(1)
	if err := alloc.AllocSequence(rootSeqID, nil, 8); err != nil {
		t.Fatalf("AllocSequence root failed: %v", err)
	}
	rootBlocks := alloc.Table(rootSeqID)
	if len(rootBlocks) != 2 {
		t.Fatalf("rootBlocks len = %d, want 2", len(rootBlocks))
	}
	// Retain root blocks to simulate shared prefix cache hold
	if err := alloc.Retain(rootBlocks); err != nil {
		t.Fatalf("Retain rootBlocks failed: %v", err)
	}

	bt := cudaspec.NewBranchTree(rootSeqID, rootBlocks, 100)

	// Allocate 2 private blocks for branch 1 and branch 2
	b1SeqID := int64(2)
	if err := alloc.AllocSequence(b1SeqID, rootBlocks, 16); err != nil {
		t.Fatalf("AllocSequence b1 failed: %v", err)
	}
	b1Table := alloc.Table(b1SeqID)
	b1Private := b1Table[len(rootBlocks):]

	b2SeqID := int64(3)
	if err := alloc.AllocSequence(b2SeqID, rootBlocks, 16); err != nil {
		t.Fatalf("AllocSequence b2 failed: %v", err)
	}
	b2Table := alloc.Table(b2SeqID)
	b2Private := b2Table[len(rootBlocks):]

	b1, err := bt.ProposeBranch(0, []int32{101, 102}, b1Private)
	if err != nil {
		t.Fatalf("ProposeBranch 1 failed: %v", err)
	}
	_, err = bt.ProposeBranch(0, []int32{201, 202}, b2Private)
	if err != nil {
		t.Fatalf("ProposeBranch 2 failed: %v", err)
	}

	// Commit b1, causing b2 to be rejected and its private blocks released
	if err := bt.CommitBranch(b1.ID, []int32{101, 102}, alloc); err != nil {
		t.Fatalf("CommitBranch b1 failed: %v", err)
	}

	// b2 private blocks should now have 0 holders and be returned to free list
	for _, blk := range b2Private {
		h, err := alloc.Holders(blk)
		if err != nil {
			t.Fatalf("Holders(%d): %v", blk, err)
		}
		// Notice b2SeqID in alloc still held it unless sequence was freed, but our releaser called alloc.Release!
		if h != 0 {
			t.Errorf("rejected block %d holders = %d, want 0", blk, h)
		}
	}

	// Shared root prefix blocks must still have holders >= 1
	for _, blk := range rootBlocks {
		h, err := alloc.Holders(blk)
		if err != nil {
			t.Fatalf("Holders(%d): %v", blk, err)
		}
		if h < 1 {
			t.Errorf("shared root block %d was prematurely freed (holders = %d)", blk, h)
		}
	}
}

// 6. TestSpeculationCacheIdentity: Verify exact match key hit and invalidation on model,
// prefix, positions, numerical mode, or generation.
func TestSpeculationCacheIdentity(t *testing.T) {
	cache := cudaspec.NewSpeculationCache(10)

	model1 := [32]byte{1, 2, 3}
	model2 := [32]byte{4, 5, 6}
	prefix1 := []int32{10, 20, 30}
	prefix2 := []int32{10, 20, 31} // differs by 1 token

	draftA := [32]byte{7}
	draftB := [32]byte{8}
	// draftKey returns a key for a result drafted with draftA.
	draftKey := func(model []byte, prefix []int32, branch cudaspec.BranchID, start int, mode string) cudaspec.SpeculationCacheKey {
		k := cudaspec.ComputeSpeculationKey(model, prefix, branch, start, 3, mode, 1)
		k.DraftDigest = draftA
		return k
	}

	// A drafted entry without a draft digest cannot be cached or reused.
	undrafted := cudaspec.ComputeSpeculationKey(model1[:], prefix1, 1, 0, 3, "FP32", 1)
	if err := cache.Put(&cudaspec.SpeculationCacheEntry{Key: undrafted, DraftPath: []int32{40, 50}}); err != cudaspec.ErrDraftDigestRequired {
		t.Errorf("Put drafted entry without draft digest: err = %v, want %v", err, cudaspec.ErrDraftDigestRequired)
	}
	if _, ok := cache.Get(undrafted); ok {
		t.Errorf("expected miss for drafted entry without draft digest")
	}

	key1 := draftKey(model1[:], prefix1, 1, 0, "FP32")
	entry1 := &cudaspec.SpeculationCacheEntry{
		Key:        key1,
		DraftPath:  []int32{40, 50},
		BonusToken: 60,
	}
	if err := cache.Put(entry1); err != nil {
		t.Fatal(err)
	}

	// Hit
	hit, ok := cache.Get(key1)
	if !ok || hit.BonusToken != 60 {
		t.Errorf("expected hit for key1")
	}

	// Key with different draft weights -> Miss
	keyDiffDraft := key1
	keyDiffDraft.DraftDigest = draftB
	if _, ok := cache.Get(keyDiffDraft); ok {
		t.Errorf("expected miss for key with different draft digest")
	}

	// Key with different prefix -> Miss
	keyDiffPrefix := draftKey(model1[:], prefix2, 1, 0, "FP32")
	if _, ok := cache.Get(keyDiffPrefix); ok {
		t.Errorf("expected miss for key with different prefix")
	}

	// Key with different numerical mode -> Miss
	keyDiffMode := draftKey(model1[:], prefix1, 1, 0, "FP16")
	if _, ok := cache.Get(keyDiffMode); ok {
		t.Errorf("expected miss for key with different numerical mode")
	}

	// Key with different branch ID -> Miss
	keyDiffBranch := draftKey(model1[:], prefix1, 2, 0, "FP32")
	if _, ok := cache.Get(keyDiffBranch); ok {
		t.Errorf("expected miss for key with different branch ID")
	}

	// Key with different start pos -> Miss
	keyDiffPos := draftKey(model1[:], prefix1, 1, 1, "FP32")
	if _, ok := cache.Get(keyDiffPos); ok {
		t.Errorf("expected miss for key with different start pos")
	}

	// Invalidation by model change
	purged := cache.InvalidateModel(model2)
	if purged != 1 {
		t.Errorf("purged = %d, want 1", purged)
	}
	if _, ok := cache.Get(key1); ok {
		t.Errorf("expected miss after model invalidation")
	}

	// Generation invalidation
	keyGen1 := cudaspec.ComputeSpeculationKey(model1[:], prefix1, 1, 0, 3, "FP32", 1)
	keyGen1.DraftDigest = draftA
	keyGen2 := cudaspec.ComputeSpeculationKey(model1[:], prefix1, 1, 0, 3, "FP32", 2)
	keyGen2.DraftDigest = draftA
	for _, k := range []cudaspec.SpeculationCacheKey{keyGen1, keyGen2} {
		if err := cache.Put(&cudaspec.SpeculationCacheEntry{Key: k}); err != nil {
			t.Fatalf("Put generation %d: %v", k.Generation, err)
		}
		if _, ok := cache.Get(k); !ok {
			t.Fatalf("missing generation %d entry before invalidation", k.Generation)
		}
	}

	purgedGen := cache.InvalidateGeneration(1)
	if purgedGen != 1 {
		t.Errorf("purgedGen = %d, want 1", purgedGen)
	}
	if _, ok := cache.Get(keyGen1); ok {
		t.Errorf("expected keyGen1 to be invalidated")
	}
	if _, ok := cache.Get(keyGen2); !ok {
		t.Errorf("expected keyGen2 to survive")
	}
}

// TestHostSpeculativeCycleReference uses a CPU oracle and mock block release.
func TestHostSpeculativeCycleReference(t *testing.T) {
	cfg := cudaspec.CycleConfig{
		NumHeads:      2,
		HeadDim:       4,
		Scale:         0.5,
		VocabSize:     100,
		NumDraftHeads: 2,
		NumericalMode: "FP32",
	}

	releaser := &mockReleaser{}
	bt := cudaspec.NewBranchTree(100, []int{1, 2}, 10)

	// Target oracle simulates target model greedy predictions
	oracle := func(ctx []int32) int32 {
		last := ctx[len(ctx)-1]
		return last + 1 // Predicts next sequential token
	}

	// Propose matching draft tokens: context ends in 10, draft proposes [11, 12]
	outcome, err := cudaspec.ExecuteSpeculativeCycle(cfg, bt, 0, []int32{11, 12}, []int{5, 6}, releaser, oracle)
	if err != nil {
		t.Fatalf("ExecuteSpeculativeCycle matching failed: %v", err)
	}
	if !outcome.Committed {
		t.Errorf("expected matching cycle to be committed")
	}
	if outcome.AcceptedRatio != 1.0 {
		t.Errorf("accepted ratio = %f, want 1.0", outcome.AcceptedRatio)
	}
	if outcome.BonusToken != 13 {
		t.Errorf("bonus token = %d, want 13", outcome.BonusToken)
	}

	// Second cycle proposing mismatched tokens: context ends in 13, draft proposes [99, 100]
	activeBranch := bt.ActiveBranch()
	outcome2, err := cudaspec.ExecuteSpeculativeCycle(cfg, bt, activeBranch, []int32{99, 100}, []int{7, 8}, releaser, oracle)
	if err != nil {
		t.Fatalf("ExecuteSpeculativeCycle mismatch failed: %v", err)
	}
	if outcome2.Committed {
		t.Errorf("expected mismatching cycle to be rolled back")
	}
	if outcome2.AcceptedRatio != 0.0 {
		t.Errorf("accepted ratio = %f, want 0.0", outcome2.AcceptedRatio)
	}
	if outcome2.BonusToken != 13 { // Target predicted 13 at context [10, 11, 12]
		t.Errorf("bonus correction token = %d, want 13", outcome2.BonusToken)
	}
	// Verify private KV blocks [7 8] were released
	if !reflect.DeepEqual(releaser.released, []int{7, 8}) {
		t.Errorf("releaser released = %v, want [7 8]", releaser.released)
	}
}

// 8. Negative controls required by prompt:
// - Swap branch IDs
// - Reuse cache after prefix/weight change
// - Accept incorrect token
// - Free shared prefix KV while retained by branch
func TestNegativeControl_SwapBranchIDs(t *testing.T) {
	bt := cudaspec.NewBranchTree(100, []int{1}, 10)
	b1, err := bt.ProposeBranch(0, []int32{20, 21}, []int{2})
	if err != nil {
		t.Fatalf("ProposeBranch 1 failed: %v", err)
	}
	b2, err := bt.ProposeBranch(0, []int32{30, 31}, []int{3})
	if err != nil {
		t.Fatalf("ProposeBranch 2 failed: %v", err)
	}

	// Attempting to commit b1 with tokens of b2 should fail
	err = bt.CommitBranch(b1.ID, b2.Tokens, nil)
	if err == nil {
		t.Fatalf("expected error when committing branch with swapped tokens")
	}

	// Swapping IDs: passing non-existent branch ID should fail
	err = bt.CommitBranch(999, b1.Tokens, nil)
	if err == nil {
		t.Fatalf("expected error when committing non-existent branch ID")
	}
}

func TestNegativeControl_ReuseCacheAfterPrefixOrWeightChange(t *testing.T) {
	cache := cudaspec.NewSpeculationCache(5)
	modelA := []byte("model_weights_v1")
	modelB := []byte("model_weights_v2")
	prefixA := []int32{1, 2, 3}
	prefixB := []int32{1, 2, 4}

	draft := [32]byte{7}

	keyA := cudaspec.ComputeSpeculationKey(modelA, prefixA, 1, 0, 3, "FP32", 1)
	keyA.DraftDigest = draft
	if err := cache.Put(&cudaspec.SpeculationCacheEntry{Key: keyA, BonusToken: 42}); err != nil {
		t.Fatal(err)
	}
	if entry, ok := cache.Get(keyA); !ok || entry.BonusToken != 42 {
		t.Fatalf("Get(keyA) = %v, %v; want bonus 42", entry, ok)
	}

	// 1. After prefix change: querying with prefixB must not return entry
	keyPrefixChanged := cudaspec.ComputeSpeculationKey(modelA, prefixB, 1, 0, 3, "FP32", 1)
	keyPrefixChanged.DraftDigest = draft
	if entry, ok := cache.Get(keyPrefixChanged); ok {
		t.Fatalf("negative control failed: cache returned entry %v after prefix change", entry)
	}

	// 2. After weight change: querying with modelB must not return entry
	keyWeightChanged := cudaspec.ComputeSpeculationKey(modelB, prefixA, 1, 0, 3, "FP32", 1)
	keyWeightChanged.DraftDigest = draft
	if entry, ok := cache.Get(keyWeightChanged); ok {
		t.Fatalf("negative control failed: cache returned entry %v after weight change", entry)
	}
}

func TestNegativeControl_AcceptIncorrectToken(t *testing.T) {
	bt := cudaspec.NewBranchTree(100, []int{1}, 10)
	b, err := bt.ProposeBranch(0, []int32{50, 51}, []int{2})
	if err != nil {
		t.Fatalf("ProposeBranch failed: %v", err)
	}

	// Attempt to commit with wrong token 52 instead of 51
	err = bt.CommitBranch(b.ID, []int32{50, 52}, nil)
	if err == nil {
		t.Fatalf("negative control failed: committed branch with mismatched token")
	}

	// Verify branch state was NOT committed
	br, _ := bt.GetBranch(b.ID)
	if br.State == cudaspec.ProposalCommitted {
		t.Fatalf("branch state marked committed after rejected commit attempt")
	}
}

func TestNegativeControl_FreeSharedPrefixKVWhileRetainedByBranch(t *testing.T) {
	blockSize := 4
	totalBlocks := 8
	alloc := cudakv.NewBlockAllocator(blockSize, totalBlocks)

	// Allocate root sequence
	rootSeqID := int64(1)
	if err := alloc.AllocSequence(rootSeqID, nil, 8); err != nil {
		t.Fatalf("AllocSequence failed: %v", err)
	}
	sharedBlocks := alloc.Table(rootSeqID)

	// Simulate branch holding a reference via Retain
	if err := alloc.Retain(sharedBlocks); err != nil {
		t.Fatalf("Retain failed: %v", err)
	}

	// Free the root sequence
	alloc.FreeSequence(rootSeqID)

	// Negative control verification: shared blocks must NOT be free because branch still holds them
	for _, blk := range sharedBlocks {
		h, err := alloc.Holders(blk)
		if err != nil {
			t.Fatalf("Holders(%d): %v", blk, err)
		}
		if h != 1 {
			t.Fatalf("negative control failed: shared block %d has %d holders, want 1 (still held by branch)", blk, h)
		}
	}

	// Now when branch releases, holders should become 0
	if err := alloc.Release(sharedBlocks); err != nil {
		t.Fatalf("Release failed: %v", err)
	}
	for _, blk := range sharedBlocks {
		h, _ := alloc.Holders(blk)
		if h != 0 {
			t.Fatalf("after branch release: shared block %d holders = %d, want 0", blk, h)
		}
	}
}
