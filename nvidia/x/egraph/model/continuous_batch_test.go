package model

import (
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
)

func TestContinuousBatchRadixPrefixHitsAndLRUEviction(t *testing.T) {
	// 6 total pages, 4 tokens per page.
	tree, err := NewRadixKVTree(6, 4)
	if err != nil {
		t.Fatalf("NewRadixKVTree: %v", err)
	}

	// Insert prompt A: 8 tokens (2 pages: [10,11,12,13], [20,21,22,23]).
	promptA := []int{10, 11, 12, 13, 20, 21, 22, 23}
	pagesA, hitsA, nodeA, err := tree.InsertPrefix(promptA)
	if err != nil {
		t.Fatalf("InsertPrefix(promptA): %v", err)
	}
	if hitsA != 0 || len(pagesA) != 2 {
		t.Fatalf("promptA got hits=%d pages=%v, want hits=0 len=2", hitsA, pagesA)
	}

	// Insert prompt B sharing first 4 tokens with prompt A + 4 new tokens (1 new page).
	promptB := []int{10, 11, 12, 13, 30, 31, 32, 33}
	pagesB, hitsB, nodeB, err := tree.InsertPrefix(promptB)
	if err != nil {
		t.Fatalf("InsertPrefix(promptB): %v", err)
	}
	if hitsB != 1 {
		t.Fatalf("promptB prefix cache hits=%d, want 1 shared page", hitsB)
	}
	if pagesB[0] != pagesA[0] {
		t.Fatalf("promptB shared page[0]=%d != promptA page[0]=%d", pagesB[0], pagesA[0])
	}

	// Insert prompt C with 12 completely new tokens (3 pages). Pool is now full (6/6 used).
	promptC := []int{40, 41, 42, 43, 50, 51, 52, 53, 60, 61, 62, 63}
	pagesC, hitsC, nodeC, err := tree.InsertPrefix(promptC)
	if err != nil {
		t.Fatalf("InsertPrefix(promptC): %v", err)
	}
	if hitsC != 0 || len(pagesC) != 3 || tree.FreePages() != 0 {
		t.Fatalf("promptC got hits=%d pages=%v free=%d, want 0, 3, 0", hitsC, pagesC, tree.FreePages())
	}

	// While all 3 sequences are pinned (RefCount > 0), allocating another uncached prompt must fail.
	if _, _, _, err := tree.InsertPrefix([]int{90, 91, 92, 93}); err == nil {
		t.Fatalf("expected error when all KV pages are pinned by active sequences")
	}

	// Release A first, then C. Both become zero-ref leaves, with A's suffix older than C.
	tree.Release(nodeA)
	tree.Release(nodeC)

	// Allocating 1 page (4 tokens) should evict A's unshared suffix leaf (1 page) first while keeping shared prefix (pinned by B) and C.
	promptD := []int{90, 91, 92, 93}
	pagesD, _, nodeD, err := tree.InsertPrefix(promptD)
	if err != nil {
		t.Fatalf("InsertPrefix(promptD) after release: %v", err)
	}
	if len(pagesD) != 1 || pagesD[0] != pagesA[1] {
		t.Fatalf("expected LRU eviction to reuse A's suffix page %d, got %v", pagesA[1], pagesD)
	}

	tree.Release(nodeB)
	tree.Release(nodeD)
}

func TestContinuousBatchHeterogeneousPackingAndBucketRouting(t *testing.T) {
	ctx := context.Background()
	tree, err := NewRadixKVTree(32, 4)
	if err != nil {
		t.Fatalf("NewRadixKVTree: %v", err)
	}
	table, err := specdecode.NewMultiBucketMegakernelTable(specdecode.MultiBucketTableConfig{
		Arch: "sm_121a",
	})
	if err != nil {
		t.Fatalf("NewMultiBucketMegakernelTable: %v", err)
	}

	sched, err := NewContinuousBatchScheduler(tree, table, 8)
	if err != nil {
		t.Fatalf("NewContinuousBatchScheduler: %v", err)
	}

	// 1) Decode request: 1 active token -> routes to Bucket M=1.
	seqDecode := &InFlightSequence{
		ID:           "req_decode",
		Phase:        PhaseDecode,
		PromptTokens: []int{1, 2, 3, 4},
		DecodeToken:  101,
	}
	if err := sched.Admit(seqDecode); err != nil {
		t.Fatalf("Admit(req_decode): %v", err)
	}

	step1, err := sched.ScheduleStep(ctx)
	if err != nil {
		t.Fatalf("ScheduleStep(1): %v", err)
	}
	if step1.ActiveTokens != 1 || step1.BucketM != 1 || step1.PaddedRows != 0 || step1.ActiveRowMask != 0b1 {
		t.Fatalf("step1 got active=%d bucket=%d padded=%d mask=%b, want 1, 1, 0, 1",
			step1.ActiveTokens, step1.BucketM, step1.PaddedRows, step1.ActiveRowMask)
	}

	// 2) Add a chunked prefill request (chunk=2 tokens) sharing the 4-token prefix -> M_active = 1 + 2 = 3 -> routes to Bucket M=4 with 1 padded row.
	seqPrefill := &InFlightSequence{
		ID:           "req_prefill",
		Phase:        PhaseChunkedPrefill,
		PromptTokens: []int{1, 2, 3, 4, 5, 6},
		ChunkBudget:  2,
		DecodeToken:  202,
	}
	if err := sched.Admit(seqPrefill); err != nil {
		t.Fatalf("Admit(req_prefill): %v", err)
	}
	if seqPrefill.CachedPages != 1 || seqPrefill.PrefillOffset != 4 {
		t.Fatalf("expected req_prefill to hit 1 cached page (offset 4), got cached=%d offset=%d",
			seqPrefill.CachedPages, seqPrefill.PrefillOffset)
	}

	step2, err := sched.ScheduleStep(ctx)
	if err != nil {
		t.Fatalf("ScheduleStep(2): %v", err)
	}
	if step2.ActiveTokens != 3 || step2.BucketM != 4 || step2.PaddedRows != 1 {
		t.Fatalf("step2 got active=%d bucket=%d padded=%d, want 3, 4, 1",
			step2.ActiveTokens, step2.BucketM, step2.PaddedRows)
	}
	if step2.ActiveRowMask != 0b0111 {
		t.Fatalf("step2 ActiveRowMask=%04b, want 0111", step2.ActiveRowMask)
	}
	// Verify valid vs padded rows.
	for r := range step2.BucketM {
		row := step2.Rows[r]
		if r < 3 {
			if !row.Valid || !row.WriteKV || row.KVPageID < 0 || !step2.RowIsActive(r) {
				t.Fatalf("row %d should be active and write KV: %+v", r, row)
			}
		} else {
			if row.Valid || row.WriteKV || row.KVPageID != -1 || step2.RowIsActive(r) {
				t.Fatalf("padded row %d must be masked out from writing KV: %+v", r, row)
			}
		}
	}

	// 3) Add a speculative draft tree sequence (1 root + 3 draft = 4 tokens).
	// Note req_prefill finished its 2-token prefill chunk in step2 and transitioned to PhaseDecode (1 token).
	// Total M_active = 1 (req_decode) + 1 (req_prefill now in decode) + 4 (req_spec) = 6 -> routes to Bucket M=8 with 2 padded rows.
	seqSpec := &InFlightSequence{
		ID:             "req_spec",
		Phase:          PhaseSpeculativeVerify,
		PromptTokens:   []int{1, 2, 3, 4},
		DecodeToken:    301,
		DraftTokens:    []int{302, 303, 304},
		AcceptanceRate: 0.85,
	}
	if err := sched.Admit(seqSpec); err != nil {
		t.Fatalf("Admit(req_spec): %v", err)
	}

	step3, err := sched.ScheduleStep(ctx)
	if err != nil {
		t.Fatalf("ScheduleStep(3): %v", err)
	}
	if step3.ActiveTokens != 6 || step3.BucketM != 8 || step3.PaddedRows != 2 {
		t.Fatalf("step3 got active=%d bucket=%d padded=%d, want 6, 8, 2",
			step3.ActiveTokens, step3.BucketM, step3.PaddedRows)
	}
	if step3.ActiveRowMask != 0b00111111 {
		t.Fatalf("step3 ActiveRowMask=%08b, want 00111111", step3.ActiveRowMask)
	}
	for r := 6; r < 8; r++ {
		if step3.Rows[r].Valid || step3.Rows[r].WriteKV || step3.Rows[r].KVPageID != -1 {
			t.Fatalf("step3 padded row %d not properly masked: %+v", r, step3.Rows[r])
		}
	}

	if err := sched.Complete("req_decode"); err != nil {
		t.Fatalf("Complete(req_decode): %v", err)
	}
	if err := sched.Complete("req_prefill"); err != nil {
		t.Fatalf("Complete(req_prefill): %v", err)
	}
	if err := sched.Complete("req_spec"); err != nil {
		t.Fatalf("Complete(req_spec): %v", err)
	}
}
