package nvl72

import (
	"fmt"
	"math"
	"testing"
)

func makeTokenPage(base int32, pageSize int) []int32 {
	p := make([]int32, pageSize)
	for i := range pageSize {
		p[i] = base + int32(i)
	}
	return p
}

func TestRadixPageAllocatorTieredC2C(t *testing.T) {
	const (
		pageSize  = 16
		pageBytes = 450000 // 450 KB/page -> 1.0 us per page over 450 GB/s Grace NVLink-C2C
		hbmCap    = 4
		c2cCap    = 4
	)

	alloc, err := NewRadixPageAllocator(pageSize, pageBytes, hbmCap, c2cCap)
	if err != nil {
		t.Fatalf("NewRadixPageAllocator failed: %v", err)
	}

	sys0 := makeTokenPage(100, pageSize)
	sys1 := makeTokenPage(200, pageSize)
	u1Suffix := makeTokenPage(300, pageSize)
	u2Suffix := makeTokenPage(400, pageSize)

	// 1. Request "req-1" allocates [sys0, sys1, u1Suffix] (3 fresh HBM pages).
	r1, err := alloc.MatchOrAllocatePrefix("req-1", [][]int32{sys0, sys1, u1Suffix})
	if err != nil {
		t.Fatalf("req-1 MatchOrAllocatePrefix failed: %v", err)
	}
	if r1.HBMHitPages != 0 || r1.C2CPromotedPages != 0 || r1.NewlyAllocatedPages != 3 || r1.EvictedToC2CPages != 0 {
		t.Fatalf("req-1 receipt = %+v, want NewlyAllocatedPages=3", r1)
	}
	if got := alloc.HBMUsedPages(); got != 3 {
		t.Fatalf("HBMUsedPages after req-1 = %d, want 3", got)
	}

	// 2. Request "req-2" shares prefix [sys0, sys1] in HBM3e and adds [u2Suffix] (2 HBM hits + 1 new HBM page).
	r2, err := alloc.MatchOrAllocatePrefix("req-2", [][]int32{sys0, sys1, u2Suffix})
	if err != nil {
		t.Fatalf("req-2 MatchOrAllocatePrefix failed: %v", err)
	}
	if r2.HBMHitPages != 2 || r2.C2CPromotedPages != 0 || r2.NewlyAllocatedPages != 1 || r2.EvictedToC2CPages != 0 {
		t.Fatalf("req-2 receipt = %+v, want HBMHitPages=2, NewlyAllocatedPages=1", r2)
	}
	if r2.PrefillTokensSaved != 2*pageSize || r2.PrefillTokensToCompute != 1*pageSize {
		t.Fatalf("req-2 token accounting = (saved=%d, compute=%d), want (%d, %d)",
			r2.PrefillTokensSaved, r2.PrefillTokensToCompute, 2*pageSize, 1*pageSize)
	}
	if got := alloc.HBMUsedPages(); got != 4 {
		t.Fatalf("HBMUsedPages after req-2 = %d, want 4 (full)", got)
	}

	// 3. Release both req-1 and req-2 so all 4 HBM3e pages become zero-refcount LRU candidates.
	if err := alloc.ReleaseRequest("req-1"); err != nil {
		t.Fatalf("ReleaseRequest(req-1) failed: %v", err)
	}
	if err := alloc.ReleaseRequest("req-2"); err != nil {
		t.Fatalf("ReleaseRequest(req-2) failed: %v", err)
	}

	// 4. Request "req-3" allocates 4 completely new pages [p0, p1, p2, p3],
	// forcing all 4 previously cached HBM3e pages ([u1Suffix, sys0, sys1, u2Suffix])
	// to be demoted to Tier 2 Grace LPDDR5X C2C.
	p0 := makeTokenPage(1000, pageSize)
	p1 := makeTokenPage(1100, pageSize)
	p2 := makeTokenPage(1200, pageSize)
	p3 := makeTokenPage(1300, pageSize)
	r3, err := alloc.MatchOrAllocatePrefix("req-3", [][]int32{p0, p1, p2, p3})
	if err != nil {
		t.Fatalf("req-3 MatchOrAllocatePrefix failed: %v", err)
	}
	if r3.NewlyAllocatedPages != 4 || r3.EvictedToC2CPages != 4 {
		t.Fatalf("req-3 receipt = %+v, want NewlyAllocatedPages=4, EvictedToC2CPages=4", r3)
	}
	if got := alloc.C2CUsedPages(); got != 4 {
		t.Fatalf("C2CUsedPages after req-3 = %d, want 4", got)
	}
	if err := alloc.ReleaseRequest("req-3"); err != nil {
		t.Fatalf("ReleaseRequest(req-3) failed: %v", err)
	}

	// 5. Request "req-4" arrives with prefix [sys0, sys1] + new suffix [u4Suffix]:
	// [sys0, sys1] must be promoted from Tier 2 Grace LPDDR5X C2C back to Tier 1 HBM3e
	// over NVLink-C2C (450 GB/s), avoiding prefill recomputation.
	u4Suffix := makeTokenPage(2000, pageSize)
	r4, err := alloc.MatchOrAllocatePrefix("req-4", [][]int32{sys0, sys1, u4Suffix})
	if err != nil {
		t.Fatalf("req-4 MatchOrAllocatePrefix failed: %v", err)
	}
	if r4.HBMHitPages != 0 || r4.C2CPromotedPages != 2 || r4.NewlyAllocatedPages != 1 || r4.EvictedToC2CPages != 3 {
		t.Fatalf("req-4 receipt = %+v, want C2CPromotedPages=2, NewlyAllocatedPages=1, EvictedToC2CPages=3", r4)
	}
	wantPromoteBytes := int64(2 * pageBytes)
	if r4.C2CPromoteBytes != wantPromoteBytes {
		t.Fatalf("req-4 C2CPromoteBytes = %d, want %d", r4.C2CPromoteBytes, wantPromoteBytes)
	}
	wantLatencyUs := float64(wantPromoteBytes) / (NVL72GraceC2CGBps * 1e3)
	if math.Abs(r4.C2CPromoteLatencyUs-wantLatencyUs) > 1e-9 {
		t.Fatalf("req-4 C2CPromoteLatencyUs = %g, want %g", r4.C2CPromoteLatencyUs, wantLatencyUs)
	}
	if r4.PrefillTokensSaved != 2*pageSize {
		t.Fatalf("req-4 PrefillTokensSaved = %d, want %d", r4.PrefillTokensSaved, 2*pageSize)
	}
}

func TestPlanChunkedPrefillBatch(t *testing.T) {
	requests := []ChunkedBatchRequest{
		{ReqID: "dec-1", IsDecode: true},
		{ReqID: "dec-2", IsDecode: true},
		{ReqID: "prefill-A", TotalPromptTokens: 1024, CachedPrefixTokens: 512}, // 512 uncached tokens
		{ReqID: "prefill-B", TotalPromptTokens: 768, CachedPrefixTokens: 256},  // 512 uncached tokens
	}
	const maxTokenBudget = 514 // 2 decode tokens + 512 prefill tokens per step

	steps, err := PlanChunkedPrefillBatch(requests, maxTokenBudget)
	if err != nil {
		t.Fatalf("PlanChunkedPrefillBatch failed: %v", err)
	}
	if len(steps) != 2 {
		t.Fatalf("len(steps) = %d, want 2: %+v", len(steps), steps)
	}

	// Step 0: dec-1, dec-2 (2 tokens) + prefill-A (512 tokens) = 514 tokens.
	if steps[0].StepIndex != 0 || len(steps[0].ScheduledDecodes) != 2 || steps[0].ScheduledPrefillChunks["prefill-A"] != 512 || steps[0].TotalTokensInStep != 514 {
		t.Fatalf("step 0 = %+v, want decodes=[dec-1 dec-2], prefill-A=512, total=514", steps[0])
	}
	// Step 1: dec-1, dec-2 (2 tokens) + prefill-B (512 tokens) = 514 tokens.
	if steps[1].StepIndex != 1 || len(steps[1].ScheduledDecodes) != 2 || steps[1].ScheduledPrefillChunks["prefill-B"] != 512 || steps[1].TotalTokensInStep != 514 {
		t.Fatalf("step 1 = %+v, want decodes=[dec-1 dec-2], prefill-B=512, total=514", steps[1])
	}

	// Also verify multi-step chunking when a single prefill spans multiple steps.
	stepsSplit, err := PlanChunkedPrefillBatch(requests, 258) // 2 decodes + 256 prefill tokens/step -> 4 steps
	if err != nil {
		t.Fatalf("PlanChunkedPrefillBatch(258) failed: %v", err)
	}
	if len(stepsSplit) != 4 {
		t.Fatalf("len(stepsSplit) = %d, want 4", len(stepsSplit))
	}
	for i, st := range stepsSplit {
		if st.TotalTokensInStep != 258 {
			t.Fatalf("stepsSplit[%d].TotalTokensInStep = %d, want 258", i, st.TotalTokensInStep)
		}
	}
}

func ExampleRadixPageAllocator() {
	alloc, _ := NewRadixPageAllocator(16, 450000, 2, 4)
	p0 := makeTokenPage(10, 16)
	p1 := makeTokenPage(20, 16)
	p2 := makeTokenPage(30, 16)
	p3 := makeTokenPage(40, 16)

	// Populate [p0, p1] in HBM3e and release.
	_, _ = alloc.MatchOrAllocatePrefix("req-init", [][]int32{p0, p1})
	_ = alloc.ReleaseRequest("req-init")

	// Allocate [p2, p3], demoting [p0, p1] from HBM3e to Grace LPDDR5X C2C.
	rEvict, _ := alloc.MatchOrAllocatePrefix("req-evict", [][]int32{p2, p3})
	_ = alloc.ReleaseRequest("req-evict")

	// Reuse [p0, p1]: promoted from Grace LPDDR5X C2C back to HBM3e over NVLink-C2C (450 GB/s).
	rPromote, _ := alloc.MatchOrAllocatePrefix("req-promote", [][]int32{p0, p1})

	steps, _ := PlanChunkedPrefillBatch([]ChunkedBatchRequest{
		{ReqID: "decode-0", IsDecode: true},
		{ReqID: "req-promote", TotalPromptTokens: 64, CachedPrefixTokens: rPromote.PrefillTokensSaved},
	}, 17)

	fmt.Printf("evicted_to_c2c=%d c2c_promoted=%d promote_us=%.1f saved_tokens=%d steps=%d\n",
		rEvict.EvictedToC2CPages,
		rPromote.C2CPromotedPages,
		rPromote.C2CPromoteLatencyUs,
		rPromote.PrefillTokensSaved,
		len(steps),
	)
	// Output:
	// evicted_to_c2c=2 c2c_promoted=2 promote_us=2.0 saved_tokens=32 steps=2
}
