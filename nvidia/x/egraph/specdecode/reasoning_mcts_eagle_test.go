package specdecode

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func TestReasoningMCTSPruneBranchesAndPrefixReuse(t *testing.T) {
	scores := []float32{0.21, 0.88, 0.45, 0.94, 0.72, 0.61, 0.15, 0.83}
	surviving, reuse := PruneReasoningBranchesOnDevice(scores, 0.60, 3)

	want := []int{3, 1, 7} // 0.94 (idx 3), 0.88 (idx 1), 0.83 (idx 7)
	if !slices.Equal(surviving, want) {
		t.Fatalf("PruneReasoningBranchesOnDevice surviving = %v, want %v", surviving, want)
	}
	if reuse != 3.0 {
		t.Fatalf("prefixTileReuseFactor = %v, want 3.0", reuse)
	}

	// Verify ReasoningMCTSTree state update and CUDA emission.
	tree, err := NewReasoningMCTSTree(512, 8, 3, 0.60)
	if err != nil {
		t.Fatalf("NewReasoningMCTSTree: %v", err)
	}
	if gotPages := tree.SharedPrefixPages(); gotPages != 32 {
		t.Fatalf("SharedPrefixPages = %d, want 32", gotPages)
	}
	gotSurv, gotReuse, err := tree.Prune(scores)
	if err != nil {
		t.Fatalf("tree.Prune: %v", err)
	}
	if !slices.Equal(gotSurv, want) || gotReuse != 3.0 {
		t.Fatalf("tree.Prune = (%v, %v), want (%v, 3.0)", gotSurv, gotReuse, want)
	}
	activeCount := 0
	for _, b := range tree.Branches {
		if b.Active {
			activeCount++
		}
	}
	if activeCount != 3 {
		t.Fatalf("active branches after Prune = %d, want 3", activeCount)
	}

	cudaSrc := tree.CUDA()
	for _, sub := range []string{
		"reasoning_mcts_prm_tree_device",
		"branch_scores",
		"min_prm_threshold",
		"surviving_indices",
		"__shfl_xor_sync",
	} {
		if !strings.Contains(cudaSrc, sub) {
			t.Fatalf("tree.CUDA() missing %q\n%s", sub, cudaSrc)
		}
	}

	// Verify fallback when all branches score below threshold: highest-scoring branch survives.
	lowScores := []float32{0.10, 0.42, 0.25, float32(math.NaN())}
	fallbackSurv, fallbackReuse := PruneReasoningBranchesOnDevice(lowScores, 0.90, 2)
	if !slices.Equal(fallbackSurv, []int{1}) || fallbackReuse != 1.0 {
		t.Fatalf("fallback surviving = (%v, %v), want ([1], 1.0)", fallbackSurv, fallbackReuse)
	}
}

func TestReasoningMCTSFusedEAGLE3TargetDraftMegakernel(t *testing.T) {
	const (
		m            = 2
		targetHidden = 256
		draftHidden  = 256
		draftSteps   = 2
		topK         = 4
		eps          = 1e-5
	)

	// Construct a representative target verification post-attention plan ending in BF16 hidden state.
	bTarget := hopper.NewBuilder()
	xIn := bTarget.Input("x_in", hopper.BF16, m, targetHidden)
	wTarget := bTarget.Input("w_target", hopper.BF16, targetHidden, targetHidden)
	proj := bTarget.MatMul(xIn, wTarget)
	normW := bTarget.Input("target_norm", hopper.F32, targetHidden)
	targetOut := bTarget.Cast(hopper.BF16, bTarget.RMSNorm(proj, normW, targetHidden, eps))

	targetExpr, err := hopper.Optimize(bTarget.G, targetOut)
	if err != nil {
		t.Fatalf("Optimize target: %v", err)
	}
	targetPlan, err := hopper.Compile(targetExpr, "target_verify", hopper.BF16)
	if err != nil {
		t.Fatalf("Compile target: %v", err)
	}
	targetStageCount := len(targetPlan.Kernels)

	fusedPlan, mega, err := BuildFusedEAGLE3TargetDraftMegakernel(targetPlan, draftHidden, draftSteps, topK)
	if err != nil {
		t.Fatalf("BuildFusedEAGLE3TargetDraftMegakernel: %v", err)
	}
	if fusedPlan == nil || mega == nil {
		t.Fatal("expected non-nil fusedPlan and mega")
	}
	if len(mega.Stages) <= targetStageCount {
		t.Fatalf("len(mega.Stages) = %d, want > %d (target stages + EAGLE-3 draft stages)", len(mega.Stages), targetStageCount)
	}

	// Verify stage attribution spans both target_ and eagle3_ stages and emits valid CUDA.
	hasTargetStage := false
	hasEagleStage := false
	for _, st := range mega.Stages {
		if strings.HasPrefix(st.Name, "target_") {
			hasTargetStage = true
		}
		if strings.HasPrefix(st.Name, "eagle3_") {
			hasEagleStage = true
		}
	}
	if !hasTargetStage || !hasEagleStage {
		t.Fatalf("expected both target_ and eagle3_ stages in megakernel, got stages: %+v", mega.Stages)
	}

	src, err := mega.Source()
	if err != nil {
		t.Fatalf("mega.Source(): %v", err)
	}
	for _, sub := range []string{
		"target_verify_eagle3_fused",
		"eg_ready_wait",
		"eg_ready_signal",
	} {
		if !strings.Contains(src, sub) {
			t.Fatalf("fused EAGLE-3 megakernel source missing %q", sub)
		}
	}
}

func TestReasoningMCTSToolCallTrapDoorCUDA(t *testing.T) {
	delims := []int32{128010, 128011, 32007}
	cuda := EmitToolCallTrapDoorCUDA(delims)

	wantSubstrings := []string{
		"eg_grammar_trap_door",
		"st.release.sys.global",
		"128010",
		"128011",
		"32007",
		"host_doorbell_sys",
		"host_payload_ring",
	}
	for _, sub := range wantSubstrings {
		if !strings.Contains(cuda, sub) {
			t.Fatalf("EmitToolCallTrapDoorCUDA missing %q\n--- source ---\n%s", sub, cuda)
		}
	}
}
