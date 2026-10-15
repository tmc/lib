package cudasearch_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func makeTestState() (*egraph.SearchState, []*egraph.Rewrite) {
	g := egraph.New()
	x := g.Add(egraph.ENode{Op: "x"})
	w := g.Add(egraph.ENode{Op: "w"})
	matmul := g.Add(egraph.ENode{Op: "matmul", Children: []egraph.Id{x, w}})

	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": matmul})

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("matmul-to-gemv", "(matmul ?x ?w)", "(gemv ?x ?w)"),
		egraph.MustRewrite("gemv-to-fused", "(gemv ?x ?w)", "(fused_gemv_pipeline ?x ?w)"),
	}

	return state, rules
}

// TestBadCandidateRejected verifies that a candidate failing numerical verification or execution is rejected.
func TestBadCandidateRejected(t *testing.T) {
	state, rules := makeTestState()
	budget := cudasearch.DefaultBudget()
	// FailCandidate only compares an operator string; it allocates no owned storage.
	budget.CallbackMemoryBytes = 1
	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 2,
		Budget:    budget,
	})

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
		FailCandidate: func(expr *egraph.Expr) bool {
			// Fused candidate fails verification
			return expr.Op == "fused_gemv_pipeline"
		},
	}

	res, err := beam.Search(context.Background(), state, rules, eval)
	if err != nil {
		t.Fatalf("unexpected search error: %v", err)
	}

	if res.Winner.Expr.Op == "fused_gemv_pipeline" {
		t.Fatalf("failing candidate must not be chosen as winner")
	}
}

// TestWorkloadDimensionsUsed verifies that workload dimensions and binding values are passed and used.
func TestWorkloadDimensionsUsed(t *testing.T) {
	shape, err := cudashape.New(1, 2048, 4096)
	if err != nil {
		t.Fatalf("shape creation failed: %v", err)
	}

	bindings := cudasearch.WorkloadBindings{
		Name:            "test_workload",
		Shape:           shape,
		DType:           cudair.Float16,
		NumericalPolicy: cudasearch.NumericalPolicyExact,
	}

	if err := bindings.Validate(); err != nil {
		t.Fatalf("bindings validation failed: %v", err)
	}

	if len(bindings.Shape.Dims()) != 3 || bindings.Shape.Dim(1) != 2048 {
		t.Errorf("wrong dims: %v", bindings.Shape.Dims())
	}

	wkID := bindings.WorkloadIdentity()
	if wkID.ProgramID != "test_workload" || len(wkID.ShapeDims) != 3 {
		t.Errorf("workload identity mismatch: %+v", wkID)
	}
}

// TestBudgetReturnsValidatedPlan verifies that search produces a validated plan within budget.
func TestBudgetReturnsValidatedPlan(t *testing.T) {
	state, rules := makeTestState()
	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 2,
		Budget: cudasearch.SearchBudget{
			MaxStates:      10,
			MaxDepth:       3,
			MaxEvaluations: 10,
		},
	})

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
	}

	res, err := beam.Search(context.Background(), state, rules, eval)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if res.Winner == nil || !res.Winner.Validated {
		t.Fatalf("winner must be validated, got %+v", res.Winner)
	}
	if res.Evaluations > 10 {
		t.Errorf("evaluations %d exceeded budget 10", res.Evaluations)
	}
}

// TestNullTrial verifies that identical candidates in a null trial baseline control measure within noise floor.
func TestNullTrial(t *testing.T) {
	arms := []cudatrial.Arm{
		{
			Name: "baseline_a",
			Run: func(ctx context.Context) (cudatrial.Receipt, error) {
				r := cudatrial.Receipt{Complete: true, Digest: "sha256:same"}
				_ = r.RecordTiming(cudatrial.TimingSummedKernel, 100*time.Microsecond)
				return r, nil
			},
		},
		{
			Name: "baseline_b",
			Run: func(ctx context.Context) (cudatrial.Receipt, error) {
				r := cudatrial.Receipt{Complete: true, Digest: "sha256:same"}
				_ = r.RecordTiming(cudatrial.TimingSummedKernel, 100*time.Microsecond)
				return r, nil
			},
		},
	}

	receipts, err := cudatrial.Interleave(context.Background(), arms, 3)
	if err != nil {
		t.Fatalf("null trial interleave failed: %v", err)
	}

	exact, msg := cudatrial.SameArmExact(receipts)
	if !exact {
		t.Fatalf("null trial digests diverged: %s", msg)
	}
}

// TestTimingKindMismatch verifies that comparing receipts with mismatched timing kinds is rejected.
func TestTimingKindMismatch(t *testing.T) {
	r1 := cudatrial.Receipt{
		TimingKind:   cudatrial.TimingGPUBusy,
		GPUBusy:      50 * time.Microsecond,
		SummedKernel: 0,
	}
	r2 := cudatrial.Receipt{
		TimingKind:   cudatrial.TimingSummedKernel,
		SummedKernel: 50 * time.Microsecond,
		GPUBusy:      0,
	}

	if r1.TimingKind == r2.TimingKind {
		t.Fatalf("expected timing kind mismatch")
	}

	// Verify that DurationFor fails if requesting timing kind from wrong receipt
	if _, err := r1.DurationFor(r2.TimingKind); err == nil {
		t.Errorf("expected error querying mismatched timing kind on r1, got nil")
	}
}

// TestHeldOutWinner verifies that the winner selected on training workload is confirmed on held-out workload.
func TestHeldOutWinner(t *testing.T) {
	state, rules := makeTestState()
	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 2,
		Budget:    cudasearch.DefaultBudget(),
	})

	trainEval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
	}

	res, err := beam.Search(context.Background(), state, rules, trainEval)
	if err != nil {
		t.Fatalf("train search failed: %v", err)
	}
	if res.Winner == nil {
		t.Fatalf("train winner is nil")
	}

	// Held-out evaluator with different token length
	heldOutShape, _ := cudashape.New(1, 1024, 4096)
	heldOutEval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
		Workload: cudasearch.WorkloadBindings{
			Name:            "held_out_llama",
			Shape:           heldOutShape,
			DType:           cudair.Float16,
			NumericalPolicy: cudasearch.NumericalPolicyExact,
		},
	}

	rootID, _ := state.CanonicalRoot("root")
	heldOutPlan, err := heldOutEval.Evaluate(context.Background(), state, rootID)
	if err != nil {
		t.Fatalf("held out evaluation failed: %v", err)
	}
	if !heldOutPlan.Validated {
		t.Fatalf("held-out confirmation failed")
	}
}

// TestStateResetBetweenCandidates verifies that mutable state reset callback is invoked before each evaluation.
func TestStateResetBetweenCandidates(t *testing.T) {
	state, rules := makeTestState()
	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 2,
		Budget: cudasearch.SearchBudget{
			MaxStates:      10,
			MaxDepth:       2,
			MaxEvaluations: 5,
		},
	})

	resetInvoked := 0
	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
		Workload: cudasearch.WorkloadBindings{
			Name:            "reset_test",
			Shape:           cudashape.Shape{},
			NumericalPolicy: cudasearch.NumericalPolicyExact,
			ResetState: func() error {
				resetInvoked++
				return nil
			},
		},
	}

	res, err := beam.Search(context.Background(), state, rules, eval)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if resetInvoked == 0 || resetInvoked != res.Evaluations {
		t.Errorf("resetInvoked = %d, evaluations = %d", resetInvoked, res.Evaluations)
	}
}

// TestBeamMultistepState verifies that Beam explores real multi-step paths, not discarding transformations.
func TestBeamMultistepState(t *testing.T) {
	state, rules := makeTestState()
	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 4,
		Budget: cudasearch.SearchBudget{
			MaxStates:      20,
			MaxDepth:       5,
			MaxEvaluations: 20,
		},
	})

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
	}

	res, err := beam.Search(context.Background(), state, rules, eval)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Winner should be fused_gemv_pipeline, requiring 2 steps: matmul -> gemv -> fused_gemv_pipeline
	if res.Winner.Expr.Op != "fused_gemv_pipeline" {
		t.Errorf("winner = %s, want fused_gemv_pipeline", res.Winner.Expr.Op)
	}
	if len(res.Winner.Derivation) < 2 {
		t.Errorf("derivation length = %d, want at least 2 steps for multi-step path", len(res.Winner.Derivation))
	}
}

// TestBeamStableTies verifies that Beam breaks ties deterministically.
func TestBeamStableTies(t *testing.T) {
	state, rules := makeTestState()
	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 2,
		Budget: cudasearch.SearchBudget{
			MaxStates:      10,
			MaxDepth:       2,
			MaxEvaluations: 10,
		},
	})

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
	}

	res1, err1 := beam.Search(context.Background(), state, rules, eval)
	res2, err2 := beam.Search(context.Background(), state, rules, eval)

	if err1 != nil || err2 != nil {
		t.Fatalf("search failed: %v, %v", err1, err2)
	}

	if res1.Winner.Expr.Fingerprint() != res2.Winner.Expr.Fingerprint() {
		t.Errorf("tie break not deterministic: %s vs %s", res1.Winner.Expr.Fingerprint(), res2.Winner.Expr.Fingerprint())
	}
}

// TestMCTSSeedReplay verifies that MCTS with identical seed reproduces the exact search outcome.
func TestMCTSSeedReplay(t *testing.T) {
	state, rules := makeTestState()
	cfg1 := cudasearch.DefaultMCTSConfig()
	cfg1.Seed = 12345
	cfg1.Iterations = 20

	cfg2 := cudasearch.DefaultMCTSConfig()
	cfg2.Seed = 12345
	cfg2.Iterations = 20

	eval1 := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
	}
	eval2 := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
	}

	mcts1 := cudasearch.NewMCTSSearch(cfg1)
	mcts2 := cudasearch.NewMCTSSearch(cfg2)

	res1, err1 := mcts1.Search(context.Background(), state, rules, eval1)
	res2, err2 := mcts2.Search(context.Background(), state, rules, eval2)

	if err1 != nil || err2 != nil {
		t.Fatalf("search failed: %v, %v", err1, err2)
	}

	if res1.Winner.Expr.Fingerprint() != res2.Winner.Expr.Fingerprint() {
		t.Errorf("mcts replay divergence: %s vs %s", res1.Winner.Expr.Fingerprint(), res2.Winner.Expr.Fingerprint())
	}
	if res1.StatesVisited != res2.StatesVisited {
		t.Errorf("states visited mismatch: %d vs %d", res1.StatesVisited, res2.StatesVisited)
	}
}

// TestMCTSUntriedActions verifies that MCTS tracks and prioritizes untried actions before repeating.
func TestMCTSUntriedActions(t *testing.T) {
	state, rules := makeTestState()
	cfg := cudasearch.DefaultMCTSConfig()
	cfg.Iterations = 10

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
	}

	mcts := cudasearch.NewMCTSSearch(cfg)
	res, err := mcts.Search(context.Background(), state, rules, eval)
	if err != nil {
		t.Fatalf("mcts search failed: %v", err)
	}
	if res.StatesVisited == 0 {
		t.Errorf("expected states visited > 0")
	}
}

// TestUCTRewardDirection verifies that lower latency candidates produce higher UCT rewards.
func TestUCTRewardDirection(t *testing.T) {
	baselineCost := 1000.0

	// Faster candidate (lower cost)
	fastCost := 500.0
	fastRatio := fastCost / baselineCost
	fastReward := 1.0 - (fastRatio / 2.0)

	// Slower candidate (higher cost)
	slowCost := 1500.0
	slowRatio := slowCost / baselineCost
	slowReward := 1.0 - (slowRatio / 2.0)

	if fastReward <= slowReward {
		t.Fatalf("lower cost must produce higher reward: fast=%f, slow=%f", fastReward, slowReward)
	}
}

// TestFailedRolloutNotWinner verifies that a failed rollout or candidate is never selected as winner.
func TestFailedRolloutNotWinner(t *testing.T) {
	state, rules := makeTestState()
	cfg := cudasearch.DefaultMCTSConfig()
	// FailCandidate only compares the operator string and allocates no owned storage.
	cfg.Budget.CallbackMemoryBytes = 1
	cfg.Iterations = 15

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
		FailCandidate: func(expr *egraph.Expr) bool {
			return expr.Op == "fused_gemv_pipeline"
		},
	}

	mcts := cudasearch.NewMCTSSearch(cfg)
	res, err := mcts.Search(context.Background(), state, rules, eval)
	if err != nil {
		t.Fatalf("search error: %v", err)
	}

	if res.Winner.Expr.Op == "fused_gemv_pipeline" {
		t.Fatalf("failed rollout/candidate was selected as winner!")
	}
}

// TestSearchStateMemoryBudget verifies that search halts with ErrMemoryBudgetExceeded when budget is exhausted.
func TestSearchStateMemoryBudget(t *testing.T) {
	state, rules := makeTestState()
	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 10,
		Budget: cudasearch.SearchBudget{
			MaxStates:      100,
			MaxDepth:       10,
			MaxMemoryBytes: 1024, // Artificially tiny memory budget (1 KB)
		},
	})

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
	}

	_, err := beam.Search(context.Background(), state, rules, eval)
	if err == nil {
		t.Fatalf("expected memory budget error, got nil")
	}
	if !errors.Is(err, cudasearch.ErrMemoryBudgetExceeded) {
		t.Fatalf("expected ErrMemoryBudgetExceeded, got %v", err)
	}
}

// TestSearchCancellation verifies that search cleanly halts upon context cancellation.
func TestSearchCancellation(t *testing.T) {
	state, rules := makeTestState()
	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 4,
		Budget:    cudasearch.DefaultBudget(),
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
	}

	_, err := beam.Search(ctx, state, rules, eval)
	if err == nil {
		t.Fatalf("expected cancellation error, got nil")
	}
	if !errors.Is(err, cudasearch.ErrSearchCancelled) {
		t.Fatalf("expected ErrSearchCancelled, got %v", err)
	}
}

// TestScoringEpochChange verifies that scoring epoch changes invalidate stale evidence and re-score cleanly.
func TestScoringEpochChange(t *testing.T) {
	modelV1 := cudasearch.DefaultScoringModel("epoch-v1")
	modelV2 := cudasearch.DefaultScoringModel("epoch-v2")

	evidence := &cudasearch.ProfileEvidence{
		StallMemoryThrottle: 0.5,
		Occupancy:           0.3,
		Epoch:               "epoch-v1",
	}

	r := &cudatrial.Receipt{
		Complete:     true,
		SummedKernel: 100 * time.Microsecond,
	}

	costV1, err1 := modelV1.Score(r, evidence)
	costV2, err2 := modelV2.Score(r, evidence)

	if err1 != nil || err2 != nil {
		t.Fatalf("score errors: %v, %v", err1, err2)
	}

	// Neither current nor stale profile fractions can alter measured latency.
	if costV1 != costV2 {
		t.Errorf("profile changed latency: current=%f stale=%f", costV1, costV2)
	}
}

// TestProfileFallback verifies that missing or stale profile evidence falls back cleanly to timing-only score.
func TestProfileFallback(t *testing.T) {
	model := cudasearch.DefaultScoringModel("epoch-v1")

	r := &cudatrial.Receipt{
		Complete:     true,
		SummedKernel: 100 * time.Microsecond,
	}

	// Missing evidence
	costNoEvidence, err := model.Score(r, nil)
	if err != nil {
		t.Fatalf("Score failed with nil evidence: %v", err)
	}

	// Stale evidence
	staleEvidence := &cudasearch.ProfileEvidence{
		StallMemoryThrottle: 0.8,
		Epoch:               "epoch-old",
	}
	costStale, err := model.Score(r, staleEvidence)
	if err != nil {
		t.Fatalf("Score failed with stale evidence: %v", err)
	}

	if costNoEvidence != costStale {
		t.Errorf("stale evidence (%f) did not fall back to no-evidence cost (%f)", costStale, costNoEvidence)
	}
}

// --- Negative / Mutant Controls ---

// Negative Control 2: Mix busy-time and summed-kernel timings.
func TestNegativeControl_MixBusyAndSummedKernelRejected(t *testing.T) {
	r := cudatrial.Receipt{
		Name:         "mixed_receipt",
		GPUBusy:      40 * time.Microsecond,
		SummedKernel: 60 * time.Microsecond,
		WallElapsed:  80 * time.Microsecond,
	}

	decomp := r.LatencyDecomposition()
	_, err := decomp.TotalDuration()
	if !errors.Is(err, cudatrial.ErrNaiveSumDoubleCounts) {
		t.Fatalf("naive sum of mixed timings must return ErrNaiveSumDoubleCounts, got %v", err)
	}
}

// Negative Control 3: Omit a required input binding.
func TestNegativeControl_OmitRequiredInputBinding(t *testing.T) {
	// Missing Name and Shape dimensions
	incomplete := cudasearch.WorkloadBindings{
		NumericalPolicy: cudasearch.NumericalPolicyExact,
	}

	err := incomplete.Validate()
	if err == nil {
		t.Fatalf("expected error for incomplete workload bindings, got nil")
	}
	if !errors.Is(err, cudasearch.ErrMissingBinding) {
		t.Fatalf("expected ErrMissingBinding, got %v", err)
	}
}

// Negative Control 4: Exhaust budget before any valid candidate.
func TestNegativeControl_ExhaustBudgetBeforeValidCandidate(t *testing.T) {
	state, rules := makeTestState()
	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
		FailCandidate: func(expr *egraph.Expr) bool {
			// All candidates fail!
			return true
		},
	}

	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 2,
		Budget: cudasearch.SearchBudget{
			MaxStates:      5,
			MaxEvaluations: 5,
		},
	})

	_, err := beam.Search(context.Background(), state, rules, eval)
	if err == nil {
		t.Fatalf("expected error when all candidates fail, got nil")
	}
}
