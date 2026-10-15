package cudasearch

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func extractionContext(t *testing.T, ctx context.Context, budget SearchBudget, eval Evaluator) (context.Context, *workBudget) {
	t.Helper()
	ctx, work, err := searchWorkBudget(ctx, budget, eval)
	if err != nil {
		t.Fatal(err)
	}
	return ctx, work
}

func TestExtractionMemoryCandidateAliases(t *testing.T) {
	state, _ := beamContractState()
	eval := &MockEvaluator{Scoring: DefaultScoringModel("epoch")}
	ctx, work := extractionContext(t, context.Background(), SearchBudget{MaxMemoryBytes: 1 << 20}, eval)
	candidate, err := evaluateSearch(ctx, eval, state, state.Roots["root"])
	if err != nil || candidate == nil || candidate.Expr == nil || candidate.extraction == nil || candidate.extractionLedger != work.memory {
		t.Fatal(candidate, err)
	}
	before := work.memory.live
	if before <= 0 {
		t.Fatal("result is uncharged")
	}
	workspace, err := work.memory.reserve(64)
	if err != nil {
		t.Fatal(err)
	}
	alias := *candidate
	result := &SearchResult{Winner: candidate, AllCandidates: []*CandidatePlan{candidate, &alias, candidate}}
	result.Close()
	after := work.memory.live
	if candidate.Expr != nil || alias.Expr != nil || after >= before+64 || after < 64 {
		t.Fatal("owner transfer or alias release failed", before, after)
	}
	result.Close()
	candidate.Close()
	alias.Close()
	if work.memory.live != after {
		t.Fatal("double release", work.memory.live, after)
	}
	workspace.release()
	if work.memory.live != after-64 {
		t.Fatal("closed unrelated workspace")
	}
	borrowed := &egraph.Expr{Op: "borrowed"}
	external := &CandidatePlan{Expr: borrowed}
	external.Close()
	if external.Expr != borrowed {
		t.Fatal("closed borrowed expression")
	}
}

func TestExtractionMemoryPublicSearchTransfer(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			state, rules := beamContractState()
			result, err := memoryStrategy(strategy, SearchBudget{MaxMemoryBytes: 1 << 20, MaxStates: 1}, state, rules, &MockEvaluator{Scoring: DefaultScoringModel("epoch")})
			if err != nil || result == nil || result.Winner == nil || result.Winner.Expr == nil || result.Winner.extraction == nil {
				t.Fatal(result, err)
			}
			ledger := result.Winner.extractionLedger
			if ledger == nil || ledger.live <= 0 {
				t.Fatal("missing retained ledger")
			}
			before := ledger.live
			result.Close()
			if ledger.live >= before {
				t.Fatal("extraction did not retain/release a reservation", before, ledger.live)
			}
			after := ledger.live
			result.Close()
			if ledger.live != after {
				t.Fatal("double result close")
			}
		})
	}
}

func TestExtractionMemoryMockFailureCleanup(t *testing.T) {
	state, _ := beamContractState()
	eval := &MockEvaluator{Scoring: DefaultScoringModel("epoch"), FailCandidate: func(*egraph.Expr) bool { return true }}
	ctx, work := extractionContext(t, context.Background(), SearchBudget{MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: 1}, eval)
	candidate, err := evaluateSearch(ctx, eval, state, state.Roots["root"])
	if candidate != nil || err == nil || work.memory.live != 0 {
		t.Fatal(candidate, err, work.memory.live)
	}
}

func TestExtractionMemoryExecutableOwnership(t *testing.T) {
	for _, stage := range []string{"success", "compile", "outputs", "cleanup", "cancel"} {
		t.Run(stage, func(t *testing.T) {
			eval, baseline, compiled, _ := executableFixture(t)
			state := executableState()
			parent, cancel := context.WithCancel(context.Background())
			defer cancel()
			failure := errors.New("fixture execution failure")
			original := eval.Compile
			if stage == "compile" {
				eval.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { return compiled, failure }
			}
			if stage == "outputs" {
				compiled.value = 2
			}
			if stage == "cleanup" {
				compiled.closeErr = failure
			}
			var captured *egraph.Expr
			if stage == "cleanup" {
				eval.Compile = func(ctx context.Context, expr *egraph.Expr, bindings WorkloadBindings) (Executable, error) {
					captured = expr
					return original(ctx, expr, bindings)
				}
			}
			if stage == "cancel" {
				eval.Compile = func(ctx context.Context, expr *egraph.Expr, bindings WorkloadBindings) (Executable, error) {
					x, err := original(ctx, expr, bindings)
					cancel()
					return x, err
				}
			}
			budget := SearchBudget{MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: 4096}
			ctx, work := extractionContext(t, parent, budget, eval)
			candidate, err := evaluateSearch(ctx, eval, state, state.Roots["root"])
			if baseline.closed != 0 || compiled.closed != 1 {
				t.Fatal("executable ownership changed", baseline.closed, compiled.closed)
			}
			if stage == "success" {
				if err != nil || candidate == nil || candidate.Expr == nil || candidate.extraction == nil {
					t.Fatal(candidate, err)
				}
				before := work.memory.live
				candidate.Close()
				if work.memory.live >= before {
					t.Fatal("executable candidate has no retained extraction")
				}
				if compiled.closed != 1 || baseline.closed != 0 {
					t.Fatal("candidate close closed executables")
				}
				return
			}
			if err == nil || candidate != nil {
				t.Fatal(candidate, err)
			}
			if stage == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			if stage == "cleanup" {
				if eval.pendingExtraction == nil || eval.pendingExtraction.Expr == nil || eval.pendingExtraction.Expr != captured || eval.pendingMemory == nil || work.memory.live <= budget.CallbackMemoryBytes+eval.pendingMemory.Bytes() {
					t.Fatal("pending executable lost extraction/slot/envelope ownership", work.memory.live)
				}
				held := work.memory.live
				if err := eval.Close(); !errors.Is(err, failure) || work.memory.live != held {
					t.Fatal("failed close released ownership", err, work.memory.live, held)
				}
				compiled.closeErr = nil
				if err := eval.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if work.memory.live != 0 {
				t.Fatal("failed evaluation retained extraction", stage, work.memory.live)
			}
		})
	}
}

func TestExtractionMemoryBeforeCompile(t *testing.T) {
	state := executableState()
	// This copied path is a storage fixture; this test does not replay it.
	state.Derivation = make([]egraph.Action, 128)
	measurement := &memoryBudget{limit: 1 << 20}
	owner, err := egraph.ExtractBoundedWithMemory(context.Background(), state.Graph, state.Roots["root"], egraph.AstSizeCost{}, egraph.ExtractionLimits{MaxWork: 1000, MaxNodes: 1000, MaxDepth: 1000}, measurement)
	if err != nil {
		t.Fatal(err)
	}
	retained := measurement.live
	owner.Close()
	if retained <= 0 || measurement.live != 0 {
		t.Fatal("invalid retained-storage measurement", retained, measurement.live)
	}
	pathBytes, err := storageBytes(state.Derivation)
	if err != nil {
		t.Fatal(err)
	}
	eval, baseline, compiled, _ := executableFixture(t)
	calls := 0
	original := eval.Compile
	eval.Compile = func(ctx context.Context, expr *egraph.Expr, bindings WorkloadBindings) (Executable, error) {
		calls++
		return original(ctx, expr, bindings)
	}
	const envelope = 4096
	limit := envelope + retained + pathBytes + int64(reflect.TypeFor[egraphcore.MemoryScope]().Size()+reflect.TypeFor[Executable]().Size()) - 1
	ctx, work := extractionContext(t, context.Background(), SearchBudget{MaxMemoryBytes: limit, CallbackMemoryBytes: envelope}, eval)
	candidate, err := evaluateSearch(ctx, eval, state, state.Roots["root"])
	if candidate != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || calls != 0 || work.compilations != 0 || work.memory.live != 0 || baseline.closed != 0 || compiled.closed != 0 {
		t.Fatal(candidate, err, calls, work.compilations, work.memory.live, baseline.closed, compiled.closed)
	}
}

type explicitMemoryCost struct{ call func() float64 }

func (m explicitMemoryCost) NodeCost(egraph.ENode, []float64) float64 { return m.call() }
func (m explicitMemoryCost) NodeCostWithMemory(egraph.ENode, []float64, *egraphcore.MemoryScope) (float64, error) {
	return m.call(), nil
}

type declaredMemoryCost struct{ call func() float64 }

func (m declaredMemoryCost) NodeCost(egraph.ENode, []float64) float64 { return m.call() }

func TestExtractionMemoryCallbackEnvelope(t *testing.T) {
	var observed [2]int64
	for i := range observed {
		state, _ := beamContractState()
		eval := &MockEvaluator{Scoring: DefaultScoringModel("epoch")}
		ctx, work := extractionContext(t, context.Background(), SearchBudget{MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: 256}, eval)
		call := func() float64 { observed[i] = work.memory.live; return 1 }
		if i == 0 {
			eval.CostModel = explicitMemoryCost{call: call}
		} else {
			eval.CostModel = declaredMemoryCost{call: call}
		}
		candidate, err := evaluateSearch(ctx, eval, state, state.Roots["root"])
		if err != nil {
			t.Fatal(err)
		}
		candidate.Close()
	}
	adapterBytes := int64(reflect.TypeFor[declaredCostModel]().Size() + reflect.TypeFor[egraphcore.MemoryScope]().Size())
	if observed[0] <= 256 || observed[1] != observed[0]+adapterBytes {
		t.Fatal("callback envelope charged more than once or absent", observed, adapterBytes)
	}
	state, _ := beamContractState()
	calls := 0
	eval := &MockEvaluator{CostModel: declaredMemoryCost{call: func() float64 { calls++; return 1 }}, Scoring: DefaultScoringModel("epoch")}
	ctx, work := extractionContext(t, context.Background(), SearchBudget{MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: 256}, eval)
	// A configured declaration is insufficient without the actual outer reserve.
	candidate, err := eval.Evaluate(ctx, state, state.Roots["root"])
	if candidate != nil || !errors.Is(err, ErrMemoryAccountingUnavailable) || calls != 0 || work.memory.live != 0 {
		t.Fatal(candidate, err, calls, work.memory.live)
	}
}

func TestExtractionMemoryAdmissionUnwind(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			state, rules := beamContractState()
			var candidates []*CandidatePlan
			model := &MockEvaluator{Scoring: DefaultScoringModel("epoch")}
			eval := beamEvaluatorFunc(func(ctx context.Context, state *egraph.SearchState, id egraph.Id) (*CandidatePlan, error) {
				candidate, err := model.Evaluate(ctx, state, id)
				if err == nil {
					candidates = append(candidates, candidate)
					if len(candidates) > 1 {
						candidate.CostSource = CostMeasured
					}
				}
				return candidate, err
			})
			result, err := memoryStrategy(strategy, SearchBudget{MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: 4096, MaxStates: 3, MaxDepth: 1, MaxEvaluations: 3}, state, rules, eval)
			if err != nil || result == nil || len(candidates) < 2 || len(result.Rejected) == 0 {
				t.Fatal(result, err, len(candidates))
			}
			if candidates[0].Expr == nil || candidates[0].extraction == nil {
				t.Fatal("closed admitted winner")
			}
			for _, candidate := range candidates[1:] {
				if candidate.Expr != nil || candidate.extraction != nil {
					t.Fatal("failed candidate owner not closed")
				}
			}
			result.Close()
			if candidates[0].Expr != nil {
				t.Fatal("winner owner not transferred")
			}
		})
	}
}
