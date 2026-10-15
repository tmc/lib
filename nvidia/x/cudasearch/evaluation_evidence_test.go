package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleEvaluationEvidence() {
	order := []string{}
	baseline := &testExecutable{id: "baseline", value: 1, order: &order}
	candidate := &testExecutable{id: "wrong-output", value: 2, order: &order}
	eval := &ExecutableEvaluator{
		Workload: WorkloadBindings{Name: "host-fixture", InputNames: []string{"x"}, InputIdentities: map[string]string{"x": "synthetic-fixture-x"}, NumericalPolicy: NumericalPolicyExact},
		Baseline: baseline, References: map[string][]float32{"out": {1}},
		Rounds: 2, TimingKind: cudatrial.TimingSummedKernel,
		Compile:             func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { return candidate, nil },
		HardwareLockChecker: func() bool { return true }, // Synthetic host executables only.
	}
	result, err := NewBeamSearch(BeamConfig{}).Search(context.Background(), executableState(), nil, eval)
	evidence := result.Rejected[0].Evidence
	fmt.Println(errors.Is(err, ErrVerificationFailed), evidence.ExecutableID, evidence.ComparisonArm)
	fmt.Println(evidence.Comparison["out"].MaxAbsolute, len(evidence.Trials))
	// Output:
	// true wrong-output candidate
	// 1 0
}

func TestSearchRetainsRejectedEvidence(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		for _, failure := range []string{"outputs", "timing", "initial"} {
			t.Run(strategy+"/"+failure, func(t *testing.T) {
				e, _, compiled, _ := executableFixture(t)
				compile := e.Compile
				calls := 0
				e.Compile = func(ctx context.Context, expr *egraph.Expr, b WorkloadBindings) (Executable, error) {
					calls++
					if calls > 1 || failure == "initial" {
						if failure == "timing" {
							compiled.kind = cudatrial.TimingGPUBusy
						} else {
							compiled.value = 2
						}
					}
					return compile(ctx, expr, b)
				}
				s := executableState()
				rules := []*egraph.Rewrite{egraph.MustRewrite("alternative", "fixture", "alternative")}
				budget := SearchBudget{MaxDepth: 1, MaxEvaluations: 2}
				var result *SearchResult
				var err error
				if strategy == "beam" {
					result, err = NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), s, rules, e)
				} else {
					result, err = NewMCTSSearch(MCTSConfig{Budget: budget}).Search(context.Background(), s, rules, e)
				}
				if failure == "initial" {
					if !errors.Is(err, ErrVerificationFailed) || result == nil || result.Winner != nil {
						t.Fatalf("result=%+v err=%v", result, err)
					}
				} else if err != nil || result == nil || result.Winner == nil {
					t.Fatalf("result=%+v err=%v", result, err)
				}
				if len(result.Rejected) != 1 {
					t.Fatalf("rejected=%+v", result.Rejected)
				}
				rejected := result.Rejected[0]
				evidence := rejected.Evidence
				if evidence == nil || evidence.ExecutableID != compiled.id || evidence.ComparisonArm != "candidate" {
					t.Fatalf("evidence=%+v", evidence)
				}
				if failure == "timing" {
					if !errors.Is(rejected.Err, ErrTimingKindMismatch) || len(evidence.Trials) != 2 || evidence.Trials[1].Err == nil {
						t.Fatalf("evidence=%+v", evidence)
					}
					e.Receipts[0].Timings[cudatrial.TimingSummedKernel] = time.Hour
					if evidence.Trials[0].Timings[cudatrial.TimingSummedKernel] == time.Hour {
						t.Fatal("aliased timings")
					}
				} else if len(evidence.Trials) != 0 || evidence.Comparison["out"].MaxAbsolute != 1 {
					t.Fatalf("evidence=%+v", evidence)
				}
				e.Comparison["out"] = OutputComparison{MaxAbsolute: 99}
				if evidence.Comparison["out"].MaxAbsolute == 99 {
					t.Fatal("aliased comparison")
				}
				if result.Winner != nil && len(result.Winner.Trials) != 4 {
					t.Fatal("lost accepted ABBA trials")
				}
			})
		}
	}
}

func TestExecutableTrialsDetached(t *testing.T) {
	e, _, _, _ := executableFixture(t)
	s := executableState()
	c, err := e.Evaluate(context.Background(), s, s.Roots["root"])
	if err != nil {
		t.Fatal(err)
	}
	e.Receipts[1].Provenance["executable_identity"] = "changed"
	e.Receipts[1].Timings[cudatrial.TimingSummedKernel] = time.Hour
	e.Comparison["out"] = OutputComparison{MaxAbsolute: 99}
	if c.Trials[1].Provenance["executable_identity"] == "changed" || c.Receipt.Provenance["executable_identity"] == "changed" || c.Trials[1].Timings[cudatrial.TimingSummedKernel] == time.Hour || c.Comparison["out"].MaxAbsolute == 99 {
		t.Fatal("candidate aliases evaluator evidence")
	}
}
