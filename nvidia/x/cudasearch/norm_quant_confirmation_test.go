package cudasearch

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// Host control for the qualification flow. Outputs are fixture/reference data;
// these executables neither compile CUDA nor measure physical device time.
type normFixtureExecutable struct {
	testExecutable
	outputs map[string][]float32
}

func (x *normFixtureExecutable) Outputs(context.Context) (map[string][]float32, error) {
	return x.outputs, nil
}

func TestNormQuantConfirmationAndProposal(t *testing.T) {
	for _, corrupt := range []bool{false, true} {
		name := "valid"
		if corrupt {
			name = "wrong-held-output"
		}
		t.Run(name, func(t *testing.T) {
			order := []string{}
			var compiled []*normFixtureExecutable
			newEvaluator := func(round int) *ExecutableEvaluator {
				w, reference := normQuantSearchFixture(round)
				compiler, err := NewNormQuantCompiler(&cuda.Device{}, "sm_121a", w, func() bool { return false })
				if err != nil {
					t.Fatal(err)
				}
				bindings := compiler.Bindings()
				baseline := &normFixtureExecutable{testExecutable: testExecutable{id: "baseline/" + bindings.Key(), kind: cudatrial.TimingGPUBusy, duration: 10 * time.Microsecond, order: &order}, outputs: reference}
				return &ExecutableEvaluator{Generation: 1, Workload: bindings, Baseline: baseline, References: reference,
					Rounds: 2, TimingKind: cudatrial.TimingGPUBusy, HardwareLockChecker: func() bool { return true }, CostModel: normQuantExtractionCost{},
					Compile: func(_ context.Context, expr *egraph.Expr, _ WorkloadBindings) (Executable, error) {
						x := &normFixtureExecutable{testExecutable: testExecutable{id: expr.Op + "/" + bindings.Key(), kind: cudatrial.TimingGPUBusy, duration: 10 * time.Microsecond, order: &order}, outputs: compiler.References()}
						if expr.Op == "norm_quant_fused" {
							x.duration = 5 * time.Microsecond
						}
						if corrupt && round == 1 {
							x.outputs["q8_bytes"][143]++
						}
						compiled = append(compiled, x)
						return x, nil
					},
				}
			}
			g := egraph.New()
			root := g.Add(egraph.ENode{Op: "norm_quant_unfused"})
			s := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
			result, err := NewBeamSearch(BeamConfig{EvaluatorID: "norm-quant-host-fixture-v1", Budget: SearchBudget{MaxDepth: 1, MaxEvaluations: 2}}).Search(context.Background(), s,
				[]*egraph.Rewrite{egraph.MustRewrite("fuse", "norm_quant_unfused", "norm_quant_fused")}, newEvaluator(0))
			if err != nil {
				t.Fatal(err)
			}
			if result.Winner.Expr.Op != "norm_quant_fused" || !result.WorkAccounted || result.Compilations != 2 {
				t.Fatalf("result=%+v", result)
			}
			held := newEvaluator(1)
			candidate, err := held.Confirm(context.Background(), result.Winner)
			if corrupt {
				if !errors.Is(err, ErrVerificationFailed) || candidate != nil || len(held.Receipts) != 0 || held.Comparison["q8_bytes"].MaxAbsolute != 1 {
					t.Fatalf("candidate=%v err=%v comparison=%v", candidate, err, held.Comparison)
				}
				if _, err := result.ProposePromotion(candidate, 1); err == nil {
					t.Fatal("proposed rejected confirmation")
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				proposal, err := result.ProposePromotion(candidate, 1)
				if err != nil {
					t.Fatal(err)
				}
				if proposal.HeldOutComparison["q8_bytes"].Elements != 144 || len(proposal.HeldOutTrials) != 4 {
					t.Fatal("lost packed-output qualification")
				}
			}
			secondary := newEvaluator(2)
			candidate, err = secondary.Evaluate(context.Background(), s, root)
			if err != nil {
				t.Fatal(err)
			}
			if candidate.ConfirmationOf != "" || candidate.Comparison["norm"].Elements != 128 {
				t.Fatal("secondary output mislabeled or omitted")
			}
			if _, err := result.ProposePromotion(candidate, 1); err == nil {
				t.Fatal("proposed unrelated secondary-output case")
			}
			for _, x := range compiled {
				if x.closed != 1 {
					t.Fatalf("owner %s closed %d times", x.id, x.closed)
				}
			}
		})
	}
}
