package cudasearch

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type numericalSearchCost struct{}

func (numericalSearchCost) NodeCost(n egraph.ENode, _ []float64) float64 {
	if n.Op == "fixture" {
		return 3
	}
	return 1
}

func TestSearchNumericalAcceptance(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		for _, exact := range []bool{false, true} {
			name := strategy + "/tolerant"
			if exact {
				name = strategy + "/exact"
			}
			t.Run(name, func(t *testing.T) {
				e, _, _, order := executableFixture(t)
				e.CostModel = numericalSearchCost{}
				e.Policy.Absolute = 0.01
				if exact {
					e.Workload.NumericalPolicy = NumericalPolicyExact
					e.Policy.Absolute = 0
				}
				legalValue := float32(1.001)
				if math.Float32bits(legalValue) == math.Float32bits(1) {
					t.Fatal("fixture must differ in bytes")
				}
				compiled := make(map[string]*testExecutable)
				e.Compile = func(_ context.Context, expr *egraph.Expr, _ WorkloadBindings) (Executable, error) {
					x := &testExecutable{id: expr.Op, value: 1, kind: cudatrial.TimingSummedKernel, duration: 10 * time.Microsecond, order: order}
					switch expr.Op {
					case "legal":
						x.value, x.duration = legalValue, 5*time.Microsecond
					case "wrong":
						x.value, x.duration = 2, time.Nanosecond
					case "fixture":
					default:
						t.Fatalf("unexpected expression %s", expr.Op)
					}
					compiled[expr.Op] = x
					return x, nil
				}
				rules := []*egraph.Rewrite{egraph.MustRewrite("legal", "fixture", "legal"), egraph.MustRewrite("wrong", "fixture", "wrong")}
				budget := SearchBudget{MaxDepth: 1, MaxEvaluations: 3}
				var result *SearchResult
				var err error
				if strategy == "beam" {
					result, err = NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), executableState(), rules, e)
				} else {
					result, err = NewMCTSSearch(MCTSConfig{Budget: budget}).Search(context.Background(), executableState(), rules, e)
				}
				if err != nil {
					t.Fatal(err)
				}
				want, rejections := "legal", 1
				if exact {
					want, rejections = "fixture", 2
				}
				if result.Winner.ID != want || len(result.Rejected) != rejections || len(compiled) != 3 {
					t.Fatalf("winner=%s rejected=%d compiled=%v", result.Winner.ID, len(result.Rejected), compiled)
				}
				if !exact && (result.Winner.Comparison["out"].MaxAbsolute <= 0 || result.Winner.Comparison["out"].MaxAbsolute > 0.01) {
					t.Fatal("lost permitted numerical deviation")
				}
				for _, r := range result.Rejected {
					if !errors.Is(r.Err, ErrVerificationFailed) || r.Evidence == nil || len(r.Evidence.Trials) != 0 {
						t.Fatalf("rejection=%+v", r)
					}
				}
				if slices.Contains(*order, "wrong run") || (exact && slices.Contains(*order, "legal run")) {
					t.Fatal("timed rejected outputs")
				}
				for _, x := range compiled {
					if x.closed != 1 {
						t.Fatalf("%s closed %d times", x.id, x.closed)
					}
				}
			})
		}
	}
}
