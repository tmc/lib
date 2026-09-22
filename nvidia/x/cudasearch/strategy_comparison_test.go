package cudasearch

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// This is the entire fixture language and its independently declared latency
// oracle, not a score inferred from either search implementation. Device times
// are synthetic; the comparison proves mechanics and accounting only.
var strategyOracle = map[string]time.Duration{"start": 9 * time.Microsecond, "left": 5 * time.Microsecond, "right": 4 * time.Microsecond, "goal": time.Microsecond}

type strategyOracleCost struct{}

func (strategyOracleCost) NodeCost(n egraph.ENode, _ []float64) float64 {
	return float64(strategyOracle[n.Op])
}

// Test-only baselines. Saturation accumulates checked rewrites in one graph,
// extracting/evaluating after each action. Random allocation selects a retained
// branch and action with a local RNG. Neither uses unbounded whole-rule Apply.
func comparisonBaseline(t *testing.T, initial *egraph.SearchState, rules []*egraph.Rewrite, e *ExecutableEvaluator, b SearchBudget, random bool) *SearchResult {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), b.Timeout)
	defer cancel()
	ctx, work, err := searchWorkBudget(ctx, b, e)
	if err != nil {
		t.Fatal(err)
	}
	r := &SearchResult{StatesVisited: 1, WorkAccounted: true}
	r.Strategy = "saturation"
	if random {
		r.Strategy, r.Seed = "random", 42
	}
	defer func() {
		r.Extractions, r.Compilations, r.ExtractionWork = work.extractions, work.compilations, work.extractionWork
	}()
	evaluate := func(s *egraph.SearchState) {
		root, _ := s.CanonicalRoot("root")
		c, err := e.Evaluate(ctx, s, root)
		r.Evaluations++
		if err != nil {
			r.Rejected = append(r.Rejected, rejection(s, e, err))
			return
		}
		if err := admitSearchCandidate(c); err != nil {
			t.Fatal(err)
		}
		r.AllCandidates = append(r.AllCandidates, c)
		if r.Winner == nil || c.Cost < r.Winner.Cost {
			r.Winner = c
		}
	}
	evaluate(initial)
	type branch struct {
		state   *egraph.SearchState
		actions []egraph.ActionCandidate
	}
	frontier := []branch{{initial, egraph.EnumerateActionsBounded(initial.Graph, rules, b.MaxActions)}}
	rng := rand.New(rand.NewSource(42))
	for len(frontier) > 0 && r.ActionsAttempted < b.MaxTotalActions && r.StatesVisited < b.MaxStates && r.Evaluations < b.MaxEvaluations && !work.exhausted {
		if err := ctx.Err(); err != nil {
			t.Fatal(err)
		}
		i := 0
		if random {
			i = rng.Intn(len(frontier))
		}
		s, actions := frontier[i].state, frontier[i].actions
		if len(s.Derivation) >= b.MaxDepth || len(actions) == 0 {
			frontier = append(frontier[:i], frontier[i+1:]...)
			continue
		}
		j := 0
		if random {
			j = rng.Intn(len(actions))
		}
		action := actions[j]
		frontier[i].actions = append(actions[:j], actions[j+1:]...)
		r.ActionsAttempted++
		r.StatesVisited++
		transition, err := s.ApplyAction(action)
		if err != nil {
			t.Fatal(err)
		}
		if transition.NoOp {
			continue
		}
		if err := admitGraphBudget(transition.State, b); err != nil {
			t.Fatal(err)
		}
		evaluate(transition.State)
		next := branch{transition.State, egraph.EnumerateActionsBounded(transition.State.Graph, rules, b.MaxActions)}
		if random {
			frontier = append(frontier, next)
		} else {
			frontier[0] = next
		}
	}
	return r
}

func TestSearchStrategiesEnumerableOracle(t *testing.T) {
	for _, strategy := range []string{"saturation", "random", "beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			g := egraph.New()
			root := g.Add(egraph.ENode{Op: "start"})
			s := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
			rules := []*egraph.Rewrite{egraph.MustRewrite("left", "start", "left"), egraph.MustRewrite("right", "start", "right"), egraph.MustRewrite("left-goal", "left", "goal"), egraph.MustRewrite("right-goal", "right", "goal")}
			b := SearchBudget{MaxStates: 16, MaxTotalActions: 16, MaxActions: 8, MaxDepth: 4, MaxNodesPerState: 8, MaxClassesPerState: 8, MaxEvaluations: 16, MaxExtractions: 16, MaxCompilations: 16, MaxExtractionWork: 4096, Timeout: time.Second}
			e, _, prototype, _ := executableFixture(t)
			e.CostModel = strategyOracleCost{}
			compileCalls := 0
			e.Compile = func(_ context.Context, expr *egraph.Expr, _ WorkloadBindings) (Executable, error) {
				d, ok := strategyOracle[expr.Op]
				if !ok || len(expr.Children) != 0 {
					t.Fatalf("expression outside finite oracle: %+v", expr)
				}
				compileCalls++
				x := *prototype
				x.id, x.duration = expr.Op, d
				return &x, nil
			}
			var r *SearchResult
			var err error
			switch strategy {
			case "saturation", "random":
				r = comparisonBaseline(t, s, rules, e, b, strategy == "random")
			case "beam":
				r, err = NewBeamSearch(BeamConfig{BeamWidth: 2, Budget: b}).Search(context.Background(), s, rules, e)
			case "mcts":
				r, err = NewMCTSSearch(MCTSConfig{Seed: 42, Iterations: 16, RolloutDepth: 2, Budget: b}).Search(context.Background(), s, rules, e)
			}
			if err != nil {
				t.Fatal(err)
			}
			if r.Winner == nil || !r.WorkAccounted || compileCalls != r.Compilations || r.Evaluations != r.Compilations || r.Extractions != r.Compilations {
				t.Fatalf("inconsistent accounting: %+v calls=%d", r, compileCalls)
			}
			if r.StatesVisited > b.MaxStates || r.ActionsAttempted > b.MaxTotalActions || r.Evaluations > b.MaxEvaluations || r.ExtractionWork > b.MaxExtractionWork {
				t.Fatalf("exceeded shared allowance: %+v", r)
			}
			for _, c := range r.AllCandidates {
				if c.Cost != float64(strategyOracle[c.Expr.Op]) || c.Cost < float64(strategyOracle["goal"]) || len(c.Trials) != 4 {
					t.Fatalf("candidate violates oracle: %+v", c)
				}
				if _, err := egraph.ReplaySearchState(s, rules, c.Derivation); err != nil {
					t.Fatalf("non-replayable candidate: %v", err)
				}
			}
			// This fixture's bounded saturation reaches the complete four-term
			// closure. It anchors the known oracle; heuristics are observations.
			if strategy == "saturation" && r.Winner.Cost != float64(strategyOracle["goal"]) {
				t.Fatal("bounded closure failed to recover enumerable optimum")
			}
			t.Logf("synthetic_ns=%g oracle_ns=%d states=%d actions=%d evaluations=%d compilations=%d extraction_work=%d rejected=%d", r.Winner.Cost, strategyOracle["goal"].Nanoseconds(), r.StatesVisited, r.ActionsAttempted, r.Evaluations, r.Compilations, r.ExtractionWork, len(r.Rejected))
		})
	}
}
