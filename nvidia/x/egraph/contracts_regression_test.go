package egraph

import (
	"context"
	"errors"
	"math"
	"testing"
)

func TestPatternExplicitData(t *testing.T) {
	g := New()
	root := g.Add(ENode{Op: "value", Data: 1})
	g.Union(root, g.Add(ENode{Op: "value", Data: 2}))
	g.Rebuild()
	for _, tt := range []struct {
		name    string
		pattern *Pattern
		want    int
	}{
		{"wildcard", &Pattern{Op: "value"}, 2},
		{"exact", &Pattern{Op: "value", HasData: true, Data: 1}, 1},
		{"nil", &Pattern{Op: "value", HasData: true}, 0},
		{"different type", &Pattern{Op: "value", HasData: true, Data: int64(1)}, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			matches, err := tt.pattern.MatchEClassBounded(g, root, make(Subst), 10)
			if err != nil || len(matches) != tt.want {
				t.Fatalf("matches=%v err=%v want=%d", matches, err, tt.want)
			}
		})
	}
}

func TestBoundedExtractionExactLeafBudget(t *testing.T) {
	for _, work := range []int{1, 2, 3} {
		t.Run(string(rune('0'+work)), func(t *testing.T) {
			g := New()
			root := g.Add(NewNode("leaf"))
			limits := DefaultExtractionLimits()
			limits.MaxWork = work
			expr, _, stats, err := ExtractBoundedWithStats(context.Background(), g, root, nil, limits)
			if work < 3 {
				if expr != nil || !errors.Is(err, ErrExtractionLimit) || !stats.WorkExhausted || stats.Work > work {
					t.Fatalf("expr=%v stats=%+v err=%v", expr, stats, err)
				}
			} else if err != nil || expr == nil || stats.Work != 3 {
				t.Fatalf("expr=%v stats=%+v err=%v", expr, stats, err)
			}
		})
	}
}

func TestInvalidFiniteCostStillRejected(t *testing.T) {
	for _, cost := range []float64{-1, math.Inf(1), math.Inf(-1), math.NaN()} {
		g := New()
		root := g.Add(NewNode("leaf"))
		expr, _, err := ExtractBounded(context.Background(), g, root, boundedCostFunc(func(ENode, []float64) float64 { return cost }), DefaultExtractionLimits())
		if err == nil || expr != nil {
			t.Fatalf("cost=%v expr=%v err=%v", cost, expr, err)
		}
	}
}

func TestInstantTunerRequiresContext(t *testing.T) {
	g := New()
	root := g.Add(NewNode("leaf"))
	tuner := NewTuner(nil)
	if got, err := tuner.SelectBest(nil, g, root, BudgetInstant); err == nil || got != nil {
		t.Fatalf("got=%v err=%v", got, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if got, err := tuner.SelectBest(ctx, g, root, BudgetInstant); !errors.Is(err, context.Canceled) || got != nil {
		t.Fatalf("got=%v err=%v", got, err)
	}
}
