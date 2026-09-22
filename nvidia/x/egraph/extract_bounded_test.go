package egraph

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

type boundedCostFunc func(ENode, []float64) float64

func (f boundedCostFunc) NodeCost(n ENode, c []float64) float64 { return f(n, c) }

func TestBoundedExtractionDAGExpansion(t *testing.T) {
	g := New()
	root := g.Add(ENode{Op: "leaf"})
	for i := 0; i < 20; i++ {
		root = g.Add(ENode{Op: "pair", Children: []Id{root, root}})
	}
	limits := DefaultExtractionLimits()
	limits.MaxNodes = 100
	if expr, _, err := ExtractBounded(context.Background(), g, root, nil, limits); expr != nil || !errors.Is(err, ErrExtractionLimit) {
		t.Fatalf("unbounded tree: %v %v", expr, err)
	}
}

func TestBoundedExtractionLimitsAndCancellation(t *testing.T) {
	for _, kind := range []string{"work", "nodes", "depth", "canceled", "callback"} {
		t.Run(kind, func(t *testing.T) {
			g := New()
			leaf := g.Add(ENode{Op: "leaf"})
			root := g.Add(ENode{Op: "wrap", Children: []Id{leaf}})
			limits := DefaultExtractionLimits()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			model := boundedCostFunc(func(n ENode, c []float64) float64 {
				calls++
				if kind == "callback" {
					cancel()
				}
				return (AstSizeCost{}).NodeCost(n, c)
			})
			want := ErrExtractionLimit
			switch kind {
			case "work":
				limits.MaxWork = 1
			case "nodes":
				limits.MaxNodes = 1
			case "depth":
				limits.MaxDepth = 1
			case "canceled":
				cancel()
				want = context.Canceled
			case "callback":
				want = context.Canceled
			}
			expr, _, err := ExtractBounded(ctx, g, root, model, limits)
			if expr != nil || !errors.Is(err, want) {
				t.Fatalf("expr=%v err=%v want=%v", expr, err, want)
			}
			if kind == "canceled" && calls != 0 {
				t.Fatal("called model after cancellation")
			}
			if kind == "callback" && calls != 1 {
				t.Fatal("continued after callback canceled")
			}
		})
	}
}

func ExampleExtractBounded() {
	g := New()
	root := g.Add(ENode{Op: "leaf"})
	expr, cost, err := ExtractBounded(context.Background(), g, root, AstSizeCost{}, DefaultExtractionLimits())
	fmt.Println(expr, cost, err)
	// Output: leaf 1 <nil>
}

func TestExtractionStatsIncludeFailedWork(t *testing.T) {
	g := New()
	root := g.Add(ENode{Op: "leaf"})
	limits := DefaultExtractionLimits()
	_, _, stats, err := ExtractBoundedWithStats(context.Background(), g, root, nil, limits)
	if err != nil || stats.Work != 3 || stats.Nodes != 1 || stats.WorkExhausted {
		t.Fatalf("stats=%+v err=%v", stats, err)
	}
	limits.MaxWork = 1
	_, _, stats, err = ExtractBoundedWithStats(context.Background(), g, root, nil, limits)
	if !errors.Is(err, ErrExtractionLimit) || stats.Work != 1 || !stats.WorkExhausted {
		t.Fatalf("failed stats=%+v err=%v", stats, err)
	}
}

func ExampleExtractBoundedWithStats() {
	g := New()
	root := g.Add(ENode{Op: "leaf"})
	expr, _, usage, err := ExtractBoundedWithStats(context.Background(), g, root, nil, DefaultExtractionLimits())
	fmt.Println(expr, usage.Work, usage.Nodes, err)
	// Output: leaf 3 1 <nil>
}
