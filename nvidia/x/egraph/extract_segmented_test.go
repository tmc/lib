package egraph

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

// residualStack builds n pre-norm layers x' = (add x (mm (rms x) w_i)),
// each also offering (fused x w_i) for the matmul. fuseAcross layers, from
// the second on, also offer their norm as (rmsadd x₋ y₋), reading the
// previous layer's operands, as an add+norm fusion across the boundary
// would. kinds names each layer's matmul operator, cycling.
func residualStack(n int, kinds []string, fuseAcross bool) (*EGraph, Id) {
	g := New()
	x := g.Add(ENode{Op: "x"})
	var prevX, prevY Id
	for i := range n {
		w := g.Add(ENode{Op: fmt.Sprintf("w%d", i)})
		h := g.Add(ENode{Op: "rms", Children: []Id{x}})
		if fuseAcross && i > 0 {
			g.Union(h, g.Add(ENode{Op: "rmsadd", Children: []Id{prevX, prevY}}))
		}
		y := g.Add(ENode{Op: kinds[i%len(kinds)], Children: []Id{h, w}})
		g.Union(y, g.Add(ENode{Op: "fused", Children: []Id{x, w}}))
		prevX, prevY = x, y
		x = g.Add(ENode{Op: "add", Children: []Id{x, y}})
	}
	g.Rebuild()
	return g, x
}

var bigLimits = ExtractionLimits{MaxWork: 1 << 40, MaxNodes: 1 << 24, MaxDepth: 1 << 20}

func TestExtractSegmentedMatchesDAG(t *testing.T) {
	tests := []struct {
		name               string
		layers             int
		kinds              []string
		fuseAcross         bool
		segments, distinct int
	}{
		{"one layer", 1, []string{"mm"}, false, 1, 1},
		{"uniform", 6, []string{"mm"}, false, 6, 2},
		{"alternating", 6, []string{"mm", "mm2"}, false, 6, 3},
		{"distinct", 4, []string{"a", "b", "c", "d"}, false, 4, 4},
		// The fusion removes every boundary but the first layer's
		// matmul, which only the second layer reads.
		{"fused across", 6, []string{"mm"}, true, 2, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, root := residualStack(tt.layers, tt.kinds, tt.fuseAcross)
			want, _, _, err := ExtractDAG(context.Background(), g, root, AstSizeCost{}, bigLimits)
			if err != nil {
				t.Fatal(err)
			}
			got, _, stats, err := ExtractSegmented(context.Background(), g, root, AstSizeCost{}, bigLimits)
			if err != nil {
				t.Fatal(err)
			}
			if !got.Equal(want) {
				t.Errorf("ExtractSegmented = %v\nExtractDAG       = %v", got, want)
			}
			if stats.Segments != tt.segments || stats.Distinct != tt.distinct {
				t.Errorf("segments %d, distinct %d; want %d, %d", stats.Segments, stats.Distinct, tt.segments, tt.distinct)
			}
		})
	}
}

// TestExtractSegmentedCost checks that a shared value is paid for once:
// each layer of a uniform stack costs the same, however deep it is.
func TestExtractSegmentedCost(t *testing.T) {
	cost := func(n int) float64 {
		g, root := residualStack(n, []string{"mm"}, false)
		_, c, _, err := ExtractSegmented(context.Background(), g, root, AstSizeCost{}, bigLimits)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	one, two := cost(1), cost(2)
	for _, n := range []int{3, 10, 100} {
		if got, want := cost(n), one+float64(n-1)*(two-one); got != want {
			t.Errorf("cost of %d layers = %g, want %g", n, got, want)
		}
	}
}

// TestExtractSegmentedLinear fails if extraction work grows faster than
// the graph. Whole-graph extraction's work grows as the square of the
// number of layers and exhausts the limit; at about 1000 layers its tree
// costs overflow float64.
func TestExtractSegmentedLinear(t *testing.T) {
	const layers = 2000
	g, root := residualStack(layers, []string{"mm"}, false)
	classes, nodes := g.Size()
	limits := ExtractionLimits{MaxWork: 20 * (classes + nodes), MaxNodes: 1 << 24, MaxDepth: 1 << 20}
	e, _, stats, err := ExtractSegmented(context.Background(), g, root, AstSizeCost{}, limits)
	if err != nil {
		t.Fatalf("ExtractSegmented: %v (work %d)", err, stats.Work)
	}
	if stats.Segments != layers || stats.Distinct != 2 {
		t.Errorf("segments %d, distinct %d; want %d, 2", stats.Segments, stats.Distinct, layers)
	}
	if e.Op != "add" {
		t.Errorf("root %s, want add", e.Op)
	}
	if _, _, _, err := ExtractDAG(context.Background(), g, root, AstSizeCost{}, limits); !errors.Is(err, ErrExtractionLimit) {
		t.Errorf("ExtractDAG of %d layers in %d work: %v; the bound no longer tells segmented from whole-graph extraction", layers, limits.MaxWork, err)
	}
}

func TestExtractSegmentedCycle(t *testing.T) {
	g := New()
	a := g.Add(ENode{Op: "a"})
	f := g.Add(ENode{Op: "f", Children: []Id{a}})
	g.Union(f, g.Add(ENode{Op: "id", Children: []Id{f}}))
	root := g.Add(ENode{Op: "add", Children: []Id{f, f}})
	g.Rebuild()
	want, wc, _, err := ExtractDAG(context.Background(), g, root, AstSizeCost{}, bigLimits)
	if err != nil {
		t.Fatal(err)
	}
	got, gc, _, err := ExtractSegmented(context.Background(), g, root, AstSizeCost{}, bigLimits)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Equal(want) {
		t.Errorf("ExtractSegmented = %v, ExtractDAG = %v", got, want)
	}
	// f is shared by root and a boundary: it is paid for once.
	if gc != wc-2 {
		t.Errorf("cost %g, want %g", gc, wc-2)
	}
}

func ExampleExtractSegmented() {
	g, root := residualStack(61, []string{"mm"}, false)
	e, _, stats, err := ExtractSegmented(context.Background(), g, root, AstSizeCost{}, DefaultExtractionLimits())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(e.Op, e.Children[1].Op)
	fmt.Println(stats.Segments, "segments,", stats.Distinct, "extracted")
	// Output:
	// add fused
	// 61 segments, 2 extracted
}

// TestExtractSegmentedAvoidable checks that a shared class some choice
// above it avoids is not a boundary: priced at zero, it would make the
// choice that reads it look as cheap as the one that does not.
func TestExtractSegmentedAvoidable(t *testing.T) {
	g := New()
	a := g.Add(ENode{Op: "a"})
	b := g.Add(ENode{Op: "b"})
	attn := g.Add(ENode{Op: "attn", Children: []Id{a, b}})
	ss := g.Add(ENode{Op: "ss", Children: []Id{attn}})
	root := g.Add(ENode{Op: "norm", Children: []Id{attn, ss}})
	g.Union(root, g.Add(ENode{Op: "fused", Children: []Id{a, b}}))
	g.Rebuild()
	got, _, stats, err := ExtractSegmented(context.Background(), g, root, AstSizeCost{}, bigLimits)
	if err != nil {
		t.Fatal(err)
	}
	if got.Op != "fused" || stats.Segments != 1 {
		t.Errorf("ExtractSegmented = %v in %d segments, want (fused a b) in 1", got, stats.Segments)
	}
}
