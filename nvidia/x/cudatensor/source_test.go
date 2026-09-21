package cudatensor_test

import (
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	. "github.com/tmc/lib/nvidia/x/cudatensor"
)

// here reports the line it is called from, so a case can state the line it
// expects without writing the number down and letting it drift as the file
// above it is edited.
func here(t *testing.T) int {
	t.Helper()
	_, _, line, ok := runtime.Caller(1)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return line
}

// TestCapturedSourceIsTheCaller pins that a graph node records where the
// caller built it, not where this package happened to call runtime.Callers.
//
// The distinction is the whole value of the field: these locations become the
// loc(#locN) attributes of the lowered Tile IR, and through them the Go end of
// the SASS -> PTX -> Tile IR -> Go chain. A location pointing into this
// package is not a weaker answer than the caller's line, it is a wrong one,
// and it looks exactly like a right one.
//
// Every exported operation is listed because the depth is not uniform: some
// call the capture directly and some reach it through withUnary or
// withBinary. A table covering only one shape passes while the other is wrong,
// which is the state this test was written for.
func TestCapturedSourceIsTheCaller(t *testing.T) {
	newPair := func(g *cudair.Graph) (Tensor, Tensor) {
		return NewInput(g, "A", []int{4, 4}, cudair.Float32), NewInput(g, "B", []int{4, 4}, cudair.Float32)
	}
	for _, tt := range []struct {
		name string
		call func(t *testing.T, g *cudair.Graph) (Tensor, int)
	}{
		{"NewInput", func(t *testing.T, g *cudair.Graph) (Tensor, int) {
			return NewInput(g, "X", []int{4, 4}, cudair.Float32), here(t)
		}},
		{"NewConstant", func(t *testing.T, g *cudair.Graph) (Tensor, int) {
			return NewConstant(g, "C", []int{1}, cudair.Float32, []float32{1}), here(t)
		}},
		{"Add", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, b := newPair(g); return a.Add(b), here(t) }},
		{"Sub", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, b := newPair(g); return a.Sub(b), here(t) }},
		{"Mul", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, b := newPair(g); return a.Mul(b), here(t) }},
		{"Div", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, b := newPair(g); return a.Div(b), here(t) }},
		{"Relu", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, _ := newPair(g); return a.Relu(), here(t) }},
		{"Gelu", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, _ := newPair(g); return a.Gelu(), here(t) }},
		{"Silu", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, _ := newPair(g); return a.Silu(), here(t) }},
		{"Exp", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, _ := newPair(g); return a.Exp(), here(t) }},
		{"Log", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, _ := newPair(g); return a.Log(), here(t) }},
		{"Sqrt", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, _ := newPair(g); return a.Sqrt(), here(t) }},
		{"Neg", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, _ := newPair(g); return a.Neg(), here(t) }},
		{"MatMul", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, b := newPair(g); return a.MatMul(b), here(t) }},
		{"ReduceSum", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, _ := newPair(g); return a.ReduceSum(1), here(t) }},
		{"ReduceMax", func(t *testing.T, g *cudair.Graph) (Tensor, int) { a, _ := newPair(g); return a.ReduceMax(1), here(t) }},
		{"Reshape", func(t *testing.T, g *cudair.Graph) (Tensor, int) {
			a, _ := newPair(g)
			return a.Reshape([]int{16}), here(t)
		}},
		{"Transpose", func(t *testing.T, g *cudair.Graph) (Tensor, int) {
			a, _ := newPair(g)
			return a.Transpose([]int{1, 0}), here(t)
		}},
		{"Slice", func(t *testing.T, g *cudair.Graph) (Tensor, int) {
			a, _ := newPair(g)
			return a.Slice([]int{0, 0}, []int{2, 2}, []int{1, 1}), here(t)
		}},
		{"RMSNorm", func(t *testing.T, g *cudair.Graph) (Tensor, int) {
			a, b := newPair(g)
			return a.RMSNorm(b, 1e-5), here(t)
		}},
		{"RoPE", func(t *testing.T, g *cudair.Graph) (Tensor, int) {
			a, b := newPair(g)
			return a.RoPE(b, 10000), here(t)
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, line := tt.call(t, cudair.NewGraph())
			if err := got.Err(); err != nil {
				t.Fatalf("%s: %v", tt.name, err)
			}
			n := got.Node()
			if n == nil {
				t.Fatalf("%s produced no node", tt.name)
			}
			loc := n.SourceLoc
			if loc.File == "" {
				t.Fatalf("%s recorded no source location", tt.name)
			}
			if !strings.HasSuffix(loc.File, "source_test.go") {
				t.Errorf("%s attributed to %s, want this test file: the caller's code, not the library's", tt.name, loc.File)
			}
			if loc.Line != line {
				t.Errorf("%s recorded line %d, want %d", tt.name, loc.Line, line)
			}
			if !strings.Contains(loc.Function, "TestCapturedSourceIsTheCaller") {
				t.Errorf("%s recorded function %q, want the calling test function", tt.name, loc.Function)
			}
		})
	}
}

// TestCapturedCallers checks that the frames above the building statement
// are kept, nearest first, so a graph built through a helper can be charged
// to the helper's caller.
func TestCapturedCallers(t *testing.T) {
	g := cudair.NewGraph()
	a := NewInput(g, "A", []int{4, 4}, cudair.Float32)
	b := NewInput(g, "B", []int{4, 4}, cudair.Float32)
	got, line := viaHelper(a, b), here(t)
	if err := got.Err(); err != nil {
		t.Fatal(err)
	}
	loc := got.Node().SourceLoc
	if !strings.Contains(loc.Function, "viaHelper") {
		t.Fatalf("leaf function = %q, want viaHelper", loc.Function)
	}
	if len(loc.Callers) == 0 {
		t.Fatal("no callers recorded")
	}
	c := loc.Callers[0]
	if !strings.HasSuffix(c.File, "source_test.go") || c.Line != line {
		t.Errorf("Callers[0] = %s:%d, want this file line %d", c.File, c.Line, line)
	}
	if !strings.Contains(c.Function, "TestCapturedCallers") {
		t.Errorf("Callers[0].Function = %q, want the test function", c.Function)
	}
	for _, c := range loc.Callers {
		if strings.HasPrefix(c.Function, "runtime.") {
			t.Errorf("runtime frame %q recorded as a caller", c.Function)
		}
	}
}

// viaHelper builds a node one frame below its caller.
func viaHelper(a, b Tensor) Tensor {
	return a.Add(b)
}
