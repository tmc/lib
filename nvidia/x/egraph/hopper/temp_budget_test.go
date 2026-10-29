package hopper

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestCompileWithTempBudget(t *testing.T) {
	build := func() (*Builder, egraph.Id) {
		b := NewBuilder()
		x := b.Input("x", BF16, 1, 256)
		w := b.Input("w", BF16, 64, 256)
		m := b.MatMul(x, w)
		return b, b.Map(MustLambda("(add $0 $1)"), m, m)
	}
	b, root := build()
	e, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Compile(e, "budget", F32)
	if err != nil {
		t.Fatal(err)
	}
	b, root = build()
	got, err := CompileWithTempBudget(b.G, root, "budget", F32, 0, false)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("zero budget differs: got %v, err %v", got, err)
	}
	for _, reuse := range []bool{false, true} {
		b, root = build()
		if _, err := CompileWithTempBudget(b.G, root, "budget", F32, 256, reuse); err != nil {
			t.Fatalf("reuse %v: exact budget: %v", reuse, err)
		}
		b, root = build()
		if _, err := CompileWithTempBudget(b.G, root, "budget", F32, 255, reuse); err == nil || !strings.Contains(err.Error(), "temporary budget") {
			t.Fatalf("reuse %v: undersized budget: %v", reuse, err)
		}
	}
}

// The synthetic equivalence below isolates plan selection: the test supplies
// the equivalence, rather than asserting a numerical rewrite between branches.
func TestCompileWithTempBudgetAlternative(t *testing.T) {
	b := NewBuilder()
	x := b.Input("x", BF16, 1, 256)
	w := b.Input("w", BF16, 64, 256)
	m := b.MatMul(x, w)
	large := b.Map(MustLambda("(add $0 $1)"), m, m)
	y := b.Input("y", F32, 1, 64)
	fn := MustLambda("(add $0 1)")
	small := b.Map(fn, b.Map(fn, b.Map(fn, y)))
	b.G.Union(large, small)
	b.G.Rebuild()
	best, err := CompileWithTempBudget(b.G, large, "alternative", F32, 0, true)
	if err != nil || len(best.Temps) == 0 {
		t.Fatalf("test optimum has no temporary: %v %v", best, err)
	}
	p, err := CompileWithTempBudget(b.G, large, "alternative", F32, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Temps) != 0 || len(p.Inputs) != 1 || p.Inputs[0].Name != "y" {
		t.Fatalf("did not choose feasible alternative: %+v", p)
	}
}

func ExampleCompileWithTempBudget() {
	b := NewBuilder()
	x := b.Input("x", BF16, 1, 256)
	w := b.Input("w", BF16, 64, 256)
	p, err := CompileWithTempBudget(b.G, b.MatMul(x, w), "projection", F32, 1, true)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(p.Temps))
	// Output: 0
}
