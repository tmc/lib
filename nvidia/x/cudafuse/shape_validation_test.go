package cudafuse

import (
	"errors"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"math"
	"testing"
)

func TestLogicalShapeCounts(t *testing.T) {
	for _, shape := range [][]int{{0, -1}, {math.MaxInt, 0, -1}} {
		if _, err := shapeElements(shape); !errors.Is(err, cudashape.ErrNegativeDimension) {
			t.Fatal(shape, err)
		}
	}
	shape := []int{0, math.MaxInt, math.MaxInt}
	if n, err := shapeElements(shape); n != 0 || err != nil {
		t.Fatal(n, err)
	}
	if _, err := cudashape.RowMajorStrides(shape); !errors.Is(err, cudashape.ErrShapeOverflow) {
		t.Fatal("logical zero mistaken for representable strides", err)
	}
	if n, err := shapeElements(nil); n != 1 || err != nil {
		t.Fatal(n, err)
	}
	if _, err := elementCount([]int{math.MaxInt, 2}); err == nil {
		t.Fatal("count overflow")
	}
}
func TestShapeErrorsPropagate(t *testing.T) {
	ctx := NewLazyContext()
	bad := NewLazyArray(ctx, []int{0, -1}, cudair.Float32)
	if !errors.Is(bad.Node.Err, cudashape.ErrNegativeDimension) {
		t.Fatal("leaf lost typed shape error", bad.Node.Err)
	}
	valid := NewLazyArray(ctx, []int{1}, cudair.Float32)
	index := NewLazyArray(ctx, []int{1}, cudair.Int32)
	for _, a := range []*LazyArray{bad.Map("neg"), bad.Add(valid), valid.Add(bad), bad.Reshape([]int{0}), bad.Transpose([]int{0, 1}), bad.Cycle([]int{2}), bad.Replicate(2), bad.Gather(index), valid.Filter(bad), valid.Keep(bad), valid.Where(bad, valid)} {
		if !errors.Is(a.Node.Err, cudashape.ErrNegativeDimension) {
			t.Fatal(a.Node.Op, a.Node.Err)
		}
	}
	for _, f := range []func() (*MaterializedPlan, error){bad.Sum, func() (*MaterializedPlan, error) { return bad.Scan("sum") }, bad.Sort, bad.Materialize} {
		if _, err := f(); !errors.Is(err, cudashape.ErrNegativeDimension) {
			t.Fatal(err)
		}
	}
	if valid.Add(NewLazyArray(NewLazyContext(), []int{1}, cudair.Float32)).Node.Err == nil {
		t.Fatal("mixed context")
	}
	if !errors.Is(valid.Add(index).Node.Err, cudair.ErrTypeMismatch) {
		t.Fatal("mixed type")
	}
}
