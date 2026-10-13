package cudair_test

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func snapshotSemanticGraph(kind cudair.NodeKind) (*cudair.Graph, *cudair.Node) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2, 2}, cudair.Float32)
	var n *cudair.Node
	switch kind {
	case cudair.KindInput:
		n = x
	case cudair.KindConstant:
		n = g.AddConstant("c", []int{4}, cudair.Float32, []float32{7, 8, 9, 10})
	case cudair.KindUnary:
		n = g.AddUnary("neg", x)
	case cudair.KindBinary:
		n = g.AddBinary("add", x, g.AddConstant("c", nil, cudair.Float32, []float32{2}))
	case cudair.KindMatMul:
		n = g.AddMatMul(x, g.AddInput("y", []int{2, 2}, cudair.Float32))
	case cudair.KindReduce:
		n = g.AddReduce("sum", x, 1)
	case cudair.KindReshape:
		n = g.AddReshape(x, []int{4})
	case cudair.KindTranspose:
		n = g.AddTranspose(x, []int{1, 0})
	case cudair.KindSlice:
		n = g.AddSlice(x, []int{0, 0}, []int{2, 2}, []int{1, 1})
	case cudair.KindConcat:
		n = g.AddConcat([]*cudair.Node{x, g.AddInput("y", []int{2, 2}, cudair.Float32)}, 0)
	}
	_ = g.SetOutputs(n)
	return g, n
}

func snapshotSetShape(n *cudair.Node, shape ...int) {
	n.Shape = shape
	n.Strides, _ = cudashape.RowMajorStrides(shape)
}

func TestSnapshotSemanticMutations(t *testing.T) {
	tests := []struct {
		name   string
		kind   cudair.NodeKind
		mutate func(*cudair.Node)
		want   error
	}{
		{"constant scalar substitute", cudair.KindConstant, func(n *cudair.Node) { n.ConstValues = []float32{7} }, cudair.ErrShapeMismatch},
		{"constant surplus", cudair.KindConstant, func(n *cudair.Node) { n.ConstValues = append(n.ConstValues, 11) }, cudair.ErrShapeMismatch},
		{"input dependency", cudair.KindInput, func(n *cudair.Node) { n.Inputs = []*cudair.Node{n} }, nil},
		{"constant dependency", cudair.KindConstant, func(n *cudair.Node) { n.Inputs = []*cudair.Node{n} }, nil},
		{"unary shape", cudair.KindUnary, func(n *cudair.Node) { snapshotSetShape(n, 4) }, cudair.ErrShapeMismatch},
		{"binary shape", cudair.KindBinary, func(n *cudair.Node) { snapshotSetShape(n, 1) }, cudair.ErrShapeMismatch},
		{"matmul shape", cudair.KindMatMul, func(n *cudair.Node) { snapshotSetShape(n, 1) }, cudair.ErrShapeMismatch},
		{"reduction shape", cudair.KindReduce, func(n *cudair.Node) { snapshotSetShape(n, 1) }, cudair.ErrShapeMismatch},
		{"reduction missing axis", cudair.KindReduce, func(n *cudair.Node) { n.Axes = nil }, cudair.ErrInvalidAxis},
		{"reduction multiple axes", cudair.KindReduce, func(n *cudair.Node) { n.Axes = []int{0, 1} }, cudair.ErrInvalidAxis},
		{"reduction duplicate axes", cudair.KindReduce, func(n *cudair.Node) { n.Axes = []int{1, 1} }, cudair.ErrInvalidAxis},
		{"reduction negative axis", cudair.KindReduce, func(n *cudair.Node) { n.Axes = []int{-1} }, cudair.ErrInvalidAxis},
		{"reduction high axis", cudair.KindReduce, func(n *cudair.Node) { n.Axes = []int{2} }, cudair.ErrInvalidAxis},
		{"reshape shape", cudair.KindReshape, func(n *cudair.Node) { snapshotSetShape(n, 3) }, cudair.ErrShapeMismatch},
		{"transpose high axis", cudair.KindTranspose, func(n *cudair.Node) { n.Perm = []int{2, 0} }, cudair.ErrInvalidAxis},
		{"transpose negative axis", cudair.KindTranspose, func(n *cudair.Node) { n.Perm = []int{-1, 0} }, cudair.ErrInvalidAxis},
		{"transpose duplicate axis", cudair.KindTranspose, func(n *cudair.Node) { n.Perm = []int{0, 0} }, cudair.ErrInvalidAxis},
		{"transpose short permutation", cudair.KindTranspose, func(n *cudair.Node) { n.Perm = []int{0} }, cudair.ErrInvalidAxis},
		{"transpose shape", cudair.KindTranspose, func(n *cudair.Node) { snapshotSetShape(n, 4) }, cudair.ErrShapeMismatch},
		{"slice short starts", cudair.KindSlice, func(n *cudair.Node) { n.SliceStarts = []int{0} }, cudair.ErrInvalidAxis},
		{"slice short ends", cudair.KindSlice, func(n *cudair.Node) { n.SliceEnds = []int{2} }, cudair.ErrInvalidAxis},
		{"slice short steps", cudair.KindSlice, func(n *cudair.Node) { n.SliceSteps = []int{1} }, cudair.ErrInvalidAxis},
		{"slice zero step", cudair.KindSlice, func(n *cudair.Node) { n.SliceSteps[0] = 0 }, cudair.ErrInvalidAxis},
		{"slice negative step", cudair.KindSlice, func(n *cudair.Node) { n.SliceSteps[0] = -1 }, cudair.ErrInvalidAxis},
		{"slice negative start", cudair.KindSlice, func(n *cudair.Node) { n.SliceStarts[0] = -1 }, cudair.ErrInvalidAxis},
		{"slice reversed range", cudair.KindSlice, func(n *cudair.Node) { n.SliceStarts[0], n.SliceEnds[0] = 2, 1 }, cudair.ErrInvalidAxis},
		{"slice high end", cudair.KindSlice, func(n *cudair.Node) { n.SliceEnds[0] = 3 }, cudair.ErrInvalidAxis},
		{"slice shape", cudair.KindSlice, func(n *cudair.Node) { snapshotSetShape(n, 1, 2) }, cudair.ErrShapeMismatch},
		{"slice arithmetic overflow", cudair.KindSlice, func(n *cudair.Node) { n.SliceSteps[0] = int(^uint(0) >> 1) }, cudashape.ErrShapeOverflow},
		{"concat negative axis", cudair.KindConcat, func(n *cudair.Node) { n.Axis = -1 }, cudair.ErrInvalidAxis},
		{"concat high axis", cudair.KindConcat, func(n *cudair.Node) { n.Axis = 2 }, cudair.ErrInvalidAxis},
		{"concat rank", cudair.KindConcat, func(n *cudair.Node) { snapshotSetShape(n.Inputs[1], 4) }, cudair.ErrShapeMismatch},
		{"concat nonaxis extent", cudair.KindConcat, func(n *cudair.Node) { snapshotSetShape(n.Inputs[1], 2, 3) }, cudair.ErrShapeMismatch},
		{"concat output shape", cudair.KindConcat, func(n *cudair.Node) { snapshotSetShape(n, 2, 4) }, cudair.ErrShapeMismatch},
		{"unknown layout", cudair.KindInput, func(n *cudair.Node) { n.Layout = cudashape.Layout(99) }, nil},
		{"column layout", cudair.KindInput, func(n *cudair.Node) { n.Layout = cudashape.LayoutColumnMajor; n.Strides = []int{1, 2} }, nil},
		{"strided layout", cudair.KindInput, func(n *cudair.Node) { n.Layout = cudashape.LayoutStrided }, nil},
		{"missing strides", cudair.KindInput, func(n *cudair.Node) { n.Strides = nil }, cudashape.ErrRankMismatch},
		{"stride rank", cudair.KindInput, func(n *cudair.Node) { n.Strides = []int{1} }, cudashape.ErrRankMismatch},
		{"negative stride", cudair.KindInput, func(n *cudair.Node) { n.Strides = []int{-1, 1} }, cudashape.ErrInvalidStride},
		{"broadcast stride", cudair.KindInput, func(n *cudair.Node) { n.Strides = []int{0, 1} }, cudashape.ErrInvalidStride},
		{"noncontiguous stride", cudair.KindInput, func(n *cudair.Node) { n.Strides = []int{3, 1} }, cudashape.ErrInvalidStride},
		{"byte stride overflow", cudair.KindInput, func(n *cudair.Node) { n.Strides = []int{int(cudashape.MaxByteLimit), 1} }, cudashape.ErrShapeOverflow},
		{"dtype byte overflow", cudair.KindInput, func(n *cudair.Node) {
			snapshotSetShape(n, int(cudashape.MaxByteLimit/8))
			n.Type = cudair.Int64
			n.Shape[0]++
		}, cudashape.ErrShapeOverflow},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, n := snapshotSemanticGraph(tt.kind)
			if g.Err() != nil {
				t.Fatal(g.Err())
			}
			if _, err := g.Snapshot(); err != nil {
				t.Fatalf("valid control: %v", err)
			}
			tt.mutate(n)
			if snapshot, err := g.Snapshot(); err == nil || snapshot != nil || tt.want != nil && !errors.Is(err, tt.want) {
				t.Fatalf("Snapshot() = %v, %v; want refusal with %v", snapshot, err, tt.want)
			}
		})
	}
	for _, kind := range []cudair.NodeKind{cudair.KindReshape, cudair.KindTranspose, cudair.KindSlice, cudair.KindConcat} {
		g, n := snapshotSemanticGraph(kind)
		n.Op = "unknown"
		if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrUnknownOperator) {
			t.Fatalf("%s operator mutation: %v", kind, err)
		}
	}
}

func TestSnapshotSemanticEvaluation(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{4}, cudair.Float32)
	c := g.AddConstant("c", []int{4}, cudair.Float32, []float32{7, 8, 9, 10})
	y := g.AddBinary("add", x, c)
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	got, err := cudaref.EvaluateSnapshot(s, map[string][]float32{"x": {1, 2, 3, 4}})
	if err != nil || !reflect.DeepEqual(got[y.Name], []float32{8, 10, 12, 14}) {
		t.Fatalf("vector constant control: %v, %v", got, err)
	}
	c.ConstValues = []float32{7}
	if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrShapeMismatch) {
		t.Fatalf("vector constant scalar substitute admitted: %v", err)
	}
	for _, kind := range []cudair.NodeKind{cudair.KindUnary, cudair.KindBinary, cudair.KindMatMul, cudair.KindReduce, cudair.KindReshape, cudair.KindTranspose, cudair.KindSlice, cudair.KindConcat} {
		t.Run(kind.String(), func(t *testing.T) {
			g, n := snapshotSemanticGraph(kind)
			s, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			want := map[cudair.NodeKind][]float32{
				cudair.KindUnary: {-1, -2, -3, -4}, cudair.KindBinary: {3, 4, 5, 6},
				cudair.KindMatMul: {19, 22, 43, 50}, cudair.KindReduce: {3, 7},
				cudair.KindReshape: {1, 2, 3, 4}, cudair.KindTranspose: {1, 3, 2, 4},
				cudair.KindSlice: {1, 2, 3, 4}, cudair.KindConcat: {1, 2, 3, 4, 5, 6, 7, 8},
			}[kind]
			// Both source and accessor mutations must leave the admitted snapshot intact.
			n.Shape, n.Strides = []int{99}, []int{99}
			s.NodeByID(n.ID).Op = "unknown"
			got, err := cudaref.EvaluateSnapshot(s, map[string][]float32{"x": {1, 2, 3, 4}, "y": {5, 6, 7, 8}})
			if err != nil || !reflect.DeepEqual(got[n.Name], want) {
				t.Fatalf("EvaluateSnapshot() = %v, %v; want %v", got, err, want)
			}
		})
	}
	for _, values := range [][]float32{nil, {2}} {
		for _, shape := range [][]int{nil, {1}, {1, 1, 1}} {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{4}, cudair.Float32)
			c := g.AddConstant("c", shape, cudair.Float32, values)
			y := g.AddBinary("add", x, c)
			s, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			got, err := cudaref.EvaluateSnapshot(s, map[string][]float32{"x": {1, 2, 3, 4}})
			want := []float32{1, 2, 3, 4}
			if len(values) != 0 {
				want = []float32{3, 4, 5, 6}
			}
			if err != nil || !reflect.DeepEqual(got[y.Name], want) {
				t.Fatalf("scalar shape %v values %v: %v, %v", shape, values, got, err)
			}
		}
	}
	for _, op := range []string{"sum", "max"} {
		g := cudair.NewGraph()
		x := g.AddInput("empty", []int{2, 0}, cudair.Float32)
		y := g.AddReduce(op, x, 1)
		s, err := g.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		got, err := cudaref.EvaluateSnapshot(s, map[string][]float32{"empty": {}})
		want := float32(0)
		if op == "max" {
			want = float32(math.Inf(-1))
		}
		if err != nil || !reflect.DeepEqual(got[y.Name], []float32{want, want}) {
			t.Fatalf("empty %s: %v, %v", op, got, err)
		}
	}
}

func TestSnapshotSemanticParameters(t *testing.T) {
	for _, tc := range []struct {
		name        string
		ends, steps []int
		want        []float32
	}{
		{"stepped slice", []int{2, 2}, []int{1, 2}, []float32{1, 3}},
		{"empty slice", []int{0, 2}, []int{1, 1}, []float32{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{2, 2}, cudair.Float32)
			y := g.AddSlice(x, []int{0, 0}, tc.ends, tc.steps)
			s, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			got, err := cudaref.EvaluateSnapshot(s, map[string][]float32{"x": {1, 2, 3, 4}})
			if err != nil || !reflect.DeepEqual(got[y.Name], tc.want) {
				t.Fatalf("slice: %v, %v; want %v", got, err, tc.want)
			}
		})
	}
	// These registered semantics remain available to their execution consumers.
	for _, op := range []string{"mean", "rmsnorm_eps=1e-05", "rope_theta=10000"} {
		g := cudair.NewGraph()
		x := g.AddInput("x", []int{2, 2}, cudair.Float32)
		switch op {
		case "mean":
			g.AddReduce(op, x, 1)
		case "rmsnorm_eps=1e-05":
			g.AddUnary(op, x)
		default:
			g.AddBinary(op, x, g.AddConstant("position", nil, cudair.Float32, []float32{1}))
		}
		if _, err := g.Snapshot(); err != nil {
			t.Fatalf("registered %s: %v", op, err)
		}
	}
}

func TestSnapshotSemanticDataTypes(t *testing.T) {
	for _, dt := range []cudair.DataType{cudair.Float32, cudair.Float16, cudair.BFloat16, cudair.Int32, cudair.Int64, cudair.Uint8, cudair.Int8} {
		g := cudair.NewGraph()
		x := g.AddInput("x", []int{2, 3}, dt)
		_ = g.SetOutputs(x)
		s, err := g.Snapshot()
		if err != nil || s.NodeByID(x.ID).SizeBytes() != 6*dt.SizeBytes() {
			t.Fatalf("dtype %s: snapshot=%v, err=%v", dt, s, err)
		}
	}
}
