package cudafuse

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// shapeSourceModel interprets emitted dataflow against actual runner pointer
// bindings. This is a host source/ownership control, never CUDA evidence.
func shapeSourceModel(f FusedKernel, inputs [][]float32, count int) ([][]float32, error) {
	if len(inputs) != len(f.Inputs) {
		return nil, fmt.Errorf("input ABI")
	}
	storage := map[string][]float32{}
	for i, name := range f.Inputs {
		storage[name] = inputs[i]
	}
	sum := strings.Contains(f.Source, "double sum_value = 0.0;")
	scan := strings.Contains(f.Source, "scan_value")
	domain := count
	if sum || scan {
		domain = 0
		if m := regexp.MustCompile(`idx < ([0-9]+)ULL`).FindStringSubmatch(f.Source); m != nil {
			domain, _ = strconv.Atoi(m[1])
		}
		if sum && (count != 1 || !strings.Contains(f.Source, "sum_value += (double)") && domain != 0) {
			return nil, fmt.Errorf("sum dispatch/policy")
		}
		if scan && domain != 0 && (!strings.Contains(f.Source, "if (idx == 0) scan_value =") || !strings.Contains(f.Source, "else scan_value = __fadd_rn")) {
			return nil, fmt.Errorf("scan policy")
		}
	}
	if m := regexp.MustCompile(`if \(n != ([0-9]+)`).FindStringSubmatch(f.Source); m != nil {
		n, _ := strconv.Atoi(m[1])
		if n != count {
			return nil, fmt.Errorf("dispatch extent %d != %d", n, count)
		}
	}
	defs := regexp.MustCompile(`(?m)^\s*(float|unsigned long long) (v_[A-Za-z0-9_]+|index_[A-Za-z0-9_]+) = (.*);$`).FindAllStringSubmatch(f.Source, -1)
	out := make([][]float32, len(f.Outputs))
	for i := range out {
		out[i] = make([]float32, count)
	}
	var total float64
	var prefix float32
	numberSuffix := regexp.MustCompile(`([0-9])f\b`)
	for idx := 0; idx < domain; idx++ {
		values := map[string]float32{}
		indices := map[string]uint64{"idx": uint64(idx)}
		for _, d := range defs {
			expr := numberSuffix.ReplaceAllString(strings.ReplaceAll(d[3], "ULL", ""), "$1")
			if m := regexp.MustCompile(`^\(?(v_[A-Za-z0-9_]+) > 0\.0(?:\)? \?| \?).*$`).FindStringSubmatch(expr); m != nil {
				expr = "relu(" + m[1] + ")"
			}
			tree, err := parser.ParseExpr(expr)
			if err != nil {
				return nil, fmt.Errorf("parse %q: %w", expr, err)
			}
			if d[1] == "unsigned long long" {
				v, err := modelIndexExpression(tree, indices)
				if err != nil {
					return nil, err
				}
				indices[d[2]] = v
				continue
			}
			v, err := shapeFloatExpression(tree, values, indices, storage)
			if err != nil {
				return nil, err
			}
			values[d[2]] = v
		}
		if sum || scan {
			re := `sum_value \+= \(double\)(v_[A-Za-z0-9_]+);`
			if scan {
				re = `if \(idx == 0\) scan_value = (v_[A-Za-z0-9_]+);`
			}
			m := regexp.MustCompile(re).FindStringSubmatch(f.Source)
			if m == nil {
				return nil, fmt.Errorf("missing boundary operand")
			}
			v, ok := values[m[1]]
			if !ok {
				return nil, fmt.Errorf("boundary operand undefined")
			}
			if sum {
				total += float64(v)
			} else {
				if idx == 0 {
					prefix = v
				} else {
					prefix = float32(prefix + v)
				}
				out[0][idx] = prefix
			}
		} else {
			for i, name := range f.Outputs {
				m := regexp.MustCompile(`out_` + regexp.QuoteMeta(name) + `\[idx\] = (v_[A-Za-z0-9_]+);`).FindStringSubmatch(f.Source)
				if m == nil {
					return nil, fmt.Errorf("missing store")
				}
				v, ok := values[m[1]]
				if !ok {
					return nil, fmt.Errorf("undefined output")
				}
				out[i][idx] = v
			}
		}
	}
	if sum {
		out[0][0] = float32(total)
	}
	return out, nil
}

func shapeFloatExpression(e ast.Expr, values map[string]float32, indices map[string]uint64, storage map[string][]float32) (float32, error) {
	switch e := e.(type) {
	case *ast.Ident:
		if v, ok := values[e.Name]; ok {
			return v, nil
		}
	case *ast.BasicLit:
		v, err := strconv.ParseFloat(e.Value, 32)
		return float32(v), err
	case *ast.ParenExpr:
		return shapeFloatExpression(e.X, values, indices, storage)
	case *ast.IndexExpr:
		name, ok := e.X.(*ast.Ident)
		if !ok {
			return 0, fmt.Errorf("invalid load")
		}
		i, err := modelIndexExpression(e.Index, indices)
		if err != nil {
			return 0, err
		}
		v, ok := storage[name.Name]
		if !ok || i >= uint64(len(v)) {
			return 0, fmt.Errorf("out-of-range source %s[%d]", name.Name, i)
		}
		return v[i], nil
	case *ast.UnaryExpr:
		v, err := shapeFloatExpression(e.X, values, indices, storage)
		if err != nil {
			return 0, err
		}
		if e.Op == token.SUB {
			return -v, nil
		}
	case *ast.BinaryExpr:
		a, err := shapeFloatExpression(e.X, values, indices, storage)
		if err != nil {
			return 0, err
		}
		b, err := shapeFloatExpression(e.Y, values, indices, storage)
		if err != nil {
			return 0, err
		}
		switch e.Op {
		case token.ADD:
			return float32(a + b), nil
		case token.SUB:
			return float32(a - b), nil
		case token.MUL:
			return float32(a * b), nil
		case token.QUO:
			return float32(a / b), nil
		}
	case *ast.CallExpr:
		name, ok := e.Fun.(*ast.Ident)
		if !ok || len(e.Args) != 1 {
			return 0, fmt.Errorf("invalid function")
		}
		a, err := shapeFloatExpression(e.Args[0], values, indices, storage)
		if err != nil {
			return 0, err
		}
		switch name.Name {
		case "relu":
			if a > 0 {
				return a, nil
			}
			return 0, nil
		case "expf":
			return float32(math.Exp(float64(a))), nil
		case "sqrtf":
			return float32(math.Sqrt(float64(a))), nil
		}
	}
	return 0, fmt.Errorf("unsupported float expression")
}

func runShapeProbe(t *testing.T, root *LazyNode, data map[*LazyNode][]float32, eager bool) (*Execution, *sumExecutionProbe, *executionPlan) {
	t.Helper()
	plan, probe, bound := prepareSumProbe(t, root, eager, data)
	e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), []func(){func() { probe.releases++ }})
	if err != nil {
		t.Fatal(err)
	}
	return e, probe, plan
}

func TestBinaryShapeExecutionModes(t *testing.T) {
	for _, scalarShape := range [][]int{nil, {1}, {1, 1}} {
		for _, op := range []string{"add", "sub", "mul", "div"} {
			for _, left := range []bool{false, true} {
				for _, eager := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/%t/%t/%v", op, left, eager, scalarShape), func(t *testing.T) {
						ctx := NewLazyContext()
						v := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
						s := NewLazyArray(ctx, scalarShape, cudair.Float32)
						a, b := v, s
						if left {
							a, b = s, v
						}
						root := a.Map2(op, b)
						vector := []float32{1, 2, 3, 4, 5, 6}
						scalar := float32(12)
						want := make([]float32, 6)
						for i, x := range vector {
							a, b := x, scalar
							if left {
								a, b = scalar, x
							}
							switch op {
							case "add":
								want[i] = a + b
							case "sub":
								want[i] = a - b
							case "mul":
								want[i] = a * b
							case "div":
								want[i] = a / b
							}
						}
						// Observe construction, direct flat source, indexed source, planner and execution separately.
						if !slices.Equal(root.Shape, []int{2, 3}) {
							t.Fatal(root.Shape)
						}
						g := cudair.NewGraph()
						ga := g.AddInput("vector", []int{2, 3}, cudair.Float32)
						gb := g.AddInput("scalar", scalarShape, cudair.Float32)
						if left {
							ga, gb = gb, ga
						}
						g.AddBinary(op, ga, gb)
						flat, err := FuseWithSourceMap(g, "shape_flat")
						if err != nil {
							t.Fatal(err)
						}
						got, err := shapeSourceModel(flat, [][]float32{vector, {scalar}}, 6)
						if err != nil || !slices.Equal(got[0], want) {
							t.Fatal("flat", got, err, want)
						}
						indexed, err := FuseLazy([]*LazyNode{v.Node, s.Node}, []*LazyNode{root.Node}, "shape_indexed")
						if err != nil {
							t.Fatal(err)
						}
						got, err = shapeSourceModel(indexed, [][]float32{vector, {scalar}}, 6)
						if err != nil || !slices.Equal(got[0], want) {
							t.Fatal("indexed", got, err, want)
						}
						e, probe, _ := runShapeProbe(t, root.Node, map[*LazyNode][]float32{v.Node: vector, s.Node: {scalar}}, eager)
						values, err := e.Readback(context.Background())
						if err != nil || !slices.Equal(values, want) || !slices.Equal(e.Shape(), []int{2, 3}) {
							t.Fatal(values, err, want)
						}
						t.Logf("host modes construction=%v flat=%x indexed=%x eager=%t values=%v steps=%v allocations=%d", root.Shape, sha256.Sum256([]byte(flat.Source)), sha256.Sum256([]byte(indexed.Source)), eager, values, probe.counts, probe.allocs)
						if probe.counts[len(probe.counts)-1] != 6 || len(probe.storage[e.output.ptr]) != 6 {
							t.Fatal("scalar first input selected output extent")
						}
						if err := e.Close(); err != nil {
							t.Fatal(err)
						}
						if probe.frees != probe.allocs || probe.releases != 1 {
							t.Fatal("ownership")
						}
					})
				}
			}
		}
	}
}

func TestScalarIntermediateAndOutputDomains(t *testing.T) {
	for _, eager := range []bool{false, true} {
		ctx := NewLazyContext()
		v := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
		s := NewLazyArray(ctx, []int{1, 1}, cudair.Float32)
		derived := s.Map("neg").Transpose([]int{1, 0}).Reshape(nil)
		root := v.Transpose([]int{1, 0}).Add(derived)
		flatGraph := cudair.NewGraph()
		flatInput := flatGraph.AddInput("scalar_view", []int{1, 1}, cudair.Float32)
		flatGraph.AddReshape(flatInput, nil)
		if _, err := FuseWithSourceMap(flatGraph, "unsupported_flat_view"); err == nil {
			t.Fatal("flat view operator shortcut")
		}
		direct, err := FuseLazy([]*LazyNode{v.Node, s.Node}, []*LazyNode{root.Node}, "derived_scalar")
		if err != nil {
			t.Fatal(err)
		}
		directValues, err := shapeSourceModel(direct, [][]float32{{1, 2, 3, 4, 5, 6}, {10}}, 6)
		if err != nil || !slices.Equal(directValues[0], []float32{-9, -6, -8, -5, -7, -4}) {
			t.Fatal(directValues, err)
		}

		e, probe, _ := runShapeProbe(t, root.Node, map[*LazyNode][]float32{v.Node: {1, 2, 3, 4, 5, 6}, s.Node: {10}}, eager)
		got, err := e.Readback(context.Background())
		if err != nil || !slices.Equal(got, []float32{-9, -6, -8, -5, -7, -4}) || !slices.Equal(e.Shape(), []int{3, 2}) {
			t.Fatal(got, err)
		}
		if err := e.Close(); err != nil || probe.frees != probe.allocs {
			t.Fatal(err)
		}
	}
}

func TestScalarFanoutDifferentDomains(t *testing.T) {
	for _, eager := range []bool{false, true} {
		ctx := NewLazyContext()
		v := NewLazyArray(ctx, []int{6}, cudair.Float32)
		s := NewLazyArray(ctx, nil, cudair.Float32)
		shared := s.Map("neg")
		root := v.Add(shared).Mul(shared)
		e, probe, plan := runShapeProbe(t, root.Node, map[*LazyNode][]float32{v.Node: {1, 2, 3, 4, 5, 6}, s.Node: {2}}, eager)
		got, err := e.Readback(context.Background())
		if err != nil || !slices.Equal(got, []float32{2, 0, -2, -4, -6, -8}) {
			t.Fatal(got, err)
		}
		ownedScalar := false
		for _, step := range plan.steps {
			for _, out := range step.outputs {
				if out == plan.originals[shared.Node] {
					ownedScalar = true
					if step.count != 1 {
						t.Fatal("scalar output stored at vector extent")
					}
				}
			}
		}
		if !ownedScalar {
			t.Fatal("shared scalar storage erased")
		}
		if err := e.Close(); err != nil || probe.frees != probe.allocs {
			t.Fatal(err)
		}
	}
}

func TestBothScalarExecutionAndSnapshotClones(t *testing.T) {
	for _, eager := range []bool{false, true} {
		ctx := NewLazyContext()
		a := NewLazyArray(ctx, []int{1}, cudair.Float32)
		b := NewLazyArray(ctx, nil, cudair.Float32)
		root := a.Sub(b)
		g := cudair.NewGraph()
		ga := g.AddInput("left", []int{1}, cudair.Float32)
		gb := g.AddInput("right", nil, cudair.Float32)
		g.AddBinary("sub", ga, gb)
		flat, err := FuseWithSourceMap(g, "both_scalar")
		if err != nil {
			t.Fatal(err)
		}
		values, err := shapeSourceModel(flat, [][]float32{{9}, {2}}, 1)
		if err != nil || !slices.Equal(values[0], []float32{7}) {
			t.Fatal(values, err)
		}
		indexed, err := FuseLazy([]*LazyNode{a.Node, b.Node}, []*LazyNode{root.Node}, "both_scalar_indexed")
		if err != nil {
			t.Fatal(err)
		}
		values, err = shapeSourceModel(indexed, [][]float32{{9}, {2}}, 1)
		if err != nil || !slices.Equal(values[0], []float32{7}) {
			t.Fatal(values, err)
		}

		plan, probe, bound := prepareSumProbe(t, root.Node, eager, map[*LazyNode][]float32{a.Node: {9}, b.Node: {2}})
		root.Node.Shape[0] = 99
		a.Node.Shape[0] = 77
		e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), nil)
		if err != nil {
			t.Fatal(err)
		}
		got, err := e.Readback(context.Background())
		if err != nil || !slices.Equal(got, []float32{7}) || !slices.Equal(e.Shape(), []int{1}) {
			t.Fatal(got, err)
		}
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestHandBuiltBinaryShapeRefusal(t *testing.T) {
	for _, shape := range [][]int{{6}, {3, 2}, {3}, {0, -1}} {
		ctx := NewLazyContext()
		a := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
		b := NewLazyArray(ctx, shape, cudair.Float32)
		n := &LazyNode{ID: ctx.newID(), Op: "add", Shape: []int{2, 3}, DType: cudair.Float32, Inputs: []*LazyNode{a.Node, b.Node}}
		if _, err := CompileDeferredGraph(ctx, n); err == nil {
			t.Fatal("hand-built shape bypass", shape)
		}
		if ctx.Tracker.Snapshot().Allocations != 0 {
			t.Fatal("failed graph allocated")
		}
		if _, err := FuseLazy([]*LazyNode{a.Node, b.Node}, []*LazyNode{n}, "forged"); err == nil {
			t.Fatal("indexed bypass", shape)
		}
	}
}

func TestScalarBoundarySumScan(t *testing.T) {
	for _, scan := range []bool{false, true} {
		for _, eager := range []bool{false, true} {
			ctx := NewLazyContext()
			a := NewLazyArray(ctx, []int{3}, cudair.Float32)
			b := NewLazyArray(ctx, nil, cudair.Float32)
			expr := a.Add(b)
			root := sumNode(ctx, expr.Node)
			want := []float32{15}
			if scan {
				root = scanNode(ctx, expr.Node)
				want = []float32{3, 8, 15}
			}
			e, probe, _ := runShapeProbe(t, root, map[*LazyNode][]float32{a.Node: {1, 3, 5}, b.Node: {2}}, eager)
			got, err := e.Readback(context.Background())
			if err != nil || !sameScan(got, want) {
				t.Fatal(got, err, want)
			}
			if err := e.Close(); err != nil || probe.frees != probe.allocs {
				t.Fatal(err)
			}
		}
	}
}

func TestBinaryFlatShapeMutationRefusal(t *testing.T) {
	for _, shape := range [][]int{{6}, {3, 2}, {3}} {
		g := cudair.NewGraph()
		a := g.AddInput("a", []int{2, 3}, cudair.Float32)
		b := g.AddInput("b", []int{2, 3}, cudair.Float32)
		g.AddBinary("add", a, b)
		b.Shape = shape
		if _, err := FuseWithSourceMap(g, "mutated_flat"); err == nil {
			t.Fatal("flat equal-count bypass", shape)
		}
	}
}

func TestScalarZeroSourceAndPlannerDomains(t *testing.T) {
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{0, math.MaxInt, math.MaxInt}, cudair.Float32)
	b := NewLazyArray(ctx, nil, cudair.Float32)
	root := a.Add(b)
	f, err := FuseLazy([]*LazyNode{a.Node, b.Node}, []*LazyNode{root.Node}, "empty_domain")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(f.Source, "float v_") || strings.Contains(f.Source, "out_"+root.Node.Name()+"[idx]") {
		t.Fatal("empty domain emitted a read/store")
	}
	p, err := root.Materialize()
	if err != nil || p.AllocatedBytes != 0 {
		t.Fatal(p, err)
	}
	// Logical no-work is distinct from direct IR concrete stride admissibility.
	g := cudair.NewGraph()
	ga := g.AddInput("empty", a.Shape, cudair.Float32)
	gb := g.AddInput("scalar", nil, cudair.Float32)
	g.AddBinary("add", ga, gb)
	if _, err := FuseWithSourceMap(g, "empty_strides"); err == nil {
		t.Fatal("unrepresentable IR layout mistaken for zero-work source")
	}
}

func TestBinaryExecutionForgedMetadataRefusal(t *testing.T) {
	for _, dtype := range []bool{false, true} {
		ctx := NewLazyContext()
		a := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
		b := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
		p, err := a.Add(b).Materialize()
		if err != nil {
			t.Fatal(err)
		}
		want := cudair.ErrShapeMismatch
		if dtype {
			b.Node.DType = cudair.Int32
			want = cudair.ErrTypeMismatch
		} else {
			b.Node.Shape = []int{6}
		}
		for _, execute := range []func(context.Context, *cuda.Stream, map[string]ExecutionBinding) (*Execution, error){p.Execute, p.ExecuteEager} {
			e, err := execute(context.Background(), nil, nil)
			if e != nil || !errors.Is(err, want) {
				t.Fatal("forged metadata reached native stage", e, err, want)
			}
		}
	}
}
