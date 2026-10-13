package egraph_test

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/egraph"
	"go/ast"
	"go/parser"
	"go/token"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func TestTypedBinaryShapeFusion(t *testing.T) {
	for _, op := range []string{"add", "sub", "mul", "div"} {
		for _, left := range []bool{false, true} {
			for _, transposed := range []bool{false, true} {
				v := indexedExpr("v", []int{2, 3})
				shape := []int{2, 3}
				if transposed {
					shape = []int{3, 2}
					v = indexedTranspose(v, shape, []int{1, 0})
				}
				s := indexedExpr("s", nil)
				a, b := v, s
				if left {
					a, b = s, v
				}
				root := indexedExpr(op, shape, a, b)
				plan, err := egraph.ExtractFusionRegions(root)
				if err != nil {
					t.Fatal(op, left, transposed, err)
				}
				f := plan.Regions[0].Kernel
				if !strings.Contains(f.Source, "n != 6") || strings.Contains(f.Source, "1024") || !strings.Contains(f.Source, "= 0ULL;") {
					t.Fatal("typed scalar shape/index discarded", f.Source)
				}
				if len(f.Inputs) != 2 || len(f.Outputs) != 1 {
					t.Fatal("typed ABI")
				}
				vector := []float32{1, 2, 3, 4, 5, 6}
				ordered := vector
				if transposed {
					ordered = []float32{1, 4, 2, 5, 3, 6}
				}
				data := map[string][]float32{}
				for i, in := range plan.Regions[0].Inputs {
					if in.Op == "v" {
						data[f.Inputs[i]] = vector
					} else if in.Op == "s" {
						data[f.Inputs[i]] = []float32{12}
					} else {
						t.Fatal("unknown input", in.Op)
					}
				}
				want := make([]float32, 6)
				for i, v := range ordered {
					a, b := v, float32(12)
					if left {
						a, b = b, a
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
				got, err := typedSourceValues(f.Source, f.Outputs[0], data, 6)
				t.Logf("typed egraph host source=%x op=%s scalar-left=%t transpose=%t values=%v", sha256.Sum256([]byte(f.Source)), op, left, transposed, got)
				if err != nil || !slices.Equal(got, want) {
					t.Fatal("typed dataflow", got, err, want)
				}

			}
		}
	}
}
func TestTypedBinaryShapeRefusal(t *testing.T) {
	for _, other := range [][]int{{6}, {3, 2}, {3}} {
		root := indexedExpr("add", []int{2, 3}, indexedExpr("v", []int{2, 3}), indexedExpr("s", other))
		if _, err := egraph.ExtractFusionRegions(root); !errors.Is(err, cudair.ErrShapeMismatch) {
			t.Fatal(other, err)
		}
	}
	root := indexedExpr("add", []int{1}, indexedExpr("v", []int{2, 3}), indexedExpr("s", nil))
	if _, err := egraph.ExtractFusionRegions(root); err == nil {
		t.Fatal("declared output discarded")
	}
	for _, which := range []string{"leaf", "root"} {
		root := indexedExpr("add", []int{2, 3}, indexedExpr("v", []int{2, 3}), indexedExpr("s", nil))
		if which == "leaf" {
			root.Children[1].Data = nil
		} else {
			root.Data = nil
		}
		if _, err := egraph.ExtractFusionRegions(root); err == nil {
			t.Fatal("partial metadata fabricated extent")
		}
	}
}

// typedSourceValues evaluates the emitted source's indices/operands, not a
// prerecorded result. This bounded host evaluator does not execute CUDA.
func typedSourceValues(source, output string, data map[string][]float32, count int) ([]float32, error) {
	defs := regexp.MustCompile(`(?m)^\s*(float|unsigned long long) ([A-Za-z0-9_]+) = (.*);$`).FindAllStringSubmatch(source, -1)
	store := regexp.MustCompile(`out_` + regexp.QuoteMeta(output) + `\[idx\] = ([A-Za-z0-9_]+);`).FindStringSubmatch(source)
	if store == nil {
		return nil, fmt.Errorf("missing store")
	}
	out := make([]float32, count)
	for i := range out {
		values := map[string]float64{"idx": float64(i)}
		for _, d := range defs {
			expr := strings.ReplaceAll(d[3], "ULL", "")
			tree, err := parser.ParseExpr(expr)
			if err != nil {
				return nil, err
			}
			v, err := typedValue(tree, values, data, d[1] == "unsigned long long")
			if err != nil {
				return nil, err
			}
			if d[1] == "float" {
				v = float64(float32(v))
			}
			values[d[2]] = v
		}
		v, ok := values[store[1]]
		if !ok {
			return nil, fmt.Errorf("undefined store")
		}
		out[i] = float32(v)
	}
	return out, nil
}
func typedValue(e ast.Expr, values map[string]float64, data map[string][]float32, integer bool) (float64, error) {
	switch e := e.(type) {
	case *ast.Ident:
		if v, ok := values[e.Name]; ok {
			return v, nil
		}
	case *ast.BasicLit:
		return strconv.ParseFloat(e.Value, 64)
	case *ast.ParenExpr:
		return typedValue(e.X, values, data, integer)
	case *ast.IndexExpr:
		name, ok := e.X.(*ast.Ident)
		if !ok {
			return 0, fmt.Errorf("bad load")
		}
		index, err := typedValue(e.Index, values, data, true)
		if err != nil {
			return 0, err
		}
		v, ok := data[name.Name]
		if !ok || index < 0 || index >= float64(len(v)) || index != float64(int(index)) {
			return 0, fmt.Errorf("bad source extent")
		}
		return float64(v[int(index)]), nil
	case *ast.UnaryExpr:
		a, err := typedValue(e.X, values, data, integer)
		if err != nil {
			return 0, err
		}
		if e.Op == token.SUB {
			return -a, nil
		}
	case *ast.BinaryExpr:
		a, err := typedValue(e.X, values, data, integer)
		if err != nil {
			return 0, err
		}
		b, err := typedValue(e.Y, values, data, integer)
		if err != nil {
			return 0, err
		}
		switch e.Op {
		case token.ADD:
			return float64(float32(a + b)), nil
		case token.SUB:
			return float64(float32(a - b)), nil
		case token.MUL:
			return a * b, nil
		case token.QUO:
			if b != 0 {
				if integer {
					return float64(uint64(a) / uint64(b)), nil
				}
				return a / b, nil
			}
		case token.REM:
			if b != 0 {
				return float64(uint64(a) % uint64(b)), nil
			}
		}
	}
	return 0, fmt.Errorf("unsupported typed expression")
}

func TestTypedCycleBoundaryConsumer(t *testing.T) {
	source := indexedExpr("x", []int{3})
	cycle := indexedExpr("cycle", []int{6}, source)
	root := indexedExpr("add", []int{6}, cycle, indexedExpr("s", nil))
	plan, err := egraph.ExtractFusionRegions(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Regions) != 1 || len(plan.Regions[0].Inputs) != 2 || plan.Regions[0].Inputs[0].Op != "cycle" || !strings.Contains(plan.Regions[0].Kernel.Source, "n != 6") {
		t.Fatal("typed boundary extent discarded", plan)
	}
	// This is external boundary-storage metadata, not proof that egraph executed Cycle.
}

func TestTypedEmptyAndNegativeShapes(t *testing.T) {
	root := indexedExpr("add", []int{2, 0}, indexedExpr("empty", []int{2, 0}), indexedExpr("scalar", nil))
	plan, err := egraph.ExtractFusionRegions(root)
	if err != nil {
		t.Fatal(err)
	}
	source := plan.Regions[0].Kernel.Source
	if strings.Contains(source, "float v_") || !strings.Contains(source, "n != 0") {
		t.Fatal("empty typed region read values")
	}
	bad := indexedExpr("add", []int{0}, indexedExpr("empty", []int{0, -1}), indexedExpr("scalar", nil))
	if _, err := egraph.ExtractFusionRegions(bad); err == nil {
		t.Fatal("typed negative-after-zero discarded")
	}
	a := indexedExpr("a", []int{1})
	b := indexedExpr("b", nil)
	root = indexedExpr("sub", []int{1}, a, b)
	plan, err = egraph.ExtractFusionRegions(root)
	if err != nil {
		t.Fatal(err)
	}
	f := plan.Regions[0].Kernel
	data := map[string][]float32{f.Inputs[0]: {9}, f.Inputs[1]: {2}}
	values, err := typedSourceValues(f.Source, f.Outputs[0], data, 1)
	if err != nil || !slices.Equal(values, []float32{7}) {
		t.Fatal(values, err)
	}
}

func TestTypedScalarIntermediate(t *testing.T) {
	v := indexedExpr("v", []int{2, 3})
	s := indexedExpr("s", []int{1, 1})
	derived := indexedExpr("reshape", nil, indexedTranspose(indexedExpr("neg", []int{1, 1}, s), []int{1, 1}, []int{1, 0}))
	root := indexedExpr("add", []int{3, 2}, indexedTranspose(v, []int{3, 2}, []int{1, 0}), derived)
	plan, err := egraph.ExtractFusionRegions(root)
	if err != nil {
		t.Fatal(err)
	}
	f := plan.Regions[0].Kernel
	data := map[string][]float32{}
	for i, input := range plan.Regions[0].Inputs {
		if input.Op == "v" {
			data[f.Inputs[i]] = []float32{1, 2, 3, 4, 5, 6}
		} else if input.Op == "s" {
			data[f.Inputs[i]] = []float32{10}
		} else {
			t.Fatal("unknown typed input")
		}
	}
	values, err := typedSourceValues(f.Source, f.Outputs[0], data, 6)
	if err != nil || !slices.Equal(values, []float32{-9, -6, -8, -5, -7, -4}) {
		t.Fatal(values, err)
	}
}
