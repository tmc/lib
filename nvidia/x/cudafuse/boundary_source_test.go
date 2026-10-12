package cudafuse

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func sumNode(ctx *LazyContext, input *LazyNode) *LazyNode {
	return &LazyNode{ID: ctx.newID(), Op: "sum", Inputs: []*LazyNode{input}, Shape: []int{1}, DType: cudair.Float32, Attrs: map[string]any{"axis": 0}}
}

// sourceModel interprets a deliberately small scalar subset of the emitted
// source. It is host evidence for emitted dataflow, not a CUDA execution model.
func sourceModel(f FusedKernel, inputs [][]float32, count int) ([][]float32, error) {
	if len(inputs) != len(f.Inputs) {
		return nil, fmt.Errorf("model input ABI")
	}
	storage := make(map[string][]float32)
	for i, name := range f.Inputs {
		storage[name] = inputs[i]
	}
	outputs := make([][]float32, len(f.Outputs))
	for i := range outputs {
		outputs[i] = make([]float32, count)
	}
	sum := strings.Contains(f.Source, "double sum_value = 0.0;")
	domain := count
	if sum {
		if count != 1 || !strings.Contains(f.Source, "if (n != 1") || !strings.Contains(f.Source, "if (blockIdx.x != 0 || threadIdx.x != 0) return;") || !strings.Contains(f.Source, "out_"+f.Outputs[0]+"[0] = (float)sum_value;") {
			return nil, fmt.Errorf("model sum dispatch or store")
		}
		domain = 0
		if m := regexp.MustCompile(`idx < ([0-9]+)ULL`).FindStringSubmatch(f.Source); m != nil {
			domain, _ = strconv.Atoi(m[1])
		}
	}
	definitions := regexp.MustCompile(`(?m)^\s*(float|unsigned long long) ([A-Za-z0-9_]+) = (.*);$`).FindAllStringSubmatch(f.Source, -1)
	var total float64
	for idx := 0; idx < domain; idx++ {
		values := make(map[string]float32)
		indices := map[string]uint64{"idx": uint64(idx)}
		for _, d := range definitions {
			expr := strings.TrimSuffix(d[3], "f")
			if d[1] == "unsigned long long" {
				tree, err := parser.ParseExpr(strings.ReplaceAll(expr, "ULL", ""))
				if err != nil {
					return nil, err
				}
				v, err := modelIndexExpression(tree, indices)
				if err != nil {
					return nil, err
				}
				indices[d[2]] = v
				continue
			}
			if load := regexp.MustCompile(`^([A-Za-z0-9_]+)\[([A-Za-z0-9_]+)\]$`).FindStringSubmatch(expr); load != nil {
				v := storage[load[1]]
				index, ok := indices[load[2]]
				if !ok || index >= uint64(len(v)) {
					return nil, fmt.Errorf("model read beyond input %s", load[1])
				}
				values[d[2]] = v[index]
				continue
			}
			tree, err := parser.ParseExpr(expr)
			if err != nil {
				return nil, fmt.Errorf("model unsupported expression %q", expr)
			}
			v, err := modelFloatExpression(tree, values)
			if err != nil {
				return nil, err
			}
			values[d[2]] = v
		}
		if sum {
			m := regexp.MustCompile(`sum_value \+= \(double\)(v_[A-Za-z0-9_]+);`).FindStringSubmatch(f.Source)
			if m == nil {
				return nil, fmt.Errorf("model missing FP64 accumulation")
			}
			v, ok := values[m[1]]
			if !ok {
				return nil, fmt.Errorf("model missing sum operand")
			}
			total += float64(v)
		} else {
			for i, name := range f.Outputs {
				m := regexp.MustCompile(`out_` + regexp.QuoteMeta(name) + `\[idx\] = (v_[A-Za-z0-9_]+);`).FindStringSubmatch(f.Source)
				if m == nil {
					return nil, fmt.Errorf("model missing output store %s", name)
				}
				v, ok := values[m[1]]
				if !ok {
					return nil, fmt.Errorf("model missing output value")
				}
				outputs[i][idx] = v
			}
		}
	}
	if sum {
		outputs[0][0] = float32(total)
	}
	return outputs, nil
}

func modelIndexExpression(e ast.Expr, indices map[string]uint64) (uint64, error) {
	switch e := e.(type) {
	case *ast.Ident:
		if v, ok := indices[e.Name]; ok {
			return v, nil
		}
	case *ast.BasicLit:
		return strconv.ParseUint(e.Value, 10, 64)
	case *ast.ParenExpr:
		return modelIndexExpression(e.X, indices)
	case *ast.BinaryExpr:
		a, err := modelIndexExpression(e.X, indices)
		if err != nil {
			return 0, err
		}
		b, err := modelIndexExpression(e.Y, indices)
		if err != nil {
			return 0, err
		}
		switch e.Op {
		case token.ADD:
			return a + b, nil
		case token.MUL:
			return a * b, nil
		case token.QUO:
			if b != 0 {
				return a / b, nil
			}
		case token.REM:
			if b != 0 {
				return a % b, nil
			}
		}
	}
	return 0, fmt.Errorf("model missing or unsupported index")
}

func modelFloatExpression(e ast.Expr, values map[string]float32) (float32, error) {
	switch e := e.(type) {
	case *ast.Ident:
		v, ok := values[e.Name]
		if ok {
			return v, nil
		}
	case *ast.ParenExpr:
		return modelFloatExpression(e.X, values)
	case *ast.UnaryExpr:
		v, err := modelFloatExpression(e.X, values)
		if e.Op == token.SUB {
			return -v, err
		}
	case *ast.BinaryExpr:
		a, err := modelFloatExpression(e.X, values)
		if err != nil {
			return 0, err
		}
		b, err := modelFloatExpression(e.Y, values)
		if err != nil {
			return 0, err
		}
		switch e.Op {
		case token.ADD:
			return a + b, nil
		case token.SUB:
			return a - b, nil
		case token.MUL:
			return a * b, nil
		case token.QUO:
			return a / b, nil
		}
	}
	return 0, fmt.Errorf("model missing or unsupported scalar operand")
}

func TestSumSourceFP64AndEmptyWrite(t *testing.T) {
	for _, n := range []int{0, 1, 3, 1003} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			ctx := NewLazyContext()
			x := NewLazyArray(ctx, []int{n}, cudair.Float32)
			root := sumNode(ctx, x.Node)
			f, err := compileSumKernel([]*LazyNode{x.Node}, x.Node, root, "sum_control")
			if err != nil {
				t.Fatal(err)
			}
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(i%11-5) / 8
			}
			if n == 3 {
				data = []float32{16777216, 1, -16777216}
			}
			got, err := sourceModel(f, [][]float32{data}, 1)
			var want float64
			for _, v := range data {
				want += float64(v)
			}
			if err != nil || got[0][0] != float32(want) || n == 0 && math.Float32bits(got[0][0]) != 0 {
				t.Fatal(got, err, want)
			}
		})
	}
}

func TestSumSourceOmissionControls(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{3}, cudair.Float32)
	y := NewLazyArray(ctx, []int{3}, cudair.Float32)
	expr := x.Add(y).Map("neg")
	root := sumNode(ctx, expr.Node)
	f, err := compileSumKernel([]*LazyNode{x.Node, y.Node}, expr.Node, root, "sum_controls")
	if err != nil {
		t.Fatal(err)
	}
	data := [][]float32{{16777216, 1, -16777216}, {2, 3, 4}}
	good, err := sourceModel(f, data, 1)
	if err != nil || good[0][0] != -10 {
		t.Fatal(good, err)
	}
	for _, omission := range []string{"out_" + root.Name() + "[0] = (float)sum_value;", "sum_value += (double)", y.Node.Name() + "[idx]", "idx < 3ULL", "if (blockIdx.x != 0 || threadIdx.x != 0) return;"} {
		bad := f
		bad.Source = strings.Replace(f.Source, omission, "/* omitted */", 1)
		if bad.Source == f.Source {
			t.Fatal("missing omission target", omission)
		}
		got, err := sourceModel(bad, data, 1)
		if err == nil && got[0][0] == good[0][0] {
			t.Fatal("source omission survived", omission)
		}
	}
}
