package cudafuse

import (
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func scanNode(ctx *LazyContext, input *LazyNode) *LazyNode {
	return &LazyNode{ID: ctx.newID(), Op: "scan", Inputs: []*LazyNode{input}, Shape: slices.Clone(input.Shape), DType: cudair.Float32, Attrs: map[string]any{"scan_op": "sum"}}
}

// scanSourceModel interprets the bounded emitted prefix form and reuses only
// the pointwise/index scalar interpreter for its input expression. It executes
// no compiler or CUDA API and is not physical or general CUDA evidence.
func scanSourceModel(f FusedKernel, inputs [][]float32, count int) ([][]float32, error) {
	seed := regexp.MustCompile(`if \(idx == 0\) scan_value = (v_[A-Za-z0-9_]+);`).FindStringSubmatch(f.Source)
	if len(f.Outputs) != 1 || seed == nil || !strings.Contains(f.Source, "float scan_value = 0.0f;") || !strings.Contains(f.Source, "else scan_value = __fadd_rn(scan_value, "+seed[1]+");") || !strings.Contains(f.Source, "out_"+f.Outputs[0]+"[idx] = scan_value;") || !strings.Contains(f.Source, "if (blockIdx.x != 0 || threadIdx.x != 0) return;") || !strings.Contains(f.Source, fmt.Sprintf("if (n != %d || n <= 0", count)) {
		return nil, fmt.Errorf("model scan seed, arithmetic, dispatch or store")
	}
	bound := regexp.MustCompile(`for \(unsigned long long idx = 0; idx < ([0-9]+)ULL; idx\+\+\)`).FindStringSubmatch(f.Source)
	if bound == nil {
		return nil, fmt.Errorf("model scan traversal")
	}
	n, _ := strconv.Atoi(bound[1])
	if n != count {
		return nil, fmt.Errorf("model scan extent")
	}
	scalar := f
	scalar.Source = strings.ReplaceAll(scalar.Source, "    float scan_value = 0.0f;", "")
	scalar.Source = strings.ReplaceAll(scalar.Source, "        if (idx == 0) scan_value = "+seed[1]+";", "")
	scalar.Source = strings.ReplaceAll(scalar.Source, "        else scan_value = __fadd_rn(scan_value, "+seed[1]+");", "")
	scalar.Source = strings.ReplaceAll(scalar.Source, "out_"+f.Outputs[0]+"[idx] = scan_value;", "out_"+f.Outputs[0]+"[idx] = "+seed[1]+";")
	values, err := sourceModel(scalar, inputs, count)
	if err != nil {
		return nil, err
	}
	var prefix float32
	for i, v := range values[0] {
		if i == 0 {
			prefix = v
		} else {
			prefix = float32(prefix + v)
		}
		values[0][i] = prefix
	}
	return values, nil
}

// ordinaryScan is independent of emitted-source parsing and plan scheduling.
func ordinaryScan(input []float32) []float32 {
	out := slices.Clone(input)
	for i := 1; i < len(out); i++ {
		out[i] = float32(out[i-1] + input[i])
	}
	return out
}
func sameScan(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) && !(math.IsNaN(float64(a[i])) && math.IsNaN(float64(b[i]))) {
			return false
		}
	}
	return true
}

func TestScanSourceOrderedPrefix(t *testing.T) {
	cases := [][]float32{nil, {float32(math.Copysign(0, -1))}, {16777216, 1, -16777216}, {-3, 2, -5, 8}, {math.SmallestNonzeroFloat32, math.SmallestNonzeroFloat32}, {float32(math.Inf(1)), float32(math.Inf(-1))}, {float32(math.NaN()), 1}}
	tail := make([]float32, 1003)
	for i := range tail {
		tail[i] = float32(i%9-4) / 8
	}
	cases = append(cases, tail)
	for _, data := range cases {
		t.Run(fmt.Sprint(len(data), data[:min(len(data), 3)]), func(t *testing.T) {
			ctx := NewLazyContext()
			a := NewLazyArray(ctx, []int{len(data)}, cudair.Float32)
			f, err := compileScanKernel([]*LazyNode{a.Node}, a.Node, scanNode(ctx, a.Node), "scan")
			if err != nil {
				t.Fatal(err)
			}
			got, err := sourceModel(f, [][]float32{data}, len(data))
			if err != nil || !sameScan(got[0], ordinaryScan(data)) {
				t.Fatal(got, err, ordinaryScan(data))
			}
			if len(data) == 0 && (strings.Contains(f.Source, "scan_value") || strings.Contains(f.Source, "[idx]")) {
				t.Fatal("empty source forms input or output accesses")
			}
		})
	}
}

func TestScanSourceSemanticOmissions(t *testing.T) {
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{3}, cudair.Float32)
	f, err := compileScanKernel([]*LazyNode{a.Node}, a.Node, scanNode(ctx, a.Node), "scan")
	if err != nil {
		t.Fatal(err)
	}
	data := [][]float32{{16777216, 1, -16777216}}
	good, err := sourceModel(f, data, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, part := range []string{"float scan_value = 0.0f;", "if (idx == 0)", "__fadd_rn(scan_value,", "[idx] = scan_value;", "idx++", "if (blockIdx.x != 0 || threadIdx.x != 0) return;"} {
		bad := f
		bad.Source = strings.Replace(f.Source, part, "/* omitted */", 1)
		if bad.Source == f.Source {
			t.Fatal(part)
		}
		got, err := sourceModel(bad, data, 3)
		if err == nil && sameScan(got[0], good[0]) {
			t.Fatal("omission survived", part)
		}
	}
}

func TestScanSourceIndexedCoordinates(t *testing.T) {
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
	v := a.Transpose([]int{1, 0}).Reshape([]int{6})
	f, err := compileScanKernel([]*LazyNode{a.Node}, v.Node, scanNode(ctx, v.Node), "indexed_scan")
	if err != nil {
		t.Fatal(err)
	}
	got, err := sourceModel(f, [][]float32{{1, 2, 3, 4, 5, 6}}, 6)
	if err != nil || !sameScan(got[0], ordinaryScan([]float32{1, 4, 2, 5, 3, 6})) {
		t.Fatal(got, err)
	}
}
