package cudafuse

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// filterSourceModel interprets the emitted bounded count/scatter forms. Scalar
// expressions use the existing Go source model. It runs no C++/CUDA compiler.
func filterSourceModel(f FusedKernel, inputs [][]float32, n int) ([][]float32, error) {
	if strings.Contains(f.Source, "unsigned int invalid_predicate") {
		if len(f.Outputs) != 4 || !strings.Contains(f.Source, fmt.Sprintf("if (n != %d || n <= 0", n)) || !strings.Contains(f.Source, "if (blockIdx.x != 0 || threadIdx.x != 0) return;") || !strings.Contains(f.Source, fmt.Sprintf("idx < %dULL; idx++", n)) {
			return nil, fmt.Errorf("Filter count dispatch/domain")
		}
		pointwise := f
		pointwise.Outputs = f.Outputs[:2]
		pointwise.Source = strings.ReplaceAll(f.Source, "scan_value", "filter_prefix_value")
		out, err := shapeSourceModel(pointwise, inputs, n)
		if err != nil {
			return nil, err
		}
		predicate := regexp.MustCompile(`out_` + regexp.QuoteMeta(f.Outputs[1]) + `\[idx\] = (v_[A-Za-z0-9_]+);`).FindStringSubmatch(f.Source)
		selection := regexp.MustCompile(`float filter_selected = (v_[A-Za-z0-9_]+) == ([01])\.0f \? 1\.0f : 0\.0f;`).FindStringSubmatch(f.Source)
		if predicate == nil || selection == nil || selection[1] != predicate[1] || !strings.Contains(f.Source, "invalid_predicate |= !("+predicate[1]+" == 0.0f || "+predicate[1]+" == 1.0f);") || !strings.Contains(f.Source, "if (idx == 0) scan_value = filter_selected;") || !strings.Contains(f.Source, "else scan_value = __fadd_rn(scan_value, filter_selected);") || !strings.Contains(f.Source, "out_"+f.Outputs[2]+"[idx] = scan_value;") {
			return nil, fmt.Errorf("Filter predicate/prefix source")
		}
		selected, _ := strconv.Atoi(selection[2])
		prefix := make([]float32, n)
		var count, invalid float32
		for i, p := range out[1] {
			if p != 0 && p != 1 {
				invalid = 1
			}
			if p == float32(selected) {
				count++
			}
			prefix[i] = count
		}
		metadata := []float32{invalid, count}
		if strings.Contains(f.Source, "out_"+f.Outputs[3]+"[0] = 0.0f;") {
			metadata[0] = 0
		} else if !strings.Contains(f.Source, "out_"+f.Outputs[3]+"[0] = (float)invalid_predicate;") {
			return nil, fmt.Errorf("Filter status write")
		}
		if strings.Contains(f.Source, "out_"+f.Outputs[3]+"[1] = scan_value + 1.0f;") {
			metadata[1]++
		} else if !strings.Contains(f.Source, "out_"+f.Outputs[3]+"[1] = scan_value;") {
			return nil, fmt.Errorf("Filter count write")
		}
		return append(out, prefix, metadata), nil
	}
	if strings.Contains(f.Source, "unsigned int destination") {
		if len(inputs) != 3 || len(f.Inputs) != 3 || len(f.Outputs) != 1 || !strings.Contains(f.Source, fmt.Sprintf("if (n != %d || n <= 0", n)) || !strings.Contains(f.Source, "idx < (unsigned int)n;") {
			return nil, fmt.Errorf("Filter scatter dispatch/domain")
		}
		bound := regexp.MustCompile(`destination < ([0-9]+)u`).FindStringSubmatch(f.Source)
		selection := regexp.MustCompile(`if \(` + regexp.QuoteMeta(f.Inputs[1]) + `\[idx\] == ([01])\.0f\)`).FindStringSubmatch(f.Source)
		if bound == nil || selection == nil || !strings.Contains(f.Source, "destination = (unsigned int)"+f.Inputs[2]+"[idx] - 1u;") || !strings.Contains(f.Source, "out_"+f.Outputs[0]+"[destination] = "+f.Inputs[0]+"[idx];") {
			return nil, fmt.Errorf("Filter stable scatter source")
		}
		size, _ := strconv.Atoi(bound[1])
		selected, _ := strconv.Atoi(selection[1])
		out := make([]float32, size)
		written := make([]bool, size)
		for i := 0; i < n; i++ {
			if inputs[1][i] == float32(selected) {
				destination := int(inputs[2][i]) - 1
				if destination >= 0 && destination < size {
					if written[destination] {
						return nil, fmt.Errorf("duplicate scatter destination")
					}
					out[destination] = inputs[0][i]
					written[destination] = true
				}
			}
		}
		for _, yes := range written {
			if !yes {
				return nil, fmt.Errorf("unwritten Filter output")
			}
		}
		return [][]float32{out}, nil
	}
	return shapeSourceModel(f, inputs, n)
}

func TestFilterSourceStableAndOmissions(t *testing.T) {
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{5}, cudair.Float32)
	p := NewLazyArray(ctx, []int{5}, cudair.Float32)
	f := a.Map("neg").Filter(p)
	count, err := compileFilterCount([]*LazyNode{a.Node, p.Node}, f.Node.Inputs[0], p.Node, f.Node, "count")
	if err != nil {
		t.Fatal(err)
	}
	got, err := filterSourceModel(count, [][]float32{{2, 5, 8, 11, 14}, {1, 0, 1, 0, 1}}, 5)
	if err != nil || !slices.Equal(got[0], []float32{-2, -5, -8, -11, -14}) || !slices.Equal(got[2], []float32{1, 1, 2, 2, 3}) || !slices.Equal(got[3], []float32{0, 3}) {
		t.Fatal(got, err)
	}
	scatter := compileFilterScatter(f.Node, 5, 3, "scatter")
	selected, err := filterSourceModel(scatter, got[:3], 5)
	if err != nil || !slices.Equal(selected[0], []float32{-2, -8, -14}) {
		t.Fatal(selected, err)
	}
	for _, part := range []string{"n != 5", "idx++", "invalid_predicate |=", "if (idx == 0)", "__fadd_rn(scan_value,", "[1] = scan_value;"} {
		bad := count
		bad.Source = strings.Replace(count.Source, part, "/* omitted */", 1)
		if bad.Source == count.Source {
			t.Fatal("missing mutation anchor", part)
		}
		if result, err := filterSourceModel(bad, [][]float32{{2, 5, 8, 11, 14}, {1, 0, 1, 0, 1}}, 5); err == nil {
			if _, err := filterCount(result[1], result[2], result[3]); err == nil {
				t.Fatal("count source omission survived", part)
			}
		}
	}
	for _, part := range []string{"idx < (unsigned int)n", "[idx] == 1.0f", "[idx] - 1u", "[destination] ="} {
		bad := scatter
		bad.Source = strings.Replace(scatter.Source, part, "/* omitted */", 1)
		if bad.Source == scatter.Source {
			t.Fatal("missing mutation anchor", part)
		}
		if result, err := filterSourceModel(bad, got[:3], 5); err == nil && slices.Equal(result[0], selected[0]) {
			t.Fatal("scatter source omission survived", part)
		}
	}
}
