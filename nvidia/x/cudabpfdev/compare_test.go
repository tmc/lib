package cudabpfdev

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestCompareOutputs(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	y := g.AddUnary("neg", x)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	inputs := map[string][]float32{"x": {1, -2}}
	exact := NumericalPolicy{Mode: "exact"}
	buf := func(a, b float32) map[string][]float32 { return map[string][]float32{y.Name: {a, b}} }
	for _, tc := range []struct {
		name                   string
		original, instrumented map[string][]float32
		policy                 NumericalPolicy
		want                   string
	}{
		{"equal", buf(-1, 2), buf(-1, 2), exact, ""},
		{"both wrong", buf(1, -2), buf(1, -2), exact, "original/reference"},
		{"instrumented wrong", buf(-1, 2), buf(-1, -2), exact, "instrumented/reference"},
		{"missing", buf(-1, 2), nil, exact, "count mismatch"},
		{"renamed", buf(-1, 2), map[string][]float32{"wrong": {-1, 2}}, exact, "missing"},
		{"short", buf(-1, 2), map[string][]float32{y.Name: {-1}}, exact, "length mismatch"},
		{"tolerant", buf(-1, 2), buf(-1.05, 2), NumericalPolicy{Mode: "tolerance", AbsoluteTolerance: .1}, ""},
		{"opposite errors", buf(-1.08, 2), buf(-.92, 2), NumericalPolicy{Mode: "tolerance", AbsoluteTolerance: .1}, "instrumented/original"},
		{"undeclared", buf(-1, 2), buf(-1, 2), NumericalPolicy{}, "mode"},
		{"negative tolerance", buf(-1, 2), buf(-1, 2), NumericalPolicy{Mode: "tolerance", AbsoluteTolerance: -1}, "invalid"},
		{"nan tolerance", buf(-1, 2), buf(-1, 2), NumericalPolicy{Mode: "tolerance", RelativeTolerance: math.NaN()}, "invalid"},
		{"infinite tolerance", buf(-1, 2), buf(-1, 2), NumericalPolicy{Mode: "tolerance", AbsoluteTolerance: math.Inf(1)}, "invalid"},
		{"relaxed exact", buf(-1, 2), buf(-1, 2), NumericalPolicy{Mode: "exact", AbsoluteTolerance: 1}, "exact comparison"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := CompareOutputs(g, inputs, tc.original, tc.instrumented, tc.policy)
			if tc.want == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v want %q", err, tc.want)
			}
		})
	}
	if err := CompareOutputs(nil, inputs, nil, nil, exact); err == nil {
		t.Fatal("nil graph accepted")
	}
	if err := CompareOutputs(g, nil, buf(-1, 2), buf(-1, 2), exact); err == nil {
		t.Fatal("missing reference inputs accepted")
	}
}

func TestNumericalSpecialValues(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	if err := g.SetOutputs(x); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name      string
		want, got float32
		policy    NumericalPolicy
		equal     bool
	}{
		{"signed zero", 0, math.Float32frombits(1 << 31), NumericalPolicy{Mode: "tolerance", AbsoluteTolerance: 1}, false},
		{"relaxed zero", 0, math.Float32frombits(1 << 31), NumericalPolicy{Mode: "tolerance", IgnoreZeroSign: true}, true},
		{"nan default", float32(math.NaN()), float32(math.NaN()), NumericalPolicy{Mode: "exact"}, false},
		{"nan explicit", math.Float32frombits(0x7fc00001), math.Float32frombits(0x7fc00002), NumericalPolicy{Mode: "exact", EqualNaNs: true}, true},
		{"unexpected nan", 1, float32(math.NaN()), NumericalPolicy{Mode: "tolerance", EqualNaNs: true}, false},
		{"infinity", float32(math.Inf(1)), float32(math.Inf(1)), NumericalPolicy{Mode: "tolerance"}, true},
		{"opposite infinity", float32(math.Inf(1)), float32(math.Inf(-1)), NumericalPolicy{Mode: "tolerance", RelativeTolerance: 100}, false},
		{"finite infinity", 1, float32(math.Inf(1)), NumericalPolicy{Mode: "tolerance", RelativeTolerance: 100}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ref := map[string][]float32{"x": {tc.want}}
			got := map[string][]float32{"x": {tc.got}}
			err := CompareOutputs(g, ref, ref, got, tc.policy)
			if (err == nil) != tc.equal {
				t.Fatalf("error=%v equal=%v", err, tc.equal)
			}
		})
	}
}
