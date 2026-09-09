package gomlx

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/testutil"
)

func TestNativeUnaryMath(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	input := []float64{math.Inf(-1), -2.5, -1.5, -.5, -1e-12, math.Copysign(0, -1), 0, 1e-12, .5, 1.5, 2.5, math.Inf(1), math.NaN()}
	for _, tc := range []struct {
		name      string
		op        func(compute.Function, compute.Value) (compute.Value, error)
		reference func(float64) float64
	}{
		{"expm1", func(f compute.Function, v compute.Value) (compute.Value, error) { return f.Expm1(v) }, math.Expm1},
		{"log1p", func(f compute.Function, v compute.Value) (compute.Value, error) { return f.Log1p(v) }, math.Log1p},
		{"ceil", func(f compute.Function, v compute.Value) (compute.Value, error) { return f.Ceil(v) }, math.Ceil},
		{"floor", func(f compute.Function, v compute.Value) (compute.Value, error) { return f.Floor(v) }, math.Floor},
		{"round", func(f compute.Function, v compute.Value) (compute.Value, error) { return f.Round(v) }, math.RoundToEven},
		{"tanh", func(f compute.Function, v compute.Value) (compute.Value, error) { return f.Tanh(v) }, math.Tanh},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := testutil.Exec1(b, []any{input}, func(f compute.Function, v []compute.Value) (compute.Value, error) { return tc.op(f, v[0]) })
			if err != nil {
				t.Fatal(err)
			}
			got := result.([]float64)
			for i, x := range input {
				want := tc.reference(x)
				g := got[i]
				if math.IsNaN(want) {
					if !math.IsNaN(g) {
						t.Fatalf("%v got%v wantNaN", x, g)
					}
					continue
				}
				if math.IsInf(want, 0) {
					if g != want {
						t.Fatalf("%v got%v want%v", x, g, want)
					}
					continue
				}
				if math.IsNaN(g) || math.Abs(g-want) > 1e-12*math.Max(math.Abs(want), 1e-15) {
					t.Fatalf("%v got%.17g want%.17g", x, g, want)
				}
				if want == 0 && math.Signbit(want) != math.Signbit(g) {
					t.Fatalf("%v zero sign differs", x)
				}
			}
		})
	}
	results, err := testutil.Exec(b, []any{input}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
		a, err := f.IsFinite(v[0])
		if err != nil {
			return nil, err
		}
		b, err := f.IsNaN(v[0])
		return []compute.Value{a, b}, err
	})
	if err != nil {
		t.Fatal(err)
	}
	finite, nan := make([]bool, len(input)), make([]bool, len(input))
	for i, x := range input {
		finite[i] = !math.IsNaN(x) && !math.IsInf(x, 0)
		nan[i] = math.IsNaN(x)
	}
	if !reflect.DeepEqual(results, []any{finite, nan}) {
		t.Fatalf("classifications %v", results)
	}
	for _, tc := range []struct{ input, want any }{
		{[]int64{math.MinInt64, -1, 0, 1, math.MaxInt64}, []int64{-1, -1, 0, 1, 1}},
		{[]uint64{0, 1, 1 << 63, ^uint64(0)}, []uint64{0, 1, 1, 1}},
	} {
		got, err := testutil.Exec1(b, []any{tc.input}, func(f compute.Function, v []compute.Value) (compute.Value, error) { return f.Sign(v[0]) })
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("sign %v want%v", got, tc.want)
		}
	}
}
