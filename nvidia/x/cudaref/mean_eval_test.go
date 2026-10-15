package cudaref_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func TestEvaluateMeanAcrossAxes(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		axis  int
		input []float32
		want  []float32
	}{
		{"axis 0", []int{2, 3}, 0, []float32{1, 3, 5, 3, 5, 7}, []float32{2, 4, 6}},
		{"axis 1", []int{2, 3}, 1, []float32{1, 3, 5, 3, 5, 7}, []float32{3, 5}},
		{"middle axis", []int{2, 2, 2}, 1, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{2, 3, 6, 7}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", tt.shape, cudair.Float32)
			mean := g.AddReduce("mean", x, tt.axis)
			if err := g.SetOutputs(mean); err != nil {
				t.Fatal(err)
			}
			snap, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}

			for name, evaluate := range map[string]func() ([]float32, error){
				"graph": func() ([]float32, error) {
					out, err := cudaref.Evaluate(g, map[string][]float32{"x": tt.input})
					return out[mean.Name], err
				},
				"snapshot": func() ([]float32, error) {
					out, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"x": tt.input})
					return out[mean.Name], err
				},
			} {
				got, err := evaluate()
				if err != nil {
					t.Fatalf("%s evaluation: %v", name, err)
				}
				if !reflect.DeepEqual(got, tt.want) {
					t.Errorf("%s mean = %v, want %v", name, got, tt.want)
				}
			}
		})
	}
}

func TestEvaluateMeanIEEEValues(t *testing.T) {
	nan := float32(math.NaN())
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	negZero := float32(math.Copysign(0, -1))
	tests := []struct {
		name  string
		input []float32
		check func(float32) bool
	}{
		{"NaN", []float32{1, nan}, func(v float32) bool { return math.IsNaN(float64(v)) }},
		{"positive infinity", []float32{posInf, 1}, func(v float32) bool { return math.IsInf(float64(v), 1) }},
		{"opposed infinities", []float32{posInf, negInf}, func(v float32) bool { return math.IsNaN(float64(v)) }},
		{"zero sign follows float64 accumulator", []float32{negZero, negZero}, func(v float32) bool { return v == 0 && !math.Signbit(float64(v)) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{len(tt.input)}, cudair.Float32)
			mean := g.AddReduce("mean", x, 0)
			if err := g.SetOutputs(mean); err != nil {
				t.Fatal(err)
			}
			snap, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			for name, evaluate := range map[string]func() ([]float32, error){
				"graph": func() ([]float32, error) {
					out, err := cudaref.Evaluate(g, map[string][]float32{"x": tt.input})
					return out[mean.Name], err
				},
				"snapshot": func() ([]float32, error) {
					out, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"x": tt.input})
					return out[mean.Name], err
				},
			} {
				got, err := evaluate()
				if err != nil {
					t.Fatalf("%s evaluation: %v", name, err)
				}
				if value := got[0]; !tt.check(value) {
					t.Errorf("%s mean = %v, signbit=%v", name, value, math.Signbit(float64(value)))
				}
			}
		})
	}
}

func TestEvaluateMeanEmptyPolicies(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		axis  int
		count int
		check func(float32) bool
	}{
		{"empty set with output cells", []int{2, 0}, 1, 2, func(v float32) bool { return math.IsNaN(float64(v)) }},
		{"empty output", []int{0, 2}, 1, 0, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", tt.shape, cudair.Float32)
			mean := g.AddReduce("mean", x, tt.axis)
			if err := g.SetOutputs(mean); err != nil {
				t.Fatal(err)
			}
			snap, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			for name, evaluate := range map[string]func() ([]float32, error){
				"graph": func() ([]float32, error) {
					out, err := cudaref.Evaluate(g, map[string][]float32{"x": {}})
					return out[mean.Name], err
				},
				"snapshot": func() ([]float32, error) {
					out, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"x": {}})
					return out[mean.Name], err
				},
			} {
				got, err := evaluate()
				if err != nil {
					t.Fatalf("%s evaluation: %v", name, err)
				}
				if len(got) != tt.count {
					t.Fatalf("%s mean output length = %d, want %d", name, len(got), tt.count)
				}
				if tt.check != nil {
					for i, value := range got {
						if !tt.check(value) {
							t.Errorf("%s mean[%d] = %v, want NaN", name, i, value)
						}
					}
				}
			}
		})
	}
}

func TestEvaluateMeanRejectsMalformedGraphAndUsesSnapshotCopy(t *testing.T) {
	t.Run("snapshot rejects malformed dimensions", func(t *testing.T) {
		g := cudair.NewGraph()
		x := g.AddInput("x", []int{2, 2}, cudair.Float32)
		g.AddReduce("mean", x, 1)
		x.Shape[1] = -1
		if _, err := g.Snapshot(); err == nil {
			t.Fatal("Snapshot() accepted a negative input extent")
		}
	})

	t.Run("detached snapshot", func(t *testing.T) {
		g := cudair.NewGraph()
		x := g.AddInput("x", []int{2, 2}, cudair.Float32)
		mean := g.AddReduce("mean", x, 1)
		if err := g.SetOutputs(mean); err != nil {
			t.Fatal(err)
		}
		snap, err := g.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		x.Shape[1] = 7
		input := []float32{1, 3, 5, 7}
		all, err := cudaref.EvaluateSnapshotAll(snap, map[string][]float32{"x": input})
		if err != nil {
			t.Fatal(err)
		}
		input[0] = 100
		if want := []float32{2, 6}; !reflect.DeepEqual(all[mean.ID], want) {
			t.Fatalf("snapshot mean = %v, want %v", all[mean.ID], want)
		}
		if want := []float32{1, 3, 5, 7}; !reflect.DeepEqual(all[x.ID], want) {
			t.Fatalf("snapshot input copy = %v, want %v", all[x.ID], want)
		}
	})
}
