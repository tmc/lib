package main

import (
	"math"
	"reflect"
	"testing"
)

func TestFFNReferenceOracle(t *testing.T) {
	for _, shape := range [][2]int{{1, 1}, {3, 5}, {5, 3}, {7, 11}, {64, 128}} {
		h, n := shape[0], shape[1]
		p, err := newFFNPlan(h, n, 1e-5, 4*(2*h+3*n))
		if err != nil {
			t.Fatal(err)
		}
		values := func(size, salt int) []float32 {
			v := make([]float32, size)
			for i := range v {
				v[i] = float32((i*7+salt)%23-11) / 31
			}
			return v
		}
		b := ffnBuffers{x: values(h, 1), norm: values(h, 3), gate: values(h*n, 5), up: values(h*n, 9), down: values(h*n, 13)}
		before := ffnBuffers{x: append([]float32(nil), b.x...), norm: append([]float32(nil), b.norm...), gate: append([]float32(nil), b.gate...), up: append([]float32(nil), b.up...), down: append([]float32(nil), b.down...)}
		// Compose the existing CPU forward primitives independently of the plan.
		x := rmsNorm(b.x, b.norm, p.epsilon)
		gate := matmulVec(x, b.gate, h, n)
		up := matmulVec(x, b.up, h, n)
		for i := range gate {
			gate[i] = gate[i] / (1 + float32(math.Exp(float64(-gate[i])))) * up[i]
		}
		want := matmulVec(gate, b.down, n, h)
		for i := range want {
			want[i] += b.x[i]
		}
		got, err := p.reference(b)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			tolerance := 1e-6 * (1 + math.Abs(float64(want[i])))
			if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > tolerance {
				t.Fatalf("shape %v output %d: got %g want %g", shape, i, got[i], want[i])
			}
		}
		if !reflect.DeepEqual(b, before) {
			t.Fatal("reference mutated input")
		}
		again, err := p.reference(b)
		if err != nil {
			t.Fatal(err)
		}
		first := append([]float32(nil), got...)
		second := append([]float32(nil), again...)
		got[0] += 99
		if !reflect.DeepEqual(again, second) || !reflect.DeepEqual(b, before) {
			t.Fatal("first output aliases second output or input")
		}
		mutatedFirst := append([]float32(nil), got...)
		again[0] -= 77
		if !reflect.DeepEqual(got, mutatedFirst) || !reflect.DeepEqual(b, before) {
			t.Fatal("second output aliases first output or input")
		}
		if reflect.DeepEqual(got, first) || reflect.DeepEqual(again, second) {
			t.Fatal("ownership mutations did not change their output")
		}
	}
}
