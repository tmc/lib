package cudaautograd_test

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
)

// This exercises the existing optimizer consumer with a small rectangular
// MLP's independent float64 finite-difference gradients. Native gradients
// enter the same consumer in cudalayer.TestMLPBackward_TrainingStepMatchesCPU.
func TestMLPTraining_AllLayersMatchReferenceUpdate(t *testing.T) {
	const in, hidden, out, samples = 5, 7, 3, 7
	w0, w1 := make([]float64, in*hidden), make([]float64, hidden*out)
	for i := range w0 {
		w0[i] = 0.03 + float64(i%5)/100
	}
	for i := range w1 {
		w1[i] = 0.02 + float64(i%7)/100
	}
	loss := func() float64 {
		sum := 0.0
		for s := 0; s < samples; s++ {
			h := make([]float64, hidden)
			for j := range h {
				for i := 0; i < in; i++ {
					h[j] += float64(s+i+1) / 10 * w0[i*hidden+j]
				}
				h[j] = math.Max(0, h[j])
			}
			for j := 0; j < out; j++ {
				y := 0.0
				for i := range h {
					y += h[i] * w1[i*out+j]
				}
				sum += y * float64(s+j+1) / 20 / samples
			}
		}
		return sum
	}
	before := loss()
	initial := map[string][]float32{}
	params := map[string][]float32{}
	grads := map[string][]float32{}
	want := map[string][]float32{}
	const lr = float32(0.001)
	for _, layer := range []struct {
		name    string
		weights []float64
	}{{"layer0", w0}, {"layer1", w1}} {
		p, g, r := make([]float32, len(layer.weights)), make([]float32, len(layer.weights)), make([]float32, len(layer.weights))
		for i, v := range layer.weights {
			const h = 1e-5
			layer.weights[i] = v + h
			high := loss()
			layer.weights[i] = v - h
			low := loss()
			layer.weights[i] = v
			p[i] = float32(v)
			g[i] = float32((high - low) / (2 * h))
			r[i] = p[i] - lr*g[i]
		}
		params[layer.name] = p
		initial[layer.name] = append([]float32(nil), p...)
		grads[layer.name] = g
		want[layer.name] = r
	}
	opt, e := cudaautograd.NewSGDOptimizer(lr, 0)
	if e != nil {
		t.Fatal(e)
	}
	if e := opt.Step(params, grads); e != nil {
		t.Fatal(e)
	}
	for _, layer := range []struct {
		name    string
		weights []float64
	}{{"layer0", w0}, {"layer1", w1}} {
		for i, v := range params[layer.name] {
			if v != want[layer.name][i] {
				t.Fatalf("%s[%d] update=%g want%g", layer.name, i, v, want[layer.name][i])
			}
			if v == float32(layer.weights[i]) {
				t.Fatalf("%s[%d] was not updated", layer.name, i)
			}
			layer.weights[i] = float64(v)
		}
	}
	if after := loss(); !(after < before) {
		t.Fatalf("declared small training fixture loss %g -> %g", before, after)
	}
	// A real optimizer invocation given only the final layer's gradient must
	// fail the all-layer update oracle, even if its scalar loss also falls.
	partial, e := cudaautograd.NewSGDOptimizer(lr, 0)
	if e != nil {
		t.Fatal(e)
	}
	if e := partial.Step(initial, map[string][]float32{"layer1": grads["layer1"]}); e != nil {
		t.Fatal(e)
	}
	matched := true
	for name, values := range initial {
		for i, value := range values {
			if value != want[name][i] {
				matched = false
			}
		}
	}
	if matched {
		t.Fatal("one-layer-only optimizer control survived all-parameter oracle")
	}
}
