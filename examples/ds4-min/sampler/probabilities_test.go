package sampler

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

func checkProbabilityDistribution(t *testing.T, probabilities []float32, extent int) {
	t.Helper()
	if len(probabilities) != extent {
		t.Fatalf("probability extent %d, want %d", len(probabilities), extent)
	}
	var sum float64
	supported := 0
	for i, p := range probabilities {
		if !isFinite(p) || p < 0 || p > 1.000001 {
			t.Fatalf("invalid probability at token %d: %g", i, p)
		}
		sum += float64(p)
		if p > 0 {
			supported++
		}
	}
	if supported == 0 || math.Abs(sum-1) > 2e-5 {
		t.Fatalf("probability normalization %g, support %d", sum, supported)
	}
}

func TestProbabilitiesValues(t *testing.T) {
	logits := []float32{0, float32(math.Log(2)), float32(math.Log(4))}
	for _, tc := range []struct {
		name string
		p    Params
		want []float32
	}{
		{"full", Params{Temperature: 1, TopP: 1}, []float32{1.0 / 7, 2.0 / 7, 4.0 / 7}},
		{"top k", Params{Temperature: 1, TopK: 2, TopP: 1}, []float32{0, 1.0 / 3, 2.0 / 3}},
		{"top p", Params{Temperature: 1, TopP: 0.5}, []float32{0, 0, 1}},
		{"min p", Params{Temperature: 1, TopP: 1, MinP: 0.4}, []float32{0, 1.0 / 3, 2.0 / 3}},
		{"ranked min p", Params{Temperature: 1, TopP: 0.99, MinP: 0.4}, []float32{0, 1.0 / 3, 2.0 / 3}},
		{"greedy", Params{}, []float32{0, 0, 1}},
		{"out of range", Params{Temperature: 1, TopP: 1, MinP: 2}, []float32{0, 0, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := New(3)
			got, err := s.Probabilities(logits, tc.p)
			if err != nil {
				t.Fatal(err)
			}
			checkProbabilityDistribution(t, got, len(logits))
			for i, w := range tc.want {
				if !isFinite(got[i]) || math.Abs(float64(got[i]-w)) > 2e-7 {
					t.Fatalf("%v want%v", got, tc.want)
				}
			}
			got[0] = 99
			again, err := s.Probabilities(logits, tc.p)
			if err != nil || again[0] == 99 {
				t.Fatal("probabilities alias scratch")
			}
			checkProbabilityDistribution(t, again, len(logits))
		})
	}
	got, err := New(2).Probabilities([]float32{1, 1}, Params{Temperature: 1, TopK: 1})
	if err != nil || !reflect.DeepEqual(got, []float32{1, 0}) {
		t.Fatalf("tie %v %v", got, err)
	}
}

func TestProbabilitiesMatchSampling(t *testing.T) {
	for _, p := range []Params{{Temperature: 0.7, TopP: 1, MinP: 0.01}, {Temperature: 1, TopK: 2, TopP: 0.99}, {Temperature: 1, TopP: 0.8, MinP: 0.1}} {
		logits := []float32{0, 1, 2, 3}
		s := New(4)
		prob, err := s.Probabilities(logits, p)
		if err != nil {
			t.Fatal(err)
		}
		checkProbabilityDistribution(t, prob, len(logits))
		counts := make([]int, 4)
		rng := RNG{State: 71}
		const trials = 100000
		for i := 0; i < trials; i++ {
			counts[s.Sample(logits, p, &rng)]++
		}
		for i, q := range prob {
			if math.Abs(float64(counts[i])/trials-float64(q)) > 0.006 {
				t.Fatalf("params%+v token%d observed%g probability%g", p, i, float64(counts[i])/trials, q)
			}
		}
	}
}

func TestProbabilitiesRejectNonfinite(t *testing.T) {
	s := New(1)
	if _, err := s.Probabilities(nil, Default); err == nil {
		t.Fatal("empty logits admitted")
	}
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, err := s.Probabilities([]float32{v}, Default); err == nil {
			t.Fatal("nonfinite logits admitted")
		}
		p := Default
		p.Temperature = v
		if _, err := s.Probabilities([]float32{0}, p); err == nil {
			t.Fatal("nonfinite params admitted")
		}
	}
}

func ExampleSampler_Probabilities() {
	p, _ := New(2).Probabilities([]float32{0, 0}, Default)
	fmt.Println(p)
	// Output: [0.5 0.5]
}
