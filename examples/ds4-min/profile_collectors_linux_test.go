//go:build linux

package main

import (
	"math"
	"os"
	"testing"
)

// This gate executes real layer producers; host tests cannot substitute for it.
func TestCUDACollectorEndpointInvariant(t *testing.T) {
	if os.Getenv("DS4_CUDA_COLLECTOR_GATE") == "" {
		t.Skip("DS4_CUDA_COLLECTOR_GATE is not set")
	}
	if os.Getenv("DS4_CUDA_COLLECTOR_GATE") != "1" || os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("collector gate requires explicit admission and gpu-lock")
	}
	model, input := os.Getenv("DS4_MODEL"), os.Getenv("DS4_CUDA_PREFILL_TOKENS")
	if model == "" || input == "" {
		t.Fatal("model and token fixture required")
	}
	data, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) < 8 {
		t.Fatal("fixture needs eight tokens")
	}
	for _, batch := range []bool{false, true} {
		t.Run(map[bool]string{false: "scalar", true: "batch"}[batch], func(t *testing.T) {
			var baseline []float32
			for _, enabled := range []bool{false, true} {
				r, err := newCUDARuntime(model, 128)
				if err != nil {
					t.Fatal(err)
				}
				var p *profileCollectors
				if enabled {
					p, err = newProfileCollectors(true, true, 1<<30)
					if err != nil {
						t.Fatal(err)
					}
					p.expertStart = 8
					r.collectors = p
				}
				if err := r.Reset(); err != nil {
					t.Fatal(err)
				}
				if batch {
					err = r.EvalBatch(tokens[:8], 0)
				} else {
					for i, token := range tokens[:8] {
						if err = r.Eval(token, i); err != nil {
							break
						}
					}
				}
				if err != nil {
					t.Fatal(err)
				}
				prefill := append([]float32(nil), r.Logits()...)
				if enabled && p.experts.Records() != 0 {
					t.Fatal("prefill leaked into decode profile")
				}
				if err := r.Eval(tokens[0], 8); err != nil {
					t.Fatal(err)
				}
				endpoint := append(prefill, r.Logits()...)
				if enabled {
					if p.experts.Records() != 43 || p.imatrix.Chunks() == 0 {
						t.Fatalf("missing producer records: expert=%d chunks=%d", p.experts.Records(), p.imatrix.Chunks())
					}
					if len(endpoint) != len(baseline) {
						t.Fatal("endpoint shape changed")
					}
					for i, v := range endpoint {
						if math.Float32bits(v) != math.Float32bits(baseline[i]) {
							t.Fatalf("off/on endpoint differs at %d", i)
						}
					}
				} else {
					baseline = endpoint
				}
				if err := r.Close(); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
