package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func TestPrefillWeightStaging(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		p, err := buildEgraphPlans(&llama3_8B, false, egraphOptions{Profile: mustProfile(t, "GB10"), Attn: true, PrefillRows: -1, SMs: 48, StageWeights: enabled})
		if err != nil {
			t.Fatal(err)
		}
		seen := make(map[string]bool)
		for _, plan := range []*hopper.Plan{p.pf.narrowIn, p.pf.narrowPost} {
			for _, k := range plan.Kernels {
				if k.MMA == nil {
					continue
				}
				var weight string
				for _, arg := range k.Args {
					switch arg {
					case "w_qkv", "w_o", "w_gu", "w_down":
						weight = arg
					}
				}
				if weight == "" || seen[weight] {
					t.Fatalf("unexpected projection arguments %v", k.Args)
				}
				seen[weight] = true
				want := enabled && weight != "w_o"
				if k.MMA.StageWeights != want {
					t.Errorf("%s staged=%v, want %v", weight, k.MMA.StageWeights, want)
				}
				shared := 34816
				if want {
					shared = 36864
				}
				if got := k.MMA.SharedBytes(); got != shared {
					t.Errorf("%s shared=%d, want %d", weight, got, shared)
				}
				if _, err := k.MMA.CUDA(); err != nil {
					t.Fatalf("%s: %v", weight, err)
				}
			}
		}
		if len(seen) != 4 {
			t.Fatalf("saw %d projections, want 4", len(seen))
		}
	}
}
