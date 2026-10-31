package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func TestEgraphTemporaryArena(t *testing.T) {
	for _, mega := range []bool{false, true} {
		opt := egraphOptions{Profile: mustProfile(t, "GB10"), Attn: true, PrefillRows: -1, Mega: mega}
		plans, err := buildEgraphPlans(&llama3_8B, false, opt)
		if err != nil {
			t.Fatal(err)
		}
		eg := egraphDecoder{plans: plans}
		arena, err := hopper.PlanArena(eg.planList(), 0, !mega)
		if err != nil {
			t.Fatal(err)
		}
		if arena.Bytes == 0 || arena.Bytes >= arena.OriginalBytes {
			t.Fatalf("mega=%v: arena %d bytes, separate %d bytes", mega, arena.Bytes, arena.OriginalBytes)
		}
		seen := make(map[string]bool)
		for _, offsets := range arena.Offsets {
			for name := range offsets {
				if seen[name] {
					t.Fatalf("duplicate temporary %q across plans", name)
				}
				seen[name] = true
			}
		}
		if _, err := hopper.PlanArena(eg.planList(), arena.Bytes-1, !mega); err == nil {
			t.Fatal("accepted arena exceeding budget")
		}
		t.Logf("mega=%v: arena=%d separate=%d saved=%d", mega, arena.Bytes, arena.OriginalBytes, arena.OriginalBytes-arena.Bytes)
	}
}
