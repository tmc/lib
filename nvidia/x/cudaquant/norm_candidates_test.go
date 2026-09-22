package cudaquant_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaquant"
)

func TestNormQuantCandidates(t *testing.T) {
	for _, weighted := range []bool{false, true} {
		for _, keep := range []bool{false, true} {
			candidates, err := cudaquant.RMSNormQ8_1Candidates(2, 64, weighted, keep)
			if err != nil {
				t.Fatal(err)
			}
			want := 2
			if keep {
				want = 1
			}
			if len(candidates) != want {
				t.Fatalf("secondary consumer=%v: %d candidates", keep, len(candidates))
			}
			for i, c := range candidates {
				for name, size := range map[string]int{"input": 512, "output": 144} {
					got, err := c.Buffers[name].FootprintBytes()
					if err != nil || got != size {
						t.Fatalf("%s: footprint=%d err=%v", name, got, err)
					}
				}
				if _, ok := c.Buffers["weight"]; ok != weighted {
					t.Fatal("weighted/unweighted layouts conflated")
				}
				if i == 0 && (c.WorkspaceBytes != 512 || len(c.Launches) != 2 || c.Launches[1].Input != "norm") {
					t.Fatalf("unfused pipeline: %+v", c)
				}
				if i == 1 && (c.WorkspaceBytes != 0 || len(c.Launches) != 1) {
					t.Fatalf("fused pipeline: %+v", c)
				}
				for _, launch := range c.Launches {
					if launch.Grid != [3]uint32{2, 1, 1} || launch.Block != [3]uint32{32, 1, 1} {
						t.Fatalf("invalid launch: %+v", launch)
					}
				}
			}
			delete(candidates[0].Buffers, "output")
			if len(candidates) == 2 {
				if _, ok := candidates[1].Buffers["output"]; !ok {
					t.Fatal("candidate maps alias")
				}
			}
		}
	}
	for _, shape := range [][2]int{{0, 32}, {1, 0}, {1, 33}, {-1, 32}, {math.MaxInt, 32}, {math.MaxInt32, math.MaxInt32 - 31}} {
		if _, err := cudaquant.RMSNormQ8_1Candidates(shape[0], shape[1], false, false); err == nil {
			t.Fatalf("accepted invalid or overflowing shape %v", shape)
		}
	}
}

func Example_normQuantCandidates() {
	candidates, err := cudaquant.RMSNormQ8_1Candidates(2, 64, true, true)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, candidate := range candidates {
		fmt.Println(candidate.Name, len(candidate.Launches), candidate.WorkspaceBytes)
	}
	// Output: norm_quant_unfused 2 512
}
