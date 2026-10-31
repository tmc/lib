package nvl72

import (
	"fmt"
	"math"
	"testing"
)

// dsv3 is a DeepSeek-V3-shaped layer on 72 ranks: 256 routed experts plus
// 32 redundant slots (4 per rank), top-8, hidden 7168, expert width 2048.
func dsv3(tokens int) EPAllToAll {
	return EPAllToAll{
		Gen: GenGB300, Ranks: 72, TokensPerRank: tokens, Hidden: 7168, TopK: 8,
		Experts: 288, ExpertInter: 2048, DispatchBytes: 1 + 1.0/128, RankDedup: true,
	}
}

func TestEstimateEPAllToAll(t *testing.T) {
	tests := []struct {
		name         string
		c            EPAllToAll
		copies       float64
		remoteCopies float64
	}{
		// One rank: everything is local and no bytes move.
		{"one rank", EPAllToAll{Gen: GenGB300, Ranks: 1, TokensPerRank: 8, Hidden: 64, TopK: 2, Experts: 4, ExpertInter: 32}, 2, 0},
		// Without dedup every route is a copy; (R-1)/R of them are remote.
		{"routes", EPAllToAll{Gen: GenGB300, Ranks: 4, TokensPerRank: 8, Hidden: 64, TopK: 2, Experts: 8, ExpertInter: 32}, 2, 1.5},
		// Dedup with one expert per rank: K distinct experts, K distinct ranks.
		{"dedup one per rank", EPAllToAll{Gen: GenGB300, Ranks: 8, TokensPerRank: 8, Hidden: 64, TopK: 2, Experts: 8, ExpertInter: 32, RankDedup: true}, 2, 2 * 7.0 / 8},
		// Dedup with every expert on two ranks, top-2 of 4: a rank is missed
		// with probability C(2,2)/C(4,2) = 1/6.
		{"dedup two per rank", EPAllToAll{Gen: GenGB300, Ranks: 2, TokensPerRank: 8, Hidden: 64, TopK: 2, Experts: 4, ExpertInter: 32, RankDedup: true}, 2 * 5.0 / 6, 5.0 / 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, err := EstimateEPAllToAll(tt.c)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(e.Copies-tt.copies) > 1e-9 || math.Abs(e.RemoteCopies-tt.remoteCopies) > 1e-9 {
				t.Errorf("copies %g remote %g, want %g %g", e.Copies, e.RemoteCopies, tt.copies, tt.remoteCopies)
			}
			wantD := float64(tt.c.TokensPerRank) * tt.remoteCopies * (float64(tt.c.Hidden)*2 + 8*float64(tt.c.TopK))
			if math.Abs(e.DispatchBytes-wantD) > 1e-6 {
				t.Errorf("dispatch bytes %g, want %g", e.DispatchBytes, wantD)
			}
			if e.DispatchNs < 3000 || e.CombineNs < 3000 || e.SequentialNs <= e.ExpertNs {
				t.Errorf("times %+v", e)
			}
		})
	}
}

func TestEstimateEPAllToAllErrors(t *testing.T) {
	for _, c := range []EPAllToAll{
		{Gen: "H100", Ranks: 8, TokensPerRank: 1, Hidden: 1, TopK: 1, Experts: 8, ExpertInter: 1},
		{Gen: GenGB300, Ranks: 73, TokensPerRank: 1, Hidden: 1, TopK: 1, Experts: 73, ExpertInter: 1},
		{Gen: GenGB300, Ranks: 72, TokensPerRank: 1, Hidden: 1, TopK: 8, Experts: 256, ExpertInter: 1},
	} {
		if _, err := EstimateEPAllToAll(c); err == nil {
			t.Errorf("EstimateEPAllToAll(%+v) succeeded", c)
		}
	}
}

// TestEPAllToAllDedupSaves checks rank dedup never sends more, and sends
// strictly less once a rank holds more than one expert.
func TestEPAllToAllDedupSaves(t *testing.T) {
	c := dsv3(128)
	with, err := EstimateEPAllToAll(c)
	if err != nil {
		t.Fatal(err)
	}
	c.RankDedup = false
	without, err := EstimateEPAllToAll(c)
	if err != nil {
		t.Fatal(err)
	}
	if with.DispatchBytes >= without.DispatchBytes {
		t.Errorf("dedup dispatch %g >= %g", with.DispatchBytes, without.DispatchBytes)
	}
}

// TestEPAllToAllSweep logs the modelled GB300 NVL72 dispatch and combine
// for a DeepSeek-V3-shaped layer from decode to prefill batch sizes.
func TestEPAllToAllSweep(t *testing.T) {
	t.Log("tokens/rank  copies  dispatchMB  combineMB  dispatch_us  combine_us  expert_us  seq_us  tbo_us  comm%")
	prev := 0.0
	for _, s := range []int{1, 8, 32, 64, 128, 256, 512, 1024, 4096, 8192} {
		e, err := EstimateEPAllToAll(dsv3(s))
		if err != nil {
			t.Fatal(err)
		}
		if e.SequentialNs <= prev {
			t.Errorf("tokens %d: sequential %g not increasing", s, e.SequentialNs)
		}
		prev = e.SequentialNs
		t.Logf("%11d  %6.2f  %10.2f  %9.2f  %11.1f  %10.1f  %9.1f  %6.1f  %6.1f  %5.1f",
			s, e.Copies, e.DispatchBytes/1e6, e.CombineBytes/1e6, e.DispatchNs/1e3, e.CombineNs/1e3,
			e.ExpertNs/1e3, e.SequentialNs/1e3, e.OverlappedNs/1e3, 100*e.CommFraction)
	}
}

func ExampleEstimateEPAllToAll() {
	e, err := EstimateEPAllToAll(EPAllToAll{
		Gen: GenGB300, Ranks: 72, TokensPerRank: 128, Hidden: 7168, TopK: 8,
		Experts: 288, ExpertInter: 2048, DispatchBytes: 1 + 1.0/128, RankDedup: true,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%.2f copies/token, dispatch %.1f MB, combine %.1f MB per GPU\n",
		e.Copies, e.DispatchBytes/1e6, e.CombineBytes/1e6)
	// Output: 7.71 copies/token, dispatch 7.1 MB, combine 14.0 MB per GPU
}
