package advise

import (
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// rankCapture is one rank of a tensor-parallel decode: each step a GEMM,
// then an all-reduce that ends at the same moment on every rank. The rank
// whose GEMM takes extra starts its all-reduce late. clock offsets the
// rank's timestamps, as an unaligned host clock would.
func rankCapture(clock, extra uint64) *cudacapture.Capture {
	var samples []*profile.Sample
	for step := uint64(0); step < 5; step++ {
		t := clock + 1_000_000 + step*1_000_000
		gemmEnd := t + 300_000 + extra
		samples = append(samples,
			microKernelSample("gemm_bf16_tn", 7, t, gemmEnd, int(2*step+1)),
			microKernelSample("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 7, gemmEnd+10_000, t+500_000, int(2*step+2)))
	}
	return &cudacapture.Capture{Profile: &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
		Sample:     samples,
	}}
}

func TestAnalyzeRanks(t *testing.T) {
	tests := []struct {
		name      string
		align     Alignment
		clocks    []uint64
		straggler string
		evidence  EvidenceLevel
	}{
		{"one clock", AlignClock, []uint64{0, 0, 0, 0}, "r2", EvidenceObserved},
		{"skewed clocks aligned on collectives", AlignCollective, []uint64{0, 7_000_000_000, 3_000, 900_000_000}, "r2", EvidenceSuspected},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ranks []Rank
			for i, clock := range tt.clocks {
				extra := uint64(0)
				if i == 2 {
					extra = 50_000
				}
				ranks = append(ranks, Rank{Name: "r" + string(rune('0'+i)), Capture: rankCapture(clock, extra)})
			}
			report, err := AnalyzeRanks(ranks, tt.align)
			if err != nil {
				t.Fatal(err)
			}
			tl := report.Ranks
			if tl.Collectives != 5 || tl.Straggler != tt.straggler {
				t.Fatalf("collectives %d straggler %q; want 5, %q", tl.Collectives, tl.Straggler, tt.straggler)
			}
			if tl.MeanSkewNanos != 50_000 || tl.MaxSkewNanos != 50_000 {
				t.Errorf("skew mean %d max %d, want 50000", tl.MeanSkewNanos, tl.MaxSkewNanos)
			}
			for i, r := range tl.Ranks {
				wantWait, wantLast := int64(5*50_000), 0
				if i == 2 {
					wantWait, wantLast = 0, 5
				}
				if r.WaitNanos != wantWait || r.Last != wantLast {
					t.Errorf("%s waited %d last %d; want %d, %d", r.Name, r.WaitNanos, r.Last, wantWait, wantLast)
				}
			}
			var skew *Item
			exposed := 0
			for i := range report.Items {
				switch report.Items[i].rule {
				case "rank-skew":
					skew = &report.Items[i]
				case "exposed-comm":
					exposed++
				}
			}
			if skew == nil || skew.EvidenceLevel != tt.evidence || !strings.Contains(skew.Finding, "Rank r2 entered 5 of them last") {
				t.Errorf("skew finding = %+v", skew)
			}
			if exposed != 4 {
				t.Errorf("%d per-rank exposed findings, want 4", exposed)
			}
			md := report.Markdown()
			if !strings.Contains(md, "## Ranks") || !strings.Contains(md, "straggler r2") {
				t.Errorf("markdown lacks the rank table:\n%s", md)
			}
		})
	}
}

func TestAnalyzeRanksErrors(t *testing.T) {
	one := []Rank{{Name: "r0", Capture: rankCapture(0, 0)}}
	if _, err := AnalyzeRanks(one, AlignClock); err == nil {
		t.Error("one rank: no error")
	}
	noComm := &cudacapture.Capture{Profile: &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{microKernelSample("gemm", 1, 10, 20, 1)},
	}}
	if _, err := AnalyzeRanks([]Rank{{Name: "a", Capture: noComm}, {Name: "b", Capture: noComm}}, AlignCollective); err == nil {
		t.Error("collective alignment without collectives: no error")
	}
	if _, err := AnalyzeRanks([]Rank{{Name: "a", Capture: noComm}, {Name: "b", Capture: noComm}}, "nope"); err == nil {
		t.Error("unknown alignment: no error")
	}
}
