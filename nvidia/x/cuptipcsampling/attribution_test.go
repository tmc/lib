package cuptipcsampling_test

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func TestAttributePCSamples(t *testing.T) {
	records := []cuptiactivity.Record{
		{
			Kind:          cupti.ActivityKindMemcpy,
			Name:          "memcpy_h2d",
			Start:         100,
			End:           200,
			Complete:      true,
			CorrelationID: 1,
		},
		{
			Kind:          cupti.ActivityKindConcurrentKernel,
			Name:          "_Z12gemm_kernelPKfS0_Pf",
			Start:         1000,
			End:           5000,
			Complete:      true,
			CorrelationID: 2,
		},
	}

	metrics := []cuptipcsampling.SASSMetric{
		{
			PC:           0x100,
			KernelName:   "_Z12gemm_kernelPKfS0_Pf",
			TotalSamples: 200,
			Selected:     50,
			StallSamples: 150,
			Stalls: []cuptipcsampling.PipelineStall{
				{Name: "long_scoreboard", Samples: 100, Percent: 66.67},
				{Name: "stall_barrier", Samples: 50, Percent: 33.33},
			},
		},
		{
			PC:           0x108,
			KernelName:   "_Z12gemm_kernelPKfS0_Pf",
			TotalSamples: 100,
			Selected:     0,
			StallSamples: 100,
			Stalls: []cuptipcsampling.PipelineStall{
				{Name: "long_scoreboard", Samples: 100, Percent: 100.0},
			},
		},
	}

	attributed := cuptipcsampling.AttributePCSamples(records, metrics)
	if len(attributed) != 1 {
		t.Fatalf("len(attributed) = %d, want 1 (memcpy should be skipped)", len(attributed))
	}

	ak := attributed[0]
	if ak.KernelName != "_Z12gemm_kernelPKfS0_Pf" {
		t.Errorf("KernelName = %q", ak.KernelName)
	}
	if ak.DurationNs != 4000 {
		t.Errorf("DurationNs = %d, want 4000", ak.DurationNs)
	}
	if ak.TotalWarps != 300 {
		t.Errorf("TotalWarps = %d, want 300", ak.TotalWarps)
	}
	if ak.IssuedWarps != 50 {
		t.Errorf("IssuedWarps = %d, want 50", ak.IssuedWarps)
	}
	if ak.StallWarps != 250 {
		t.Errorf("StallWarps = %d, want 250", ak.StallWarps)
	}
	if ak.DominantStall != "long_scoreboard" {
		t.Errorf("DominantStall = %q, want long_scoreboard", ak.DominantStall)
	}
	if math.Abs(ak.DominantPct-80.0) > 0.01 { // 200 / 250 = 80%
		t.Errorf("DominantPct = %.2f, want 80.0", ak.DominantPct)
	}

	breakdown := ak.StallBreakdown()
	if math.Abs(breakdown["long_scoreboard"]-80.0) > 0.01 {
		t.Errorf("breakdown[long_scoreboard] = %.2f, want 80.0", breakdown["long_scoreboard"])
	}
	if math.Abs(breakdown["stall_barrier"]-20.0) > 0.01 {
		t.Errorf("breakdown[stall_barrier] = %.2f, want 20.0", breakdown["stall_barrier"])
	}
}
