package qualify

import (
	"math"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

func TestClassifyBottlenecks(t *testing.T) {
	obs := KernelObservation{
		SharedBankConflict: 0.25,
		WGMMASerialized:    true,
		SpillStoreBytes:    8,
		AchievedOccupancy:  0.12,
		Registers:          192,
		GridBlocks:         150,
		DeviceSMs:          132,
		DurationNs:         5000,
		LaunchGapNs:        1500,
	}
	got := ClassifyBottlenecks(obs)
	want := []BottleneckKind{
		BottleneckBankConflict,
		BottleneckWGMMASerial,
		BottleneckRegisterSpill,
		BottleneckLowOccupancy,
		BottleneckWaveTail,
		BottleneckLaunchOverhead,
	}
	if !slices.Equal(got, want) {
		t.Fatalf("ClassifyBottlenecks = %v, want %v", got, want)
	}
}

func TestObservationFromPTXAS(t *testing.T) {
	fn := ptxasinfo.Function{
		Name:        "_Z12tile_gemm_k0iPKtS0_Pf",
		SpillStores: 24,
		SpillLoads:  32,
		Registers:   184,
		Diagnostics: []ptxasinfo.Diagnostic{
			{Code: "C7519", Text: "wgmma.mma_async instructions are serialized"},
		},
	}
	obs := ObservationFromPTXAS(fn, 160, 132)
	if !obs.WGMMASerialized || obs.SpillStoreBytes != 24 || obs.Registers != 184 {
		t.Fatalf("unexpected observation: %+v", obs)
	}
}

func TestRefineKnobs(t *testing.T) {
	knobs := ScheduleKnobs{
		Name:   "tile_k0",
		Op:     "tile_gemm",
		BM:     128,
		BN:     64,
		Stages: 2,
		WGMMA:  true,
	}
	refined, reasons := RefineKnobs(knobs, KernelObservation{
		Name:               "tile_k0",
		SharedBankConflict: 0.22,
		WGMMASerialized:    true,
	})
	if !refined.Swizzle128B || !refined.WarpSpecialized || len(reasons) < 2 {
		t.Fatalf("unexpected refined knobs: %+v (reasons: %v)", refined, reasons)
	}
}

func TestPGOObservationFromCountersAndPTXAS(t *testing.T) {
	f64 := func(v float64) *float64 { return &v }
	b := func(v bool) *bool { return &v }

	fn := ptxasinfo.Function{
		Name:      "gemm_k0",
		Registers: 168,
	}

	tests := []struct {
		name     string
		cm       CounterMetrics
		wantBank float64
		wantOcc  float64
		wantKV   bool
		wantBns  []BottleneckKind
	}{
		{
			name:     "unmeasured_nil_counters",
			cm:       CounterMetrics{DurationNs: 10000, LaunchGapNs: 500},
			wantBank: 0,
			wantOcc:  0,
			wantKV:   false,
			wantBns:  nil,
		},
		{
			name: "measured_zero_vs_positive",
			cm: CounterMetrics{
				DurationNs:         10000,
				LaunchGapNs:        2000,
				SharedBankConflict: f64(0.0),
				AchievedOccupancy:  f64(0.18),
				MemoryBoundKV:      b(true),
			},
			wantBank: 0,
			wantOcc:  0.18,
			wantKV:   true,
			wantBns: []BottleneckKind{
				BottleneckLowOccupancy,
				BottleneckMemoryBoundKV,
				BottleneckLaunchOverhead,
			},
		},
		{
			name: "clamping_and_invalid_counter_values",
			cm: CounterMetrics{
				SharedBankConflict: f64(1.4),
				AchievedOccupancy:  f64(math.NaN()),
				MemoryBoundKV:      b(false),
			},
			wantBank: 1.0,
			wantOcc:  0,
			wantKV:   false,
			wantBns:  []BottleneckKind{BottleneckBankConflict},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obs := ObservationFromCountersAndPTXAS(fn, tt.cm, 264, 132)
			if obs.SharedBankConflict != tt.wantBank || obs.AchievedOccupancy != tt.wantOcc || obs.MemoryBoundKV != tt.wantKV {
				t.Fatalf("got obs=%+v, want bank=%v occ=%v kv=%v", obs, tt.wantBank, tt.wantOcc, tt.wantKV)
			}
			gotBns := ClassifyBottlenecks(obs)
			if !slices.Equal(gotBns, tt.wantBns) {
				t.Fatalf("ClassifyBottlenecks = %v, want %v", gotBns, tt.wantBns)
			}
		})
	}
}

func TestPGOHypothesesFromStalls(t *testing.T) {
	tests := []struct {
		name   string
		stalls StallSample
		want   []BottleneckKind
	}{
		{
			name:   "zero_samples",
			stalls: StallSample{},
			want:   nil,
		},
		{
			name: "math_throttle_only_not_bottleneck",
			stalls: StallSample{
				TotalSamples: 100,
				MathThrottle: 85,
				MIOThrottle:  10, // 0.10 < 0.15
			},
			want: nil,
		},
		{
			name: "descending_share_order",
			stalls: StallSample{
				TotalSamples:   100,
				MIOThrottle:    20, // 0.20 -> BottleneckBankConflict
				LongScoreboard: 45, // 0.45 -> BottleneckMemoryBoundKV
				WaitMBarrier:   15, // 0.15 -> BottleneckWGMMASerial
				WaitTMEM:       10, // 0.10 < 0.15
			},
			want: []BottleneckKind{
				BottleneckMemoryBoundKV,
				BottleneckBankConflict,
				BottleneckWGMMASerial,
			},
		},
		{
			name: "deterministic_tie_break",
			stalls: StallSample{
				TotalSamples:   100,
				WaitTMEM:       25,
				WaitMBarrier:   25,
				MIOThrottle:    25,
				LongScoreboard: 25,
			},
			want: []BottleneckKind{
				BottleneckBankConflict,
				BottleneckMemoryBoundKV,
				BottleneckWGMMASerial,
				BottleneckLowOccupancy,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hyps := HypothesesFromStalls(tt.stalls)
			var got []BottleneckKind
			for _, h := range hyps {
				got = append(got, h.Bottleneck)
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("HypothesesFromStalls(%+v) = %v, want %v", tt.stalls, got, tt.want)
			}
		})
	}
}
