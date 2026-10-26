package qualify

import (
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
