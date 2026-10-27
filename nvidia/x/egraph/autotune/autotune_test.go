package autotune_test

import (
	"encoding/json"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
)

func TestTwinForTargetAndAutoDiscover(t *testing.T) {
	tests := []struct {
		target         string
		major          int
		minor          int
		tmemCols       int
		maxCluster     int
		tcgen05        bool
		scaleVec4X     bool
		lutB           bool
		syncRestrictA  bool
		hwCvtE2M1ToF16 bool
		fabricAsync    bool
		ptxISA         string
	}{
		{
			target:         "sm_90a",
			major:          9,
			minor:          0,
			tmemCols:       0,
			maxCluster:     8,
			tcgen05:        false,
			scaleVec4X:     false,
			lutB:           false,
			syncRestrictA:  false,
			hwCvtE2M1ToF16: false,
			fabricAsync:    false,
			ptxISA:         "8.5",
		},
		{
			target:         "sm_100a",
			major:          10,
			minor:          0,
			tmemCols:       512,
			maxCluster:     16,
			tcgen05:        true,
			scaleVec4X:     false,
			lutB:           false,
			syncRestrictA:  false,
			hwCvtE2M1ToF16: true,
			fabricAsync:    false,
			ptxISA:         "8.5",
		},
		{
			target:         "sm_103a",
			major:          10,
			minor:          3,
			tmemCols:       512,
			maxCluster:     16,
			tcgen05:        true,
			scaleVec4X:     true,
			lutB:           false,
			syncRestrictA:  false,
			hwCvtE2M1ToF16: true,
			fabricAsync:    false,
			ptxISA:         "9.0",
		},
		{
			target:         "sm_107a",
			major:          10,
			minor:          7,
			tmemCols:       512,
			maxCluster:     32,
			tcgen05:        true,
			scaleVec4X:     true,
			lutB:           true,
			syncRestrictA:  true,
			hwCvtE2M1ToF16: true,
			fabricAsync:    true,
			ptxISA:         "9.4",
		},
		{
			target:         "sm_107f",
			major:          10,
			minor:          7,
			tmemCols:       512,
			maxCluster:     32,
			tcgen05:        true,
			scaleVec4X:     true,
			lutB:           true,
			syncRestrictA:  true,
			hwCvtE2M1ToF16: true,
			fabricAsync:    true,
			ptxISA:         "9.4",
		},
		{
			target:         "sm_120a",
			major:          12,
			minor:          0,
			tmemCols:       0,
			maxCluster:     1,
			tcgen05:        false,
			scaleVec4X:     false,
			lutB:           false,
			syncRestrictA:  false,
			hwCvtE2M1ToF16: true,
			fabricAsync:    false,
			ptxISA:         "8.5",
		},
		{
			target:         "sm_121a",
			major:          12,
			minor:          1,
			tmemCols:       0,
			maxCluster:     1,
			tcgen05:        false,
			scaleVec4X:     false,
			lutB:           false,
			syncRestrictA:  false,
			hwCvtE2M1ToF16: false,
			fabricAsync:    false,
			ptxISA:         "9.0",
		},
	}

	for _, tc := range tests {
		t.Run(tc.target, func(t *testing.T) {
			twin, err := autotune.TwinForTarget(tc.target)
			if err != nil {
				t.Fatalf("TwinForTarget(%q): %v", tc.target, err)
			}
			if twin.Target != tc.target {
				t.Errorf("Target = %q, want %q", twin.Target, tc.target)
			}
			if twin.ComputeMajor != tc.major || twin.ComputeMinor != tc.minor {
				t.Errorf("ComputeCapability = %d.%d, want %d.%d", twin.ComputeMajor, twin.ComputeMinor, tc.major, tc.minor)
			}
			if twin.TMEMColumns != tc.tmemCols {
				t.Errorf("TMEMColumns = %d, want %d", twin.TMEMColumns, tc.tmemCols)
			}
			if twin.MaxClusterCTAs != tc.maxCluster {
				t.Errorf("MaxClusterCTAs = %d, want %d", twin.MaxClusterCTAs, tc.maxCluster)
			}
			if twin.SupportsTCGen05 != tc.tcgen05 {
				t.Errorf("SupportsTCGen05 = %v, want %v", twin.SupportsTCGen05, tc.tcgen05)
			}
			if twin.SupportsScaleVec4X != tc.scaleVec4X {
				t.Errorf("SupportsScaleVec4X = %v, want %v", twin.SupportsScaleVec4X, tc.scaleVec4X)
			}
			if twin.SupportsLUTDecompressB != tc.lutB {
				t.Errorf("SupportsLUTDecompressB = %v, want %v", twin.SupportsLUTDecompressB, tc.lutB)
			}
			if twin.SupportsSyncRestrictReadA != tc.syncRestrictA {
				t.Errorf("SupportsSyncRestrictReadA = %v, want %v", twin.SupportsSyncRestrictReadA, tc.syncRestrictA)
			}
			if twin.SupportsHardwareCvtE2M1ToF16 != tc.hwCvtE2M1ToF16 {
				t.Errorf("SupportsHardwareCvtE2M1ToF16 = %v, want %v", twin.SupportsHardwareCvtE2M1ToF16, tc.hwCvtE2M1ToF16)
			}
			if twin.SupportsFabricAsyncTensor != tc.fabricAsync {
				t.Errorf("SupportsFabricAsyncTensor = %v, want %v", twin.SupportsFabricAsyncTensor, tc.fabricAsync)
			}
			if twin.PTXISAVersion != tc.ptxISA {
				t.Errorf("PTXISAVersion = %q, want %q", twin.PTXISAVersion, tc.ptxISA)
			}
		})
	}

	// Verify AutoDiscoverTarget with exclusive TMEM and NVLink fabric flags.
	rubinFabric, err := autotune.AutoDiscoverTarget(10, 7, true, true)
	if err != nil {
		t.Fatalf("AutoDiscoverTarget(10, 7, true, true): %v", err)
	}
	if rubinFabric.Target != "sm_107f" || rubinFabric.TMEMColumns != 576 || rubinFabric.RackRanks != 144 {
		t.Errorf("rubinFabric = %+v, want Target=sm_107f TMEMColumns=576 RackRanks=144", rubinFabric)
	}

	spark, err := autotune.AutoDiscoverTarget(12, 1, false, false)
	if err != nil {
		t.Fatalf("AutoDiscoverTarget(12, 1, false, false): %v", err)
	}
	if spark.Target != "sm_121a" || spark.SupportsHardwareCvtE2M1ToF16 {
		t.Errorf("spark = %+v, want Target=sm_121a SupportsHardwareCvtE2M1ToF16=false", spark)
	}

	if _, err := autotune.TwinForTarget("sm_80"); err == nil {
		t.Errorf("TwinForTarget(sm_80) succeeded, want error")
	}
	if _, err := autotune.AutoDiscoverTarget(8, 0, false, false); err == nil {
		t.Errorf("AutoDiscoverTarget(8, 0) succeeded, want error")
	}
}

func TestDiagnoseBottleneck(t *testing.T) {
	rubinTwin, err := autotune.TwinForTarget("sm_107a")
	if err != nil {
		t.Fatalf("TwinForTarget(sm_107a): %v", err)
	}

	tests := []struct {
		name             string
		twin             autotune.HardwareTwin
		stalls           autotune.StallProfile
		metrics          autotune.RangeMetrics
		wantBottleneck   autotune.BottleneckKind
		wantStages       int
		wantCTAGroup     int
		wantEarlyRelease bool
		wantSparse2to4   bool
		wantPreferLUT    bool
		wantProducerRegs int
		wantConsumerRegs int
	}{
		{
			name: "register_spill",
			twin: rubinTwin,
			stalls: autotune.StallProfile{
				TotalSamples:        1000,
				StallLongScoreboard: 700,
			},
			metrics: autotune.RangeMetrics{
				DurationNS:          45000,
				RegisterCount:       255,
				LocalSpillLoadBytes: 128,
			},
			wantBottleneck:   autotune.BottleneckRegisterSpill,
			wantStages:       2,
			wantCTAGroup:     1,
			wantProducerRegs: 40,
			wantConsumerRegs: 168,
		},
		{
			name: "memory_latency",
			twin: rubinTwin,
			stalls: autotune.StallProfile{
				TotalSamples:        1000,
				StallLongScoreboard: 620,
				StallNotSelected:    380,
			},
			metrics: autotune.RangeMetrics{
				DurationNS:    30000,
				RegisterCount: 168,
			},
			wantBottleneck:   autotune.BottleneckMemoryLatency,
			wantStages:       4,
			wantCTAGroup:     1,
			wantEarlyRelease: true,
			wantPreferLUT:    true,
			wantProducerRegs: 56,
			wantConsumerRegs: 184,
		},
		{
			name: "math_pipe_bound",
			twin: rubinTwin,
			stalls: autotune.StallProfile{
				TotalSamples:          1000,
				StallMathPipeThrottle: 550,
				StallNotSelected:      450,
			},
			metrics: autotune.RangeMetrics{
				DurationNS:    20000,
				RegisterCount: 192,
			},
			wantBottleneck:   autotune.BottleneckMathPipeBound,
			wantStages:       3,
			wantCTAGroup:     2,
			wantSparse2to4:   true,
			wantProducerRegs: 48,
			wantConsumerRegs: 208,
		},
		{
			name: "mio_smem_conflict",
			twin: rubinTwin,
			stalls: autotune.StallProfile{
				TotalSamples:         1000,
				StallMIOThrottle:     300,
				StallShortScoreboard: 200,
				StallNotSelected:     500,
			},
			metrics: autotune.RangeMetrics{
				DurationNS:    25000,
				RegisterCount: 160,
			},
			wantBottleneck:   autotune.BottleneckMIOSMEMConflict,
			wantStages:       2,
			wantCTAGroup:     1,
			wantProducerRegs: 56,
			wantConsumerRegs: 176,
		},
		{
			name: "barrier_skew",
			twin: rubinTwin,
			stalls: autotune.StallProfile{
				TotalSamples:     1000,
				StallBarrier:     480,
				StallNotSelected: 520,
			},
			metrics: autotune.RangeMetrics{
				DurationNS:    22000,
				RegisterCount: 144,
			},
			wantBottleneck:   autotune.BottleneckBarrierSkew,
			wantStages:       3,
			wantCTAGroup:     1,
			wantProducerRegs: 48,
			wantConsumerRegs: 168,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := autotune.DiagnoseBottleneck(tc.twin, tc.stalls, tc.metrics)
			if rec.Bottleneck != tc.wantBottleneck {
				t.Errorf("Bottleneck = %q, want %q", rec.Bottleneck, tc.wantBottleneck)
			}
			if rec.SuggestedStages != tc.wantStages {
				t.Errorf("SuggestedStages = %d, want %d", rec.SuggestedStages, tc.wantStages)
			}
			if rec.SuggestedCTAGroup != tc.wantCTAGroup {
				t.Errorf("SuggestedCTAGroup = %d, want %d", rec.SuggestedCTAGroup, tc.wantCTAGroup)
			}
			if rec.EnableEarlyReleaseSyncRestrict != tc.wantEarlyRelease {
				t.Errorf("EnableEarlyReleaseSyncRestrict = %v, want %v", rec.EnableEarlyReleaseSyncRestrict, tc.wantEarlyRelease)
			}
			if rec.EnableSparse2to4 != tc.wantSparse2to4 {
				t.Errorf("EnableSparse2to4 = %v, want %v", rec.EnableSparse2to4, tc.wantSparse2to4)
			}
			if rec.PreferSub4BitLUT != tc.wantPreferLUT {
				t.Errorf("PreferSub4BitLUT = %v, want %v", rec.PreferSub4BitLUT, tc.wantPreferLUT)
			}
			if rec.ProducerMaxRegs != tc.wantProducerRegs || rec.ConsumerMaxRegs != tc.wantConsumerRegs {
				t.Errorf("Regs = (%d, %d), want (%d, %d)", rec.ProducerMaxRegs, rec.ConsumerMaxRegs, tc.wantProducerRegs, tc.wantConsumerRegs)
			}
		})
	}
}

func TestOnlineCostModelAdaptationAndJSON(t *testing.T) {
	twin, err := autotune.TwinForTarget("sm_107a")
	if err != nil {
		t.Fatalf("TwinForTarget(sm_107a): %v", err)
	}

	baseline := autotune.ScheduleCandidate{
		Name:             "gemm_2stage_dense",
		Target:           "sm_107a",
		TileM:            128,
		TileN:            256,
		TileK:            64,
		Stages:           2,
		CTAGroup:         1,
		AnalyticalCostUS: 14.0,
	}
	optimized := autotune.ScheduleCandidate{
		Name:             "gemm_4stage_lut_early_release",
		Target:           "sm_107a",
		TileM:            128,
		TileN:            256,
		TileK:            128,
		Stages:           4,
		CTAGroup:         1,
		UseLUTB:          true,
		UseEarlyRelease:  true,
		AnalyticalCostUS: 16.5,
	}
	candidates := []autotune.ScheduleCandidate{baseline, optimized}

	model := autotune.NewOnlineCostModel()
	const shapeKey = "M=128,N=4096,K=14336"

	// Before any observations, the analytically cheaper baseline wins.
	best0, _, err := model.SelectBest(shapeKey, twin, candidates)
	if err != nil {
		t.Fatalf("SelectBest initial: %v", err)
	}
	if best0.Name != baseline.Name {
		t.Fatalf("initial SelectBest = %q, want %q", best0.Name, baseline.Name)
	}

	// Record online CUPTI observation showing baseline suffers heavy HBM long-scoreboard
	// stalls and local memory register spills.
	model.RecordObservation(
		shapeKey,
		baseline,
		autotune.StallProfile{
			TotalSamples:        1000,
			StallLongScoreboard: 680,
			StallNotSelected:    320,
		},
		autotune.RangeMetrics{
			DurationNS:          32000,
			RegisterCount:       254,
			LocalSpillLoadBytes: 64,
		},
	)
	// Also record a clean observation for a second shape with pure memory latency stalls.
	const memShapeKey = "M=64,N=4096,K=14336"
	model.RecordObservation(
		memShapeKey,
		baseline,
		autotune.StallProfile{
			TotalSamples:        1000,
			StallLongScoreboard: 640,
			StallNotSelected:    360,
		},
		autotune.RangeMetrics{
			DurationNS:    26000,
			RegisterCount: 168,
		},
	)

	bestAfterSpill, recSpill, err := model.SelectBest(shapeKey, twin, candidates)
	if err != nil {
		t.Fatalf("SelectBest after spill: %v", err)
	}
	if bestAfterSpill.Name != optimized.Name {
		t.Errorf("SelectBest after spill = %q, want %q", bestAfterSpill.Name, optimized.Name)
	}
	if recSpill.Bottleneck != autotune.BottleneckRegisterSpill {
		t.Errorf("recSpill.Bottleneck = %q, want %q", recSpill.Bottleneck, autotune.BottleneckRegisterSpill)
	}

	bestAfterMem, recMem, err := model.SelectBest(memShapeKey, twin, candidates)
	if err != nil {
		t.Fatalf("SelectBest after memory latency stall: %v", err)
	}
	if bestAfterMem.Name != optimized.Name {
		t.Errorf("SelectBest after memory latency stall = %q, want %q", bestAfterMem.Name, optimized.Name)
	}
	if recMem.Bottleneck != autotune.BottleneckMemoryLatency || !recMem.EnableEarlyReleaseSyncRestrict || !recMem.PreferSub4BitLUT {
		t.Errorf("recMem = %+v, want memory_latency with EnableEarlyReleaseSyncRestrict and PreferSub4BitLUT", recMem)
	}

	// Verify JSON round-trip persistence.
	rawJSON, err := json.Marshal(model)
	if err != nil {
		t.Fatalf("json.Marshal(model): %v", err)
	}
	var restored autotune.OnlineCostModel
	if err := json.Unmarshal(rawJSON, &restored); err != nil {
		t.Fatalf("json.Unmarshal(model): %v", err)
	}

	bestRestored, recRestored, err := restored.SelectBest(memShapeKey, twin, candidates)
	if err != nil {
		t.Fatalf("restored.SelectBest: %v", err)
	}
	if bestRestored.Name != optimized.Name {
		t.Errorf("restored.SelectBest = %q, want %q", bestRestored.Name, optimized.Name)
	}
	if recRestored.Bottleneck != autotune.BottleneckMemoryLatency {
		t.Errorf("recRestored.Bottleneck = %q, want %q", recRestored.Bottleneck, autotune.BottleneckMemoryLatency)
	}
}
