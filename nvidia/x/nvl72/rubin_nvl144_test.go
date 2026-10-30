package nvl72

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestRubinRackSpecs(t *testing.T) {
	nvl144 := DefaultVeraRubinNVL144Spec()
	if err := nvl144.Validate(); err != nil {
		t.Fatalf("DefaultVeraRubinNVL144Spec Validate: %v", err)
	}
	if nvl144.Tier != TierVeraRubinNVL144 {
		t.Fatalf("Tier = %q, want %q", nvl144.Tier, TierVeraRubinNVL144)
	}
	if nvl144.ComputeDiesPerRack != 144 || nvl144.PackagesPerRack != 72 || nvl144.DiesPerPackage != 2 {
		t.Fatalf("NVL144 topology = (%d dies, %d pkgs, %d dies/pkg), want (144, 72, 2)",
			nvl144.ComputeDiesPerRack, nvl144.PackagesPerRack, nvl144.DiesPerPackage)
	}
	if nvl144.VeraCPUsPerRack != 36 || nvl144.NVLinkVersion != 6 {
		t.Fatalf("VeraCPUsPerRack=%d NVLinkVersion=%d, want 36, 6", nvl144.VeraCPUsPerRack, nvl144.NVLinkVersion)
	}
	if nvl144.NVLinkUnidirGBpsPerDie != 1800.0 || nvl144.NVLinkBidirGBpsPerDie() != 3600.0 {
		t.Fatalf("NVLink6 = (%v unidir, %v bidir), want (1800, 3600)",
			nvl144.NVLinkUnidirGBpsPerDie, nvl144.NVLinkBidirGBpsPerDie())
	}
	if nvl144.RackSwitchBisectionTBps != 259.2 {
		t.Fatalf("RackSwitchBisectionTBps = %v, want 259.2", nvl144.RackSwitchBisectionTBps)
	}
	if nvl144.VeraC2CBandwidthGBps != 900.0 || nvl144.VeraC2CBidirGBps() != 1800.0 {
		t.Fatalf("VeraC2C = (%v unidir, %v bidir), want (900, 1800)",
			nvl144.VeraC2CBandwidthGBps, nvl144.VeraC2CBidirGBps())
	}
	if nvl144.HBM4BytesPerDie != 288<<30 || nvl144.HBM4BandwidthGBpsPerDie != 13000.0 {
		t.Fatalf("HBM4 = (%d bytes, %v GB/s), want (%d, 13000)",
			nvl144.HBM4BytesPerDie, nvl144.HBM4BandwidthGBpsPerDie, int64(288)<<30)
	}
	if nvl144.ConnectX9GbPerSecPerDie != 1600.0 || nvl144.ConnectX9UnidirGBpsPerDie() != 200.0 {
		t.Fatalf("ConnectX-9 = (%v Gb/s, %v GB/s), want (1600, 200)",
			nvl144.ConnectX9GbPerSecPerDie, nvl144.ConnectX9UnidirGBpsPerDie())
	}
	if nvl144.FP4DensePFLOPSPerDie != 50.0 {
		t.Fatalf("FP4DensePFLOPSPerDie = %v, want 50", nvl144.FP4DensePFLOPSPerDie)
	}

	nvl576 := DefaultRubinUltraNVL576Spec()
	if err := nvl576.Validate(); err != nil {
		t.Fatalf("DefaultRubinUltraNVL576Spec Validate: %v", err)
	}
	if nvl576.Tier != TierRubinUltraNVL576 {
		t.Fatalf("Tier = %q, want %q", nvl576.Tier, TierRubinUltraNVL576)
	}
	if nvl576.ComputeDiesPerRack != 576 || nvl576.PackagesPerRack != 144 || nvl576.DiesPerPackage != 4 {
		t.Fatalf("NVL576 topology = (%d dies, %d pkgs, %d dies/pkg), want (576, 144, 4)",
			nvl576.ComputeDiesPerRack, nvl576.PackagesPerRack, nvl576.DiesPerPackage)
	}
	if nvl576.RackSwitchBisectionTBps != 1500.0 {
		t.Fatalf("NVL576 RackSwitchBisectionTBps = %v, want 1500.0", nvl576.RackSwitchBisectionTBps)
	}
	if nvl576.HBM4BytesPerDie != 576<<30 || nvl576.HBM4BandwidthGBpsPerDie != 22000.0 {
		t.Fatalf("NVL576 HBM4e = (%d bytes, %v GB/s), want (%d, 22000)",
			nvl576.HBM4BytesPerDie, nvl576.HBM4BandwidthGBpsPerDie, int64(576)<<30)
	}
	if nvl576.FP4DensePFLOPSPerDie != 100.0 {
		t.Fatalf("NVL576 FP4DensePFLOPSPerDie = %v, want 100", nvl576.FP4DensePFLOPSPerDie)
	}
}

func TestEstimateRubinPodWideEP(t *testing.T) {
	// 1. Single-rack Vera Rubin NVL144 (NumRacks=1, 144 dies, InterRackDispatchUS == 0).
	singleRack, err := EstimateRubinPodWideEP(RubinPodEPConfig{
		Spec:                          DefaultVeraRubinNVL144Spec(),
		NumRacks:                      1,
		BatchTokensPerDie:             64,
		HiddenDim:                     7168,
		TopK:                          8,
		ExpertsPerDie:                 2,
		UseHierarchicalPackageCombine: true,
		UseFP4Dispatch:                true,
	})
	if err != nil {
		t.Fatalf("EstimateRubinPodWideEP single rack: %v", err)
	}
	if singleRack.TotalDies != 144 {
		t.Fatalf("singleRack.TotalDies = %d, want 144", singleRack.TotalDies)
	}
	if singleRack.TotalExperts != 288 {
		t.Fatalf("singleRack.TotalExperts = %d, want 288", singleRack.TotalExperts)
	}
	if singleRack.InterRackDispatchUS != 0 {
		t.Fatalf("singleRack.InterRackDispatchUS = %v, want 0", singleRack.InterRackDispatchUS)
	}
	if singleRack.IntraRackDispatchUS <= 0 || singleRack.CombineLatencyUS <= 0 || singleRack.TotalCommLatencyUS <= 0 {
		t.Fatalf("invalid single-rack latencies: %+v", singleRack)
	}
	if singleRack.SpeedupVsGB300Pod <= 1.8 {
		t.Fatalf("singleRack.SpeedupVsGB300Pod = %.3f, want > 1.8", singleRack.SpeedupVsGB300Pod)
	}

	// 2. Multi-rack 4x Vera Rubin NVL144 pod (576 dies): hierarchical package+rack
	// combine beats flat all-to-all and achieves > 1.8x speedup vs GB300 pod.
	multiHier, err := EstimateRubinPodWideEP(RubinPodEPConfig{
		Spec:                          DefaultVeraRubinNVL144Spec(),
		NumRacks:                      4,
		BatchTokensPerDie:             64,
		HiddenDim:                     7168,
		TopK:                          8,
		ExpertsPerDie:                 1,
		UseHierarchicalPackageCombine: true,
		UseFP4Dispatch:                true,
	})
	if err != nil {
		t.Fatalf("EstimateRubinPodWideEP multi-rack hierarchical: %v", err)
	}
	multiFlat, err := EstimateRubinPodWideEP(RubinPodEPConfig{
		Spec:                          DefaultVeraRubinNVL144Spec(),
		NumRacks:                      4,
		BatchTokensPerDie:             64,
		HiddenDim:                     7168,
		TopK:                          8,
		ExpertsPerDie:                 1,
		UseHierarchicalPackageCombine: false,
		UseFP4Dispatch:                true,
	})
	if err != nil {
		t.Fatalf("EstimateRubinPodWideEP multi-rack flat: %v", err)
	}
	if multiHier.TotalDies != 576 || multiHier.TotalExperts != 576 {
		t.Fatalf("multiHier dies/experts = %d/%d, want 576/576", multiHier.TotalDies, multiHier.TotalExperts)
	}
	if multiHier.InterRackDispatchUS <= 0 {
		t.Fatalf("multiHier.InterRackDispatchUS = %v, want > 0", multiHier.InterRackDispatchUS)
	}
	if multiHier.TotalCommLatencyUS >= multiFlat.TotalCommLatencyUS {
		t.Fatalf("expected UseHierarchicalPackageCombine (%v us) < flat (%v us)",
			multiHier.TotalCommLatencyUS, multiFlat.TotalCommLatencyUS)
	}
	if multiHier.SpeedupVsGB300Pod <= 1.8 {
		t.Fatalf("multiHier.SpeedupVsGB300Pod = %.3f, want > 1.8", multiHier.SpeedupVsGB300Pod)
	}
}

func TestPlanRubinDisaggregation(t *testing.T) {
	plan, err := PlanRubinDisaggregation(RubinDisaggConfig{
		PrefillSpec:  DefaultVeraRubinNVL144Spec(),
		DecodeSpec:   DefaultRubinUltraNVL576Spec(),
		PrefillRacks: 2,
		DecodeRacks:  1,
		Model: PDModelSpec{
			NumLayers:            61,
			HiddenSize:           7168,
			NumQHeads:            128,
			HeadDim:              128,
			CacheKind:            CacheMLA,
			KVLoRARank:           512,
			RoPEDim:              64,
			TotalExperts:         576,
			TopK:                 8,
			ActiveParamsBillions: 37.0,
			TotalParamsBillions:  671.0,
		},
		SLA: PDSLA{
			PromptTokens:      16384,
			DecodeBatchPerGPU: 64,
			MaxTTFTMs:         250.0,
			MaxTPOTMs:         15.0,
		},
		PageTokens:                16,
		UseLUTDecompressedPrefill: true,
		UseVeraC2CStaging:         true,
	})
	if err != nil {
		t.Fatalf("PlanRubinDisaggregation: %v", err)
	}
	if plan.PrefillTier != TierVeraRubinNVL144 || plan.DecodeTier != TierRubinUltraNVL576 {
		t.Fatalf("tiers = (%q, %q), want (%q, %q)",
			plan.PrefillTier, plan.DecodeTier, TierVeraRubinNVL144, TierRubinUltraNVL576)
	}
	if plan.PrefillDies != 288 || plan.DecodeDies != 576 {
		t.Fatalf("dies = (%d prefill, %d decode), want (288, 576)", plan.PrefillDies, plan.DecodeDies)
	}
	if plan.PrefillTargetArch != "sm_107f" {
		t.Fatalf("PrefillTargetArch = %q, want sm_107f", plan.PrefillTargetArch)
	}
	if plan.KVPageTransferUS <= 0 || plan.TotalKVTransferMS <= 0 {
		t.Fatalf("KV transfer times = (%v us/page, %v ms total), want > 0",
			plan.KVPageTransferUS, plan.TotalKVTransferMS)
	}
	if !plan.MeetsTTFT || !plan.MeetsTPOT || !plan.MeetsSLA {
		t.Fatalf("plan failed SLA: TTFT=%.3f ms (max %.1f), TPOT=%.3f ms (max %.1f)",
			plan.EstimatedTTFTMS, 250.0, plan.EstimatedTPOTMS, 15.0)
	}
	if plan.ClusterTokensPerSec <= 0 {
		t.Fatalf("ClusterTokensPerSec = %v, want > 0", plan.ClusterTokensPerSec)
	}
}

func TestRubinNVL144AllReduceResidualRMSNorm(t *testing.T) {
	// 1. Verify 144-rank default configuration and Reference across 144 ranks.
	const (
		ranks144 = 144
		rows     = 2
		cols     = 64
	)
	k144 := RubinNVL144AllReduceResidualRMSNorm{
		HiddenDim: cols,
		Eps:       1e-5,
		Threads:   64,
	}
	cudaSrc, err := k144.CUDA()
	if err != nil {
		t.Fatalf("k144.CUDA: %v", err)
	}
	for _, want := range []string{
		"NUM_RANKS = 144",
		"sm_107f",
		"eg_rubin_multimem_ld_reduce_relaxed_sys_add_f32",
		"eg_rubin_multimem_st_relaxed_sys_f32",
	} {
		if !strings.Contains(cudaSrc, want) {
			t.Fatalf("CUDA() missing %q", want)
		}
	}
	fileSrc, err := k144.CUDAFile()
	if err != nil {
		t.Fatalf("k144.CUDAFile: %v", err)
	}
	for _, want := range []string{
		"multimem.ld_reduce.relaxed.sys.global.add.f32",
		"multimem.st.relaxed.sys.global.f32",
	} {
		if !strings.Contains(fileSrc, want) {
			t.Fatalf("CUDAFile() missing %q", want)
		}
	}

	totalElems := rows * cols
	partials144 := make([][]float32, ranks144)
	for r := 0; r < ranks144; r++ {
		partials144[r] = make([]float32, totalElems)
		for i := 0; i < totalElems; i++ {
			partials144[r][i] = float32(((r+i)%5)-2) * 0.125
		}
	}
	residual := make([]float32, totalElems)
	for i := range residual {
		residual[i] = float32((i%7)-3) * 0.25
	}
	weight := make([]float32, cols)
	for c := range weight {
		weight[c] = 1.0 + float32((c%5)-2)*0.125
	}
	sum144, norm144, err := k144.Reference(partials144, residual, weight)
	if err != nil {
		t.Fatalf("k144.Reference: %v", err)
	}
	if len(sum144) != totalElems || len(norm144) != totalElems {
		t.Fatalf("Reference output lengths = (%d, %d), want %d", len(sum144), len(norm144), totalElems)
	}

	// 2. Execute across 8 simulated ranks on host via cudahostemu.RunMultiRank.
	const emuRanks = 8
	k8 := RubinNVL144AllReduceResidualRMSNorm{
		Name:      "rubin_nvl144_ar_rmsnorm_emu8",
		Target:    "sm_107f",
		NumRanks:  emuRanks,
		HiddenDim: cols,
		Eps:       1e-5,
		Threads:   64,
	}
	src8, err := k8.CUDAFile()
	if err != nil {
		t.Fatalf("k8.CUDAFile: %v", err)
	}

	partials8 := make([][]float32, emuRanks)
	symPartialsFlat := make([]float32, emuRanks*totalElems)
	for r := 0; r < emuRanks; r++ {
		partials8[r] = make([]float32, totalElems)
		for i := 0; i < totalElems; i++ {
			// Exact dyadic rationals for bit-accurate float32 reduction across ranks.
			v := float32(((r+1)*(i+1))%7-3) * 0.25
			partials8[r][i] = v
			symPartialsFlat[r*totalElems+i] = v
		}
	}
	wantSum, wantNorm, err := k8.Reference(partials8, residual, weight)
	if err != nil {
		t.Fatalf("k8.Reference: %v", err)
	}

	symInBytes := f32Bytes(symPartialsFlat)
	resBytes := f32Bytes(residual)
	wBytes := f32Bytes(weight)
	sumOutBytes := cudahostemu.Poison(emuRanks * totalElems * 4)
	normOutBytes := cudahostemu.Poison(emuRanks * totalElems * 4)

	grid, block := k8.Grid(rows)
	err = cudahostemu.RunMultiRank(context.Background(), src8, emuRanks, cudahostemu.Launch{
		Kernel: k8.Name,
		Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
		Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
	},
		cudahostemu.IntArg(rows),
		cudahostemu.BufferArg("sym_partials", "float", symInBytes),
		cudahostemu.BufferArg("residual", "float", resBytes),
		cudahostemu.BufferArg("weight", "float", wBytes),
		cudahostemu.BufferArg("sum_out", "float", sumOutBytes),
		cudahostemu.BufferArg("norm_out", "float", normOutBytes),
	)
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatalf("RunMultiRank: %v", err)
	}

	gotSum := bytesF32(sumOutBytes)
	gotNorm := bytesF32(normOutBytes)
	for r := 0; r < emuRanks; r++ {
		for i := 0; i < totalElems; i++ {
			gSum := gotSum[r*totalElems+i]
			wSum := wantSum[i]
			if math.Float32bits(gSum) != math.Float32bits(wSum) {
				t.Fatalf("rank %d sum_out[%d] = %v (%#08x), want %v (%#08x)",
					r, i, gSum, math.Float32bits(gSum), wSum, math.Float32bits(wSum))
			}
			gNorm := gotNorm[r*totalElems+i]
			wNorm := wantNorm[i]
			if math.Abs(float64(gNorm-wNorm)) > 1e-6 {
				t.Fatalf("rank %d norm_out[%d] = %v, want %v", r, i, gNorm, wNorm)
			}
		}
	}
}

func ExampleEstimateRubinPodWideEP() {
	est, err := EstimateRubinPodWideEP(RubinPodEPConfig{
		Spec:                          DefaultVeraRubinNVL144Spec(),
		NumRacks:                      4,
		BatchTokensPerDie:             64,
		HiddenDim:                     7168,
		TopK:                          8,
		ExpertsPerDie:                 1,
		UseHierarchicalPackageCombine: true,
		UseFP4Dispatch:                true,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("dies=%d experts=%d hbm4_tib=%d speedup_vs_gb300>1.8=%v\n",
		est.TotalDies, est.TotalExperts, est.TotalHBM4Bytes>>40, est.SpeedupVsGB300Pod > 1.8)
	// Output:
	// dies=576 experts=576 hbm4_tib=162 speedup_vs_gb300>1.8=true
}
