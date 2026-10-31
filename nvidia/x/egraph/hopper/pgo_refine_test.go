package hopper

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

func TestPGORefineClassifyBottlenecks(t *testing.T) {
	tests := []struct {
		name string
		obs  KernelObservation
		want []BottleneckKind
	}{
		{
			name: "healthy",
			obs: KernelObservation{
				Name:               "healthy_k",
				DurationNs:         20000,
				LaunchGapNs:        1000,
				SharedBankConflict: 0.02,
				Registers:          96,
				AchievedOccupancy:  0.50,
				GridBlocks:         264,
				DeviceSMs:          132,
			},
			want: nil,
		},
		{
			name: "bank_conflict",
			obs:  KernelObservation{SharedBankConflict: 0.12},
			want: []BottleneckKind{BottleneckBankConflict},
		},
		{
			name: "wgmma_serialized",
			obs:  KernelObservation{WGMMASerialized: true},
			want: []BottleneckKind{BottleneckWGMMASerial},
		},
		{
			name: "register_spill_store",
			obs:  KernelObservation{SpillStoreBytes: 16},
			want: []BottleneckKind{BottleneckRegisterSpill},
		},
		{
			name: "register_spill_load",
			obs:  KernelObservation{SpillLoadBytes: 32},
			want: []BottleneckKind{BottleneckRegisterSpill},
		},
		{
			name: "low_occupancy_high_regs",
			obs:  KernelObservation{AchievedOccupancy: 0.18, Registers: 168},
			want: []BottleneckKind{BottleneckLowOccupancy},
		},
		{
			name: "memory_bound_kv_explicit",
			obs:  KernelObservation{MemoryBoundKV: true},
			want: []BottleneckKind{BottleneckMemoryBoundKV},
		},
		{
			name: "memory_bound_kv_low_occ_low_regs",
			obs:  KernelObservation{AchievedOccupancy: 0.15, Registers: 64},
			want: []BottleneckKind{BottleneckMemoryBoundKV},
		},
		{
			name: "wave_tail_quantization",
			obs:  KernelObservation{GridBlocks: 160, DeviceSMs: 132}, // 1.212 waves
			want: []BottleneckKind{BottleneckWaveTail},
		},
		{
			name: "wave_tail_boundary_035",
			obs:  KernelObservation{GridBlocks: 27, DeviceSMs: 20}, // 1.35 waves
			want: []BottleneckKind{BottleneckWaveTail},
		},
		{
			name: "wave_full_no_tail",
			obs:  KernelObservation{GridBlocks: 264, DeviceSMs: 132}, // 2.0 waves
			want: nil,
		},
		{
			name: "launch_overhead",
			obs:  KernelObservation{DurationNs: 10000, LaunchGapNs: 2000}, // 0.20 >= 0.15
			want: []BottleneckKind{BottleneckLaunchOverhead},
		},
		{
			name: "multiple_combined",
			obs: KernelObservation{
				SharedBankConflict: 0.25,
				WGMMASerialized:    true,
				SpillStoreBytes:    8,
				AchievedOccupancy:  0.12,
				Registers:          192,
				GridBlocks:         150,
				DeviceSMs:          132,
				DurationNs:         5000,
				LaunchGapNs:        1500,
			},
			want: []BottleneckKind{
				BottleneckBankConflict,
				BottleneckWGMMASerial,
				BottleneckRegisterSpill,
				BottleneckLowOccupancy,
				BottleneckWaveTail,
				BottleneckLaunchOverhead,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ClassifyBottlenecks(tt.obs)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("ClassifyBottlenecks(%+v) = %v, want %v", tt.obs, got, tt.want)
			}
		})
	}
}

func TestPGORefineObservationFromPTXAS(t *testing.T) {
	fn := ptxasinfo.Function{
		Name:        "_Z12tile_gemm_k0iPKtS0_Pf",
		Arch:        "sm_90a",
		Entry:       true,
		SpillStores: 24,
		SpillLoads:  32,
		Registers:   184,
		Diagnostics: []ptxasinfo.Diagnostic{
			{
				Code: "C7519",
				Text: "(C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path in the function '_Z12tile_gemm_k0iPKtS0_Pf'",
			},
		},
	}
	obs := ObservationFromPTXAS(fn, 160, 132)
	if obs.Name != fn.Name {
		t.Fatalf("Name = %q, want %q", obs.Name, fn.Name)
	}
	if !obs.WGMMASerialized {
		t.Fatal("expected WGMMASerialized = true from C7519 diagnostic")
	}
	if obs.SpillStoreBytes != 24 || obs.SpillLoadBytes != 32 || obs.Registers != 184 {
		t.Fatalf("unexpected resource counts: %+v", obs)
	}
	bns := ClassifyBottlenecks(obs)
	for _, want := range []BottleneckKind{BottleneckWGMMASerial, BottleneckRegisterSpill, BottleneckWaveTail} {
		if !slices.Contains(bns, want) {
			t.Fatalf("ClassifyBottlenecks(%+v) = %v, missing %v", obs, bns, want)
		}
	}
}

func TestPGORefineTileGEMM(t *testing.T) {
	t.Run("bank_conflict_and_wgmma_serial_to_warpspec", func(t *testing.T) {
		base := TileGEMM{
			Name:   "pgo_tile_ws",
			N:      128,
			K:      128,
			Weight: WeightBF16,
			Out:    F32,
			BM:     128,
			BN:     64,
			Stages: 2,
			WGMMA:  true,
		}
		step := Step{Tile: &base, M: 130}
		obs := KernelObservation{
			Name:               base.Name,
			SharedBankConflict: 0.22,
			WGMMASerialized:    true,
		}
		refined, reasons := RefineStep(step, obs)
		if refined.Tile == nil {
			t.Fatal("refined.Tile is nil")
		}
		if !refined.Tile.Swizzle128B || !refined.Tile.WarpSpecialized {
			t.Fatalf("refined TileGEMM = %+v, want Swizzle128B=true and WarpSpecialized=true (reasons: %v)", *refined.Tile, reasons)
		}
		if len(reasons) < 2 {
			t.Fatalf("expected >= 2 rewrite rationales, got %v", reasons)
		}

		// Verify refined WarpSpecialized CUDA source compiles and runs accurately via cudahostemu.Run.
		src, err := refined.Tile.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		wc := newWSCase(t, *refined.Tile, step.M)
		if ok, err := wc.run(t, src); !ok || err != nil {
			t.Fatalf("emulated refined WarpSpecialized TileGEMM failed: ok=%v err=%v", ok, err)
		}
	})

	t.Run("register_spill_stepdowns", func(t *testing.T) {
		// 1. DequantWords 2 -> 1
		g2w := TileGEMM{
			Name:         "pgo_tile_2w",
			N:            128,
			K:            128,
			Weight:       WeightNVFP4,
			Out:          F32,
			BM:           128,
			BN:           128,
			Stages:       2,
			WGMMA:        true,
			Repacked:     true,
			Swizzle128B:  true,
			DequantWords: 2,
		}
		s1, r1 := RefineStep(Step{Tile: &g2w, M: 64}, KernelObservation{SpillStoreBytes: 16})
		if s1.Tile.DequantWords != 1 || len(r1) == 0 {
			t.Fatalf("expected DequantWords=1, got %d (reasons: %v)", s1.Tile.DequantWords, r1)
		}

		// 2. BN 128 -> 64
		gBN := TileGEMM{
			Name:        "pgo_tile_bn",
			N:           128,
			K:           128,
			Weight:      WeightBF16,
			Out:         F32,
			BM:          128,
			BN:          128,
			Stages:      3,
			WGMMA:       true,
			Swizzle128B: true,
		}
		s2, r2 := RefineStep(Step{Tile: &gBN, M: 64}, KernelObservation{AchievedOccupancy: 0.15, Registers: 196})
		if s2.Tile.BN != 64 || len(r2) == 0 {
			t.Fatalf("expected BN=64, got %d (reasons: %v)", s2.Tile.BN, r2)
		}

		// 3. Stages 3 -> 2 when BN is already 64
		s3, r3 := RefineStep(s2, KernelObservation{SpillLoadBytes: 8})
		if s3.Tile.Stages != 2 || len(r3) == 0 {
			t.Fatalf("expected Stages=2, got %d (reasons: %v)", s3.Tile.Stages, r3)
		}

		// Verify bit-accurate execution on exact integer inputs via cudahostemu.Run.
		r := rand.New(rand.NewPCG(701, 702))
		exact := newTileInputs(r, *s3.Tile, 64, 0, true)
		baseSrc, err := gBN.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		refSrc, err := s3.Tile.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		wantOut := emuTileGEMM(t, gBN, baseSrc, exact, 1, false)["out"]
		gotOut := emuTileGEMM(t, *s3.Tile, refSrc, exact, 1, false)["out"]
		if !bytes.Equal(gotOut, wantOut) {
			t.Fatalf("refined TileGEMM output differs from baseline at %s", firstDiff(F32, gotOut, wantOut))
		}
	})

	t.Run("wave_tail_to_streamk", func(t *testing.T) {
		base := TileGEMM{
			Name:        "pgo_tile_sk",
			N:           192,
			K:           192,
			Weight:      WeightBF16,
			Out:         F32,
			BM:          128,
			BN:          64,
			Stages:      2,
			WGMMA:       true,
			Swizzle128B: true,
		}
		// 6 tiles across 5 SMs = 1.2 waves (fractional tail 0.20 in (0, 0.35]).
		obs := KernelObservation{
			Name:       base.Name,
			GridBlocks: 6,
			DeviceSMs:  5,
		}
		refined, reasons := RefineStep(Step{Tile: &base, M: 150}, obs)
		if refined.Tile.StreamK != 5 || len(reasons) == 0 {
			t.Fatalf("expected StreamK=5, got %d (reasons: %v)", refined.Tile.StreamK, reasons)
		}

		r := rand.New(rand.NewPCG(703, 704))
		exact := newTileInputs(r, *refined.Tile, 150, 0, true)
		baseSrc, err := base.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		skSrc, err := refined.Tile.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		wantOut := emuTileGEMM(t, base, baseSrc, exact, 1, false)["out"]
		gotOut := emuTileGEMM(t, *refined.Tile, skSrc, exact, 2, false)["out"]
		if !bytes.Equal(gotOut, wantOut) {
			t.Fatalf("StreamK refined TileGEMM differs from baseline at %s", firstDiff(F32, gotOut, wantOut))
		}
	})
}

func TestPGORefineSkinnyGEMM(t *testing.T) {
	base := SkinnyGEMM{
		Name:         "pgo_skinny_nvfp4",
		N:            4,
		K:            2048,
		MaxM:         2,
		Weight:       WeightNVFP4,
		Out:          F32,
		RowsPerBlock: 2,
		Warps:        1,
	}
	step := Step{GEMM: &base, M: 2}
	obs := KernelObservation{
		Name:              base.Name,
		AchievedOccupancy: 0.14,
		Registers:         64,
		MemoryBoundKV:     true,
	}
	refined, reasons := RefineStep(step, obs)
	if refined.GEMM.Prefetch <= 0 || !refined.GEMM.BlackwellFP4 {
		t.Fatalf("expected Prefetch > 0 and BlackwellFP4=true, got %+v (reasons: %v)", *refined.GEMM, reasons)
	}

	// Verify register spill step-down disables Prefetch and reduces RowsPerBlock.
	spilled, spillReasons := RefineStep(refined, KernelObservation{SpillStoreBytes: 16, SpillLoadBytes: 16})
	if spilled.GEMM.Prefetch != 0 || spilled.GEMM.RowsPerBlock != 1 || len(spillReasons) == 0 {
		t.Fatalf("expected Prefetch=0 and RowsPerBlock=1 after spill refinement, got %+v (reasons: %v)", *spilled.GEMM, spillReasons)
	}

	// Run base, refined, and spilled-refined SkinnyGEMM on CPU via cudahostemu.Run.
	r := rand.New(rand.NewPCG(801, 802))
	m := 2
	x := randBF16(r, m*base.K, 1)
	w := randNVFP4(r, base.N, base.K)
	want, err := base.Reference(m, x, w, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name string
		g    SkinnyGEMM
	}{
		{"base", base},
		{"refined_prefetch_blackwell", *refined.GEMM},
		{"refined_spill_stepdown", *spilled.GEMM},
	} {
		src, err := tc.g.CUDA()
		if err != nil {
			t.Fatalf("%s CUDA: %v", tc.name, err)
		}
		grid, block, err := tc.g.Grid()
		if err != nil {
			t.Fatalf("%s Grid: %v", tc.name, err)
		}
		launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, x, w, w_sf, w_gs, out)",
			grid[0], grid[1], block[0], tc.g.Name, m)
		res := runEmu(t, src, launch,
			emuBuf{"x", "uint16_t", u16Bytes(x)},
			emuBuf{"w", "uint8_t", w.Packed},
			emuBuf{"w_sf", "uint8_t", w.Scales},
			emuBuf{"w_gs", "float", f32Bytes([]float32{w.GlobalScal})},
			emuBuf{"out", "float", poison(4 * m * base.N)},
		)
		got := bytesF32(res["out"])
		if idx, ok := closeTo(got, want, 1e-5, 1e-6); !ok {
			t.Fatalf("%s out[%d] = %g, want %g", tc.name, idx, got[idx], want[idx])
		}
	}
}

func TestPGORefinePagedAttention(t *testing.T) {
	base := PagedAttention{
		Name:       "pgo_attn",
		Hq:         16,
		Hkv:        2,
		QLen:       1,
		Page:       16,
		Q:          E4M3,
		Out:        F32,
		Chunk:      256,
		Stages:     3,
		TableAhead: 1,
	}
	step := Step{
		Attention: &base,
		Batch:     2,
		MaxKeys:   512,
	}
	obs := KernelObservation{
		Name:               base.Name,
		SharedBankConflict: 0.18,
		MemoryBoundKV:      true,
		GridBlocks:         160,
		DeviceSMs:          132, // 1.212 waves -> BottleneckWaveTail
	}
	refined, reasons := RefineStep(step, obs)
	got := *refined.Attention
	if !got.Swizzle {
		t.Fatalf("expected Swizzle=true, got %+v (reasons: %v)", got, reasons)
	}
	if got.TableAhead < 2 {
		t.Fatalf("expected TableAhead >= 2, got %d (reasons: %v)", got.TableAhead, reasons)
	}
	if !got.WGMMA || !got.QKFP8 {
		t.Fatalf("expected WGMMA=true and QKFP8=true for G=8 E4M3, got %+v", got)
	}
	if got.Chunk != 128 && got.Chunk != 64 {
		t.Fatalf("expected Chunk stepped down to 128 or 64, got %d", got.Chunk)
	}

	// Verify both base and refined PagedAttention compile and execute accurately via cudahostemu.Run.
	r := rand.New(rand.NewPCG(901, 902))
	seqlens := []int32{300, 512}
	q, cache := pagedCase(r, got, seqlens, 0.25)
	want, err := got.Reference(q, cache)
	if err != nil {
		t.Fatal(err)
	}
	baseOut, _, _ := runPaged(t, base, q, cache, step.MaxKeys, 1, 16)
	refOut, _, _ := runPaged(t, got, q, cache, step.MaxKeys, 1, 16)
	for i := range want {
		if diff := absDiff(refOut[i], float64(want[i])); diff > 5e-4 {
			t.Fatalf("refined PagedAttention out[%d] = %g, want %g (diff %g)", i, refOut[i], want[i], diff)
		}
		if diff := absDiff(refOut[i], baseOut[i]); diff > 5e-4 {
			t.Fatalf("refined vs base PagedAttention out[%d] = %g vs %g (diff %g)", i, refOut[i], baseOut[i], diff)
		}
	}
}

func TestPGORefinePlanMegakernel(t *testing.T) {
	const (
		tokens = 2
		kDim   = 256
		nOut   = 256
	)
	b := NewBuilder()
	x := b.Input("x", BF16, tokens, kDim)
	w := b.Input("w", BF16, nOut, kDim)
	qkv := b.MatMul(x, w)
	q := b.Reshape(b.Slice(qkv, 0, 128), tokens, 2, 64)
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), q))
	norm := b.Map(MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), q, ss)
	gate := b.Reshape(b.Slice(qkv, 128, 128), tokens, 2, 64)
	root := b.Map(MustLambda("(mul $0 (sigmoid $1))"), norm, gate)

	expr, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(expr, "pgo_qkv_norm", BF16)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Kernels) != 2 || plan.Kernels[0].GEMM == nil || plan.Kernels[1].Row == nil {
		t.Fatalf("unexpected plan kernels: %+v", plan.Kernels)
	}

	receipt := ProfileReceipt{
		Device: "NVIDIA H200",
		Observations: []KernelObservation{
			{
				Name:              plan.Kernels[0].GEMM.Name,
				DurationNs:        12000,
				LaunchGapNs:       500,
				AchievedOccupancy: 0.16,
				Registers:         64,
				MemoryBoundKV:     true,
			},
			{
				Name:        plan.Kernels[1].Row.Name,
				DurationNs:  6000,
				LaunchGapNs: 2100, // 35% launch gap -> BottleneckLaunchOverhead
			},
		},
	}

	refinedPlan, mega, actions, err := RefineFromProfile(plan, receipt)
	if err != nil {
		t.Fatal(err)
	}
	if mega == nil {
		t.Fatalf("expected non-nil MegakernelSource when consumer has BottleneckLaunchOverhead (actions: %v)", actions)
	}
	if mega.Name != "pgo_qkv_norm_mega" {
		t.Fatalf("mega.Name = %q, want %q", mega.Name, "pgo_qkv_norm_mega")
	}
	if refinedPlan.Kernels[0].GEMM.Prefetch <= 0 {
		t.Fatalf("expected refined step 0 SkinnyGEMM Prefetch > 0, got %d", refinedPlan.Kernels[0].GEMM.Prefetch)
	}
	if plan.Kernels[0].GEMM.Prefetch != 0 {
		t.Fatal("original plan was mutated in place")
	}
	if len(actions) < 2 {
		t.Fatalf("expected >= 2 action log entries, got %v", actions)
	}

	// Verify numerical equivalence across 2 epochs between unrefined multi-step Plan and refined Megakernel.
	r := rand.New(rand.NewPCG(1001, 1002))
	envs := []map[string]Value{
		randEnv(r, Tensor{"x", BF16, []int{tokens, kDim}}, Tensor{"w", BF16, []int{nOut, kDim}}),
		randEnv(r, Tensor{"x", BF16, []int{tokens, kDim}}, Tensor{"w", BF16, []int{nOut, kDim}}),
	}
	megaOuts, _, err := runMegakernelEpochs(t, *mega, envs)
	if err != nil {
		t.Fatal(err)
	}
	for ep, env := range envs {
		unrefinedOut := runPlan(t, plan, env)
		refinedPlanOut := runPlan(t, refinedPlan, env)
		agree(t, fmt.Sprintf("epoch_%d_refined_plan", ep+1), BF16, refinedPlanOut, unrefinedOut)
		agree(t, fmt.Sprintf("epoch_%d_megakernel", ep+1), BF16, megaOuts[ep], unrefinedOut)
	}
}

func absDiff(a, b float64) float64 {
	if a > b {
		return a - b
	}
	return b - a
}

func TestPGORefineProposeStepCandidates(t *testing.T) {
	baseTile := TileGEMM{
		Name:   "pgo_tile_cand",
		N:      128,
		K:      128,
		Weight: WeightBF16,
		Out:    F32,
		BM:     128,
		BN:     128,
		Stages: 3,
		WGMMA:  true,
	}
	step := Step{Tile: &baseTile, M: 128}

	fn := ptxasinfo.Function{
		Name:      baseTile.Name,
		Registers: 128,
	}
	causal := ObservationFromCountersAndPTXAS(fn, CounterMetrics{
		DurationNs:  15000,
		LaunchGapNs: 500,
	}, 264, 132)
	if causal.SharedBankConflict != 0 {
		t.Fatalf("expected unmeasured SharedBankConflict == 0, got %v", causal.SharedBankConflict)
	}
	if bns := ClassifyBottlenecks(causal); len(bns) != 0 {
		t.Fatalf("expected no causal bottlenecks, got %v", bns)
	}

	stalls := StallSample{
		TotalSamples: 100,
		MIOThrottle:  35, // 0.35 -> BottleneckBankConflict
		WaitTMEM:     20, // 0.20 -> BottleneckLowOccupancy
	}
	proposals := ProposeStepCandidates(step, causal, stalls)
	if causal.SharedBankConflict != 0 {
		t.Fatalf("ProposeStepCandidates mutated causal observation: %+v", causal)
	}
	if len(proposals) != 2 {
		t.Fatalf("expected 2 stall-hypothesis proposals, got %d: %+v", len(proposals), proposals)
	}
	if proposals[0].Causal || proposals[0].Hypothesis != BottleneckBankConflict || !proposals[0].Step.Tile.Swizzle128B {
		t.Fatalf("unexpected first proposal: %+v", proposals[0])
	}
	if proposals[1].Causal || proposals[1].Hypothesis != BottleneckLowOccupancy || proposals[1].Step.Tile.BN != 64 {
		t.Fatalf("unexpected second proposal: %+v", proposals[1])
	}

	// When a causal counter is measured (SharedBankConflict = 0.25), the causal
	// proposal appears first with Causal: true, and duplicate stall hypotheses
	// producing the identical schedule are deduplicated.
	bank := 0.25
	causalWithBank := ObservationFromCountersAndPTXAS(fn, CounterMetrics{
		SharedBankConflict: &bank,
	}, 264, 132)
	proposalsWithCausal := ProposeStepCandidates(step, causalWithBank, stalls)
	if len(proposalsWithCausal) != 2 {
		t.Fatalf("expected 1 causal + 1 distinct stall proposal, got %d: %+v", len(proposalsWithCausal), proposalsWithCausal)
	}
	if !proposalsWithCausal[0].Causal || !proposalsWithCausal[0].Step.Tile.Swizzle128B {
		t.Fatalf("expected first proposal to be Causal: true Swizzle128B, got %+v", proposalsWithCausal[0])
	}
	if proposalsWithCausal[1].Causal || proposalsWithCausal[1].Hypothesis != BottleneckLowOccupancy {
		t.Fatalf("expected second proposal to be non-causal BottleneckLowOccupancy, got %+v", proposalsWithCausal[1])
	}
}

func ExampleRefinePlan() {
	b := NewBuilder()
	x := b.Input("x", BF16, 2, 256)
	w := b.Input("w", BF16, 64, 256)
	y := b.MatMul(x, w)
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), y))
	root := b.Map(MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), y, ss)
	expr, _ := Optimize(b.G, root)
	plan, _ := Compile(expr, "proj_norm", BF16)

	receipt := ProfileReceipt{
		Device: "NVIDIA H200",
		Observations: []KernelObservation{
			{Name: plan.Kernels[0].GEMM.Name, MemoryBoundKV: true},
			{Name: plan.Kernels[1].Row.Name, DurationNs: 5000, LaunchGapNs: 1500},
		},
	}
	refined, mega, actions, _ := RefinePlan(plan, receipt)
	fmt.Println(refined.Kernels[0].GEMM.Prefetch > 0, mega != nil, len(actions), strings.HasSuffix(mega.Name, "_mega"))
	// Output:
	// true true 2 true
}
