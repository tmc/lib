package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

func TestFilterVariantsByPTXAS(t *testing.T) {
	swept, err := SweepTileGEMM(TileGEMM{Name: "proj", N: 256, K: 256, Weight: WeightBF16, Out: BF16})
	if err != nil {
		t.Fatal(err)
	}
	byName := make(map[string]Variant, len(swept))
	for _, v := range swept {
		byName[v.Name] = v
	}
	vClean := byName["proj_wg_n128s2"]      // 256 threads, 64 KiB dynamic smem
	vSpill := byName["proj_wg_n128s3"]      // spills registers
	vSerial := byName["proj_wg_n64s2"]      // C7519 wgmma serialized
	vHighReg := byName["proj_wg_n256s2"]    // 220 registers (> MaxRegisters 168 and 1 CTA/SM)
	vLowOccSmem := byName["proj_wg_n256s4"] // 192 KiB dynamic smem -> 1 CTA/SM (< MinCTAsPerSM 2)

	const ptxasLog = `
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z14proj_wg_n128s2iPKtS0_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z14proj_wg_n128s2iPKtS0_Pt
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 120 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14proj_wg_n128s3iPKtS0_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z14proj_wg_n128s3iPKtS0_Pt
    32 bytes stack frame, 24 bytes spill stores, 24 bytes spill loads
ptxas info    : Used 120 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : (C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path in the function '_Z13proj_wg_n64s2iPKtS0_Pt' in line 95
ptxas info    : Compiling entry function '_Z13proj_wg_n64s2iPKtS0_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z13proj_wg_n64s2iPKtS0_Pt
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 96 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14proj_wg_n256s2iPKtS0_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z14proj_wg_n256s2iPKtS0_Pt
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 220 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14proj_wg_n256s4iPKtS0_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z14proj_wg_n256s4iPKtS0_Pt
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 120 registers, 1024 bytes smem, 384 bytes cmem[0]
`
	rep, err := ptxasinfo.Parse(strings.NewReader(ptxasLog))
	if err != nil {
		t.Fatal(err)
	}

	candidates := []Variant{vClean, vSpill, vSerial, vHighReg, vLowOccSmem}

	tests := []struct {
		name         string
		gate         StaticGate
		wantKept     []string
		wantRejected map[string]string
	}{
		{
			name:     "strict_2cta_per_sm_168_regs",
			gate:     StaticGate{NoSpills: true, MaxRegisters: 168, MinCTAsPerSM: 2},
			wantKept: []string{"proj_wg_n128s2"},
			wantRejected: map[string]string{
				"proj_wg_n128s3": "spills registers",
				"proj_wg_n64s2":  "C7519",
				"proj_wg_n256s2": "registers 220 > max 168",
				"proj_wg_n256s4": "resident CTAs/SM 1 < min 2",
			},
		},
		{
			name:     "single_cta_per_sm_allows_large_smem",
			gate:     StaticGate{NoSpills: true, MaxRegisters: 255, MinCTAsPerSM: 1},
			wantKept: []string{"proj_wg_n128s2", "proj_wg_n256s2", "proj_wg_n256s4"},
			wantRejected: map[string]string{
				"proj_wg_n128s3": "spills registers",
				"proj_wg_n64s2":  "C7519",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kept, rejected := FilterVariantsByPTXAS(candidates, *rep, "sm_90a", tt.gate)
			var gotKept []string
			for _, v := range kept {
				gotKept = append(gotKept, v.Name)
				if v.Launch.Registers == 0 {
					t.Errorf("kept variant %s missing attached ptxas Registers", v.Name)
				}
			}
			if fmt.Sprint(gotKept) != fmt.Sprint(tt.wantKept) {
				t.Errorf("kept = %v, want %v", gotKept, tt.wantKept)
			}
			if len(rejected) != len(tt.wantRejected) {
				t.Fatalf("len(rejected) = %d, want %d: %+v", len(rejected), len(tt.wantRejected), rejected)
			}
			for _, r := range rejected {
				wantSub, ok := tt.wantRejected[r.Variant.Name]
				if !ok {
					t.Errorf("unexpected rejected variant %s (%s)", r.Variant.Name, r.Reason())
					continue
				}
				if !strings.Contains(r.Reason(), wantSub) {
					t.Errorf("rejected %s reason = %q, want substring %q", r.Variant.Name, r.Reason(), wantSub)
				}
			}
		})
	}
}

func TestRankVariantsWaveQuantized(t *testing.T) {
	cfg := CostConfig{Device: "NVIDIA H200"} // 132 SMs

	t.Run("paged_attention_c320_1wave_beats_c256_1_19wave_tail", func(t *testing.T) {
		base := PagedAttention{
			Name: "attn", Hq: 16, Hkv: 2, QLen: 1, Page: 1,
			Q: E4M3, Out: BF16, QKFP8: true, Swizzle: true,
			Gate: true, GateStride: 2304, GateOffset: 1280,
		}
		vs, err := SweepPagedAttention(base)
		if err != nil {
			t.Fatal(err)
		}
		// At Batch=1, Hkv=2, MaxKeys=80000 on 132 SMs (4 CTAs/SM -> 528 slots):
		//   C=256 launches 2 * ceil(80000/256) = 626 CTAs -> 626/528 = 1.1856 (~1.19) waves (2 quantized waves).
		//   C=320 launches 2 * ceil(80000/320) = 500 CTAs -> 500/528 = 0.9470 waves (1 quantized wave).
		ranked := RankVariants(vs, TuningShape{Batch: 1, MaxKeys: 80000}, cfg)
		if len(ranked) == 0 {
			t.Fatal("no ranked variants")
		}
		byName := make(map[string]RankedVariant, len(ranked))
		for _, r := range ranked {
			byName[r.Variant.Name] = r
		}
		r256 := byName["attn_kv256"]
		r320 := byName["attn_kv320"]

		if r256.NumCTAs != 626 || r256.Waves != 2 || math.Abs(r256.FractionalWaves-626.0/528.0) > 1e-6 {
			t.Errorf("kv256 = {NumCTAs:%d Waves:%d FractionalWaves:%.4f}, want {626 2 1.1856}",
				r256.NumCTAs, r256.Waves, r256.FractionalWaves)
		}
		if r320.NumCTAs != 500 || r320.Waves != 1 || math.Abs(r320.FractionalWaves-500.0/528.0) > 1e-6 {
			t.Errorf("kv320 = {NumCTAs:%d Waves:%d FractionalWaves:%.4f}, want {500 1 0.9470}",
				r320.NumCTAs, r320.Waves, r320.FractionalWaves)
		}
		if r320.EstimatedNs >= r256.EstimatedNs {
			t.Errorf("kv320 EstimatedNs %.1f >= kv256 EstimatedNs %.1f", r320.EstimatedNs, r256.EstimatedNs)
		}
		if ranked[0].Variant.Launch.SplitKV != 320 {
			t.Errorf("top ranked variant = %s (split_kv=%d), want split_kv=320",
				ranked[0].Variant.Name, ranked[0].Variant.Launch.SplitKV)
		}
	})

	t.Run("tile_gemm_streamk_beats_1_19wave_data_parallel_tail", func(t *testing.T) {
		// M=1792 (14 tiles of 128), N=2816 (22 tiles of 128) -> 308 tiles.
		// With 2 resident CTAs/SM on 132 SMs (264 slots), 308/264 = 1.167 waves (2 quantized waves).
		// With 4 stages (128 KiB smem -> 1 resident CTA/SM, 132 slots), M=1408 (11 tiles), N=1792 (14 tiles)
		// gives 154 tiles -> 154/132 = 1.167 waves (2 quantized waves), while StreamK=132 runs in 1 wave.
		dp, err := TileGEMM{
			Name: "gemm_dp", N: 1792, K: 2048, Weight: WeightBF16, Out: BF16,
			BM: 128, BN: 128, Stages: 4, WGMMA: true, Swizzle128B: true,
		}.Variant()
		if err != nil {
			t.Fatal(err)
		}
		sk, err := TileGEMM{
			Name: "gemm_sk", N: 1792, K: 2048, Weight: WeightBF16, Out: BF16,
			BM: 128, BN: 128, Stages: 4, WGMMA: true, Swizzle128B: true, StreamK: 132,
		}.Variant()
		if err != nil {
			t.Fatal(err)
		}

		ranked := RankVariants([]Variant{dp, sk}, TuningShape{M: 1408}, cfg)
		if ranked[0].Variant.Name != "gemm_sk" {
			t.Fatalf("winner = %s (%.1f ns), want gemm_sk over gemm_dp (%.1f ns)",
				ranked[0].Variant.Name, ranked[0].EstimatedNs, ranked[1].EstimatedNs)
		}
		if ranked[0].Waves != 1 || ranked[1].Waves != 2 {
			t.Errorf("waves = (sk:%d, dp:%d), want (1, 2)", ranked[0].Waves, ranked[1].Waves)
		}
		if ranked[0].CombineBytes == 0 {
			t.Errorf("StreamK CombineBytes = 0, want > 0")
		}
	})

	t.Run("cluster_combine_reduces_combine_traffic", func(t *testing.T) {
		noCluster, err := PagedAttention{
			Name: "attn_global", Hq: 16, Hkv: 2, QLen: 1, Page: 1,
			Q: E4M3, Out: BF16, Chunk: 256, Swizzle: true,
		}.Variant()
		if err != nil {
			t.Fatal(err)
		}
		cluster, err := PagedAttention{
			Name: "attn_cluster", Hq: 16, Hkv: 2, QLen: 1, Page: 1,
			Q: E4M3, Out: BF16, Chunk: 256, Swizzle: true, ClusterCombine: true,
		}.Variant()
		if err != nil {
			t.Fatal(err)
		}
		ranked := RankVariants([]Variant{noCluster, cluster}, TuningShape{Batch: 2, MaxKeys: 4096}, cfg)
		if ranked[0].Variant.Name != "attn_cluster" {
			t.Errorf("winner = %s, want attn_cluster", ranked[0].Variant.Name)
		}
		if ranked[0].CombineBytes >= ranked[1].CombineBytes {
			t.Errorf("cluster CombineBytes %d >= non-cluster %d", ranked[0].CombineBytes, ranked[1].CombineBytes)
		}
	})
}

func TestTunerSelectByPairedTiming(t *testing.T) {
	vBase := Variant{Name: "v_base"}
	vSub := Variant{Name: "v_sub_threshold"}
	vNoisy := Variant{Name: "v_noisy"}
	vBad := Variant{Name: "v_below_intercept"}
	vFast := Variant{Name: "v_fast"}
	vFaster := Variant{Name: "v_faster"}

	tests := []struct {
		name      string
		variants  []Variant
		timings   map[string][]pairedtiming.Block
		intercept float64
		wantName  string
		wantErr   bool
	}{
		{
			name:     "promotes_clean_winner_rejects_noisy_and_subthreshold",
			variants: []Variant{vBase, vSub, vNoisy, vBad, vFast, vFaster},
			intercept: 2.0,
			timings: map[string][]pairedtiming.Block{
				// 0.8% gain (< 1.015 threshold) -> rejected.
				"v_sub_threshold": {
					{A1: 102.0, B1: 101.2, B2: 101.2, A2: 102.0},
					{A1: 102.0, B1: 101.2, B2: 101.2, A2: 102.0},
					{A1: 102.0, B1: 101.2, B2: 101.2, A2: 102.0},
				},
				// High median from one outlier block, but loses in two blocks (MinRatio < 1.0) -> rejected.
				"v_noisy": {
					{A1: 102.0, B1: 60.0, B2: 60.0, A2: 102.0},
					{A1: 102.0, B1: 110.0, B2: 110.0, A2: 102.0},
					{A1: 102.0, B1: 108.0, B2: 108.0, A2: 102.0},
				},
				// Sample below intercept -> rejected by pairedtiming.Analyze.
				"v_below_intercept": {
					{A1: 102.0, B1: 1.5, B2: 90.0, A2: 102.0},
				},
				// Clean 1.08x speedup across all blocks -> valid.
				"v_fast": {
					{A1: 110.0, B1: 102.0, B2: 102.0, A2: 110.0},
					{A1: 110.0, B1: 101.8, B2: 102.2, A2: 110.0},
					{A1: 110.0, B1: 102.0, B2: 102.0, A2: 110.0},
				},
				// Clean 1.15x speedup across all blocks -> promoted as overall winner.
				"v_faster": {
					{A1: 117.0, B1: 102.0, B2: 102.0, A2: 117.0},
					{A1: 117.0, B1: 101.5, B2: 102.5, A2: 117.0},
					{A1: 117.0, B1: 102.0, B2: 102.0, A2: 117.0},
				},
			},
			wantName: "v_faster",
		},
		{
			name:      "keeps_baseline_when_all_candidates_subthreshold",
			variants:  []Variant{vBase, vSub},
			intercept: 2.0,
			timings: map[string][]pairedtiming.Block{
				"v_sub_threshold": {
					{A1: 102.0, B1: 101.2, B2: 101.2, A2: 102.0},
					{A1: 102.0, B1: 101.3, B2: 101.1, A2: 102.0},
				},
			},
			wantName: "v_base",
		},
		{
			name:      "errors_when_no_baseline_and_subthreshold",
			variants:  []Variant{vSub},
			intercept: 2.0,
			timings: map[string][]pairedtiming.Block{
				"v_sub_threshold": {
					{A1: 102.0, B1: 101.2, B2: 101.2, A2: 102.0},
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, sum, err := SelectByPairedTiming(tt.variants, tt.timings, tt.intercept)
			if (err != nil) != tt.wantErr {
				t.Fatalf("SelectByPairedTiming() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if got.Name != tt.wantName {
				t.Errorf("winner = %q (ratio %.4f), want %q", got.Name, sum.Ratio, tt.wantName)
			}
		})
	}
}

func TestBuildShapeDispatcherEmulated(t *testing.T) {
	const n, k = 64, 512
	vM1, err := SkinnyGEMM{
		Name: "proj_m1", N: n, K: k, MaxM: 4, Weight: WeightBF16, Out: F32,
		RowsPerBlock: 1, Warps: 4, SplitK: 1, Prefetch: 2,
	}.Variant()
	if err != nil {
		t.Fatal(err)
	}
	vM4, err := SkinnyGEMM{
		Name: "proj_m4", N: n, K: k, MaxM: 8, Weight: WeightBF16, Out: F32,
		RowsPerBlock: 2, Warps: 8, SplitK: 2,
	}.Variant()
	if err != nil {
		t.Fatal(err)
	}
	vM16, err := TileGEMM{
		Name: "proj_m16", N: n, K: k, Weight: WeightBF16, Out: F32,
		BM: 64, BN: 64, Stages: 2,
	}.Variant()
	if err != nil {
		t.Fatal(err)
	}

	shapes := []int{16, 1, 4} // intentionally unsorted to test bucket sorting
	winners := map[int]Variant{
		1:  vM1,
		4:  vM4,
		16: vM16,
	}
	src, err := BuildShapeDispatcher("proj_shape_dispatch", shapes, winners)
	if err != nil {
		t.Fatal(err)
	}
	for _, wantSub := range []string{
		"if (m <= 1) { proj_m1_launch(m, x, w, out, stream); return; }",
		"if (m <= 4) { proj_m4_launch(m, x, w, out, ws, stream); return; }",
		"proj_m16_launch(m, x, w, out, stream); return;",
	} {
		if !strings.Contains(src, wantSub) {
			t.Errorf("dispatcher source missing %q:\n%s", wantSub, src)
		}
	}

	r := rand.New(rand.NewPCG(701, 702))
	w := GEMMWeights{BF16: randBF16(r, n*k, 0.05)}
	wsBytes := SkinnyGEMM{
		Name: "s", N: n, K: k, MaxM: 8, Weight: WeightBF16, Out: F32,
		RowsPerBlock: 2, Warps: 8, SplitK: 2,
	}.WorkspaceBytes()

	for _, m := range []int{1, 4, 16} {
		t.Run(fmt.Sprintf("m=%d", m), func(t *testing.T) {
			x := randBF16(r, m*k, 1.0)
			res := runEmu(t, src, fmt.Sprintf("proj_shape_dispatch(%d, x, w, out, ws, nullptr)", m),
				emuBuf{"x", "uint16_t", u16Bytes(x)},
				emuBuf{"w", "uint16_t", u16Bytes(w.BF16)},
				emuBuf{"out", "float", poison(4 * m * n)},
				emuBuf{"ws", "float", make([]byte, wsBytes)})
			want, err := TileGEMM{
				Name: "ref", N: n, K: k, Weight: WeightBF16, Out: F32, BN: 64,
			}.Reference(m, x, nil, nil, []GEMMWeights{w}, nil)
			if err != nil {
				t.Fatal(err)
			}
			got := bytesF32(res["out"])
			if i, ok := closeTo(got, want, 1e-5, 1e-6); !ok {
				t.Fatalf("m=%d: out[%d] = %g, want %g", m, i, got[i], want[i])
			}
		})
	}
}

func TestTunerClosedLoop(t *testing.T) {
	skinny, err := SweepSkinnyGEMM(SkinnyGEMM{Name: "proj_skinny", N: 256, K: 512, MaxM: 8, Weight: WeightBF16, Out: BF16, Prefetch: 1})
	if err != nil {
		t.Fatal(err)
	}
	tile, err := SweepTileGEMM(TileGEMM{Name: "proj_tile", N: 256, K: 512, Weight: WeightBF16, Out: BF16})
	if err != nil {
		t.Fatal(err)
	}
	all := append(skinny, tile...)

	// Simulate ptxas -v log where one skinny variant spills and one wgmma tile variant serializes (C7519).
	log := fmt.Sprintf(`
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z%d%sILi1EEvPKtS1_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z%d%sILi1EEvPKtS1_Pt
    16 bytes stack frame, 16 bytes spill stores, 16 bytes spill loads
ptxas info    : Used 128 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : (C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized in the function '_Z%d%siPKtS0_Pt' in line 80
ptxas info    : Compiling entry function '_Z%d%siPKtS0_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z%d%siPKtS0_Pt
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 128 registers, 1024 bytes smem, 384 bytes cmem[0]
`,
		len(skinny[0].Name), skinny[0].Name,
		len(skinny[0].Name), skinny[0].Name,
		len(tile[0].Name), tile[0].Name,
		len(tile[0].Name), tile[0].Name,
		len(tile[0].Name), tile[0].Name,
	)
	rep, err := ptxasinfo.Parse(strings.NewReader(log))
	if err != nil {
		t.Fatal(err)
	}

	tuner := Tuner{
		Arch:      "sm_90a",
		Gate:      StaticGate{NoSpills: true, MaxRegisters: 255, MinCTAsPerSM: 1},
		Cost:      CostConfig{Device: "NVIDIA H200"},
		MinGain:   DefaultMinGain,
		Intercept: 1.0,
	}
	shapes := []int{1, 8, 32, 128}
	src, winners, rejected, err := tuner.TuneShapes("proj_closedloop", all, *rep, shapes)
	if err != nil {
		t.Fatal(err)
	}
	if len(rejected) != 2 {
		t.Fatalf("len(rejected) = %d, want 2: %+v", len(rejected), rejected)
	}
	if winners[1].Op != "skinny_gemm" {
		t.Errorf("m=1 winner op = %q (%s), want skinny_gemm", winners[1].Op, winners[1].Name)
	}
	if winners[128].Op != "tile_gemm" {
		t.Errorf("m=128 winner op = %q (%s), want tile_gemm", winners[128].Op, winners[128].Name)
	}
	if !strings.Contains(src, "inline void proj_closedloop(") {
		t.Errorf("missing proj_closedloop entry in dispatcher source")
	}
}

func ExampleFilterVariantsByPTXAS() {
	vs, _ := SweepTileGEMM(TileGEMM{Name: "proj", N: 256, K: 256, Weight: WeightBF16, Out: BF16})
	const log = `
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z13proj_wg_n64s2iPKtS0_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z13proj_wg_n64s2iPKtS0_Pt
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 112 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z13proj_wg_n64s3iPKtS0_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z13proj_wg_n64s3iPKtS0_Pt
    16 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads
ptxas info    : Used 128 registers, 1024 bytes smem, 384 bytes cmem[0]
`
	rep, _ := ptxasinfo.Parse(strings.NewReader(log))
	kept, rejected := FilterVariantsByPTXAS(vs[:2], *rep, "sm_90a", StaticGate{NoSpills: true})
	fmt.Println(len(kept), kept[0].Name, len(rejected), rejected[0].Variant.Name)
	// Output: 1 proj_wg_n64s2 1 proj_wg_n64s3
}

func ExampleRankVariants() {
	vs, _ := SweepPagedAttention(PagedAttention{
		Name: "attn", Hq: 16, Hkv: 2, QLen: 1, Page: 1,
		Q: E4M3, Out: BF16, QKFP8: true, Swizzle: true,
	})
	ranked := RankVariants(vs, TuningShape{Batch: 1, MaxKeys: 80000}, CostConfig{Device: "NVIDIA H200"})
	fmt.Println(ranked[0].Variant.Name, ranked[0].Waves, ranked[0].NumCTAs)
	// Output: attn_kv320 1 500
}
