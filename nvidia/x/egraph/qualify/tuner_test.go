package qualify

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

func TestFilterVariantsByPTXAS(t *testing.T) {
	vClean := Variant{
		Name: "proj_wg_n128s2",
		Op:   "tile_gemm",
		Launch: Launch{
			Block:       256,
			Warps:       8,
			Stages:      2,
			BM:          128,
			BN:          128,
			SharedBytes: 64 * 1024,
			MMA:         "wgmma",
		},
	}
	vSpill := Variant{
		Name:   "proj_wg_n128s3",
		Op:     "tile_gemm",
		Launch: Launch{Block: 256, Warps: 8, Stages: 3, BM: 128, BN: 128, SharedBytes: 96 * 1024, MMA: "wgmma"},
	}
	vSerial := Variant{
		Name:   "proj_wg_n64s2",
		Op:     "tile_gemm",
		Launch: Launch{Block: 256, Warps: 8, Stages: 2, BM: 128, BN: 64, SharedBytes: 48 * 1024, MMA: "wgmma"},
	}
	vHighReg := Variant{
		Name:   "proj_wg_n256s2",
		Op:     "tile_gemm",
		Launch: Launch{Block: 256, Warps: 8, Stages: 2, BM: 128, BN: 256, SharedBytes: 96 * 1024, MMA: "wgmma"},
	}
	vLowOccSmem := Variant{
		Name:   "proj_wg_n256s4",
		Op:     "tile_gemm",
		Launch: Launch{Block: 256, Warps: 8, Stages: 4, BM: 128, BN: 256, SharedBytes: 192 * 1024, MMA: "wgmma"},
	}

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
	kept, rejected := FilterVariantsByPTXAS(candidates, *rep, "sm_90a", StaticGate{NoSpills: true, MaxRegisters: 168, MinCTAsPerSM: 2})
	if len(kept) != 1 || kept[0].Name != "proj_wg_n128s2" {
		t.Fatalf("kept = %+v, want [proj_wg_n128s2]", kept)
	}
	if len(rejected) != 4 {
		t.Fatalf("len(rejected) = %d, want 4", len(rejected))
	}
}

func TestRankVariantsAndSelectByPairedTiming(t *testing.T) {
	cfg := CostConfig{Device: "NVIDIA H200"}
	v256 := Variant{
		Name:   "attn_kv256",
		Op:     "paged_attention",
		Key:    "paged_attention/Hq=16/Hkv=2/qlen=1/window=0/page=1/q=e4m3/out=bf16/qkfp8/gate=2304:1280:1e-05",
		Launch: Launch{Block: 128, Warps: 4, Stages: 4, SplitKV: 256, KVDepth: 1, Swizzle: true},
	}
	v320 := Variant{
		Name:   "attn_kv320",
		Op:     "paged_attention",
		Key:    "paged_attention/Hq=16/Hkv=2/qlen=1/window=0/page=1/q=e4m3/out=bf16/qkfp8/gate=2304:1280:1e-05",
		Launch: Launch{Block: 128, Warps: 4, Stages: 4, SplitKV: 320, KVDepth: 1, Swizzle: true},
	}
	ranked := RankVariants([]Variant{v256, v320}, TuningShape{Batch: 1, MaxKeys: 80000}, cfg)
	if len(ranked) != 2 || ranked[0].Variant.Name != "attn_kv320" {
		t.Fatalf("ranked = %+v, want attn_kv320 first", ranked)
	}
	if ranked[0].Waves != 1 || ranked[1].Waves != 2 || math.Abs(ranked[0].FractionalWaves-500.0/528.0) > 1e-6 {
		t.Fatalf("unexpected wave metrics: %+v vs %+v", ranked[0], ranked[1])
	}

	timings := map[string][]pairedtiming.Block{
		"attn_kv320": {
			{A1: 115.0, B1: 100.0, B2: 100.0, A2: 115.0},
			{A1: 115.0, B1: 100.5, B2: 99.5, A2: 115.0},
		},
	}
	win, sum, err := SelectByPairedTiming([]Variant{v256, v320}, timings, 1.0)
	if err != nil {
		t.Fatalf("SelectByPairedTiming: %v", err)
	}
	if win.Name != "attn_kv320" || sum.Ratio < 1.14 {
		t.Fatalf("winner = %s (ratio %.4f), want attn_kv320", win.Name, sum.Ratio)
	}
}

func TestBuildShapeDispatcher(t *testing.T) {
	v1 := Variant{
		Name:   "proj_m1",
		Op:     "skinny_gemm",
		Key:    "gemm/N=64/K=512/w=bf16/out=f32/epi=$0",
		Entry:  "proj_m1_launch",
		Params: []string{"int m", "const uint16_t* x", "const uint16_t* w", "float* out"},
		Source: "inline void proj_m1_launch(int m, const uint16_t* x, const uint16_t* w, float* out, cudaStream_t stream) {}\n",
	}
	v16 := Variant{
		Name:   "proj_m16",
		Op:     "tile_gemm",
		Key:    "gemm/N=64/K=512/w=bf16/out=f32/epi=$0",
		Entry:  "proj_m16_launch",
		Params: []string{"int m", "const uint16_t* x", "const uint16_t* w", "float* out"},
		Source: "inline void proj_m16_launch(int m, const uint16_t* x, const uint16_t* w, float* out, cudaStream_t stream) {}\n",
	}
	src, err := BuildShapeDispatcher("proj_dispatch", []int{16, 1}, map[int]Variant{1: v1, 16: v16})
	if err != nil {
		t.Fatalf("BuildShapeDispatcher: %v", err)
	}
	for _, want := range []string{
		"if (m <= 1) { proj_m1_launch(m, x, w, out, stream); return; }",
		"proj_m16_launch(m, x, w, out, stream); return;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("dispatcher missing %q:\n%s", want, src)
		}
	}
	_ = fmt.Sprint(src)
}
