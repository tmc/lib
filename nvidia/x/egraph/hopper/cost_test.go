package hopper

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestCompileWithCostSchedules(t *testing.T) {
	cfg := CostConfig{Device: "NVIDIA H200"}

	t.Run("skinny_gemm_prefetch_m4", func(t *testing.T) {
		b := NewBuilder()
		x := b.Input("x", BF16, 4, 2048)
		w := b.Input("w", BF16, 256, 2048)
		bias := b.Input("bias", F32, 256)
		mm := b.MatMul(x, w)
		root := b.Map(MustLambda("(silu (add $0 $1))"), mm, bias)

		// Extract unoptimized tree before Optimize (MatMul + Map = 2 kernels).
		unoptExpr, _, err := egraph.ExtractBest(b.G, root, egraph.AstSizeCost{})
		if err != nil {
			t.Fatal(err)
		}
		unoptPlan, err := Compile(unoptExpr, "skinny_unopt", BF16)
		if err != nil {
			t.Fatal(err)
		}

		optExpr, err := Optimize(b.G, root)
		if err != nil {
			t.Fatal(err)
		}
		optPlan, err := CompileWithCost(optExpr, "skinny_opt", BF16, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if len(optPlan.Kernels) != 1 || optPlan.Kernels[0].GEMM == nil {
			t.Fatalf("plan kernels = %+v, want 1 SkinnyGEMM", optPlan.Kernels)
		}
		if optPlan.Kernels[0].GEMM.Prefetch < 2 {
			t.Errorf("SkinnyGEMM.Prefetch = %d, want >= 2 for M=4 on H200", optPlan.Kernels[0].GEMM.Prefetch)
		}
		if optNs, unoptNs := optPlan.EstimateNs(cfg), unoptPlan.EstimateNs(cfg); optNs >= unoptNs {
			t.Errorf("optimized EstimateNs %.1f >= unoptimized EstimateNs %.1f", optNs, unoptNs)
		}

		r := rand.New(rand.NewPCG(101, 102))
		env := randEnv(r,
			Tensor{"x", BF16, []int{4, 2048}},
			Tensor{"w", BF16, []int{256, 2048}},
			Tensor{"bias", F32, []int{256}})
		want, err := Eval(optExpr, env)
		if err != nil {
			t.Fatal(err)
		}
		got := runPlan(t, optPlan, env)
		agree(t, "skinny_opt", BF16, got, roundAll(BF16, want.Data))
	})

	t.Run("tile_gemm_wgmma_swizzle128b_groupm_m128", func(t *testing.T) {
		b := NewBuilder()
		x := b.Input("x", BF16, 128, 128)
		w := b.Input("w", BF16, 128, 128)
		resid := b.Input("resid", F32, 128, 128)
		mm := b.MatMul(x, w)
		root := b.Map(MustLambda("(add $0 $1)"), mm, resid)

		unoptExpr, _, err := egraph.ExtractBest(b.G, root, egraph.AstSizeCost{})
		if err != nil {
			t.Fatal(err)
		}
		unoptPlan, err := Compile(unoptExpr, "tile_unopt", F32)
		if err != nil {
			t.Fatal(err)
		}

		optExpr, err := Optimize(b.G, root)
		if err != nil {
			t.Fatal(err)
		}
		optPlan, err := CompileWithCost(optExpr, "tile_opt", F32, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if len(optPlan.Kernels) != 1 || optPlan.Kernels[0].Tile == nil {
			t.Fatalf("plan kernels = %+v, want 1 TileGEMM", optPlan.Kernels)
		}
		tk := optPlan.Kernels[0].Tile
		if !tk.WGMMA || !tk.Swizzle128B || tk.GroupM != 8 {
			t.Errorf("TileGEMM schedule = {WGMMA:%t Swizzle128B:%t GroupM:%d}, want {true true 8}",
				tk.WGMMA, tk.Swizzle128B, tk.GroupM)
		}
		if optNs, unoptNs := optPlan.EstimateNs(cfg), unoptPlan.EstimateNs(cfg); optNs >= unoptNs {
			t.Errorf("optimized EstimateNs %.1f >= unoptimized EstimateNs %.1f", optNs, unoptNs)
		}

		r := rand.New(rand.NewPCG(103, 104))
		env := randEnv(r,
			Tensor{"x", BF16, []int{128, 128}},
			Tensor{"w", BF16, []int{128, 128}},
			Tensor{"resid", F32, []int{128, 128}})
		want, err := Eval(optExpr, env)
		if err != nil {
			t.Fatal(err)
		}
		got := runPlan(t, optPlan, env)
		agree(t, "tile_opt", F32, got, want.Data)
	})

	t.Run("row_kernel_cluster_cols4096", func(t *testing.T) {
		const rows, cols = 4, 4096
		b := NewBuilder()
		x := b.Input("x", BF16, rows, cols)
		w := b.Input("w", BF16, cols)
		ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), x))
		norm := b.Map(MustLambda("(mul $0 (rsqrt (add (div $1 4096) 1e-5)))"), x, ss)
		root := b.Map(MustLambda("(mul $0 $1)"), norm, w)

		expr, err := Optimize(b.G, root)
		if err != nil {
			t.Fatal(err)
		}
		basePlan, err := Compile(expr, "row_base", BF16)
		if err != nil {
			t.Fatal(err)
		}
		optPlan, err := CompileWithCost(expr, "row_opt", BF16, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if len(optPlan.Kernels) != 1 || optPlan.Kernels[0].Row == nil {
			t.Fatalf("plan kernels = %+v, want 1 RowKernel", optPlan.Kernels)
		}
		if !optPlan.Kernels[0].Row.Cluster {
			t.Errorf("RowKernel.Cluster = false, want true for Cols=%d", cols)
		}
		if optNs, baseNs := optPlan.EstimateNs(cfg), basePlan.EstimateNs(cfg); optNs >= baseNs {
			t.Errorf("clustered EstimateNs %.1f >= base EstimateNs %.1f", optNs, baseNs)
		}

		r := rand.New(rand.NewPCG(105, 106))
		env := randEnv(r,
			Tensor{"x", BF16, []int{rows, cols}},
			Tensor{"w", BF16, []int{cols}})
		want, err := Eval(expr, env)
		if err != nil {
			t.Fatal(err)
		}
		got := runPlan(t, optPlan, env)
		agree(t, "row_opt", BF16, got, roundAll(BF16, want.Data))
	})

	t.Run("paged_attention_swizzle_table_ahead", func(t *testing.T) {
		b := NewBuilder()
		q := b.Input("q", E4M3, 2, 16, HeadDim)
		kCache := b.Input("k_cache", U8, 64, 2, HeadDim)
		vCache := b.Input("v_cache", U8, 64, 2, HeadDim)
		pageTable := b.Input("page_table", I32, 2, 32)
		seqLens := b.Input("seqlens", I32, 2)
		kScale := b.Input("k_scale", F32, 1)
		vScale := b.Input("v_scale", F32, 1)
		gate := b.Input("gate", BF16, 2, 16, HeadDim)

		attn := b.PagedAttention(PagedAttention{
			Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Chunk: 64,
		}, q, kCache, vCache, pageTable, seqLens, kScale, vScale)
		root := b.RMSNormGate(attn, gate, 1e-5)

		unfusedExpr, _, err := egraph.ExtractBest(b.G, root, Cost{})
		if err != nil {
			t.Fatal(err)
		}
		unfusedPlan, err := Compile(unfusedExpr, "attn_unopt", BF16)
		if err != nil {
			t.Fatal(err)
		}

		fusedExpr, err := Optimize(b.G, root)
		if err != nil {
			t.Fatal(err)
		}
		// 32 pages * 16 = 512 maxKeys -> 512 / 64 = 8 splits (multiple of 8 -> ClusterCombine).
		optPlan, err := CompileWithCost(fusedExpr, "attn_opt", BF16, cfg)
		if err != nil {
			t.Fatal(err)
		}
		ak := optPlan.Kernels[0].Attention
		if ak == nil || !ak.Gate || !ak.Swizzle || ak.TableAhead != ak.Stages-1 || !ak.ClusterCombine || ak.WGMMA {
			t.Fatalf("PagedAttention schedule = %+v, want Gate, Swizzle, TableAhead=Stages-1, ClusterCombine, WGMMA=false", ak)
		}
		if optNs, unoptNs := optPlan.EstimateNs(cfg), unfusedPlan.EstimateNs(cfg); optNs >= unoptNs {
			t.Errorf("optimized EstimateNs %.1f >= unoptimized EstimateNs %.1f", optNs, unoptNs)
		}
	})

	t.Run("paged_attention_wgmma_qlen8", func(t *testing.T) {
		b := NewBuilder()
		q := b.Input("q", E4M3, 2, 16, 8, HeadDim)
		kCache := b.Input("k_cache", U8, 64, 2, HeadDim)
		vCache := b.Input("v_cache", U8, 64, 2, HeadDim)
		pageTable := b.Input("page_table", I32, 2, 32)
		seqLens := b.Input("seqlens", I32, 2)
		kScale := b.Input("k_scale", F32, 1)
		vScale := b.Input("v_scale", F32, 1)
		gate := b.Input("gate", BF16, 2, 16, 8, HeadDim)

		attn := b.PagedAttention(PagedAttention{
			Hq: 16, Hkv: 2, QLen: 8, Page: 16, Q: E4M3, Chunk: 64,
		}, q, kCache, vCache, pageTable, seqLens, kScale, vScale)
		root := b.RMSNormGate(attn, gate, 1e-5)

		unoptExpr, _, err := egraph.ExtractBest(b.G, root, Cost{})
		if err != nil {
			t.Fatal(err)
		}
		unoptPlan, err := Compile(unoptExpr, "attn_wg_unopt", BF16)
		if err != nil {
			t.Fatal(err)
		}

		fusedExpr, err := Optimize(b.G, root)
		if err != nil {
			t.Fatal(err)
		}
		optPlan, err := CompileWithCost(fusedExpr, "attn_wg_opt", BF16, cfg)
		if err != nil {
			t.Fatal(err)
		}
		ak := optPlan.Kernels[0].Attention
		if ak == nil || !ak.WGMMA || !ak.QKFP8 || !ak.Gate || !ak.Swizzle || ak.TableAhead != ak.Stages-1 || !ak.ClusterCombine {
			t.Fatalf("PagedAttention schedule = %+v, want WGMMA, QKFP8, Gate, Swizzle, TableAhead=Stages-1, ClusterCombine", ak)
		}
		if optNs, unoptNs := optPlan.EstimateNs(cfg), unoptPlan.EstimateNs(cfg); optNs >= unoptNs {
			t.Errorf("optimized EstimateNs %.1f >= unoptimized EstimateNs %.1f", optNs, unoptNs)
		}
	})
}

func ExampleCompileWithCost() {
	b := NewBuilder()
	x := b.Input("x", BF16, 128, 256)
	w := b.Input("w", BF16, 128, 256)
	bias := b.Input("bias", F32, 128)
	root := b.Map(MustLambda("(silu (add $0 $1))"), b.MatMul(x, w), bias)

	expr, _ := Optimize(b.G, root)
	plan, _ := CompileWithCost(expr, "proj", BF16, CostConfig{Device: "NVIDIA H200"})
	tk := plan.Kernels[0].Tile
	fmt.Println(tk.WGMMA, tk.Swizzle128B, tk.GroupM)
	// Output: true true 8
}
