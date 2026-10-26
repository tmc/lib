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

	t.Run("blackwell_gb10_nvfp4_tile_and_skinny", func(t *testing.T) {
		gb10Cfg := CostConfig{Device: "NVIDIA GB10"}
		r := rand.New(rand.NewPCG(201, 202))
		nvfp4Env := func(m, n, k int) map[string]Value {
			env := randEnv(r,
				Tensor{"x", BF16, []int{m, k}},
				Tensor{"bias", F32, []int{n}})
			w := randNVFP4(r, n, k)
			packed := make([]float64, len(w.Packed))
			for i, b := range w.Packed {
				packed[i] = float64(b)
			}
			scales := make([]float64, len(w.Scales))
			for i := range w.Scales {
				scales[i] = float64(E4M3Bits(float32(0.25 + r.Float64()*2.0)))
			}
			env["wq"] = Value{Shape: []int{n, k / 2}, Data: packed}
			env["wsf"] = Value{Shape: []int{n, k / 16}, Data: scales}
			env["wgs"] = Value{Shape: []int{1}, Data: []float64{float64(w.GlobalScal)}}
			return env
		}

		// 1. TileGEMM (M = 128) on NVIDIA GB10 selects BlackwellFP4: true.
		b := NewBuilder()
		x := b.Input("x", BF16, 128, 128)
		wq := b.Input("wq", U8, 128, 64)
		wsf := b.Input("wsf", U8, 128, 8)
		wgs := b.Input("wgs", F32, 1)
		w := b.NVFP4(wq, wsf, wgs)
		bias := b.Input("bias", F32, 128)
		root := b.Map(MustLambda("(add $0 $1)"), b.MatMul(x, w), bias)

		expr, err := Optimize(b.G, root)
		if err != nil {
			t.Fatal(err)
		}
		plan, err := CompileWithCost(expr, "gb10_tile", F32, gb10Cfg)
		if err != nil {
			t.Fatal(err)
		}
		if len(plan.Kernels) != 1 || plan.Kernels[0].Tile == nil {
			t.Fatalf("gb10 tile plan kernels = %+v, want 1 TileGEMM", plan.Kernels)
		}
		tk := plan.Kernels[0].Tile
		if !tk.BlackwellFP4 || tk.WGMMA || tk.GroupM != 8 {
			t.Fatalf("gb10 TileGEMM schedule = {BlackwellFP4:%t WGMMA:%t GroupM:%d}, want {true false 8}",
				tk.BlackwellFP4, tk.WGMMA, tk.GroupM)
		}
		env := nvfp4Env(128, 128, 128)
		want, err := Eval(expr, env)
		if err != nil {
			t.Fatal(err)
		}
		got := runPlan(t, plan, env)
		agree(t, "gb10_tile", F32, got, want.Data)

		// 2. SkinnyGEMM (M = 4) on NVIDIA GB10 selects BlackwellFP4: true and Prefetch >= 2.
		b2 := NewBuilder()
		xs := b2.Input("x", BF16, 4, 2048)
		wqs := b2.Input("wq", U8, 16, 1024)
		wsfs := b2.Input("wsf", U8, 16, 128)
		wgss := b2.Input("wgs", F32, 1)
		ws := b2.NVFP4(wqs, wsfs, wgss)
		expr2, err := Optimize(b2.G, b2.MatMul(xs, ws))
		if err != nil {
			t.Fatal(err)
		}
		plan2, err := CompileWithCost(expr2, "gb10_skinny", F32, gb10Cfg)
		if err != nil {
			t.Fatal(err)
		}
		if len(plan2.Kernels) != 1 || plan2.Kernels[0].GEMM == nil {
			t.Fatalf("gb10 skinny plan kernels = %+v, want 1 SkinnyGEMM", plan2.Kernels)
		}
		sk := plan2.Kernels[0].GEMM
		if !sk.BlackwellFP4 || sk.Prefetch < 2 {
			t.Fatalf("gb10 SkinnyGEMM schedule = {BlackwellFP4:%t Prefetch:%d}, want {true >=2}",
				sk.BlackwellFP4, sk.Prefetch)
		}
		env2 := nvfp4Env(4, 16, 2048)
		want2, err := Eval(expr2, env2)
		if err != nil {
			t.Fatal(err)
		}
		got2 := runPlan(t, plan2, env2)
		agree(t, "gb10_skinny", F32, got2, want2.Data)
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
