package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func buildPagedAttnEnv(r *rand.Rand, a PagedAttention, seqlens []int32, kvScale float32, gateStride int) map[string]Value {
	q, c := pagedCase(r, a, seqlens, kvScale)
	batch := len(seqlens)
	tokens := batch * a.QLen
	kf := make([]float64, len(c.K))
	for i, b := range c.K {
		kf[i] = float64(b)
	}
	vf := make([]float64, len(c.V))
	for i, b := range c.V {
		vf[i] = float64(b)
	}
	ptf := make([]float64, len(c.PageTable))
	for i, v := range c.PageTable {
		ptf[i] = float64(v)
	}
	slf := make([]float64, len(c.SeqLens))
	for i, v := range c.SeqLens {
		slf[i] = float64(v)
	}
	cacheRows := len(c.K) / (a.Hkv * HeadDim)
	env := map[string]Value{
		"q":          {Shape: []int{tokens, a.Hq, HeadDim}, Data: f64s(q)},
		"k_cache":    {Shape: []int{cacheRows, a.Hkv, HeadDim}, Data: kf},
		"v_cache":    {Shape: []int{cacheRows, a.Hkv, HeadDim}, Data: vf},
		"page_table": {Shape: []int{batch, c.MaxPages}, Data: ptf},
		"seqlens":    {Shape: []int{batch}, Data: slf},
		"k_scale":    {Shape: []int{1}, Data: []float64{float64(c.KScale)}},
		"v_scale":    {Shape: []int{1}, Data: []float64{float64(c.VScale)}},
	}
	if gateStride > 0 {
		g := make([]float64, tokens*gateStride)
		for i := range g {
			g[i] = float64(BF16.Round(float32(r.NormFloat64())))
		}
		if gateStride == a.Hq*HeadDim {
			env["gate"] = Value{Shape: []int{tokens, a.Hq, HeadDim}, Data: g}
		} else {
			env["qkv"] = Value{Shape: []int{tokens, gateStride}, Data: g}
		}
	}
	return env
}

// TestTermPagedAttentionGateFusion verifies that an unfused PagedAttention +
// RowKernel(rms_norm_gate) term graph optimizes into a single
// PagedAttention{Gate: true} launch and matches Eval on the host emulator.
func TestTermPagedAttentionGateFusion(t *testing.T) {
	tests := []struct {
		name        string
		primitiveOp bool // express rms_norm_gate using primitive Map + ReduceSum
		slicedGate  bool // read gate as a slice/reshape of a wider [tokens, 2304] tensor
	}{
		{"rms_norm_gate_term", false, false},
		{"primitive_map_reduce_sliced_gate", true, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const (
				batch, qlen = 2, 2
				tokens      = batch * qlen
				hq, hkv     = 16, 2
			)
			seqlens := []int32{150, 20}
			baseAttn := PagedAttention{
				Hq: hq, Hkv: hkv, QLen: qlen, Page: 8,
				Q: E4M3, QScale: 5.4730064863875, Chunk: 64,
			}
			r := rand.New(rand.NewPCG(77, 88))
			gateStride := hq * HeadDim
			if tt.slicedGate {
				gateStride = 2304
			}
			env := buildPagedAttnEnv(r, baseAttn, seqlens, 1.0, gateStride)
			cacheRows := env["k_cache"].Shape[0]
			maxPages := env["page_table"].Shape[1]

			b := NewBuilder()
			q := b.Input("q", E4M3, tokens, hq, HeadDim)
			kCache := b.Input("k_cache", U8, cacheRows, hkv, HeadDim)
			vCache := b.Input("v_cache", U8, cacheRows, hkv, HeadDim)
			pageTable := b.Input("page_table", I32, batch, maxPages)
			seqLens := b.Input("seqlens", I32, batch)
			kScale := b.Input("k_scale", F32, 1)
			vScale := b.Input("v_scale", F32, 1)

			var gate egraph.Id
			if tt.slicedGate {
				qkv := b.Input("qkv", BF16, tokens, 2304)
				gate = b.Reshape(b.Slice(qkv, 1280, hq*HeadDim), tokens, hq, HeadDim)
			} else {
				gate = b.Input("gate", BF16, tokens, hq, HeadDim)
			}

			attn := b.PagedAttention(baseAttn, q, kCache, vCache, pageTable, seqLens, kScale, vScale)
			var root egraph.Id
			if tt.primitiveOp {
				ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), attn))
				norm := b.Map(MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), attn, ss)
				root = b.Map(MustLambda("(mul $0 (sigmoid $1))"), norm, gate)
			} else {
				root = b.RMSNormGate(attn, gate, 1e-5)
			}

			// Extract the unfused expression before Optimize.
			unfusedExpr, _, err := egraph.ExtractBest(b.G, root, Cost{})
			if err != nil {
				t.Fatal(err)
			}
			unfusedPlan, err := Compile(unfusedExpr, "attn_unfused", BF16)
			if err != nil {
				t.Fatal(err)
			}
			if len(unfusedPlan.Kernels) != 2 || unfusedPlan.Kernels[0].Attention == nil || unfusedPlan.Kernels[0].Attention.Gate || unfusedPlan.Kernels[1].Row == nil {
				t.Fatalf("unfused plan kernels = %+v, want [PagedAttention{Gate:false}, RowKernel]", unfusedPlan.Kernels)
			}

			// Optimize fuses PagedAttention + RMSNormGate into PagedAttention{Gate: true}.
			fusedExpr, err := Optimize(b.G, root)
			if err != nil {
				t.Fatal(err)
			}
			if fusedExpr.Op != OpPagedAttnGated {
				t.Fatalf("optimized op = %s, want %s", fusedExpr.Op, OpPagedAttnGated)
			}
			fusedPlan, err := Compile(fusedExpr, "attn_fused", BF16)
			if err != nil {
				t.Fatal(err)
			}
			if len(fusedPlan.Kernels) != 1 || fusedPlan.Kernels[0].Attention == nil || !fusedPlan.Kernels[0].Attention.Gate {
				t.Fatalf("fused plan kernels = %+v, want 1 PagedAttention{Gate:true}", fusedPlan.Kernels)
			}
			if _, err := fusedPlan.Source(); err != nil {
				t.Fatal(err)
			}

			cfg := CostConfig{Device: "NVIDIA H200"}
			if fusedNs, unfusedNs := fusedPlan.EstimateNs(cfg), unfusedPlan.EstimateNs(cfg); fusedNs >= unfusedNs {
				t.Errorf("fused EstimateNs %.1f >= unfused EstimateNs %.1f", fusedNs, unfusedNs)
			}

			wantUnfused, err := Eval(unfusedExpr, env)
			if err != nil {
				t.Fatal(err)
			}
			wantFused, err := Eval(fusedExpr, env)
			if err != nil {
				t.Fatal(err)
			}
			evalTol := 1e-12
			if tt.primitiveOp {
				evalTol = 1e-6 // primitive Map evaluates in FP32; PagedAttention.Reference in FP64
			}
			for i := range wantFused.Data {
				if math.Abs(wantFused.Data[i]-wantUnfused.Data[i]) > evalTol {
					t.Fatalf("Eval mismatch at %d: %g vs %g", i, wantFused.Data[i], wantUnfused.Data[i])
				}
			}

			gotUnfused := runPlan(t, unfusedPlan, env)
			gotFused := runPlan(t, fusedPlan, env)
			if e := relErr(gotFused, wantFused.Data); e > 4e-3 || math.IsNaN(e) {
				t.Errorf("fused relative L2 error %.2e > 4e-3", e)
			}
			if e := relErr(gotUnfused, wantFused.Data); e > 4e-3 || math.IsNaN(e) {
				t.Errorf("unfused relative L2 error %.2e > 4e-3", e)
			}
			if e := relErr(gotFused, gotUnfused); e > 4e-3 || math.IsNaN(e) {
				t.Errorf("fused vs unfused relative L2 error %.2e > 4e-3", e)
			}
		})
	}
}

// TestTermMoEFusionEmulated verifies that an unfused MoE chain:
//
//	RowTopK -> ExpertAlign -> GroupedMatMul(W13) -> SwiGLU -> GroupedMatMul(W2) -> PostExpertNorm -> WeightedTopKSum
//
// optimizes into the fused 5-kernel plan:
//
//	RowTopK -> ExpertAlign -> TileGEMM(GLU) -> TileGEMM(SumSq) -> ExpertSum
//
// eliminating intermediate full-width FP32 buffers, reducing EstimateNs, and
// matching Eval on the host emulator.
func TestTermMoEFusionEmulated(t *testing.T) {
	const (
		tokens, experts, local, start = 6, 32, 8, 8
		dModel, dExpert, kRoutes      = 128, 64, 8
	)
	r := rand.New(rand.NewPCG(9, 10))

	b := NewBuilder()
	logits := b.Input("logits", F32, tokens, experts)
	bias := b.Input("bias", F32, experts)
	localStart := b.Input("local_start", I32, 1)
	x := b.Input("x", BF16, tokens, dModel)

	w13q := b.Input("w13_q", U8, local, 2*dExpert, dModel/2)
	w13sf := b.Input("w13_sf", U8, local, 2*dExpert, dModel/16)
	w13gs := b.Input("w13_gs", F32, local)
	w13 := b.NVFP4(w13q, w13sf, w13gs)

	w2q := b.Input("w2_q", U8, local, dModel, dExpert/2)
	w2sf := b.Input("w2_sf", U8, local, dModel, dExpert/16)
	w2gs := b.Input("w2_gs", F32, local)
	w2 := b.NVFP4(w2q, w2sf, w2gs)
	postW := b.Input("post_w", BF16, dModel)

	routeCfg := RowTopK{
		D: experts, K: kRoutes, X: F32, Bias: F32,
		Score:     MustLambda("(sigmoid (mul $0 0.1767766952966369))"),
		Key:       MustLambda("(add $0 $1)"),
		Normalize: true, Eps: 1e-15, Scale: 1,
	}
	alignCfg := ExpertAlign{K: kRoutes, Experts: experts, Local: local, Warps: 4}

	ids, weights := b.RowTopK(routeCfg, logits, bias)
	offsets, rows, slots := b.ExpertAlign(alignCfg, ids, localStart)

	mm13 := b.GroupedMatMul(x, w13, offsets, rows)
	inter := b.SwiGLU(mm13)
	mm2 := b.GroupedMatMul(inter, w2, offsets)
	normed := b.PostExpertNorm(mm2, postW, 1e-8)
	root := b.WeightedTopKSum(normed, slots, weights)

	// Extract unfused expression before Optimize.
	unfusedExpr, _, err := egraph.ExtractBest(b.G, root, Cost{})
	if err != nil {
		t.Fatal(err)
	}
	unfusedPlan, err := Compile(unfusedExpr, "moe_unfused", BF16)
	if err != nil {
		t.Fatal(err)
	}
	if len(unfusedPlan.Kernels) != 7 {
		t.Fatalf("unfused plan has %d kernels, want 7 (RowTopK, ExpertAlign, TileGEMM, RowKernel, TileGEMM, RowKernel, ExpertSum)", len(unfusedPlan.Kernels))
	}

	// Optimize fuses W13+SwiGLU into TileGEMM{GLU:true} and W2+PostExpertNorm+WeightedTopKSum into TileGEMM{SumSq:true} + ExpertSum.
	fusedExpr, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	if fusedExpr.Op != OpExpertSum {
		t.Fatalf("optimized root = %s, want %s", fusedExpr.Op, OpExpertSum)
	}
	fusedPlan, err := Compile(fusedExpr, "moe_fused", BF16)
	if err != nil {
		t.Fatal(err)
	}
	if len(fusedPlan.Kernels) != 5 {
		t.Fatalf("fused plan has %d kernels, want 5", len(fusedPlan.Kernels))
	}
	k0, k1, k2, k3, k4 := fusedPlan.Kernels[0], fusedPlan.Kernels[1], fusedPlan.Kernels[2], fusedPlan.Kernels[3], fusedPlan.Kernels[4]
	if k0.TopK == nil || k1.Align == nil ||
		k2.Tile == nil || !k2.Tile.Grouped || !k2.Tile.Gather || !k2.Tile.GLU ||
		k3.Tile == nil || !k3.Tile.Grouped || !k3.Tile.SumSq || k3.Tile.Epilogue.String() != "(mul $0 (add 1 $1))" ||
		k4.ExpertSum == nil || k4.ExpertSum.PreNormed {
		t.Fatalf("unexpected fused kernel sequence: %+v", fusedPlan.Kernels)
	}
	if _, err := fusedPlan.Source(); err != nil {
		t.Fatal(err)
	}

	cfg := CostConfig{Device: "NVIDIA H200"}
	if fusedNs, unfusedNs := fusedPlan.EstimateNs(cfg), unfusedPlan.EstimateNs(cfg); fusedNs >= unfusedNs {
		t.Errorf("fused EstimateNs %.1f >= unfused EstimateNs %.1f", fusedNs, unfusedNs)
	}

	// Build deterministic environment with token 0 routing to no local experts.
	env := make(map[string]Value)
	logitData := make([]float64, tokens*experts)
	for i := range logitData {
		logitData[i] = float64(float32(r.NormFloat64() * 4))
		if i < experts && i >= start && i < start+local {
			logitData[i] = -40
		}
	}
	env["logits"] = Value{Shape: []int{tokens, experts}, Data: logitData}
	biasData := make([]float64, experts)
	for i := range biasData {
		biasData[i] = float64(float32(r.NormFloat64() * 0.1))
	}
	env["bias"] = Value{Shape: []int{experts}, Data: biasData}
	env["local_start"] = Value{Shape: []int{1}, Data: []float64{start}}

	xData := make([]float64, tokens*dModel)
	for i := range xData {
		xData[i] = float64(BF16.Round(float32(r.NormFloat64())))
	}
	env["x"] = Value{Shape: []int{tokens, dModel}, Data: xData}

	fillNVFP4 := func(prefix string, n, k int) {
		var packed, scales, gs []float64
		for range local {
			w := randNVFP4(r, n, k)
			for _, b := range w.Packed {
				packed = append(packed, float64(b))
			}
			for _, b := range w.Scales {
				scales = append(scales, float64(b))
			}
			gs = append(gs, float64(float32(0.002+r.Float64()*0.002)))
		}
		env[prefix+"_q"] = Value{Shape: []int{local, n, k / 2}, Data: packed}
		env[prefix+"_sf"] = Value{Shape: []int{local, n, k / 16}, Data: scales}
		env[prefix+"_gs"] = Value{Shape: []int{local}, Data: gs}
	}
	fillNVFP4("w13", 2*dExpert, dModel)
	fillNVFP4("w2", dModel, dExpert)

	pwData := make([]float64, dModel)
	for i := range pwData {
		pwData[i] = float64(BF16.Round(float32(r.NormFloat64() * 0.1)))
	}
	env["post_w"] = Value{Shape: []int{dModel}, Data: pwData}

	wantUnfused, err := Eval(unfusedExpr, env)
	if err != nil {
		t.Fatal(err)
	}
	wantFused, err := Eval(fusedExpr, env)
	if err != nil {
		t.Fatal(err)
	}
	for i := range wantFused.Data {
		if wantFused.Data[i] != wantUnfused.Data[i] {
			t.Fatalf("Eval fused vs unfused differs at %d: %g vs %g", i, wantFused.Data[i], wantUnfused.Data[i])
		}
	}

	gotFused := runPlan(t, fusedPlan, env)
	gotUnfused := runPlan(t, unfusedPlan, env)
	wantRounded := roundAll(BF16, wantFused.Data)
	if e := relErr(gotFused, wantRounded); e > 5e-3 || math.IsNaN(e) {
		t.Errorf("fused MoE relative L2 error %.2e > 5e-3", e)
	}
	if e := relErr(gotUnfused, wantRounded); e > 5e-3 || math.IsNaN(e) {
		t.Errorf("unfused MoE relative L2 error %.2e > 5e-3", e)
	}
	for c := range dModel {
		if gotFused[c] != 0 {
			t.Fatalf("token 0 (no local route) out[%d] = %g, want 0", c, gotFused[c])
		}
	}
}

func ExampleBuilder_GroupedMatMul() {
	b := NewBuilder()
	logits := b.Input("logits", F32, 4, 32)
	ls := b.Input("local_start", I32, 1)
	x := b.Input("x", BF16, 4, 128)
	w13 := b.Input("w13", BF16, 8, 128, 128)
	w2 := b.Input("w2", BF16, 8, 128, 64)
	postW := b.Input("post_w", BF16, 128)

	ids, weights := b.RowTopK(RowTopK{D: 32, K: 4, X: F32, Normalize: true}, logits, 0)
	offsets, rows, slots := b.ExpertAlign(ExpertAlign{K: 4, Experts: 32, Local: 8}, ids, ls)
	inter := b.SwiGLU(b.GroupedMatMul(x, w13, offsets, rows))
	norm := b.PostExpertNorm(b.GroupedMatMul(inter, w2, offsets), postW, 1e-8)
	root := b.WeightedTopKSum(norm, slots, weights)

	expr, _ := Optimize(b.G, root)
	plan, _ := Compile(expr, "moe", BF16)
	fmt.Println(len(plan.Kernels), plan.Kernels[2].Tile.GLU, plan.Kernels[3].Tile.SumSq, plan.Kernels[4].ExpertSum != nil)
	// Output: 5 true true true
}
