package qualify

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func buildTestCrossLayerMegakernel(t *testing.T) hopper.Megakernel {
	t.Helper()

	const (
		m = 4
		d = 256
	)
	buildPost := func(prefix string) *hopper.Plan {
		b := hopper.NewBuilder()
		x := b.Input("xn", hopper.BF16, m, d)
		wo := b.Input(prefix+"_wo", hopper.BF16, d, d)
		res := b.Input(prefix+"_res", hopper.F32, m, d)
		attnOut := b.MatMul(x, wo)
		hid := b.Map(hopper.MustLambda("(add $0 $1)"), attnOut, res)
		nw1 := b.Input(prefix+"_nw1", hopper.F32, d)
		mlpIn := b.Cast(hopper.BF16, b.RMSNorm(hid, nw1, d, 1e-5))

		wgu := b.Input(prefix+"_wgu", hopper.BF16, 2*d, d)
		act := b.SwiGLU(b.MatMul(mlpIn, wgu))
		wd := b.Input(prefix+"_wd", hopper.BF16, d, d)
		down := b.MatMul(act, wd)
		hid2 := b.Map(hopper.MustLambda("(add $0 $1)"), down, hid)
		nw2 := b.Input(prefix+"_nw2", hopper.F32, d)
		out := b.Cast(hopper.BF16, b.RMSNorm(hid2, nw2, d, 1e-5))

		expr, err := hopper.Optimize(b.G, out)
		if err != nil {
			t.Fatalf("Optimize(%s): %v", prefix, err)
		}
		pl, err := hopper.CompileFor(expr, prefix+"_post", hopper.BF16, "sm_121a")
		if err != nil {
			t.Fatalf("CompileFor(%s): %v", prefix, err)
		}
		return pl
	}

	buildNextAttn := func(prefix string) *hopper.Plan {
		b := hopper.NewBuilder()
		xn := b.Input("xn", hopper.BF16, m, d)
		wq := b.Input(prefix+"_wq", hopper.BF16, d, d)
		q := b.MatMul(xn, wq)
		scale := b.Input(prefix+"_scale", hopper.F32, m, d)
		scaled := b.Map(hopper.MustLambda("(mul $0 $1)"), q, scale)
		nw := b.Input(prefix+"_nw", hopper.F32, d)
		out := b.Cast(hopper.BF16, b.RMSNorm(scaled, nw, d, 1e-5))

		expr, err := hopper.Optimize(b.G, out)
		if err != nil {
			t.Fatalf("Optimize(%s): %v", prefix, err)
		}
		pl, err := hopper.CompileFor(expr, prefix+"_attn_in", hopper.BF16, "sm_121a")
		if err != nil {
			t.Fatalf("CompileFor(%s): %v", prefix, err)
		}
		return pl
	}

	post0 := buildPost("l0")
	attn1 := buildNextAttn("l1")
	_, mega, err := hopper.BuildCrossLayerMegakernel(post0, attn1, hopper.CrossLayerGlue{
		Name:               "cross_l0_l1_formal",
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	})
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel: %v", err)
	}
	return mega
}

func TestFormalBarrierVerifierValidCrossLayerMegakernel(t *testing.T) {
	mega := buildTestCrossLayerMegakernel(t)

	rep, err := VerifyMegakernelBarrierSafety(&mega)
	if err != nil {
		t.Fatalf("VerifyMegakernelBarrierSafety valid cross-layer megakernel failed: %v", err)
	}
	if !rep.Passed() {
		t.Fatalf("expected report.Passed() == true, got %+v", rep)
	}
	if rep.CheckedRAWEdges == 0 {
		t.Fatalf("expected CheckedRAWEdges > 0, got 0")
	}

	// Also verify a full decoder layer megakernel with sub-CTA (Threads = 128)
	// QKNormRoPEKVAppend + PagedAttention stages to exercise Property 3
	// (named-barrier thread uniformity and entry guard).
	const (
		tokens   = 2
		h        = 256
		hq       = 4
		hkv      = 2
		dHead    = 128
		qDim     = hq * dHead
		qkvDim   = (hq + 2*hkv) * dHead
		pageSize = 16
		maxPages = 2
		maxKeys  = 16
	)
	qkvGEMM := hopper.SkinnyGEMM{
		Name:         "layer_qkv",
		N:            qkvDim,
		K:            h,
		MaxM:         tokens,
		RowsPerBlock: 32,
		Weight:       hopper.WeightBF16,
		X:            hopper.BF16,
		Out:          hopper.BF16,
	}
	rope := hopper.QKNormRoPEKVAppend{
		Name:      "layer_rope_kv",
		NumHeads:  hq,
		KVHeads:   hkv,
		HeadDim:   dHead,
		RotaryDim: dHead,
		PageSize:  pageSize,
		MaxPages:  maxPages,
		DType:     hopper.BF16,
		QKNorm:    true,
		KVFP8:     true,
		Eps:       1e-5,
	}
	attn := hopper.PagedAttention{
		Name: "layer_attn",
		Hq:   hq,
		Hkv:  hkv,
		QLen: 1,
		Page: pageSize,
		Q:    hopper.BF16,
		Out:  hopper.BF16,
	}
	oProj := hopper.SkinnyGEMM{
		Name:         "layer_o_proj",
		N:            h,
		K:            qDim,
		MaxM:         tokens,
		RowsPerBlock: 32,
		Weight:       hopper.WeightBF16,
		X:            hopper.BF16,
		Out:          hopper.BF16,
	}
	kvCacheElems := 2 * pageSize * hkv * dHead
	subCTAPlan := &hopper.Plan{
		Name: "decoder_subcta",
		Inputs: []hopper.Tensor{
			{Name: "x", DType: hopper.BF16, Shape: []int{tokens, h}},
			{Name: "w_qkv", DType: hopper.BF16, Shape: []int{qkvDim, h}},
			{Name: "q_norm_w", DType: hopper.BF16, Shape: []int{dHead}},
			{Name: "k_norm_w", DType: hopper.BF16, Shape: []int{dHead}},
			{Name: "cos_sin", DType: hopper.F32, Shape: []int{16, dHead}},
			{Name: "positions", DType: hopper.I32, Shape: []int{tokens}},
			{Name: "seq_ids", DType: hopper.I32, Shape: []int{tokens}},
			{Name: "seqlens", DType: hopper.I32, Shape: []int{tokens}},
			{Name: "kv_positions", DType: hopper.I32, Shape: []int{tokens}},
			{Name: "page_table", DType: hopper.I32, Shape: []int{tokens, maxPages}},
			{Name: "k_scale", DType: hopper.F32, Shape: []int{1}},
			{Name: "v_scale", DType: hopper.F32, Shape: []int{1}},
			{Name: "w_o", DType: hopper.BF16, Shape: []int{h, qDim}},
		},
		Temps: []hopper.Tensor{
			{Name: "qkv_out", DType: hopper.BF16, Shape: []int{tokens, qkvDim}},
			{Name: "q_out", DType: hopper.BF16, Shape: []int{tokens, qDim}},
			{Name: "k_cache", DType: hopper.E4M3, Shape: []int{kvCacheElems}},
			{Name: "v_cache", DType: hopper.E4M3, Shape: []int{kvCacheElems}},
			{Name: "attn_out", DType: hopper.BF16, Shape: []int{tokens, qDim}},
			{Name: "attn_ws", DType: hopper.E4M3, Shape: []int{64}},
		},
		Output: hopper.Tensor{Name: "o_out", DType: hopper.BF16, Shape: []int{tokens, h}},
		Kernels: []hopper.PlanKernel{
			{GEMM: &qkvGEMM, M: tokens, Args: []string{"x", "w_qkv", "qkv_out"}},
			{
				RoPE: &rope,
				M:    tokens,
				Args: []string{
					"qkv_out", "q_norm_w", "k_norm_w", "cos_sin",
					"positions", "seq_ids", "seqlens", "kv_positions", "page_table",
					"k_scale", "v_scale", "q_out", "k_cache", "v_cache",
				},
			},
			{
				Attention: &attn,
				Batch:     tokens,
				MaxKeys:   maxKeys,
				MaxPages:  maxPages,
				Args: []string{
					"q_out", "k_cache", "v_cache", "page_table",
					"seqlens", "k_scale", "v_scale", "attn_out", "attn_ws",
				},
			},
			{GEMM: &oProj, M: tokens, Args: []string{"attn_out", "w_o", "o_out"}},
		},
	}
	subCTAMega, err := subCTAPlan.Megakernel("decoder_subcta_mega")
	if err != nil {
		t.Fatalf("subCTAPlan.Megakernel: %v", err)
	}
	subCTARep, err := VerifyMegakernelBarrierSafety(&subCTAMega)
	if err != nil {
		t.Fatalf("VerifyMegakernelBarrierSafety sub-CTA megakernel failed: %v", err)
	}
	if !subCTARep.Passed() || subCTARep.CheckedSubCTABarrierStages != 2 {
		t.Fatalf("expected CheckedSubCTABarrierStages=2 and Passed()=true, got %+v", subCTARep)
	}

	// Verify ping-pong arena rewrite with WAR wait edges also passes all 4 properties.
	arenaMega, err := RewriteMegakernelPingPongArena(&mega, true)
	if err != nil {
		t.Fatalf("RewriteMegakernelPingPongArena(addWARWaitEdges=true): %v", err)
	}
	arenaRep, err := VerifyMegakernelBarrierSafety(arenaMega)
	if err != nil {
		t.Fatalf("VerifyMegakernelBarrierSafety safe ping-pong arena failed: %v", err)
	}
	if !arenaRep.Passed() || arenaRep.CheckedArenaAliases == 0 {
		t.Fatalf("expected safe ping-pong arena to pass with CheckedArenaAliases > 0, got %+v", arenaRep)
	}
}

func TestFormalBarrierVerifierDetectsMissingRAWWaitStage(t *testing.T) {
	mega := buildTestCrossLayerMegakernel(t)
	broken := mega
	broken.Stages = append([]hopper.MegaStage(nil), mega.Stages...)

	// Break stage 1's WaitSlots so it no longer waits on stage 0's output.
	broken.Stages[1].WaitSlots = nil

	rep, err := VerifyMegakernelBarrierSafety(&broken)
	if err == nil {
		t.Fatalf("expected error for missing RAW WaitSlots on stage 1, got nil")
	}
	if rep == nil || rep.RAWBarrierDominanceOK || rep.Passed() {
		t.Fatalf("expected RAWBarrierDominanceOK=false, got %+v", rep)
	}

	foundStage1RAW := false
	for _, v := range rep.Violations {
		if v.Kind == ViolationRAWDominance && v.Stage == 1 && v.UpstreamStage == 0 {
			foundStage1RAW = true
			break
		}
	}
	if !foundStage1RAW {
		t.Fatalf("expected ViolationRAWDominance with Stage=1 UpstreamStage=0, got %+v", rep.Violations)
	}
}

func TestFormalBarrierVerifierDetectsPrematureArenaReuse(t *testing.T) {
	mega := buildTestCrossLayerMegakernel(t)

	// Construct a ping-pong arena megakernel where stage 0 writes eg_arena_even,
	// stage 1 reads eg_arena_even, and stage 2 overwrites eg_arena_even while only
	// waiting on stage 0 (not waiting on reader stage 1 -> WAR hazard!).
	unsafeArena, err := RewriteMegakernelPingPongArena(&mega, false)
	if err != nil {
		t.Fatalf("RewriteMegakernelPingPongArena: %v", err)
	}
	// Find the second writer of eg_arena_even and make its WaitSlots bypass the
	// intermediate reader stage.
	firstWriter := -1
	reader := -1
	secondWriter := -1
	for s, st := range unsafeArena.Stages {
		reads, writes, _ := extractStageBufferAccesses(st)
		for _, r := range reads {
			if r == "eg_arena_even" && firstWriter >= 0 && secondWriter < 0 {
				reader = s
			}
		}
		for _, w := range writes {
			if w == "eg_arena_even" {
				if firstWriter < 0 {
					firstWriter = s
				} else if secondWriter < 0 {
					secondWriter = s
				}
			}
		}
	}
	if firstWriter < 0 || reader < 0 || secondWriter < 0 {
		t.Fatalf("expected eg_arena_even writer/reader/secondWriter chain, got %d/%d/%d", firstWriter, reader, secondWriter)
	}

	// Force secondWriter to wait only on firstWriter (satisfying WAW and any
	// upstream inputs from firstWriter) while omitting reader -> premature WAR reuse!
	unsafeArena.Stages[secondWriter].WaitSlots = []int{firstWriter}

	rep, err := VerifyMegakernelBarrierSafety(unsafeArena)
	if err == nil {
		t.Fatalf("expected error for premature WAR arena reuse, got nil")
	}
	if rep == nil || rep.ArenaAliasingSafetyOK || rep.Passed() {
		t.Fatalf("expected ArenaAliasingSafetyOK=false, got %+v", rep)
	}

	foundWAR := false
	for _, v := range rep.Violations {
		if v.Kind == ViolationArenaAliasing &&
			v.Stage == secondWriter &&
			v.UpstreamStage == firstWriter &&
			v.ReaderStage == reader &&
			v.Buffer == "eg_arena_even" {
			foundWAR = true
			break
		}
	}
	if !foundWAR {
		t.Fatalf("expected ViolationArenaAliasing (Stage=%d, ReaderStage=%d, UpstreamStage=%d, Buffer=eg_arena_even), got %+v",
			secondWriter, reader, firstWriter, rep.Violations)
	}
}

func TestFormalBarrierVerifierDetectsNonWarpMultipleThreadsAndEpochOverlap(t *testing.T) {
	mega := buildTestCrossLayerMegakernel(t)
	broken := mega
	broken.Stages = append([]hopper.MegaStage(nil), mega.Stages...)

	// 1. Set stage 0 Threads to 100 (not a multiple of warp size 32).
	broken.Stages[0].Threads = 100
	// 2. Set stage 1 SignalSlot to broken.EpochSlot() (colliding with device epoch slot).
	broken.Stages[1].SignalSlot = broken.EpochSlot()

	rep, err := VerifyMegakernelBarrierSafety(&broken)
	if err == nil {
		t.Fatalf("expected error for non-warp-multiple Threads and epoch slot collision, got nil")
	}
	if rep == nil || rep.NamedBarrierUniformityOK || rep.EpochSlotNonOverlapOK {
		t.Fatalf("expected NamedBarrierUniformityOK=false and EpochSlotNonOverlapOK=false, got %+v", rep)
	}
	if !strings.Contains(err.Error(), "non-warp-multiple Threads=100") {
		t.Fatalf("expected error message to report non-warp-multiple Threads=100, got %v", err)
	}
	if !strings.Contains(err.Error(), "collides with epoch slot") {
		t.Fatalf("expected error message to report epoch slot collision, got %v", err)
	}
}
