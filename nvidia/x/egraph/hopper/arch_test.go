package hopper

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestArchParseConfig(t *testing.T) {
	tests := []struct {
		name        string
		json        string
		wantErr     bool
		checkConfig func(t *testing.T, cfg ArchConfig)
	}{
		{
			name: "dense_gqa",
			json: `{
				"architectures": ["TransformerForCausalLM"],
				"hidden_size": 4096,
				"intermediate_size": 14336,
				"num_hidden_layers": 32,
				"num_attention_heads": 32,
				"num_key_value_heads": 8,
				"rms_norm_eps": 1e-5
			}`,
			checkConfig: func(t *testing.T, cfg ArchConfig) {
				if cfg.HiddenSize != 4096 || cfg.IntermediateSize != 14336 {
					t.Errorf("dims = (%d, %d), want (4096, 14336)", cfg.HiddenSize, cfg.IntermediateSize)
				}
				if cfg.NumAttentionHeads != 32 || cfg.NumKeyValueHeads != 8 || cfg.HeadDim != 128 {
					t.Errorf("heads = (%d, %d, d=%d), want (32, 8, 128)",
						cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim)
				}
				if cfg.IsMoE() {
					t.Errorf("IsMoE() = true, want false")
				}
				if cfg.WeightDType != BF16 || cfg.KVDType != BF16 || cfg.WeightFormat() != WeightBF16 {
					t.Errorf("dtypes = (w=%v, kv=%v, wf=%v), want (bf16, bf16, bf16)",
						cfg.WeightDType, cfg.KVDType, cfg.WeightFormat())
				}
			},
		},
		{
			name: "moe_gated_attn_nvfp4",
			json: `{
				"architectures": ["MoEDecoderForCausalLM"],
				"hidden_size": 4096,
				"intermediate_size": 14336,
				"moe_intermediate_size": 2048,
				"num_hidden_layers": 48,
				"num_attention_heads": 32,
				"num_key_value_heads": 4,
				"head_dim": 64,
				"num_local_experts": 8,
				"num_experts": 64,
				"num_experts_per_tok": 8,
				"rms_norm_eps": 1e-6,
				"qk_norm": true,
				"attention_gate": true,
				"quantization_config": {
					"quant_method": "nvfp4"
				}
			}`,
			checkConfig: func(t *testing.T, cfg ArchConfig) {
				if !cfg.IsMoE() {
					t.Fatalf("IsMoE() = false, want true")
				}
				if cfg.NumLocalExperts != 8 || cfg.NumExperts != 64 || cfg.NumExpertsPerTok != 8 {
					t.Errorf("experts = (%d/%d top-%d), want (8/64 top-8)",
						cfg.NumLocalExperts, cfg.NumExperts, cfg.NumExpertsPerTok)
				}
				if !cfg.QKNorm || !cfg.AttentionGate {
					t.Errorf("QKNorm=%v AttentionGate=%v, want true, true", cfg.QKNorm, cfg.AttentionGate)
				}
				if cfg.WeightDType != NVFP4 || cfg.WeightFormat() != WeightNVFP4 {
					t.Errorf("WeightDType=%v WeightFormat=%v, want nvfp4", cfg.WeightDType, cfg.WeightFormat())
				}
				if cfg.KVDType != E4M3 {
					t.Errorf("KVDType=%v, want e4m3", cfg.KVDType)
				}
			},
		},
		{
			name: "fp8_quant_config",
			json: `{
				"hidden_size": 2048,
				"intermediate_size": 8192,
				"num_attention_heads": 16,
				"num_key_value_heads": 4,
				"quantization_config": {
					"quant_method": "fp8"
				}
			}`,
			checkConfig: func(t *testing.T, cfg ArchConfig) {
				if cfg.WeightDType != E4M3 || cfg.WeightFormat() != WeightE4M3 || cfg.KVDType != E4M3 {
					t.Errorf("got WeightDType=%v WeightFormat=%v KVDType=%v, want e4m3",
						cfg.WeightDType, cfg.WeightFormat(), cfg.KVDType)
				}
			},
		},
		{
			name: "modelopt_quant_config",
			json: `{
				"hidden_size": 2048,
				"intermediate_size": 8192,
				"num_attention_heads": 16,
				"num_key_value_heads": 4,
				"quantization_config": {
					"quant_method": "modelopt"
				}
			}`,
			checkConfig: func(t *testing.T, cfg ArchConfig) {
				if cfg.WeightDType != NVFP4 || cfg.WeightFormat() != WeightNVFP4 {
					t.Errorf("got WeightDType=%v WeightFormat=%v, want nvfp4",
						cfg.WeightDType, cfg.WeightFormat())
				}
			},
		},
		{
			name:    "invalid_missing_hidden_size",
			json:    `{"intermediate_size": 1024, "num_attention_heads": 8}`,
			wantErr: true,
		},
		{
			name:    "invalid_bad_gqa_ratio",
			json:    `{"hidden_size": 1024, "intermediate_size": 2048, "num_attention_heads": 8, "num_key_value_heads": 3}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := ParseArchConfig([]byte(tt.json))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ParseArchConfig succeeded, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseArchConfig: %v", err)
			}
			if tt.checkConfig != nil {
				tt.checkConfig(t, cfg)
			}
		})
	}
}

func TestCompileLayerSchedules(t *testing.T) {
	moeCfg := ArchConfig{
		HiddenSize:          1024,
		IntermediateSize:    2048,
		MoEIntermediateSize: 512,
		NumHiddenLayers:     24,
		NumAttentionHeads:   16,
		NumKeyValueHeads:    2,
		HeadDim:             64,
		NumLocalExperts:     4,
		NumExperts:          16,
		NumExpertsPerTok:    4,
		RMSNormEps:          1e-5,
		AttentionGate:       true,
		WeightDType:         NVFP4,
		KVDType:             E4M3,
	}

	t.Run("h200_hopper_schedules", func(t *testing.T) {
		clDecode, err := CompileLayer(moeCfg, 4, 1024, "NVIDIA H200")
		if err != nil {
			t.Fatalf("CompileLayer H200 decode: %v", err)
		}
		if clDecode.Mode != ModePersistentMegakernel {
			t.Errorf("Mode = %q, want %q", clDecode.Mode, ModePersistentMegakernel)
		}
		if clDecode.MegakernelSource == "" || len(clDecode.Megakernels) == 0 {
			t.Errorf("expected non-empty MegakernelSource for batchM=4")
		}
		if clDecode.CUDASource == "" || clDecode.TVMFFISource == "" {
			t.Errorf("expected non-empty CUDASource and TVMFFISource")
		}

		var (
			sawGatedAttn    bool
			sawWGMMADequant bool
			sawExpertSum    bool
		)
		for _, k := range clDecode.Plan.Kernels {
			if k.Attention != nil && k.Attention.Gate && k.Attention.Swizzle {
				sawGatedAttn = true
			}
			if k.Tile != nil && k.Tile.Weight == WeightNVFP4 {
				if k.Tile.WGMMA && k.Tile.Repacked && k.Tile.DequantWords == 2 && !k.Tile.BlackwellFP4 {
					sawWGMMADequant = true
				}
			}
			if k.ExpertSum != nil {
				sawExpertSum = true
			}
		}
		if !sawGatedAttn {
			t.Errorf("H200 plan did not fuse gated PagedAttention{Gate: true, Swizzle: true}")
		}
		if !sawWGMMADequant {
			t.Errorf("H200 plan did not select WGMMA + Repacked + DequantWords=2 on NVFP4 TileGEMM")
		}
		if !sawExpertSum {
			t.Errorf("H200 MoE plan did not fuse ExpertSum")
		}

		clPrefill, err := CompileLayer(moeCfg, 128, 1024, "NVIDIA H200")
		if err != nil {
			t.Fatalf("CompileLayer H200 prefill: %v", err)
		}
		if clPrefill.Mode != ModeStreamKWarpSpec {
			t.Errorf("prefill Mode = %q, want %q", clPrefill.Mode, ModeStreamKWarpSpec)
		}
		for _, k := range clPrefill.Plan.Kernels {
			if k.Tile != nil && (!k.Tile.WGMMA || k.Tile.DequantWords != 2) {
				t.Errorf("H200 prefill TileGEMM %s: WGMMA=%v DequantWords=%d, want true, 2",
					k.Tile.Name, k.Tile.WGMMA, k.Tile.DequantWords)
			}
		}
	})

	t.Run("gb10_blackwell_schedules", func(t *testing.T) {
		clGB10, err := CompileLayer(moeCfg, 4, 1024, "NVIDIA GB10")
		if err != nil {
			t.Fatalf("CompileLayer GB10 decode: %v", err)
		}
		var (
			sawSkinnyBlackwellFP4 bool
			sawTileBlackwellFP4   bool
		)
		for _, k := range clGB10.Plan.Kernels {
			if k.GEMM != nil && k.GEMM.Weight == WeightNVFP4 {
				if !k.GEMM.BlackwellFP4 {
					t.Errorf("GB10 SkinnyGEMM %s: BlackwellFP4 = false, want true", k.GEMM.Name)
				}
				sawSkinnyBlackwellFP4 = true
			}
			if k.Tile != nil && k.Tile.Weight == WeightNVFP4 {
				if !k.Tile.BlackwellFP4 || k.Tile.WGMMA {
					t.Errorf("GB10 TileGEMM %s: BlackwellFP4=%v WGMMA=%v, want true, false",
						k.Tile.Name, k.Tile.BlackwellFP4, k.Tile.WGMMA)
				}
				sawTileBlackwellFP4 = true
			}
		}
		if !sawSkinnyBlackwellFP4 || !sawTileBlackwellFP4 {
			t.Errorf("GB10 schedules: sawSkinnyBlackwellFP4=%v sawTileBlackwellFP4=%v, want both true",
				sawSkinnyBlackwellFP4, sawTileBlackwellFP4)
		}
	})
}

func TestCompileLayerMultiBucket(t *testing.T) {
	cfg := ArchConfig{
		HiddenSize:        1024,
		IntermediateSize:  2048,
		NumHiddenLayers:   16,
		NumAttentionHeads: 16,
		NumKeyValueHeads:  4,
		HeadDim:           64,
		RMSNormEps:        1e-5,
	}
	buckets := []int{1, 16, 128}
	bundle, err := CompileMultiBucketLayer(cfg, buckets, 1024, "NVIDIA H200")
	if err != nil {
		t.Fatalf("CompileMultiBucketLayer: %v", err)
	}
	if len(bundle.Layers) != 3 {
		t.Fatalf("len(Layers) = %d, want 3", len(bundle.Layers))
	}
	if bundle.Layers[1].Mode != ModePersistentMegakernel {
		t.Errorf("M=1 mode = %q, want %q", bundle.Layers[1].Mode, ModePersistentMegakernel)
	}
	if bundle.Layers[16].Mode != ModeMultiLaunch {
		t.Errorf("M=16 mode = %q, want %q", bundle.Layers[16].Mode, ModeMultiLaunch)
	}
	if bundle.Layers[128].Mode != ModeStreamKWarpSpec {
		t.Errorf("M=128 mode = %q, want %q", bundle.Layers[128].Mode, ModeStreamKWarpSpec)
	}
	for _, want := range []string{
		"inline void dispatch_layer_m(int M,",
		"extern \"C\" void dispatch_layer_m_c(int M,",
		"if (M <= 1) {",
		"layer_m1_run(",
		"if (M <= 16) {",
		"layer_m16_run(",
		"layer_m128_run(",
	} {
		if !strings.Contains(bundle.CUDASource, want) {
			t.Errorf("CUDASource missing %q", want)
		}
	}
	if !strings.Contains(bundle.TVMFFISource, "namespace eg {") {
		t.Errorf("TVMFFISource missing namespace eg")
	}
}

// TestCompileLayerDenseEmulated compiles a complete dense GQA transformer
// decoder layer on small dimensions (M=2, H=64, I=128, G=4, KVH=2, D=16),
// executes every step on the CPU via cudahostemu.Run, and verifies finite,
// accurate outputs against both TermProgram.Eval and an independent Go
// reference implementation.
func TestCompileLayerDenseEmulated(t *testing.T) {
	const (
		M      = 2
		H      = 64
		I      = 128
		G      = 4
		KVH    = 2
		D      = 16
		seqLen = 32
		eps    = 1e-5
	)
	cfg := ArchConfig{
		HiddenSize:        H,
		IntermediateSize:  I,
		NumHiddenLayers:   2,
		NumAttentionHeads: G,
		NumKeyValueHeads:  KVH,
		HeadDim:           D,
		RMSNormEps:        eps,
		WeightDType:       BF16,
		KVDType:           BF16,
	}

	prog, err := BuildLayerProgram(cfg, M, seqLen)
	if err != nil {
		t.Fatalf("BuildLayerProgram: %v", err)
	}
	cl, err := CompileLayer(cfg, M, seqLen, "NVIDIA H200")
	if err != nil {
		t.Fatalf("CompileLayer: %v", err)
	}
	if cl.Mode != ModePersistentMegakernel {
		t.Fatalf("Mode = %q, want %q", cl.Mode, ModePersistentMegakernel)
	}
	if len(cl.Estimates) != len(cl.Plan.Kernels) || cl.TotalNs <= 0 {
		t.Fatalf("Estimates len=%d (plan=%d) TotalNs=%v", len(cl.Estimates), len(cl.Plan.Kernels), cl.TotalNs)
	}

	// Construct deterministic inputs for all layer tensors.
	r := rand.New(rand.NewPCG(501, 502))
	env := make(map[string]Value)
	for _, in := range cl.Plan.Inputs {
		switch in.Name {
		case "k_cache", "v_cache":
			data := make([]float64, size(in.Shape))
			for i := range data {
				v := float32(r.NormFloat64() * 0.25)
				data[i] = float64(E4M3Bits(v))
			}
			env[in.Name] = Value{Shape: in.Shape, Data: data}
		case "page_table":
			maxPages := in.Shape[1]
			data := make([]float64, M*maxPages)
			for i := range data {
				data[i] = float64(i)
			}
			env[in.Name] = Value{Shape: in.Shape, Data: data}
		case "seqlens":
			env[in.Name] = Value{Shape: in.Shape, Data: []float64{24, 19}}
		case "k_scale", "v_scale":
			env[in.Name] = Value{Shape: in.Shape, Data: []float64{0.125}}
		default:
			data := make([]float64, size(in.Shape))
			for i := range data {
				v := float32(r.NormFloat64() * 0.2)
				data[i] = float64(in.DType.Round(v))
			}
			env[in.Name] = Value{Shape: in.Shape, Data: data}
		}
	}

	// 1. Evaluate via TermProgram.Eval.
	evalVal, err := prog.Eval(env)
	if err != nil {
		t.Fatalf("prog.Eval: %v", err)
	}
	wantEval := roundAll(BF16, evalVal.Data)

	// 2. Evaluate via standalone Go reference function.
	wantRef := denseLayerGoReference(t, cfg, M, env)
	if e := relErr(wantEval, wantRef); e > 1e-4 || math.IsNaN(e) {
		t.Fatalf("TermProgram.Eval vs Go reference relative L2 error = %.2e > 1e-4", e)
	}

	// 3. Execute all compiled layer kernels on CPU via cudahostemu.Run (runPlan).
	got := runPlan(t, cl.Plan, env)
	if len(got) != M*H {
		t.Fatalf("len(got) = %d, want %d", len(got), M*H)
	}
	var normSq float64
	for i, v := range got {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("emulated output[%d] = %v is not finite", i, v)
		}
		normSq += v * v
	}
	if normSq == 0 {
		t.Fatalf("emulated output is all zeros")
	}
	if e := relErr(got, wantRef); e > 5e-3 || math.IsNaN(e) {
		t.Fatalf("emulated layer vs Go reference relative L2 error = %.2e > 5e-3\ngot:  %v\nwant: %v",
			e, got[:8], wantRef[:8])
	}
	if e := relErr(got, wantEval); e > 5e-3 || math.IsNaN(e) {
		t.Fatalf("emulated layer vs Eval relative L2 error = %.2e > 5e-3", e)
	}
}

func denseLayerGoReference(t *testing.T, cfg ArchConfig, batchM int, env map[string]Value) []float64 {
	t.Helper()
	H := cfg.HiddenSize
	I := cfg.IntermediateSize
	hq := cfg.NumAttentionHeads
	hkv := cfg.NumKeyValueHeads
	qDim := hq * HeadDim
	eps := float64(cfg.RMSNormEps)

	rmsNormRef := func(x []float64, w []float64, rows, cols int) []float64 {
		out := make([]float64, rows*cols)
		for r := range rows {
			var ss float64
			for c := range cols {
				v := float64(float32(x[r*cols+c]))
				ss += v * v
			}
			inv := float64(1.0 / float32(math.Sqrt(ss/float64(cols)+eps)))
			for c := range cols {
				val := float32(x[r*cols+c]) * float32(inv) * float32(w[c])
				out[r*cols+c] = float64(BF16.Round(val))
			}
		}
		return out
	}

	matmulRef := func(x, w []float64, m, n, k int) []float64 {
		out := make([]float64, m*n)
		for r := range m {
			for col := range n {
				var sum float64
				for c := range k {
					sum += float64(BF16.Round(float32(x[r*k+c]))) * float64(BF16.Round(float32(w[col*k+c])))
				}
				out[r*n+col] = float64(float32(sum))
			}
		}
		return out
	}

	x := env["x"].Data
	norm1 := rmsNormRef(x, env["attn_norm_w"].Data, batchM, H)
	qRaw := matmulRef(norm1, env["wq"].Data, batchM, qDim, H)
	qBF16 := make([]float32, len(qRaw))
	for i, v := range qRaw {
		qBF16[i] = BF16.Round(float32(v))
	}

	// PagedAttention reference.
	kv := env["k_cache"].Data
	vv := env["v_cache"].Data
	pt := env["page_table"]
	sl := env["seqlens"].Data
	kb := make([]uint8, len(kv))
	vb := make([]uint8, len(vv))
	for i := range kv {
		kb[i] = uint8(kv[i])
		vb[i] = uint8(vv[i])
	}
	pti := make([]int32, len(pt.Data))
	for i, v := range pt.Data {
		pti[i] = int32(v)
	}
	sli := make([]int32, len(sl))
	for i, v := range sl {
		sli[i] = int32(v)
	}
	attnCfg := PagedAttention{
		Name: "ref_attn",
		Hq:   hq,
		Hkv:  hkv,
		QLen: 1,
		Page: 16,
		Q:    BF16,
		Out:  F32,
	}
	attnOutF64, err := attnCfg.Reference(qBF16, PagedCache{
		K:         kb,
		V:         vb,
		PageTable: pti,
		MaxPages:  pt.Shape[1],
		SeqLens:   sli,
		KScale:    float32(env["k_scale"].Data[0]),
		VScale:    float32(env["v_scale"].Data[0]),
	})
	if err != nil {
		t.Fatalf("PagedAttention.Reference: %v", err)
	}
	attnBF16 := roundAll(BF16, attnOutF64)

	// OProj + residual add.
	oProj := matmulRef(attnBF16, env["wo"].Data, batchM, H, qDim)
	h1 := make([]float64, batchM*H)
	for i := range h1 {
		h1[i] = float64(float32(oProj[i]) + float32(x[i]))
	}

	// Post-attention RMSNorm + Dense MLP (GateUp -> SwiGLU -> Down).
	norm2 := rmsNormRef(h1, env["ffn_norm_w"].Data, batchM, H)
	gateUp := matmulRef(norm2, env["w_gate_up"].Data, batchM, 2*I, H)
	silu := MustLambda("(mul (silu $0) $1)")
	act := make([]float64, batchM*I)
	for r := range batchM {
		for c := range I {
			g := float32(gateUp[r*2*I+c])
			u := float32(gateUp[r*2*I+I+c])
			act[r*I+c] = float64(BF16.Round(silu.Eval(g, u)))
		}
	}
	down := matmulRef(act, env["w_down"].Data, batchM, H, I)

	// ResidualRMSNorm.
	res := make([]float64, batchM*H)
	for i := range res {
		res[i] = float64(float32(down[i]) + float32(h1[i]))
	}
	return rmsNormRef(res, env["post_norm_w"].Data, batchM, H)
}

// TestCompileLayerMoEEmulated compiles a complete routed-MoE + gated-attention
// transformer decoder layer on small dimensions (M=2, H=64, MoEI=64, G=4, KVH=2,
// Local=2, Experts=32, TopK=2), executes every step on the CPU via cudahostemu.Run,
// and verifies outputs against TermProgram.Eval.
func TestCompileLayerMoEEmulated(t *testing.T) {
	const (
		M      = 2
		H      = 64
		MoEI   = 64
		G      = 4
		KVH    = 2
		D      = 16
		seqLen = 32
		eps    = 1e-5
	)
	cfg := ArchConfig{
		HiddenSize:          H,
		IntermediateSize:    128,
		MoEIntermediateSize: MoEI,
		NumHiddenLayers:     2,
		NumAttentionHeads:   G,
		NumKeyValueHeads:    KVH,
		HeadDim:             D,
		NumLocalExperts:     2,
		NumExperts:          32,
		NumExpertsPerTok:    2,
		RMSNormEps:          eps,
		AttentionGate:       true,
		WeightDType:         BF16,
		KVDType:             BF16,
	}

	prog, err := BuildLayerProgram(cfg, M, seqLen)
	if err != nil {
		t.Fatalf("BuildLayerProgram: %v", err)
	}
	cl, err := CompileLayer(cfg, M, seqLen, "NVIDIA H200")
	if err != nil {
		t.Fatalf("CompileLayer: %v", err)
	}

	r := rand.New(rand.NewPCG(601, 602))
	env := make(map[string]Value)
	for _, in := range cl.Plan.Inputs {
		switch in.Name {
		case "k_cache", "v_cache":
			data := make([]float64, size(in.Shape))
			for i := range data {
				v := float32(r.NormFloat64() * 0.25)
				data[i] = float64(E4M3Bits(v))
			}
			env[in.Name] = Value{Shape: in.Shape, Data: data}
		case "page_table":
			maxPages := in.Shape[1]
			data := make([]float64, M*maxPages)
			for i := range data {
				data[i] = float64(i)
			}
			env[in.Name] = Value{Shape: in.Shape, Data: data}
		case "seqlens":
			env[in.Name] = Value{Shape: in.Shape, Data: []float64{20, 16}}
		case "k_scale", "v_scale":
			env[in.Name] = Value{Shape: in.Shape, Data: []float64{0.125}}
		case "local_start":
			env[in.Name] = Value{Shape: in.Shape, Data: []float64{0}}
		case "router_logits":
			data := make([]float64, size(in.Shape))
			for i := range data {
				data[i] = float64(float32(r.NormFloat64() * 0.5))
			}
			// Boost local experts 0 and 1 so top-2 routes hit local experts.
			for tok := range M {
				data[tok*in.Shape[1]+0] = 4.0
				data[tok*in.Shape[1]+1] = 3.5
			}
			env[in.Name] = Value{Shape: in.Shape, Data: data}
		case "router_bias":
			env[in.Name] = Value{Shape: in.Shape, Data: make([]float64, size(in.Shape))}
		default:
			data := make([]float64, size(in.Shape))
			for i := range data {
				v := float32(r.NormFloat64() * 0.2)
				data[i] = float64(in.DType.Round(v))
			}
			env[in.Name] = Value{Shape: in.Shape, Data: data}
		}
	}

	evalVal, err := prog.Eval(env)
	if err != nil {
		t.Fatalf("prog.Eval: %v", err)
	}
	wantEval := roundAll(BF16, evalVal.Data)

	got := runPlan(t, cl.Plan, env)
	if len(got) != M*H {
		t.Fatalf("len(got) = %d, want %d", len(got), M*H)
	}
	for i, v := range got {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("emulated MoE output[%d] = %v is not finite", i, v)
		}
	}
	if e := relErr(got, wantEval); e > 5e-3 || math.IsNaN(e) {
		t.Fatalf("emulated MoE layer vs Eval relative L2 error = %.2e > 5e-3\ngot:  %v\nwant: %v",
			e, got[:8], wantEval[:8])
	}
}
