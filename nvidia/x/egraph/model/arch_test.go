package model

import (
	"fmt"
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
				if cfg.AttentionKind() != AttentionGQA {
					t.Errorf("AttentionKind() = %v, want %v", cfg.AttentionKind(), AttentionGQA)
				}
				if cfg.FFNKind() != FFNSwiGLU {
					t.Errorf("FFNKind() = %v, want %v", cfg.FFNKind(), FFNSwiGLU)
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
				if cfg.AttentionKind() != AttentionGatedGQA {
					t.Errorf("AttentionKind() = %v, want %v", cfg.AttentionKind(), AttentionGatedGQA)
				}
				if cfg.FFNKind() != FFNRoutedMoE {
					t.Errorf("FFNKind() = %v, want %v", cfg.FFNKind(), FFNRoutedMoE)
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

func TestBuildLayerProgramAndCost(t *testing.T) {
	cfg := ArchConfig{
		HiddenSize:        4096,
		IntermediateSize:  14336,
		NumHiddenLayers:   32,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8,
		HeadDim:           128,
		RMSNormEps:        1e-5,
		WeightDType:       NVFP4,
		KVDType:           E4M3,
	}
	prog, err := BuildLayerProgram(cfg, 4, 1024)
	if err != nil {
		t.Fatalf("BuildLayerProgram: %v", err)
	}
	if len(prog.Ops) == 0 {
		t.Fatalf("expected non-empty Ops")
	}

	breakdown, err := EstimateLayerCost(cfg, 4, 1024, "NVIDIA H200")
	if err != nil {
		t.Fatalf("EstimateLayerCost: %v", err)
	}
	if breakdown.Mode != ModePersistentMegakernel {
		t.Errorf("Mode = %q, want %q", breakdown.Mode, ModePersistentMegakernel)
	}
	if breakdown.TotalNs <= 0 || breakdown.TotalBytes == 0 || breakdown.TotalFLOPs == 0 {
		t.Errorf("unexpected zero cost totals: %+v", breakdown)
	}
}

func TestBuildMultiBucketDispatcherSource(t *testing.T) {
	buckets, err := ValidateBuckets([]int{128, 1, 16, 1})
	if err != nil {
		t.Fatalf("ValidateBuckets: %v", err)
	}
	if len(buckets) != 3 || buckets[0] != 1 || buckets[1] != 16 || buckets[2] != 128 {
		t.Fatalf("ValidateBuckets = %v, want [1 16 128]", buckets)
	}
	plans := []BucketPlanSpec{
		{M: 1, PlanName: "layer_m1", Mode: ModePersistentMegakernel, TotalNs: 100, PlanSource: "__global__ void k1() {}\n"},
		{M: 16, PlanName: "layer_m16", Mode: ModeMultiLaunch, TotalNs: 200, PlanSource: "__global__ void k16() {}\n"},
		{M: 128, PlanName: "layer_m128", Mode: ModeStreamKWarpSpec, TotalNs: 500, PlanSource: "__global__ void k128() {}\n"},
	}
	src := BuildMultiBucketDispatcherSource("// prelude\n", plans, []string{"const uint16_t* x", "uint16_t* out"}, []string{"x", "out"})
	for _, want := range []string{
		"inline void dispatch_layer_m(int M, const uint16_t* x, uint16_t* out, cudaStream_t stream)",
		"if (M <= 1) {",
		"layer_m1_run(x, out, stream);",
		"if (M <= 16) {",
		"layer_m16_run(x, out, stream);",
		"layer_m128_run(x, out, stream);",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("dispatcher source missing %q", want)
		}
	}
}

func ExampleParseArchConfig() {
	cfg, err := ParseArchConfig([]byte(`{
		"hidden_size": 4096,
		"intermediate_size": 14336,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"quantization_config": {"quant_method": "nvfp4"}
	}`))
	if err != nil {
		panic(err)
	}
	fmt.Println(cfg.HiddenSize, cfg.HeadDim, cfg.WeightFormat(), SelectLayerExecutionMode(4))
	// Output:
	// 4096 128 nvfp4 persistent_megakernel
}
