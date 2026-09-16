package cudacmp

import (
	"testing"
)

func TestFamilyClassifierNormalize(t *testing.T) {
	tests := []struct {
		name   string
		symbol string
		want   string
	}{
		{
			name:   "itanium simple",
			symbol: "_Z10atomicDropPfPKfi",
			want:   "atomicDrop",
		},
		{
			name:   "itanium nested namespace and template",
			symbol: "_ZN3mlx4core2cu14rms_norm_smallI13__nv_bfloat16Li128ELi2ELb1ELi8EEEvPKT_S6_PS4_fjjl",
			want:   "rms_norm_small",
		},
		{
			name:   "itanium rope",
			symbol: "_ZN3mlx4core2cu11rope_singleI13__nv_bfloat16Lb0ELb1EEEvPKT_PS4_PKiffl5uint2",
			want:   "rope_single",
		},
		{
			name:   "demangled with unnamed namespace and template args",
			symbol: "void <unnamed>::gateup_iq2_swiglu_q8_d2r_kernel<(int)32, (int)32>(float*, const float*, int)",
			want:   "gateup_iq2_swiglu_q8_d2r_kernel",
		},
		{
			name:   "demangled with anonymous namespace",
			symbol: "__global__ void (anonymous namespace)::down_iq2_swiglu_q8_d2r_kernel(int, float*)",
			want:   "down_iq2_swiglu_q8_d2r_kernel",
		},
		{
			name:   "demangled simple template",
			symbol: "void repack_weights_aligned_kernel<float>",
			want:   "repack_weights_aligned_kernel",
		},
		{
			name:   "plain identifier",
			symbol: "mul_mat_q",
			want:   "mul_mat_q",
		},
		{
			name:   "empty string",
			symbol: "",
			want:   "",
		},
	}

	fc := DefaultFamilyClassifier
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := fc.Normalize(tt.symbol)
			if got != tt.want {
				t.Errorf("Normalize(%q) = %q, want %q", tt.symbol, got, tt.want)
			}
		})
	}
}

func TestFamilyClassifierClassify(t *testing.T) {
	tests := []struct {
		name   string
		symbol string
		want   string
	}{
		{
			name:   "expert gateup",
			symbol: "void <unnamed>::gateup_iq2_swiglu_q8_d2r_kernel<(int)32, (int)32>",
			want:   FamilyExperts,
		},
		{
			name:   "expert down",
			symbol: "down_iq2_swiglu_q8_d2r_kernel",
			want:   FamilyExperts,
		},
		{
			name:   "expert router",
			symbol: "moe_router_gemv",
			want:   FamilyExperts,
		},
		{
			name:   "dense q8 mul_mat_q",
			symbol: "mul_mat_q",
			want:   FamilyDenseQ8,
		},
		{
			name:   "dense q8 explicit",
			symbol: "dense_q8_kernel",
			want:   FamilyDenseQ8,
		},
		{
			name:   "dense q8 mmq",
			symbol: "mmq_q8_0",
			want:   FamilyDenseQ8,
		},
		{
			name:   "f16 gemm",
			symbol: "mul_mat_f16",
			want:   FamilyF16,
		},
		{
			name:   "attention token tile",
			symbol: "attention_tokentile",
			want:   FamilyAttention,
		},
		{
			name:   "attention rope single",
			symbol: "_ZN3mlx4core2cu11rope_singleI13__nv_bfloat16Lb0ELb1EEEvPKT_PS4_PKiffl5uint2",
			want:   FamilyAttention,
		},
		{
			name:   "attention sdpa",
			symbol: "_ZN3mlx4core2cu59custom_kernel_prealloc_decode_sdpav_bfloat16_t_128_256_16_8EPK13__nv_bfloat16",
			want:   FamilyAttention,
		},
		{
			name:   "norm prep rms norm",
			symbol: "_ZN3mlx4core2cu14rms_norm_smallI13__nv_bfloat16Li128ELi2ELb1ELi8EEEvPKT_S6_PS4_fjjl",
			want:   FamilyNormPrep,
		},
		{
			name:   "norm prep dequantize",
			symbol: "dequantize_row_q8_0",
			want:   FamilyNormPrep,
		},
		{
			name:   "model prep repack",
			symbol: "repack_weights_aligned_kernel",
			want:   FamilyModelPrep,
		},
		{
			name:   "residual elementwise",
			symbol: "binary_vv_add",
			want:   FamilyResidual,
		},
		{
			name:   "residual unclassified",
			symbol: "atomicDrop",
			want:   FamilyResidual,
		},
	}

	fc := DefaultFamilyClassifier
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := fc.Classify(tt.symbol)
			if got != tt.want {
				t.Errorf("Classify(%q) = %q, want %q", tt.symbol, got, tt.want)
			}
		})
	}
}

func BenchmarkFamilyClassifierNormalize(b *testing.B) {
	fc := DefaultFamilyClassifier
	sym := "void <unnamed>::gateup_iq2_swiglu_q8_d2r_kernel<(int)32, (int)32>(float*, const float*, int)"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fc.Normalize(sym)
	}
}

func BenchmarkFamilyClassifierClassify(b *testing.B) {
	fc := DefaultFamilyClassifier
	sym := "_ZN3mlx4core2cu14rms_norm_smallI13__nv_bfloat16Li128ELi2ELb1ELi8EEEvPKT_S6_PS4_fjjl"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fc.Classify(sym)
	}
}
