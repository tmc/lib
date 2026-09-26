package main

import "testing"

func TestDefaultOpRules(t *testing.T) {
	rules, err := readOpRules(defaultOps)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		kernel string
		want   string
	}{
		{"cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params)", "gemm"},
		{"nvjet_sm121_sss_tf32_mma_256x128x32_2_64x64x32_tmaAB_alignCD4_bz_TNNN", "gemm"},
		{"_ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4EEvNT_6ParamsE", "gemm"},
		{"magma_sgemmEx_kernel", "gemm"},
		{"cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, true, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*)", "gemm"},
		{"mlx::core::cu::gemv_single<float, 4, 32>(float const*, float const*, float*, int, int)", "gemm"},
		{"cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_1_128x64x64_4x1x1_cga1x1x1_kernel0_0", "attention"},
		{"FlashAttention", "attention"},
		{"AttnBackwardKV", "attention"},
		{"mlx::core::cu::softmax<float, float, 256, 4, false>(float const*, float*, int)", "softmax"},
		{"SoftmaxBackward", "softmax"},
		{"mlx::core::cu::rope<float, true, false>(float const*, float*, int)", "rope"},
		{"SplitQKVRope", "rope"},
		{"LayerNormBackward", "layernorm"},
		{"mlx::core::cu::layer_norm_vjp<float, true, 1024, 4>(float const*, float*, int)", "layernorm"},
		{"mlx::core::cu::rms_norm<float, 1024, 4>(float const*, float*, int)", "layernorm"},
		{"mlx::core::cu::Db1IBroadcastAEf2OSelectDBC_VVV_b1f2f2_6142509188972423790_strided", "mask"},
		{"mlx::core::cu::ternary_g_nd<mlx::core::Select, bool, float, int, 3>(bool const*, float const*, float const*, float*)", "mask"},
		{"mlx::core::cu::Ff2IBroadcastBGf2IBroadcastCHf2IBroadcastDIf2IDivideAHJf2IErfIKf2IAddGJLf2IMultiplyFKMf2IMultiplyALNf2OMultiplyME_VSSSV_f2f2f2f2f2_6142509188972423790_strided", "activation"},
		{"GeGLUBackward", "activation"},
		{"ReLU", "activation"},
		{"cudaoptim_adamw", "optimizer"},
		{"mlx::core::cu::copy_gg_nd<float, float, int, 2>(float const*, float*)", "copy"},
		{"mlx::core::cu::col_reduce_looped<float, float, mlx::core::cu::Sum, int>(float const*, float*)", "reduce"},
		{"ColumnSumPartial", "reduce"},
		{"mlx::core::cu::gather<float, int, 1, 2>(float const*, float*)", "gather"},
		{"Embed", "gather"},
		{"mlx::core::cu::binary_vv<mlx::core::cu::Add, float, float, 4>(float const*, float const*, float*, int)", "elementwise"},
		{"mlx::core::cu::Cf2IBroadcastBDf2OAddAC_VV_f2f2_6142509188972423790_strided", "elementwise"},
		{"AddBias", "elementwise"},
		{"some_unknown_kernel", "other"},
	}
	for _, tt := range tests {
		key := opKey(tt.kernel, false)
		if got, _ := rules.kernelOp(key); got != tt.want {
			t.Errorf("default op of %q (key %q) = %q, want %q", tt.kernel, key, got, tt.want)
		}
	}
}
