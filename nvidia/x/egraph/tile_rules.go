package egraph

// TransformerTileRules returns rewrite rules that pattern match and collapse
// composite transformer subgraphs into unified declarative Tile IR operations
// targeting __tile_global__ execution.
func TransformerTileRules() []*Rewrite {
	return []*Rewrite{
		// 1. RMSNorm -> QKV Projections (monolithic / batched projection)
		MustRewrite("fuse_rmsnorm_qkv_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_qkv)",
			"(tile_rmsnorm_qkv ?x ?w_norm ?eps ?w_qkv)"),
		MustRewrite("fuse_rmsnorm_qkv_matmul",
			"(matmul (rmsnorm ?x ?w_norm ?eps) ?w_qkv)",
			"(tile_rmsnorm_qkv ?x ?w_norm ?eps ?w_qkv)"),

		// 2. RMSNorm -> Split Q, K, V Projections
		MustRewrite("fuse_rmsnorm_qkv_split_gemv",
			"(add (gemv (rmsnorm ?x ?w_norm ?eps) ?w_q) (add (gemv (rmsnorm ?x ?w_norm ?eps) ?w_k) (gemv (rmsnorm ?x ?w_norm ?eps) ?w_v)))",
			"(tile_rmsnorm_qkv_split ?x ?w_norm ?eps ?w_q ?w_k ?w_v)"),
		MustRewrite("fuse_rmsnorm_qkv_split_matmul",
			"(add (matmul (rmsnorm ?x ?w_norm ?eps) ?w_q) (add (matmul (rmsnorm ?x ?w_norm ?eps) ?w_k) (matmul (rmsnorm ?x ?w_norm ?eps) ?w_v)))",
			"(tile_rmsnorm_qkv_split ?x ?w_norm ?eps ?w_q ?w_k ?w_v)"),

		// 3. RMSNorm -> Single Projection (e.g. Attention Out or MLP In)
		MustRewrite("fuse_rmsnorm_proj_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(tile_rmsnorm_proj ?x ?w_norm ?eps ?w_proj)"),
		MustRewrite("fuse_rmsnorm_proj_matmul",
			"(matmul (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(tile_rmsnorm_proj ?x ?w_norm ?eps ?w_proj)"),

		// 4. Matmul / GEMV -> SwiGLU FFN (Llama / DeepSeek ds4)
		MustRewrite("fuse_swiglu_gemv",
			"(gemv (mul (silu (gemv ?x ?w_g)) (gemv ?x ?w_u)) ?w_d)",
			"(tile_swiglu ?x ?w_g ?w_u ?w_d)"),
		MustRewrite("fuse_swiglu_matmul",
			"(matmul (mul (silu (matmul ?x ?w_g)) (matmul ?x ?w_u)) ?w_d)",
			"(tile_swiglu ?x ?w_g ?w_u ?w_d)"),
		MustRewrite("fuse_swiglu_swish_gemv",
			"(gemv (mul (swish (gemv ?x ?w_g)) (gemv ?x ?w_u)) ?w_d)",
			"(tile_swiglu ?x ?w_g ?w_u ?w_d)"),
		MustRewrite("fuse_swiglu_swish_matmul",
			"(matmul (mul (swish (matmul ?x ?w_g)) (matmul ?x ?w_u)) ?w_d)",
			"(tile_swiglu ?x ?w_g ?w_u ?w_d)"),

		// 5. Matmul / GEMV -> GeLU FFN
		MustRewrite("fuse_gelu_gemv",
			"(gemv (gelu (gemv ?x ?w_in)) ?w_out)",
			"(tile_gelu ?x ?w_in ?w_out)"),
		MustRewrite("fuse_gelu_matmul",
			"(matmul (gelu (matmul ?x ?w_in)) ?w_out)",
			"(tile_gelu ?x ?w_in ?w_out)"),

		// 6. Gated GeLU (GeGLU)
		MustRewrite("fuse_geglu_gemv",
			"(gemv (mul (gelu (gemv ?x ?w_g)) (gemv ?x ?w_u)) ?w_d)",
			"(tile_geglu ?x ?w_g ?w_u ?w_d)"),
		MustRewrite("fuse_geglu_matmul",
			"(matmul (mul (gelu (matmul ?x ?w_g)) (matmul ?x ?w_u)) ?w_d)",
			"(tile_geglu ?x ?w_g ?w_u ?w_d)"),
	}
}

// TMALayoutRules returns rewrite rules that introduce tile layout hints and
// ensure memory accesses satisfy hardware alignment and TMA constraints,
// preventing the TMA instruction selection failures recorded in tileiras remarks.
func TMALayoutRules() []*Rewrite {
	return []*Rewrite{
		// 1. Enforce 16-byte alignment and 2D box constraints for global memory loads
		MustRewrite("tma_align_load",
			"(tile_load ?p ?shape)",
			"(tile_tma_load (assume_aligned ?p 16) (tma_box ?shape 16))"),

		// 2. Enforce 16-byte alignment and 2D box constraints for global memory stores
		MustRewrite("tma_align_store",
			"(tile_store ?p ?val ?shape)",
			"(tile_tma_store (assume_aligned ?p 16) ?val (tma_box ?shape 16))"),

		// 3. Coalesce strided memory to contiguous inner dimensions to eliminate
		// "non-contiguous stride along dim 1" TMA failures
		MustRewrite("tma_contiguous_stride",
			"(load_strided ?p ?stride)",
			"(tile_tma_load (coalesce_contiguous ?p) (tma_stride_1 ?stride))"),

		// 4. Align partition views to TMA hardware-aligned spans
		MustRewrite("tma_partition_align",
			"(partition_view ?span ?shape)",
			"(tma_partition_view (assume_aligned ?span 16) (tma_box ?shape 16))"),

		// 5. Transform unaligned / scalar matrix multiplications into Tensor Core MMA
		// preventing the FMA fallback recorded in tileiras remarks
		MustRewrite("tensor_core_shape_align",
			"(tile_mma (shape 1 1 1) ?a ?b ?c)",
			"(tile_mma_aligned (shape 16 16 16) ?a ?b ?c)"),

		// 6. Generic unaligned tile MMA to hardware-aligned Tensor Core MMA
		MustRewrite("tma_mma_align",
			"(tile_mma ?a ?b ?c)",
			"(tile_mma_aligned ?a ?b ?c)"),
	}
}

// SubByteRules returns rewrite rules for sub-byte and quantized weight patterns
// (Q8_0, IQ2_XXS, FP8 e4m3/e5m2, FP4/e2m1). These rules cleanly separate PTX
// bit-unswizzling and unpacking from Tensor Core tile matrix multiplies (HMMA / MMA).
func SubByteRules() []*Rewrite {
	return []*Rewrite{
		// 1. Q8_0: 8-bit quantization with half scale
		MustRewrite("q8_0_matmul_unswizzle",
			"(matmul ?x (dequantize_q8_0 ?w ?scale))",
			"(tile_mma ?x (ptx_unpack_q8_0 ?w ?scale))"),
		MustRewrite("q8_0_gemv_unswizzle",
			"(gemv ?x (dequantize_q8_0 ?w ?scale))",
			"(tile_mma ?x (ptx_unpack_q8_0 ?w ?scale))"),
		MustRewrite("q8_0_matvec_unswizzle",
			"(q8_0_matvec ?w ?x)",
			"(tile_mma ?x (ptx_unpack_q8_0 ?w))"),
		MustRewrite("q8_0_swiglu_unswizzle",
			"(fused_swiglu_q8_0 ?x ?wg ?wu ?wd)",
			"(tile_swiglu ?x (ptx_unpack_q8_0 ?wg) (ptx_unpack_q8_0 ?wu) (ptx_unpack_q8_0 ?wd))"),

		// 2. IQ2_XXS: 2-bit quantization for DeepSeek-V4 (ds4-min) inference
		MustRewrite("iq2_xxs_matmul_unswizzle",
			"(matmul ?x (dequantize_iq2_xxs ?w ?scales ?grid))",
			"(tile_mma ?x (ptx_unswizzle_iq2_xxs ?w ?scales ?grid))"),
		MustRewrite("iq2_xxs_gemv_unswizzle",
			"(gemv ?x (dequantize_iq2_xxs ?w ?scales ?grid))",
			"(tile_mma ?x (ptx_unswizzle_iq2_xxs ?w ?scales ?grid))"),
		MustRewrite("iq2_simd_unswizzle",
			"(iq2_simd ?w ?x ?scale)",
			"(tile_mma ?x (ptx_unswizzle_iq2_xxs ?w ?scale))"),
		MustRewrite("iq2_scalar_unswizzle",
			"(iq2_scalar ?w ?x ?scale)",
			"(tile_mma ?x (ptx_unswizzle_iq2_xxs ?w ?scale))"),
		MustRewrite("iq2_dp4a_lowering",
			"(ptx_unswizzle_iq2_xxs ?w ?scale)",
			"(ptx_dp4a_unpack ?w ?scale)"),

		// 3. FP8 (E4M3 / E5M2): 8-bit floating point formats
		MustRewrite("fp8_e4m3_matmul_unswizzle",
			"(matmul ?x (dequantize_fp8_e4m3 ?w ?scale))",
			"(tile_mma_fp8 ?x (ptx_unpack_fp8_e4m3 ?w) ?scale)"),
		MustRewrite("fp8_e5m2_matmul_unswizzle",
			"(matmul ?x (dequantize_fp8_e5m2 ?w ?scale))",
			"(tile_mma_fp8 ?x (ptx_unpack_fp8_e5m2 ?w) ?scale)"),
		MustRewrite("fp8_gemv_unswizzle",
			"(gemv ?x (dequantize_fp8 ?w ?scale))",
			"(tile_mma_fp8 ?x (ptx_unpack_fp8 ?w) ?scale)"),
		MustRewrite("fp8_scaled_gemv_unswizzle",
			"(scaled_fp8_gemv ?x ?w ?scale)",
			"(tile_mma_fp8 ?x (ptx_unpack_fp8 ?w) ?scale)"),

		// 4. FP4 / E2M1: 4-bit floating point format (NVFP4)
		MustRewrite("fp4_e2m1_matmul_unswizzle",
			"(matmul ?x (dequantize_fp4_e2m1 ?w ?scale))",
			"(mmaf_scaled ?x (ptx_unswizzle_fp4_e2m1 ?w) ?scale)"),
		MustRewrite("fp4_e2m1_gemv_unswizzle",
			"(gemv ?x (dequantize_fp4_e2m1 ?w ?scale))",
			"(mmaf_scaled ?x (ptx_unswizzle_fp4_e2m1 ?w) ?scale)"),
		MustRewrite("fp4_prmt_lowering",
			"(ptx_unswizzle_fp4_e2m1 ?w)",
			"(ptx_prmt_unpack ?w)"),
	}
}

// AllTileRules returns the combined set of transformer layer, TMA layout,
// and sub-byte format rewrite rules.
func AllTileRules() []*Rewrite {
	var rules []*Rewrite
	rules = append(rules, TransformerTileRules()...)
	rules = append(rules, TMALayoutRules()...)
	rules = append(rules, SubByteRules()...)
	return rules
}
