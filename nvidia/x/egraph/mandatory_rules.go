package egraph

// MandatoryRewriteRules returns the equality saturation rewrite rules required
// by Lane 07:
//
//	a. Weight & Scale Hoisting (Q2_K, IQ2_XXS)
//	b. Pointwise Norm/Quant Fusion (RMSNorm -> Quant, SwiGLU -> Q8_1)
//	c. Vectorized MoE Gather Tile Scaling (ld.global.v4)
//	d. Prefill Group Width Expansion (W=256)
func MandatoryRewriteRules() []*Rewrite {
	return []*Rewrite{
		// a. Weight & Scale Hoisting: Preload/hoist scales and weights once per block iteration
		MustRewrite("hoist_q2k_scales_weights",
			"(q2_k_matvec (load_unhoisted_q2k ?w ?scales) ?x)",
			"(q2_k_matvec (hoist_block_q2k ?w ?scales) ?x)"),
		MustRewrite("hoist_iq2xxs_scales_weights",
			"(iq2_xxs_matvec (load_unhoisted_iq2xxs ?w ?scales ?grid) ?x)",
			"(iq2_xxs_matvec (hoist_block_iq2xxs ?w ?scales ?grid) ?x)"),

		// b. Pointwise Norm/Quant Fusion:
		// RMSNorm -> Quantization (rms_quant_q8_1) to eliminate intermediate DRAM activation roundtrips
		MustRewrite("fuse_rmsnorm_quant_q8",
			"(quantize_q8 (rmsnorm ?x ?w_norm ?eps))",
			"(rms_quant_q8_1 ?x ?w_norm ?eps)"),
		// Epsilon is optional in the compact IR used by older frontends.
		MustRewrite("fuse_rmsnorm_quant_q8_default_eps",
			"(quantize_q8 (rmsnorm ?x ?w_norm))",
			"(rms_quant_q8_1 ?x ?w_norm)"),
		// SwiGLU -> Q8_1 quantization fusion
		MustRewrite("fuse_swiglu_quant_q8_1",
			"(quantize_q8_1 (mul (silu (gemv ?x ?w_g)) (gemv ?x ?w_u)))",
			"(fused_swiglu_q8_1 ?x ?w_g ?w_u)"),

		// c. Vectorized MoE Gather Tile Scaling:
		// Rewrite expert gather tile width from 8 to 16/32 tokens with 128-bit vectorized loads (ld.global.v4 / uint4)
		MustRewrite("vectorize_moe_gather_w16",
			"(moe_gather ?tokens ?indices 8)",
			"(moe_gather_vec128 ?tokens ?indices 16)"),
		MustRewrite("vectorize_moe_gather_w32",
			"(moe_gather ?tokens ?indices 16)",
			"(moe_gather_vec128 ?tokens ?indices 32)"),

		// d. Prefill Group Width Expansion:
		// Rewrite rules expanding prefill group width (W=256) to enable batched cross-token matrix GEMMs
		MustRewrite("prefill_group_expand_w256",
			"(prefill_group_gemm ?x ?w ?width)",
			"(prefill_group_gemm ?x ?w 256)"),
	}
}
