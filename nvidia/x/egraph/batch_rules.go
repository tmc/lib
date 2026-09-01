package egraph

// BatchGEMMRules returns equality saturation rewrite rules for batching,
// transpositions, and large GEMM tile selection.
func BatchGEMMRules() []*Rewrite {
	return []*Rewrite{
		// 1. Convert chained elementwise activations into batched tensor fusions
		MustRewrite("fuse_batch_swiglu",
			"(gemv (mul (silu (gemv ?x ?w_g)) (gemv ?x ?w_u)) ?w_d)",
			"(fused_swiglu_gemv ?x ?w_g ?w_u ?w_d)"),

		// 2. Col-Major x Col-Major Matmul swap trick for cuBLASLt:
		// C^T = (A x B)^T = B^T x A^T
		MustRewrite("cublaslt_swap_transpose",
			"(matmul ?a ?b)",
			"(cublaslt_gemm ?b ?a)"),

		// 3. Batched Multi-Head Projection Fusion
		MustRewrite("fuse_batched_qkv",
			"(add (gemv ?x ?w_q) (add (gemv ?x ?w_k) (gemv ?x ?w_v)))",
			"(fused_batched_qkv ?x ?w_q ?w_k ?w_v)"),
	}
}
