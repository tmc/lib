// Package tensorparallel shards a dense LLaMA decoder layer over TP ranks
// the way Megatron-LM does and builds each rank's kernels with the e-graph
// generator in [github.com/tmc/lib/nvidia/x/egraph/hopper].
//
// [Shard] splits a [Layer] into per-rank plans and two all-reduces:
//
//	attn_in: qkv_r   = cast_bf16(rmsnorm(x)) · w_qkv_rᵀ            column-parallel: heads/TP q, k, v heads
//	o_proj:  part_r  = attn_r · w_o_rᵀ [+ x on rank 0]             row-parallel: columns of w_o for rank r's heads
//	         h       = Σ_r part_r                                   all-reduce
//	ffn:     part_r  = swiglu(cast_bf16(rmsnorm(h)) · w_gu_rᵀ)
//	                   · w_down_rᵀ [+ h on rank 0]                  gate_up column-, down row-parallel
//	         out     = Σ_r part_r                                   all-reduce
//
// Rank 0 adds the residual in its epilogue, so each all-reduce is a plain
// sum. Every rank of a TP group shares one shape, so ranks 1..TP-1 share
// plans; rank 0's differ only in the residual operand. Each all-reduce is a
// [CollectiveNode] carrying one [egraph.CollectiveIntent] per rank, which
// the collective runtime ([egraph.CollectiveSchedule]) validates.
//
// With [FuseReduceAdd] or [FuseMultimem] the row-parallel GEMMs write their
// partials with [hopper.StoreReduceAdd] or [hopper.StoreMultimemReduceAdd]
// into one output that the caller zeroes before the layer, fusing the
// all-reduce into the GEMM epilogue. The caller must still order every
// rank's reads after every rank's adds (a stream barrier, or an NVLink
// barrier across GPUs). Fusion needs the decode shapes (at most 16 rows);
// [FuseMultimem] needs a target of sm_90 or later.
//
// [ShardRows], [ShardCols] and the per-rank helpers [QKV], [GateUp], [O]
// and [Down] cut a checkpoint's row-major weights into a rank's shards.
//
// Evidence: the plans and kernels are generated here; the tests check the
// sharding numerically on the host (a TP=N layer against TP=1) and the
// fused store under cudahostemu. On one GB10 on 2026-10-28,
// examples/nvidia/llama-infer -tp ran Llama-3-8B at TP=1, 2 and 4 in serial,
// streams and green modes, with the all-reduce as a kernel and fused as
// [FuseReduceAdd], and generated the same 8 greedy tokens as the
// single-GPU path. [FuseMultimem] kernels compile for sm_90a and sm_103a
// but have not run. No multi-GPU run has checked any of it.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch
// releases.
package tensorparallel
