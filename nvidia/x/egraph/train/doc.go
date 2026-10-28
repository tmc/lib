// Package train emits fused CUDA C++ backward and optimizer kernels for
// training workflows, including RMSNorm backward, SwiGLU backward,
// cross-entropy with optional logit softcapping backward, Blackwell hardware
// stochastic rounding (cvt.rs.satfinite.e2m1x2.f32), dual-axis NVFP4 gradient
// quantization, Blackwell tcgen05 dGrad/wGrad backward GEMMs, and fused AdamW
// with FP32 master weights.
//
// Each kernel descriptor validates its configuration and emits a self-contained
// CUDA C++ translation unit via CUDA, Launcher, and Emit (or Source):
//
//	k := train.RMSNormBackward{Name: "rmsnorm_bwd", Cols: 128, Eps: 1e-5, DType: train.BF16}
//	src, err := k.Emit()
//
// [FusedGRPOChunk] and [MuonNewtonSchulz] emit FP32 SIMT kernels for the
// GRPO loss over an LM head without materializing logits and for Muon's
// Newton–Schulz orthogonalization; [ComputeFusedGRPOChunk] and
// [MuonOrthogonalize] are their host references. These kernels run under
// cudahostemu; they do not use tcgen05.
//
// The package also plans training on the host, without launching anything:
// [SolveJointRematC2C] chooses per-activation HBM storage, inline
// rematerialization or Grace/Vera C2C offload under memory budgets, and
// [PlanZeroBubble] and [PlanDualPipe] build pipeline-parallel schedules
// that [VerifyScheduleDAG] checks.
package train
