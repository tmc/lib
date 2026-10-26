// Package train emits fused CUDA C++ backward and optimizer kernels for
// training workflows, including RMSNorm backward, SwiGLU backward,
// cross-entropy with optional logit softcapping backward, and fused AdamW
// with FP32 master weights.
//
// Each kernel descriptor validates its configuration and emits a self-contained
// CUDA C++ translation unit via CUDA, Launcher, and Emit:
//
//	k := train.RMSNormBackward{Name: "rmsnorm_bwd", Cols: 128, Eps: 1e-5, DType: train.BF16}
//	src, err := k.Emit()
package train
