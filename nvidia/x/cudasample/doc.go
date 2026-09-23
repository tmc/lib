// Package cudasample provides a CUDA greedy-argmax kernel and its CPU reference.
//
// [SampleArgmaxCPU] selects a token from host logits without loading CUDA.
// It ignores NaNs, selects the first maximum on ties, and returns -1 when no
// value exceeds negative infinity. It does not apply temperature or top-k sampling.
//
// [Compile] uses NVRTC to compile [ArgmaxKernelTemplate] for a specified compute
// capability. It returns a compiled artifact; callers must load it into their
// CUDA context and launch ArgmaxLogits themselves. The kernel takes a float32
// logits pointer, an int32 output pointer, and an int32 vocabulary size. Launch
// one block of 256 threads, keep both buffers alive through completion, and
// synchronize before reading the output on the host.
package cudasample
