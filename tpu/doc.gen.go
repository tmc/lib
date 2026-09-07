// Package tpu provides Go bindings for Google Cloud TPU via the OpenXLA PJRT C API.
//
// PJRT (Platform Runtime) is OpenXLA's uniform device runtime interface used by JAX, TensorFlow,
// and PyTorch to execute computations across TPUs, GPUs, and CPUs.
//
// The Cloud TPU runtime is packaged as libtpu.so and exposes the PJRT C API entry point
// GetPjrtApi(). This package binds libtpu.so using purego, enabling pure-Go execution
// on Google Cloud TPU instances (v2, v3, v4, v5e, v5p, etc.) without requiring CGO.
package tpu
