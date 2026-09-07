// Package openxla provides Go bindings for the OpenXLA PJRT C API.
//
// PJRT (Platform Runtime) is OpenXLA's uniform device runtime interface used by JAX, TensorFlow,
// and PyTorch to execute computations across TPUs, GPUs, and CPUs.
//
// PJRT plugins (such as libtpu.so for Cloud TPU, or pjrt_c_api_gpu.so for CUDA) expose the
// C entrypoint GetPjrtApi() returning an Api function table. This package provides high-level,
// idiomatic Go wrappers for managing clients, compiling StableHLO/HLO programs, transferring
// buffers, and executing computations on accelerator hardware without requiring CGO.
package openxla
