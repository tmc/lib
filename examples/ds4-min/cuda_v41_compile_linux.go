//go:build linux

package main

import "github.com/tmc/lib/nvidia/x/nvrtcx"

// These are CUDA device integer widths, not host libc typedefs. This follows
// compileFlashVisionCUDA's owned NVRTC header pattern; int64_t is needed by
// the pair-pooling kernel's negative carry-row sentinel.
const cudaV41DeviceStdint = `
#ifndef DS4_V41_DEVICE_STDINT_H
#define DS4_V41_DEVICE_STDINT_H
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
#endif
`

const cudaV41CompilePrefix = `
#include <stdint.h>
static_assert(sizeof(uint32_t)==4, "uint32 width");
static_assert(sizeof(uint64_t)==8, "uint64 width");
static_assert(sizeof(int64_t)==8, "int64 width");
static_assert(int64_t(-1)<0, "int64 signed");
static_assert(sizeof(float)==4, "float width");
`

func compileCUDAV41Device(source string, options ...nvrtcx.Option) (*nvrtcx.CompiledModule, error) {
	options = append(options, nvrtcx.WithHeader("stdint.h", cudaV41DeviceStdint))
	return nvrtcx.Compile(cudaV41CompilePrefix+source, options...)
}
