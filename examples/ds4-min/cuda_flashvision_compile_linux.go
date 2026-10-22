//go:build linux

package main

import "github.com/tmc/lib/nvidia/x/nvrtcx"

const flashVisionDeviceStdint = `
#ifndef DS4_VISION_DEVICE_STDINT_H
#define DS4_VISION_DEVICE_STDINT_H
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#endif
`

const flashVisionCompilePrefix = `
#include <stdint.h>
#define INFINITY __int_as_float(0x7f800000)
static_assert(sizeof(uint16_t)==2, "uint16 width");
static_assert(sizeof(uint32_t)==4, "uint32 width");
static_assert(sizeof(uint64_t)==8, "uint64 width");
static_assert(sizeof(float)==4, "float width");
`

// compileFlashVisionCUDA supplies only integer widths and exact positive
// float32 infinity. Both actual and literal reference kernels use this support;
// host libc headers are neither required nor admitted by NVRTC.
func compileFlashVisionCUDA(source string, options ...nvrtcx.Option) (*nvrtcx.CompiledModule, error) {
	options = append(options, nvrtcx.WithHeader("stdint.h", flashVisionDeviceStdint))
	return nvrtcx.Compile(flashVisionCompilePrefix+source, options...)
}
