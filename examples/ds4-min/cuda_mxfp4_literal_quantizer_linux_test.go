//go:build linux

package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func literalMXFP4Quantizer(dir string) (string, error) {
	quant, err := literalMXFP4Read(dir, "quantize.cu", "752c02fc5a17c1f884ba6acc6eed9419da7fc1d79bb6aa5f45a9837874744029")
	if err != nil {
		return "", err
	}
	common, err := literalMXFP4Read(dir, "common.cuh", "ae55a86b4b7b3dd9a2478cfa82df4dbebf61f496f12eb61a5844d79a6a06e079")
	if err != nil {
		return "", err
	}
	scale, err := literalMXFP4Part(quant, "__device__ __forceinline__ uint8_t compute_e8m0_scale", "static __global__ void quantize_mmq_nvfp4")
	if err != nil {
		return "", err
	}
	exponent, err := literalMXFP4Part(common, "static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32", "static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32")
	if err != nil {
		return "", err
	}
	kernel, err := literalMXFP4Part(quant, "static __global__ void quantize_mmq_mxfp4", "template <mmq_q8_1_ds_layout ds_layout>")
	if err != nil {
		return "", err
	}
	// Only linkage changes. Preserve literal C arithmetic, hardware FP4
	// conversion, warp geometry, source row mapping and block serialization.
	kernel = strings.Replace(kernel, "static __global__", "extern \"C\" __global__", 1)
	return literalMXFP4Types + "#include <cuda_fp4.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#define CUDART_VERSION 13040\n#define WARP_SIZE 32\n#define QK_MXFP4 32\nstruct block_fp4_mmq { uint32_t d4[4]; uint8_t qs[128]; };\n" + scale + exponent + kernel, nil
}

// NVRTC has no host stdint.h search path. Only the oracle's prefix changes;
// the hash-pinned C arithmetic below retains its original exact-width names.
const literalMXFP4Types = `#if defined(__CUDACC_RTC__)
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;
#else
#include <stdint.h>
#endif
static_assert(sizeof(uint8_t) == 1 && uint8_t(-1) > 0, "uint8_t");
static_assert(sizeof(uint32_t) == 4 && uint32_t(-1) > 0, "uint32_t");
static_assert(sizeof(int32_t) == 4 && int32_t(-1) < 0, "int32_t");
static_assert(sizeof(int64_t) == 8 && int64_t(-1) < 0, "int64_t");
`

func literalMXFP4Read(dir, name, want string) (string, error) {
	b, err := os.ReadFile(filepath.Join(dir, "cuda", "mmq", name))
	if err != nil {
		return "", err
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != want {
		return "", fmt.Errorf("pinned C %s hash mismatch", name)
	}
	return string(b), nil
}

func literalMXFP4Part(source, start, end string) (string, error) {
	i := strings.Index(source, start)
	if i < 0 {
		return "", fmt.Errorf("missing literal C start %q", start)
	}
	j := strings.Index(source[i:], end)
	if j < 0 {
		return "", fmt.Errorf("missing literal C end %q", end)
	}
	return source[i : i+j], nil
}
