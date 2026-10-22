//go:build linux

package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Builds literal pinned C kernels with bounded single-column ABI adaptation.
// No C arithmetic expressions are replaced; the CUDA toolchain remains part
// of the gate, not an assumed reproduction of a retained upstream object.
func v41AlignedLiteralModule(t *testing.T, r *cudaFoundation) (driver.Module, driver.Function, driver.Function) {
	t.Helper()
	repo := os.Getenv("DS41_ORACLE")
	if repo == "" {
		t.Fatal("DS41_ORACLE required for aligned literal device oracle")
	}
	read := func(path, want string) string {
		b, e := exec.Command("git", "-C", repo, "show", "11e8cad4473716a41baf4d62e92ed9764a00fce6:"+path).Output()
		if e != nil {
			t.Fatal(e)
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != want {
			t.Fatalf("source pin mismatch %s", path)
		}
		return string(b)
	}
	quant := read("cuda/mmq/quantize.cu", "752c02fc5a17c1f884ba6acc6eed9419da7fc1d79bb6aa5f45a9837874744029")
	common := read("cuda/mmq/common.cuh", "ae55a86b4b7b3dd9a2478cfa82df4dbebf61f496f12eb61a5844d79a6a06e079")
	mmq := read("cuda/mmq/ds4_mmq.cu", "5c414b7840bbaa564df3c78f5ced73816cc280718f2a0189a4cbdb7abd545acd")
	function := func(s, marker string) string {
		start := strings.Index(s, marker)
		if start < 0 {
			t.Fatalf("literal function absent %s", marker)
		}
		body := strings.Index(s[start:], "{")
		if body < 0 {
			t.Fatal("literal body absent")
		}
		depth := 0
		for i := start + body; i < len(s); i++ {
			if s[i] == '{' {
				depth++
			}
			if s[i] == '}' {
				depth--
				if depth == 0 {
					return s[start:i+1] + "\n"
				}
			}
		}
		t.Fatal("unterminated literal body")
		return ""
	}
	max := function(common, "template<int width = WARP_SIZE>\nstatic __device__ __forceinline__ float warp_reduce_max(float x)")
	sum := function(common, "template<int width = WARP_SIZE>\nstatic __device__ __forceinline__ float warp_reduce_sum(float x)")
	q := function(quant, "__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)\nstatic __global__ void quantize_q8_1(")
	q = strings.Replace(q, "__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)\nstatic __global__ void quantize_q8_1(", "extern \"C\" __global__ __launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1) void ds4_v41_literal_aligned_quantize(", 1)
	if strings.Count(q, "const uint3 ne2") != 1 || strings.Count(q, "fastdiv(blockIdx.z, ne2)") != 1 {
		t.Fatal("literal quantizer ABI adaptation sites changed")
	}
	q = strings.Replace(q, "const uint3 ne2", "const uint32_t ne2_z", 1)
	q = strings.Replace(q, "fastdiv(blockIdx.z, ne2)", "(blockIdx.z / ne2_z)", 1)
	q = strings.ReplaceAll(q, "ne2.z", "ne2_z")
	m := function(mmq, "template <int WARPS_PER_BLOCK>\n__global__ void q8_0_aligned_dense_vec_kernel(")
	m = strings.Replace(m, "template <int WARPS_PER_BLOCK>\n__global__ void q8_0_aligned_dense_vec_kernel(", "extern \"C\" __global__ void ds4_v41_literal_aligned_matvec(", 1)
	m = strings.ReplaceAll(m, "WARPS_PER_BLOCK", "(blockDim.x / 32)")
	prefixEnd := strings.Index(cudaV41AlignedQ8Source, "// Pinned ds4_repack.cu")
	if prefixEnd < 0 {
		t.Fatal("owned type prefix absent")
	}
	source := cudaV41AlignedQ8Source[:prefixEnd] + "\n#define WARP_SIZE 32\n#define QK8_1 32\n#define CUDA_QUANTIZE_BLOCK_SIZE 256\n#define ggml_cuda_dp4a __dp4a\ntypedef v41_q81 block_q8_1;\n" + max + sum + q + m
	p, e := compileCUDAV41Device(source, nvrtcx.WithName("literal_v41_aligned_q8.cu"), nvrtcx.WithDeviceArch(r.device), nvrtcx.WithFlag("--use_fast_math"))
	if e != nil {
		t.Fatal(e)
	}
	module, _, e := p.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if module != 0 {
		t.Cleanup(func() {
			if e := r.device.WithContext(func() error {
				if e := syncStream(r.stream()); e != nil {
					return e
				}
				return module.Close()
			}); e != nil {
				t.Error(e)
			}
		})
	}
	if e != nil {
		t.Fatal(e)
	}
	var qfn, mfn driver.Function
	if e := r.device.WithContext(func() error {
		var e error
		qfn, e = module.Function("ds4_v41_literal_aligned_quantize")
		if e != nil {
			return e
		}
		mfn, e = module.Function("ds4_v41_literal_aligned_matvec")
		return e
	}); e != nil {
		t.Fatal(e)
	}
	return module, qfn, mfn
}
