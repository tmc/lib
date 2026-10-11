package cudamoe

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// PrefillW256Layouts returns row-major A, B, C layouts for PrefillW256Source.
// A and B contain IEEE binary16 values; C contains float32 values. Conversion
// and expert routing happen before this kernel and must be measured separately.
func PrefillW256Layouts(k, n int) ([3]cudalayout.Tile, error) {
	var tiles [3]cudalayout.Tile
	if k <= 0 || n <= 0 || k%16 != 0 || n%16 != 0 || uint64(k) > math.MaxUint32 || uint64(n) > math.MaxUint32 {
		return tiles, fmt.Errorf("cudamoe: prefill dimensions must be positive uint32 multiples of 16")
	}
	for i, d := range [][3]int{{256, k, 2}, {k, n, 2}, {256, n, 4}} {
		tile, err := cudalayout.NewTile(d[0], d[1], d[2], cudalayout.WithAlignment(32))
		if err != nil {
			return [3]cudalayout.Tile{}, fmt.Errorf("cudamoe: prefill matrix %d: %w", i, err)
		}
		if _, err := tile.FootprintBytes(); err != nil {
			return [3]cudalayout.Tile{}, fmt.Errorf("cudamoe: prefill matrix %d footprint: %w", i, err)
		}
		tiles[i] = tile
	}
	return tiles, nil
}

// PrefillW256Artifact compiles the W=256 kernel for sm_100a or sm_121a.
// It requires NVRTC and CUDA headers but does not load or launch GPU work.
// Input values must be finite. FP32 accumulation may differ from a scalar
// reference by rounding; this kernel does not promise bitwise equivalence.
func PrefillW256Artifact(target string) (*cudakernel.Artifact, error) {
	return compilePrefill(target, prefillW256Source, "moe_prefill_w256")
}

// PrefillW256ScalarArtifact compiles an ordinary scalar alternative with the
// same matrix layouts and ABI as PrefillW256Artifact. Launch 256*N logical
// threads in a one-dimensional grid. It uses FP32 accumulation over FP16 inputs.
func PrefillW256ScalarArtifact(target string) (*cudakernel.Artifact, error) {
	return compilePrefill(target, prefillW256ScalarSource, "moe_prefill_scalar")
}

func compilePrefill(target, source, entry string) (*cudakernel.Artifact, error) {
	if target != "sm_100a" && target != "sm_121a" {
		return nil, fmt.Errorf("cudamoe: unsupported prefill target %q", target)
	}
	schema, err := prefillW256Schema()
	if err != nil {
		return nil, err
	}
	opts := []nvrtcx.Option{nvrtcx.WithPhysicalArchTarget(target), nvrtcx.WithStandard("c++17"), nvrtcx.WithToolkitIncludes(), nvrtcx.WithOutput(nvrtcx.OutputCUBIN)}
	flags, err := nvrtcx.CompileFlags(opts...)
	if err != nil {
		return nil, fmt.Errorf("cudamoe: prefill compiler options: %w", err)
	}
	cm, err := nvrtcx.Compile(source, opts...)
	if err != nil {
		return nil, fmt.Errorf("cudamoe: compile prefill: %w", err)
	}
	a, err := cm.ToArtifact(entry, schema, target)
	if err != nil {
		return nil, fmt.Errorf("cudamoe: prefill artifact: %w", err)
	}
	a.CompilerOpts = flags
	return a, nil
}

const prefillW256ScalarSource = `#include <cuda_fp16.h>
extern "C" __global__ void moe_prefill_scalar(const half* a, const half* b,
    float* c, unsigned k, unsigned n) {
  if (!a || !b || !c || !k || !n || k%16 || n%16) return;
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= (size_t)256*n) return;
  size_t row=i/n, col=i%n;
  float sum=0;
  for (unsigned p=0;p<k;p++) sum += __half2float(a[row*k+p])*__half2float(b[(size_t)p*n+col]);
  c[i]=sum;
}
`

func prefillW256Schema() (cudakernel.Schema, error) {
	return cudakernel.NewSchema(
		cudakernel.Param{Name: "a", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "b", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "c", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "k", Type: "u32"},
		cudakernel.Param{Name: "n", Type: "u32"},
	)
}

const prefillW256Source = `#include <cuda_fp16.h>
#include <mma.h>
extern "C" __global__ void moe_prefill_w256(const half* a, const half* b,
    float* c, unsigned k, unsigned n) {
  if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
      blockIdx.z != 0 || k == 0 || n == 0 || k % 16 || n % 16 ||
      !a || !b || !c ||
      (((unsigned long long)a | (unsigned long long)b | (unsigned long long)c) & 31)) return;
  if (blockIdx.y >= 4 || blockIdx.x >= n / 16) return;
  unsigned row = blockIdx.y * 64 + (threadIdx.x / 32) * 16;
  unsigned col = blockIdx.x * 16;
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> av;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bv;
  wmma::fragment<wmma::accumulator,16,16,16,float> cv;
  wmma::fill_fragment(cv, 0.0f);
  for (unsigned p = 0; p < k; p += 16) {
    wmma::load_matrix_sync(av, a + (size_t)row*k + p, k);
    wmma::load_matrix_sync(bv, b + (size_t)p*n + col, n);
    wmma::mma_sync(cv, av, bv, cv);
  }
  wmma::store_matrix_sync(c + (size_t)row*n + col, cv, n, wmma::mem_row_major);
}
`
