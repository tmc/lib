package cudashared

import (
	"fmt"
	"math"
)

// StagedCopyFixtureSource returns a synchronous global-to-shared-to-global
// kernel for a bounded host or native compile fixture. It uses no TMA or
// vendor-specific instructions; native execution remains separately gated.
func StagedCopyFixtureSource(target string, elements int) (string, error) {
	if elements <= 0 || int64(elements) > math.MaxInt64/4 {
		return "", fmt.Errorf("elements must be positive and bounded")
	}
	return stagedCopyFixtureSource(target, elements, 0, int64(elements)*4)
}

func stagedCopyFixtureSource(target string, elements int, offset, sharedBytes int64) (string, error) {
	if target != "sm_100a" && target != "sm_121a" {
		return "", fmt.Errorf("unsupported target %q", target)
	}
	if elements <= 0 {
		return "", fmt.Errorf("elements must be positive")
	}
	if offset < 0 || sharedBytes <= 0 || offset > sharedBytes || int64(elements) > (sharedBytes-offset)/4 {
		return "", fmt.Errorf("shared fixture allocation is too small")
	}
	return fmt.Sprintf(`// cudashared fixture target=%s elements=%d shared_offset=%d shared_bytes=%d
extern "C" __global__ void staged_copy(const float *in, float *out) {
  extern __shared__ unsigned char raw[];
  float *tile = reinterpret_cast<float *>(raw + %d);
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < %d) tile[threadIdx.x] = in[i];
  __syncthreads();
  if (i < %d) out[i] = tile[threadIdx.x];
}
`, target, elements, offset, sharedBytes, offset, elements, elements), nil
}

// StagedCopyFixture consumes a validated plan region and returns the source
// plus the dynamic byte count required by the launch contract.
func StagedCopyFixture(plan *Plan, target string, elements int) (string, int64, error) {
	if plan == nil {
		return "", 0, fmt.Errorf("nil shared plan")
	}
	var region Region
	for _, r := range plan.Regions() {
		if r.Name == "stage0" {
			region = r
			break
		}
	}
	if region.Size == 0 || !region.Dynamic {
		return "", 0, fmt.Errorf("plan lacks dynamic stage0 region")
	}
	if elements <= 0 || int64(elements) > region.Size/4 || int64(elements) > math.MaxInt64/4 {
		return "", 0, fmt.Errorf("stage0 region too small for fixture")
	}
	if region.Offset > int64(^uint(0)>>1)-int64(elements)*4 {
		return "", 0, fmt.Errorf("stage0 fixture offset overflows")
	}
	if err := plan.ValidateBase(0); err != nil {
		return "", 0, err
	}
	if region.Offset%4 != 0 || region.Alignment < 4 {
		return "", 0, fmt.Errorf("stage0 is not aligned for float32")
	}
	launchBytes := plan.DynamicBytes()
	src, err := stagedCopyFixtureSource(target, elements, region.Offset, launchBytes)
	return src, launchBytes, err
}
