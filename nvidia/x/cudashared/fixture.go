package cudashared

import (
	"fmt"
	"math"
)

// StagedCopyFixtureSource returns a synchronous global-to-shared-to-global
// kernel for a bounded host or native compile fixture. It uses no TMA or
// vendor-specific instructions; native execution remains separately gated.
// The target must have probed shared-memory limits. The source declares a
// four-byte-aligned dynamic pool; compiler and device qualification are separate.
func StagedCopyFixtureSource(target string, elements int) (string, error) {
	if elements <= 0 || int64(elements) > math.MaxInt64/4 {
		return "", fmt.Errorf("elements must be positive and bounded")
	}
	bytes := int64(elements) * 4
	if err := checkStagedCopyResources(target, 0, bytes, bytes); err != nil {
		return "", err
	}
	return stagedCopyFixtureSource(target, elements, 0, bytes, 4)
}

func checkStagedCopyResources(target string, static, dynamic, total int64) error {
	if target != "sm_100a" && target != "sm_121a" {
		return fmt.Errorf("unsupported target %q", target)
	}
	limits, err := LimitsFromDevice(target)
	if err != nil {
		return fmt.Errorf("staged copy target: %w", err)
	}
	if static > limits.StaticMax || dynamic > limits.DynamicMax || total > limits.TotalMax {
		return fmt.Errorf("staged copy exceeds target shared-memory limits")
	}
	return nil
}

func stagedCopyFixtureSource(target string, elements int, offset, sharedBytes, alignment int64) (string, error) {
	if elements <= 0 {
		return "", fmt.Errorf("elements must be positive")
	}
	if offset < 0 || sharedBytes <= 0 || offset > sharedBytes || int64(elements) > (sharedBytes-offset)/4 {
		return "", fmt.Errorf("shared fixture allocation is too small")
	}
	if alignment < 4 || alignment > 128 || alignment&(alignment-1) != 0 {
		return "", fmt.Errorf("staged copy shared alignment must be a power of two from 4 to 128 bytes")
	}
	return fmt.Sprintf(`// cudashared fixture target=%s elements=%d shared_offset=%d shared_bytes=%d
extern "C" __global__ void staged_copy(const float *in, float *out) {
  extern __shared__ __align__(%d) unsigned char raw[];
  float *tile = reinterpret_cast<float *>(raw + %d);
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < %d) tile[threadIdx.x] = in[i];
  __syncthreads();
  if (i < %d) out[i] = tile[threadIdx.x];
}
`, target, elements, offset, sharedBytes, alignment, offset, elements, elements), nil
}

// StagedCopyFixture consumes a validated plan region and returns the source
// plus the dynamic byte count required by the launch contract.
// Only the dynamic pool is supported, with alignments at most 128 bytes.
// Target limits are checked independently of the plan's supplied limits.
// The caller owns input and output buffers through completion. Emitting an
// alignment declaration does not qualify its compiler or device behavior.
func StagedCopyFixture(plan *Plan, target string, elements int) (string, int64, error) {
	if plan == nil {
		return "", 0, fmt.Errorf("nil shared plan")
	}
	if err := checkStagedCopyResources(target, plan.StaticBytes(), plan.DynamicBytes(), plan.FootprintBytes()); err != nil {
		return "", 0, err
	}
	if plan.StaticBytes() != 0 {
		return "", 0, fmt.Errorf("staged copy supports only the dynamic shared pool")
	}
	var region Region
	alignment := int64(4)
	for _, r := range plan.Regions() {
		if r.Dynamic && r.Alignment > alignment {
			alignment = r.Alignment
		}
		if r.Name == "stage0" {
			region = r
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
	src, err := stagedCopyFixtureSource(target, elements, region.Offset, launchBytes, alignment)
	if err != nil {
		return "", 0, err
	}
	return src, launchBytes, nil
}
