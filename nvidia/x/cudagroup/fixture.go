package cudagroup

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudablock"
)

// Fixture is a generated warp reduction kernel and its launch contract.
type Fixture struct {
	Source     string
	Group      Group
	Collective Collective
	Target     string
	Launch     LaunchRequirements
}

// GenerateReductionFixture emits a CUDA C++ warp reduction with full-mask
// participation. It is a source fixture; compilation and execution require a
// separately authorized CUDA toolchain and device.
func GenerateReductionFixture(group Group) (Fixture, error) {
	return GenerateReductionFixtureForTarget(group, "sm_90")
}

// GenerateReductionFixtureForTarget emits the fixture for an explicitly
// admitted SIMT target. Target names are kept as an allowlist until a
// capability provider supplies newer target records.
func GenerateReductionFixtureForTarget(group Group, target string) (Fixture, error) {
	if group.kind != Warp {
		return Fixture{}, errors.New("reduction fixture requires a warp group")
	}
	switch target {
	case "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_121", "sm_121a":
	default:
		return Fixture{}, fmt.Errorf("unsupported reduction target %q", target)
	}
	c, err := NewCollective(ReduceSum, group, MemoryThread)
	if err != nil {
		return Fixture{}, err
	}
	src := fmt.Sprintf("#include <cuda_runtime.h>\n#include <cstdio>\n%s\n%s", cudablock.Header, fmt.Sprintf(`
extern "C" __global__ void cudagroup_reduce_sum(const float* in, float* out) {
  if (blockDim.x != 32 || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 || threadIdx.x >= 32) return;
  float value = in[threadIdx.x];
  value = warp_reduce_sum<float>(value);
  if (threadIdx.x == 0) *out = value;
}
// generated for %s (%d participants)
	// target=%s
`, group.kind, group.Size(), target))
	launch := LaunchRequirements{Grid: driver.Dim3{X: 1, Y: 1, Z: 1}, Block: driver.Dim3{X: 32, Y: 1, Z: 1}}
	return Fixture{Source: src, Group: group, Collective: c, Target: target, Launch: launch}, nil
}

// ValidateLaunch checks the exact single-warp launch required by this fixture.
func (f Fixture) ValidateLaunch() error {
	if f.Group.kind != Warp || f.Target == "" {
		return errors.New("invalid reduction fixture")
	}
	switch f.Target {
	case "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_121", "sm_121a":
	default:
		return errors.New("unsupported reduction fixture target")
	}
	if f.Launch.Cluster != (driver.Dim3{}) {
		return errors.New("reduction fixture requires an unclustered launch")
	}
	if f.Launch.Grid != (driver.Dim3{X: 1, Y: 1, Z: 1}) || f.Launch.Block != (driver.Dim3{X: 32, Y: 1, Z: 1}) {
		return errors.New("reduction fixture requires a 1x1x1 grid and 32x1x1 block")
	}
	return ValidateLaunch(f.Launch, f.Group)
}

// ReduceSumReference sums in input order. The GPU uses a different addition
// tree; arbitrary floating-point inputs require an explicit tolerance policy.
func ReduceSumReference(values []float32) (float32, error) {
	if len(values) != 32 {
		return 0, errors.New("reference reduction requires 32 values")
	}
	var sum float32
	for _, v := range values {
		sum += v
	}
	return sum, nil
}
