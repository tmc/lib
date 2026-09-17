package cudatmem

import (
	"errors"
	"fmt"
	"strings"
)

// Fixture is a compile-only generated protocol fixture. PTX text is retained
// for an authorized toolchain check; this package does not claim execution.
type Fixture struct {
	Target Target
	PTX    string
	CUDA   string
}

func GenerateFixture(target Target) (Fixture, error) {
	if _, err := capability(target); err != nil {
		return Fixture{}, err
	}
	ptx := fmt.Sprintf(`.version 9.4
.target %s
.address_size 64
.visible .entry cudatmem_fixture() {
  .reg .b32 %%r<5>;
  .shared .align 4 .b32 alloc_slot;
  // One warp in CTA group 1 executes each aligned collective.
  tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [alloc_slot], 32;
  ld.shared.b32 %%r2, [alloc_slot];
  mov.b32 %%r3, 0;
  mov.b32 %%r4, 0;
  tcgen05.st.sync.aligned.32x32b.x2.b32 [%%r2], {%%r3, %%r4};
  tcgen05.wait::st.sync.aligned;
  tcgen05.dealloc.cta_group::1.sync.aligned.b32 %%r2, 32;
  tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
  ret;
}
`, target)
	return Fixture{Target: target, PTX: ptx, CUDA: cudaFixture()}, nil
}

// ValidateFixture checks the required protocol instruction sequence and exact
// target fence. It is a source-level control, not proof of driver execution.
func ValidateFixture(f Fixture) error {
	if f.Target != SM100A {
		return ErrUnsupportedTarget
	}
	expected := []string{"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [alloc_slot], 32;", "tcgen05.st.sync.aligned.32x32b.x2.b32 [%r2], {%r3, %r4};", "tcgen05.wait::st.sync.aligned;", "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r2, 32;", "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"}
	for _, op := range expected {
		if !strings.Contains(f.PTX, op) {
			return fmt.Errorf("cudatmem: fixture missing %s", op)
		}
	}
	if !strings.Contains(f.CUDA, "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32") || !strings.Contains(f.CUDA, "tcgen05.wait::ld.sync.aligned") {
		return errors.New("cudatmem: cuda fixture lacks allocation/load completion")
	}
	return nil
}

func cudaFixture() string {
	return `#include <cuda_runtime.h>
extern "C" __global__ void cudatmem_fixture(float *out) {
  __shared__ unsigned alloc_slot;
  unsigned taddr, a = threadIdx.x + 1, b = threadIdx.x + 101;
  unsigned slot = __cvta_generic_to_shared(&alloc_slot);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" :: "r"(slot));
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr) : "r"(slot) : "memory");
  asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1, %2};" :: "r"(taddr), "r"(a), "r"(b));
  asm volatile("tcgen05.wait::st.sync.aligned;");
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];" : "=r"(a), "=r"(b) : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(taddr));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  out[threadIdx.x] = float(a + b);
}
int main() {
  float *d = nullptr, got[32];
  if (cudaMalloc(&d, 32 * sizeof(float)) != cudaSuccess) return 2;
  if (cudaMemset(d, 0xA5, 32 * sizeof(float)) != cudaSuccess) return 2;
  cudatmem_fixture<<<1, 32>>>(d);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  if (cudaMemcpy(got, d, sizeof(got), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
  cudaFree(d);
  for (int i = 0; i < 32; ++i) if (got[i] != float(102 + 2*i)) return 5;
  return 0;
}
`
}

// GenerateMutantCUDA returns deliberate source mutants for the native oracle.
// Both must fail the all-lane populated-output comparison when executed.
func GenerateMutantCUDA(kind string) (string, error) {
	s := cudaFixture()
	switch kind {
	case "omit-store":
		s = strings.Replace(s, `asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1, %2};" :: "r"(taddr), "r"(a), "r"(b));`, `/* store omitted */`, 1)
	case "perturb-load":
		s = strings.Replace(s, `asm volatile("tcgen05.wait::ld.sync.aligned;");`, `asm volatile("tcgen05.wait::ld.sync.aligned;"); a += 1;`, 1)
	default:
		return "", fmt.Errorf("cudatmem: unknown fixture mutant %q", kind)
	}
	return s, nil
}
