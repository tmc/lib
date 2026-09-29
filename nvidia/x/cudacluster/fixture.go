package cudacluster

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudashared"
)

const fixtureLaunchFailureCode = 6

// Mutant names a deliberate defect injected into the reduction fixture.
//
// Each one exists to make a passing test falsifiable. A DSMEM implementation
// that quietly hands back the calling CTA's own window passes every test that
// only checks for a plausible number, so the test that would catch it must be
// shown to fail when the defect is present.
type Mutant string

// The mutants. MutantNone is the fixture under test.
const (
	MutantNone Mutant = ""

	// MutantLocalWindow maps every rank to the calling CTA's own window.
	// The reduction then sums the local seed cluster_size times instead of
	// the distinct per-rank seeds, so the expected total is wrong.
	// TestNativeClusterRemoteWindowIsRemote must fail under it.
	MutantLocalWindow Mutant = "local-window"

	// MutantNoSync drops the cluster barrier between publishing a seed and
	// reading a peer's. The reduction then races. If it still produces the
	// right total deterministically, the fixture is not exercising a
	// cross-CTA dependency and proves nothing.
	MutantNoSync Mutant = "no-sync"
)

// ReductionFixture returns CUDA C++ for the all-to-one reduction: every CTA in
// a cluster publishes a distinguishable seed into its own shared window, the
// cluster synchronizes, and rank 0 reads every peer's window and sums them.
//
// The seeds are distinguishable on purpose. Equal seeds would make a local
// window and a remote window produce the same total, which is exactly the
// failure the per-rank values rule out.
//
// clusterSize is compiled in via __cluster_dims__, so an ordinary launch
// configuration carries the cluster shape. target selects a recorded cluster
// capability and is checked before the source is generated. Sizes from 2 to 8
// are accepted because the fixture does not opt into non-portable clusters.
func ReductionFixture(target string, clusterSize int, mut Mutant) (string, error) {
	if clusterSize < 2 {
		return "", fmt.Errorf("cudacluster: reduction fixture needs at least 2 CTAs, got %d", clusterSize)
	}
	if clusterSize > 8 {
		return "", fmt.Errorf("cudacluster: reduction fixture supports at most 8 CTAs, got %d", clusterSize)
	}
	switch mut {
	case MutantNone, MutantLocalWindow, MutantNoSync:
	default:
		return "", fmt.Errorf("cudacluster: unknown mutant %q", mut)
	}

	plan, err := cudashared.NewPlan(cudashared.Limits{StaticMax: 49152, DynamicMax: 102400, TotalMax: 102400, Source: "fixture"})
	if err != nil {
		return "", err
	}
	region, err := plan.Add(cudashared.Request{Name: "window", Size: 4, Alignment: 4})
	if err != nil {
		return "", err
	}
	c, err := Open(target)
	if err != nil {
		return "", err
	}
	sharedAddr, err := EmitSharedAddr("%0", "%1", plan, region)
	if err != nil {
		return "", err
	}
	rank := "%3"
	if mut == MutantLocalWindow {
		rank = "%4"
	}
	read, err := c.EmitReadU32("%0", "%1", "%2", plan, region, rank)
	if err != nil {
		return "", err
	}
	sync := strings.Join(EmitClusterSync(), "\\n")
	if mut == MutantNoSync {
		sync = ""
	}
	readAsm := strings.Join(read, "\\n")

	var b strings.Builder
	rankArg := ""
	if mut == MutantLocalWindow {
		rankArg = ", \"r\"(rank)"
	}
	fmt.Fprintf(&b, `#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// seed_for gives each rank a distinct value. Distinctness is the control: a
// local window and a remote window are indistinguishable when every CTA
// publishes the same number.
__device__ __forceinline__ unsigned seed_for(unsigned rank) { return rank * 100u + 7u; }

__global__ void __cluster_dims__(%d, 1, 1) dsmem_reduce(unsigned *out) {
    __shared__ unsigned window;
    cg::cluster_group cluster = cg::this_cluster();
	unsigned rank = cluster.block_rank();
	if (threadIdx.x == 0) window = seed_for(rank);
	unsigned base = 0;
	asm volatile("%s" : "=r"(base) : "r"((unsigned)__cvta_generic_to_shared(&window)));
	asm volatile("%s" ::: "memory");

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        unsigned total = 0;
        for (unsigned r = 0; r < cluster.num_blocks(); ++r) {
            unsigned value = 0;
            unsigned mapped = 0;
            asm volatile("%s" : "=r"(value), "=&r"(mapped) : "r"(base), "r"(r)%s);
            total += value;
        }
        *out = total;
	    }
	asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;" ::: "memory");
}
	`, clusterSize, sharedAddr, sync, readAsm, rankArg)
	return strings.TrimRight(b.String(), "\t\r\n") + "\n", nil
}

// ExpectedReduction returns the total the unmutated fixture must produce: the
// sum of the distinct per-rank seeds.
func ExpectedReduction(clusterSize int) uint32 {
	var total uint32
	for r := 0; r < clusterSize; r++ {
		total += uint32(r)*100 + 7
	}
	return total
}

// cooperativeReductionFixture is an independent CUDA C++ oracle for the
// emitted-PTX fixture. It uses cooperative_groups for mapping and sync.
func cooperativeReductionFixture(clusterSize int) string {
	return fmt.Sprintf(`#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
namespace cg = cooperative_groups;
__global__ void __cluster_dims__(%d,1,1) oracle(unsigned *out) {
  __shared__ unsigned window;
  cg::cluster_group c = cg::this_cluster();
  if (threadIdx.x == 0) window = c.block_rank() * 100u + 7u;
  c.sync();
  if (threadIdx.x == 0 && c.block_rank() == 0) {
    unsigned total = 0;
    for (unsigned r = 0; r < c.num_blocks(); ++r) total += *c.map_shared_rank(&window, r);
    *out = total;
  }
  c.sync();
}
int main() {
  unsigned *d = 0, h = 0;
  if (cudaMalloc(&d, sizeof h) != cudaSuccess) return 2;
  if (cudaMemset(d, 0, sizeof h) != cudaSuccess) return 3;
  oracle<<<%d,32>>>(d);
  if (cudaGetLastError() != cudaSuccess) return %d;
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;
  if (cudaMemcpy(&h,d,sizeof h,cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
  printf("total %%u\n", h); cudaFree(d); return 0;
}
`, clusterSize, clusterSize, fixtureLaunchFailureCode)
}
