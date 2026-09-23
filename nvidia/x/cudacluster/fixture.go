package cudacluster

import (
	"fmt"
	"strings"
)

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
// configuration carries the cluster shape.
func ReductionFixture(clusterSize int, mut Mutant) (string, error) {
	if clusterSize < 2 {
		return "", fmt.Errorf("cudacluster: reduction fixture needs at least 2 CTAs, got %d", clusterSize)
	}
	switch mut {
	case MutantNone, MutantLocalWindow, MutantNoSync:
	default:
		return "", fmt.Errorf("cudacluster: unknown mutant %q", mut)
	}

	rankExpr := "r"
	if mut == MutantLocalWindow {
		rankExpr = "cluster.block_rank()"
	}
	sync := "    cluster.sync();"
	if mut == MutantNoSync {
		sync = "    // MutantNoSync: cluster.sync() removed"
	}

	var b strings.Builder
	fmt.Fprintf(&b, `#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// seed_for gives each rank a distinct value. Distinctness is the control: a
// local window and a remote window are indistinguishable when every CTA
// publishes the same number.
__device__ __forceinline__ unsigned seed_for(unsigned rank) { return rank * 100u + 7u; }

__global__ void __cluster_dims__(%d, 1, 1) dsmem_reduce(unsigned *out) {
    __shared__ unsigned window;
    cg::cluster_group cluster = cg::this_cluster();

    if (threadIdx.x == 0) window = seed_for(cluster.block_rank());
%s

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        unsigned total = 0;
        for (unsigned r = 0; r < cluster.num_blocks(); ++r) {
            unsigned *peer = cluster.map_shared_rank(&window, %s);
            total += *peer;
        }
        *out = total;
    }
%s
}
`, clusterSize, sync, rankExpr, sync)
	return b.String(), nil
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
