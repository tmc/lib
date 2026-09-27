package cudacluster

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// clusterTarget returns the architecture the reduction fixture is compiled
// for. CUDA_CLUSTER_NATIVE_ARCH names another.
func clusterTarget() string {
	if target := os.Getenv("CUDA_CLUSTER_NATIVE_ARCH"); target != "" {
		return target
	}
	return "sm_121a"
}

// runReduction compiles and runs the fixture and returns the total it produced.
// A non-zero exit from the fixture is returned as an error naming the stage
// that failed, so that a compile failure and a wrong answer are never confused.
func runReduction(t *testing.T, clusterSize int, mut Mutant) (uint32, error) {
	t.Helper()
	if os.Getenv("CUDA_CLUSTER_NATIVE") != "1" {
		t.Skip("native qualification pending: CUDA_CLUSTER_NATIVE=1 required")
	}
	// gpu-lock exports GPU_LOCK_HELD=$$, the holder's pid, so it is never the
	// literal "1". Test for a value, as cudashared and cudatrial do.
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("native gate enabled but no GPU_LOCK_HELD: run under gpu-lock")
	}
	target := clusterTarget()
	if err := Check(target); err != nil {
		t.Fatalf("Check(%q): %v", target, err)
	}
	nvcc, _ := cudatest.NVCC(t)

	src, err := ReductionFixture(clusterSize, mut)
	if err != nil {
		t.Fatal(err)
	}
	src += fmt.Sprintf(`
#include <cuda_runtime.h>
int main() {
  unsigned *d = 0, h = 0;
  if (cudaMalloc(&d, sizeof h) != cudaSuccess) return 2;
  if (cudaMemset(d, 0, sizeof h) != cudaSuccess) return 3;
  dsmem_reduce<<<%d, 32>>>(d);
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;
  if (cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
  cudaFree(d);
  return (int)h;
}
`, clusterSize)

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudacluster-native-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	file := filepath.Join(dir, "reduce.cu")
	if err := os.WriteFile(file, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "reduce")

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	build := exec.CommandContext(ctx, nvcc, "-arch="+target, "-o", bin, file)
	if out, err := build.CombinedOutput(); err != nil {
		return 0, fmt.Errorf("nvcc %s: %v: %s", target, err, out)
	}

	runCtx, runCancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer runCancel()
	run := exec.CommandContext(runCtx, bin)
	out, err := run.CombinedOutput()
	code := run.ProcessState.ExitCode()
	if runCtx.Err() != nil {
		return 0, fmt.Errorf("fixture did not finish: a cluster barrier in divergent flow hangs: %s", out)
	}
	switch code {
	case 2, 3, 4, 5:
		return 0, fmt.Errorf("fixture failed at CUDA stage %d: %v: %s", code, err, out)
	}
	return uint32(code), nil
}

// TestNativeClusterAllToOneReduction is lane 56 slice 6.
func TestNativeClusterAllToOneReduction(t *testing.T) {
	const size = 2
	got, err := runReduction(t, size, MutantNone)
	if err != nil {
		t.Fatal(err)
	}
	if want := ExpectedReduction(size); got != want {
		t.Fatalf("reduction = %d, want %d", got, want)
	}
}

// TestNativeClusterRemoteWindowIsRemote is the test lane 56 says must not be
// weakened, and TestNativeControlLocalWindowMutantFails is what proves it can
// fail. Running them apart would leave the first one unfalsifiable.
func TestNativeClusterRemoteWindowIsRemote(t *testing.T) {
	const size = 4
	got, err := runReduction(t, size, MutantNone)
	if err != nil {
		t.Fatal(err)
	}
	want := ExpectedReduction(size)
	if got != want {
		t.Fatalf("reduction = %d, want %d: the per-rank seeds did not all arrive", got, want)
	}
	// The local-window total is the rank-0 seed repeated. If the two totals
	// were equal the test could not distinguish them, so assert the fixture
	// is capable of the distinction before trusting the result above.
	if local := uint32(size) * 7; local == want {
		t.Fatalf("fixture cannot distinguish remote from local: both totals are %d", want)
	}
}

func TestNativeControlLocalWindowMutantFails(t *testing.T) {
	const size = 4
	got, err := runReduction(t, size, MutantLocalWindow)
	if err != nil {
		t.Fatal(err)
	}
	if want := ExpectedReduction(size); got == want {
		t.Fatalf("local-window mutant produced the correct total %d: "+
			"the remote-window test cannot fail and proves nothing", want)
	}
}

// TestNativeControlNoSyncMutantIsNotDeterministicallyCorrect asserts the
// end-to-end slice exercises a real cross-CTA dependency. A mutant that drops
// the barrier and still gives the right answer every time means the CTAs were
// never actually depending on each other.
//
// It is deliberately tolerant of a hang: removing a barrier can deadlock as
// easily as it can race, and both outcomes falsify determinism. What it will
// not accept is repeated silent correctness.
func TestNativeControlNoSyncMutantIsNotDeterministicallyCorrect(t *testing.T) {
	const size = 4
	want := ExpectedReduction(size)
	correct := 0
	const reps = 5
	for i := 0; i < reps; i++ {
		got, err := runReduction(t, size, MutantNoSync)
		if err != nil {
			t.Logf("rep %d: %v (a failure here also falsifies determinism)", i, err)
			return
		}
		if got == want {
			correct++
		}
	}
	if correct == reps {
		t.Fatalf("no-sync mutant produced the correct total %d in %d of %d runs: "+
			"the reduction is not exercising a cross-CTA dependency", want, correct, reps)
	}
}

// TestControl_NoSimulationWithoutHardware asserts that with the native gate
// closed no code path in this package produces a reduction total. The package
// has no simulation path; this test exists so that adding one fails.
func TestControl_NoSimulationWithoutHardware(t *testing.T) {
	if os.Getenv("CUDA_CLUSTER_NATIVE") == "1" {
		t.Skip("native gate open; this control covers the closed case")
	}
	src, err := ReductionFixture(4, MutantNone)
	if err != nil {
		t.Fatal(err)
	}
	if src == "" {
		t.Fatal("ReductionFixture returned no source")
	}
	// The fixture is CUDA source text, not a result. Nothing in this package
	// converts it into a number without nvcc and a device.
	if err := CheckDSMEM(clusterTarget()); err == nil {
		t.Fatal("CheckDSMEM reported DSMEM established with no instrument having probed it")
	}
}

// TestCluster_RankGeometry verifies that BlockRank and ClusterSize agree with
// launched cluster dimensions.
func TestCluster_RankGeometry(t *testing.T) {
	if os.Getenv("CUDA_CLUSTER_NATIVE") != "1" {
		t.Skip("native qualification pending: CUDA_CLUSTER_NATIVE=1 required")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("native gate enabled but no GPU_LOCK_HELD: run under gpu-lock")
	}
	for _, size := range []int{2, 4} {
		if err := runRankGeometry(t, size); err != nil {
			t.Fatalf("cluster size %d: %v", size, err)
		}
	}
}

const rankGeometryTemplate = `#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned ptx_block_rank() {
    unsigned r;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ unsigned ptx_cluster_size() {
    unsigned s;
    asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(s));
    return s;
}

__global__ void __cluster_dims__({{CLUSTER_SIZE}}, 1, 1) rank_geometry_kernel(unsigned *ranks, unsigned *err) {
    if (threadIdx.x == 0) {
        cg::cluster_group cluster = cg::this_cluster();
        unsigned cg_rank = cluster.block_rank();
        unsigned cg_size = cluster.num_blocks();
        unsigned ptx_rank = ptx_block_rank();
        unsigned ptx_size = ptx_cluster_size();

        if (cg_size != {{CLUSTER_SIZE}} || ptx_size != {{CLUSTER_SIZE}}) {
            atomicExch(err, 1);
            return;
        }
        if (cg_rank != ptx_rank) {
            atomicExch(err, 2);
            return;
        }
        if (ptx_rank >= {{CLUSTER_SIZE}}) {
            atomicExch(err, 3);
            return;
        }
        atomicAdd(&ranks[ptx_rank], 1);
    }
}

int main() {
    const int cluster_size = {{CLUSTER_SIZE}};
    unsigned *d_ranks = NULL;
    unsigned *d_err = NULL;

    if (cudaMalloc(&d_ranks, sizeof(unsigned) * cluster_size) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_ranks failed\n");
        return 2;
    }
    if (cudaMalloc(&d_err, sizeof(unsigned)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_err failed\n");
        return 2;
    }
    if (cudaMemset(d_ranks, 0, sizeof(unsigned) * cluster_size) != cudaSuccess) {
        fprintf(stderr, "cudaMemset d_ranks failed\n");
        return 3;
    }
    if (cudaMemset(d_err, 0, sizeof(unsigned)) != cudaSuccess) {
        fprintf(stderr, "cudaMemset d_err failed\n");
        return 3;
    }

    rank_geometry_kernel<<<cluster_size, 32>>>(d_ranks, d_err);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed\n");
        return 4;
    }

    unsigned h_err = 0;
    if (cudaMemcpy(&h_err, d_err, sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_err failed\n");
        return 5;
    }
    if (h_err != 0) {
        fprintf(stderr, "kernel reported geometry mismatch code %u\n", h_err);
        return 6;
    }

    unsigned h_ranks[cluster_size];
    if (cudaMemcpy(h_ranks, d_ranks, sizeof(h_ranks), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_ranks failed\n");
        return 5;
    }
    for (int i = 0; i < cluster_size; i++) {
        if (h_ranks[i] != 1) {
            fprintf(stderr, "rank %d observed %u times, want 1\n", i, h_ranks[i]);
            return 7;
        }
    }

    cudaFree(d_ranks);
    cudaFree(d_err);
    return 0;
}
`

// runRankGeometry compiles and runs a fixture verifying that BlockRank and
// ClusterSize agree with the launched cluster dimensions.
func runRankGeometry(t *testing.T, clusterSize int) error {
	t.Helper()
	target := clusterTarget()
	if err := Check(target); err != nil {
		t.Fatalf("Check(%q): %v", target, err)
	}
	nvcc, _ := cudatest.NVCC(t)

	src := strings.ReplaceAll(rankGeometryTemplate, "{{CLUSTER_SIZE}}", strconv.Itoa(clusterSize))

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudacluster-native-geom-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	file := filepath.Join(dir, "geom.cu")
	if err := os.WriteFile(file, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "geom")

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	build := exec.CommandContext(ctx, nvcc, "-arch="+target, "-o", bin, file)
	if out, err := build.CombinedOutput(); err != nil {
		return fmt.Errorf("nvcc %s: %w: %s", target, err, out)
	}

	runCtx, runCancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer runCancel()
	run := exec.CommandContext(runCtx, bin)
	out, err := run.CombinedOutput()
	if runCtx.Err() != nil {
		return fmt.Errorf("fixture timed out: %s", out)
	}
	if err != nil {
		return fmt.Errorf("rank geometry fixture failed: %w: %s", err, out)
	}
	return nil
}
