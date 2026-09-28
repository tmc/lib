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

	"github.com/tmc/lib/nvidia/x/cudalock"
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
	if err := requireNativeLock(os.Getenv); err != nil {
		t.Fatalf("native gate enabled without inherited GPU lock: %v", err)
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
  printf("total %%u\n", h);
  return 0;
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

	got, err := compileRunTotal(t, nvcc, target, dir, "reduce", src)
	if err != nil {
		return 0, err
	}
	if mut == MutantNone {
		oracle := cooperativeReductionFixture(clusterSize)
		want, err := compileRunTotal(t, nvcc, target, dir, "oracle", oracle)
		if err != nil {
			return 0, fmt.Errorf("cooperative-groups oracle: %w", err)
		}
		if got != want {
			return 0, fmt.Errorf("emitted PTX total %d differs from cooperative-groups oracle %d", got, want)
		}
	}
	return got, nil
}

func requireNativeLock(getenv func(string) string) error {
	if getenv("CUDA_CLUSTER_NATIVE") != "1" {
		return fmt.Errorf("cudacluster: native gate is closed")
	}
	_, err := cudalock.Inherited(getenv)
	return err
}

func compileRunTotal(t *testing.T, nvcc, target, dir, name, src string) (uint32, error) {
	t.Helper()
	src += "\n#include <cuda_runtime.h>\n#include <stdio.h>\n"
	file := filepath.Join(dir, name+".cu")
	if err := os.WriteFile(file, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, name)

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
	if run.ProcessState == nil {
		return 0, fmt.Errorf("fixture process has no exit status: %s", out)
	}
	code := run.ProcessState.ExitCode()
	if runCtx.Err() != nil {
		return 0, fmt.Errorf("fixture did not finish: a cluster barrier in divergent flow hangs: %s", out)
	}
	switch code {
	case 2, 3, 4, 5:
		return 0, fmt.Errorf("fixture failed at CUDA stage %d: %v: %s", code, err, out)
	case 0:
		return parseTotal(string(out))
	default:
		return 0, fmt.Errorf("fixture exited %d: %v: %s", code, err, out)
	}
}

func parseTotal(out string) (uint32, error) {
	fields := strings.Fields(out)
	for i := 0; i+1 < len(fields); i++ {
		if fields[i] != "total" {
			continue
		}
		n, err := strconv.ParseUint(fields[i+1], 10, 32)
		if err != nil {
			return 0, fmt.Errorf("parse total: %w", err)
		}
		return uint32(n), nil
	}
	return 0, fmt.Errorf("fixture output has no total: %q", out)
}

func TestParseTotal(t *testing.T) {
	for _, tc := range []struct {
		out  string
		want uint32
		bad  bool
	}{
		{"total 628\n", 628, false}, {"", 0, true}, {"total -1", 0, true}, {"total 4294967296", 0, true},
	} {
		got, err := parseTotal(tc.out)
		if (err != nil) != tc.bad || (!tc.bad && got != tc.want) {
			t.Errorf("parseTotal(%q) = %d, %v", tc.out, got, err)
		}
	}
}

func TestNativeGateRejectsPresenceOnlyLockBeforeToolLookup(t *testing.T) {
	env := func(k string) string {
		switch k {
		case "CUDA_CLUSTER_NATIVE":
			return "1"
		case "GPU_LOCK_HELD":
			return "1"
		}
		return ""
	}
	if err := requireNativeLock(env); err == nil {
		t.Fatal("native gate accepted GPU_LOCK_HELD=1")
	}
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
			if !strings.Contains(err.Error(), "fixture did not finish") {
				t.Fatalf("rep %d failed outside runtime timeout: %v", i, err)
			}
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
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled without inherited GPU lock: %v", err)
	}
	if err := runRankGeometry(t, 4); err != nil {
		t.Fatalf("cluster size 4: %v", err)
	}
}

const rankGeometryTemplate = `#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned ptx_block_rank() {
    unsigned r;
    asm("{{BLOCK_RANK}}" : "=r"(r));
    return r;
}

__device__ __forceinline__ unsigned ptx_cluster_size() {
    unsigned s;
    asm("{{CLUSTER_SIZE}}" : "=r"(s));
    return s;
}

__device__ __forceinline__ unsigned ptx_cluster_ctaid_x() {
    unsigned id; asm("{{CTAID_X}}" : "=r"(id)); return id;
}
__device__ __forceinline__ unsigned ptx_cluster_nctaid_x() {
    unsigned n; asm("{{NCTAID_X}}" : "=r"(n)); return n;
}
__device__ __forceinline__ unsigned ptx_cluster_nctaid_y() {
    unsigned n; asm("{{NCTAID_Y}}" : "=r"(n)); return n;
}

__global__ void __cluster_dims__(2, 2, 1) rank_geometry_kernel(unsigned *ranks, unsigned *err) {
    if (threadIdx.x == 0) {
        cg::cluster_group cluster = cg::this_cluster();
        unsigned cg_rank = cluster.block_rank();
        unsigned cg_size = cluster.num_blocks();
        unsigned ptx_rank = ptx_block_rank();
        unsigned ptx_size = ptx_cluster_size();
		unsigned cta_x = ptx_cluster_ctaid_x();
		unsigned ncta_x = ptx_cluster_nctaid_x();
		unsigned ncta_y = ptx_cluster_nctaid_y();

        if (cg_size != 4 || ptx_size != 4) {
            atomicExch(err, 1);
            return;
        }
		if (ncta_x != 2 || ncta_y != 2 || cta_x >= ncta_x) { atomicExch(err, 4); return; }
        if (cg_rank != ptx_rank) {
            atomicExch(err, 2);
            return;
        }
        if (ptx_rank >= {{EXPECTED_SIZE}}) {
            atomicExch(err, 3);
            return;
        }
        atomicAdd(&ranks[ptx_rank], 1);
    }
}

int main() {
    const int cluster_size = 4;
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

	src := rankGeometrySource(clusterSize)

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

func rankGeometrySource(clusterSize int) string {
	src := strings.ReplaceAll(rankGeometryTemplate, "{{BLOCK_RANK}}", strings.ReplaceAll(EmitBlockRank("%0"), "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{CLUSTER_SIZE}}", strings.ReplaceAll(EmitClusterSize("%0"), "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{EXPECTED_SIZE}}", strconv.Itoa(clusterSize))
	cta, _ := EmitClusterCTAID("%0", DimX)
	nctaX, _ := EmitClusterNCTAID("%0", DimX)
	nctaY, _ := EmitClusterNCTAID("%0", DimY)
	src = strings.ReplaceAll(src, "{{CTAID_X}}", strings.ReplaceAll(cta, "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{NCTAID_X}}", strings.ReplaceAll(nctaX, "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{NCTAID_Y}}", strings.ReplaceAll(nctaY, "%cluster", "%%cluster"))
	return src
}

func TestRankGeometrySourceUsesEmitters(t *testing.T) {
	src := rankGeometrySource(4)
	cta, _ := EmitClusterCTAID("%0", DimX)
	nctaX, _ := EmitClusterNCTAID("%0", DimX)
	nctaY, _ := EmitClusterNCTAID("%0", DimY)
	for _, emitted := range []string{strings.ReplaceAll(EmitBlockRank("%0"), "%cluster", "%%cluster"), strings.ReplaceAll(EmitClusterSize("%0"), "%cluster", "%%cluster"), strings.ReplaceAll(cta, "%cluster", "%%cluster"), strings.ReplaceAll(nctaX, "%cluster", "%%cluster"), strings.ReplaceAll(nctaY, "%cluster", "%%cluster")} {
		if !strings.Contains(src, emitted) {
			t.Errorf("rank geometry source lacks emitter output %q", emitted)
		}
	}
}
