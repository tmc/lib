package cudacluster

import (
	"context"
	"errors"
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

	src, err := ReductionFixture(target, clusterSize, mut)
	if err != nil {
		t.Fatal(err)
	}
	src += reductionMain(clusterSize)

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
		if err := compareOracleTotal(got, want, err); err != nil {
			return 0, err
		}
	}
	return got, nil
}

func compareOracleTotal(emitted, oracle uint32, oracleErr error) error {
	if oracleErr != nil {
		return fmt.Errorf("cooperative-groups oracle: %w", oracleErr)
	}
	if emitted != oracle {
		return fmt.Errorf("emitted PTX total %d differs from cooperative-groups oracle %d", emitted, oracle)
	}
	return nil
}

func reductionMain(clusterSize int) string {
	return fmt.Sprintf(`#include <cuda_runtime.h>
#include <stdio.h>
int main() {
  unsigned *d = 0, h = 0;
  if (cudaMalloc(&d, sizeof h) != cudaSuccess) return 2;
  if (cudaMemset(d, 0, sizeof h) != cudaSuccess) return 3;
  dsmem_reduce<<<%d, 32>>>(d);
  if (cudaGetLastError() != cudaSuccess) return %d;
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;
  if (cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
  cudaFree(d);
  printf("total %%u\n", h);
  return 0;
}
`, clusterSize, fixtureLaunchFailureCode)
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
	out, _ := run.CombinedOutput()
	if run.ProcessState == nil {
		return classifyFixtureResult(false, fixtureTimedOut(runCtx), 0, string(out))
	}
	return classifyFixtureResult(true, fixtureTimedOut(runCtx), run.ProcessState.ExitCode(), string(out))
}

func fixtureTimedOut(ctx context.Context) bool {
	return errors.Is(ctx.Err(), context.DeadlineExceeded)
}

var errFixtureTimeout = errors.New("fixture did not finish")

func classifyFixtureResult(hasProcessState, timedOut bool, exitCode int, output string) (uint32, error) {
	if !hasProcessState {
		return 0, fmt.Errorf("fixture process has no exit status: %s", output)
	}
	if timedOut {
		return 0, fmt.Errorf("%w: a cluster barrier in divergent flow may hang: %s", errFixtureTimeout, output)
	}
	switch exitCode {
	case 0:
		return parseTotal(output)
	case 2, 3, 4, 5, fixtureLaunchFailureCode:
		return 0, fmt.Errorf("fixture failed at CUDA stage %d: %s", exitCode, output)
	default:
		return 0, fmt.Errorf("fixture exited %d: %s", exitCode, output)
	}
}

func TestClassifyFixtureResult(t *testing.T) {
	tests := []struct {
		name        string
		hasState    bool
		timedOut    bool
		exitCode    int
		output      string
		want        uint32
		wantErr     bool
		wantTimeout bool
		wantErrText string
	}{
		{name: "success total", hasState: true, output: "total 628", want: 628},
		{name: "success without total", hasState: true, wantErr: true},
		{name: "cuda exit 2", hasState: true, exitCode: 2, output: "total 628", wantErr: true, wantErrText: "CUDA stage 2"},
		{name: "cuda exit 3", hasState: true, exitCode: 3, output: "total 628", wantErr: true, wantErrText: "CUDA stage 3"},
		{name: "cuda exit 4", hasState: true, exitCode: 4, output: "total 628", wantErr: true, wantErrText: "CUDA stage 4"},
		{name: "cuda exit 5", hasState: true, exitCode: 5, output: "total 628", wantErr: true, wantErrText: "CUDA stage 5"},
		{name: "launch error", hasState: true, exitCode: 6, output: "total 628", wantErr: true, wantErrText: "CUDA stage 6"},
		{name: "ordinary error", hasState: true, exitCode: 1, output: "total 628", wantErr: true, wantErrText: "fixture exited 1"},
		{name: "signal", hasState: true, exitCode: -1, output: "total 628", wantErr: true, wantErrText: "fixture exited -1"},
		{name: "no process state", output: "total 628", wantErr: true, wantErrText: "no exit status"},
		{name: "timeout", hasState: true, timedOut: true, exitCode: -1, output: "total 628", wantErr: true, wantTimeout: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := classifyFixtureResult(tc.hasState, tc.timedOut, tc.exitCode, tc.output)
			if got != tc.want || (err != nil) != tc.wantErr || errors.Is(err, errFixtureTimeout) != tc.wantTimeout {
				t.Fatalf("classifyFixtureResult = (%d, %v), want (%d, err=%v, timeout=%v)", got, err, tc.want, tc.wantErr, tc.wantTimeout)
			}
			if tc.wantErrText != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErrText)) {
				t.Fatalf("classifyFixtureResult error = %v, want substring %q", err, tc.wantErrText)
			}
		})
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

func TestNativeGateAcceptsGenuineInheritedLockOnHost(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "gpu-lock")
	h, err := cudalock.Acquire(context.Background(), cudalock.WithDir(dir), cudalock.WithLabel("cudacluster-host-test"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := h.Release(); err != nil {
			t.Errorf("release test lock: %v", err)
		}
	})
	env := func(key string) string {
		switch key {
		case "CUDA_CLUSTER_NATIVE":
			return "1"
		case "GPU_LOCK_HELD":
			return strconv.Itoa(os.Getpid())
		case "GPU_LOCKDIR":
			return dir
		default:
			return ""
		}
	}
	if err := requireNativeLock(env); err != nil {
		t.Fatalf("requireNativeLock with inherited lock: %v", err)
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
	if err := checkLocalWindowTotal(size, got); err != nil {
		t.Fatal(err)
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
	const reps = 5
	runs := make([]noSyncRun, 0, reps)
	for i := 0; i < reps; i++ {
		got, err := runReduction(t, size, MutantNoSync)
		runs = append(runs, noSyncRun{total: got, err: err})
		if errors.Is(err, errFixtureTimeout) {
			t.Logf("rep %d: %v (a failure here also falsifies determinism)", i, err)
			break
		}
		if err != nil {
			break
		}
	}
	if err := checkNoSyncControl(want, runs); err != nil {
		t.Fatal(err)
	}
}

type noSyncRun struct {
	total uint32
	err   error
}

func checkNoSyncControl(want uint32, runs []noSyncRun) error {
	if len(runs) == 0 {
		return fmt.Errorf("no-sync control produced no runs")
	}
	correct := 0
	for i, run := range runs {
		if run.err != nil {
			if errors.Is(run.err, errFixtureTimeout) {
				return nil
			}
			return fmt.Errorf("rep %d failed outside runtime timeout: %w", i, run.err)
		}
		if run.total == want {
			correct++
		}
	}
	if correct == len(runs) {
		return fmt.Errorf("no-sync mutant produced the correct total %d in all %d runs: the reduction is not exercising a cross-CTA dependency", want, correct)
	}
	return nil
}

func checkLocalWindowTotal(size int, got uint32) error {
	want := uint32(size) * 7
	if got != want {
		return fmt.Errorf("local-window mutant total = %d, want %d", got, want)
	}
	return nil
}

func TestCompareOracleTotal(t *testing.T) {
	oracleErr := errors.New("oracle failed")
	for _, tc := range []struct {
		name      string
		emitted   uint32
		oracle    uint32
		oracleErr error
		wantErr   bool
	}{
		{name: "equal", emitted: 42, oracle: 42},
		{name: "different", emitted: 42, oracle: 43, wantErr: true},
		{name: "oracle failed", emitted: 42, oracleErr: oracleErr, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := compareOracleTotal(tc.emitted, tc.oracle, tc.oracleErr)
			if (err != nil) != tc.wantErr {
				t.Fatalf("compareOracleTotal error = %v, wantErr=%t", err, tc.wantErr)
			}
			if tc.name == "oracle failed" && !errors.Is(err, oracleErr) {
				t.Fatalf("oracle error = %v, want wrapped source error", err)
			}
		})
	}
}

func TestCheckNoSyncControl(t *testing.T) {
	timeoutErr := fmt.Errorf("%w: fixture", errFixtureTimeout)
	ordinaryErr := errors.New("compiler failed")
	for _, tc := range []struct {
		name    string
		runs    []noSyncRun
		wantErr bool
	}{
		{name: "one wrong total", runs: []noSyncRun{{total: 1}, {total: 2}}},
		{name: "all correct", runs: []noSyncRun{{total: 9}, {total: 9}}, wantErr: true},
		{name: "timeout falsifies determinism", runs: []noSyncRun{{err: timeoutErr}}},
		{name: "other error is fatal", runs: []noSyncRun{{err: ordinaryErr}}, wantErr: true},
		{name: "no runs", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkNoSyncControl(9, tc.runs)
			if (err != nil) != tc.wantErr {
				t.Fatalf("checkNoSyncControl error = %v, wantErr=%t", err, tc.wantErr)
			}
			if tc.name == "other error is fatal" && !errors.Is(err, ordinaryErr) {
				t.Fatalf("ordinary error = %v, want wrapped source error", err)
			}
		})
	}
}

func TestCheckLocalWindowTotal(t *testing.T) {
	if err := checkLocalWindowTotal(4, 28); err != nil {
		t.Fatalf("correct local total: %v", err)
	}
	if err := checkLocalWindowTotal(4, 628); err == nil {
		t.Fatal("remote total was accepted as the local-window mutant total")
	}
}

func TestFixtureTimedOut(t *testing.T) {
	if fixtureTimedOut(context.Background()) {
		t.Fatal("background context reported timeout")
	}
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	if !fixtureTimedOut(ctx) {
		t.Fatal("expired run deadline was not recorded")
	}
}

// TestControl_NoSimulationWithoutHardware asserts that with the native gate
// closed no code path in this package produces a reduction total. The package
// has no simulation path; this test exists so that adding one fails.
func TestControl_NoSimulationWithoutHardware(t *testing.T) {
	if os.Getenv("CUDA_CLUSTER_NATIVE") == "1" {
		t.Skip("native gate open; this control covers the closed case")
	}
	src, err := ReductionFixture(clusterTarget(), 4, MutantNone)
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
__device__ __forceinline__ unsigned ptx_cluster_ctaid_y() {
    unsigned id; asm("{{CTAID_Y}}" : "=r"(id)); return id;
}
__device__ __forceinline__ unsigned ptx_cluster_nctaid_x() {
    unsigned n; asm("{{NCTAID_X}}" : "=r"(n)); return n;
}
__device__ __forceinline__ unsigned ptx_cluster_nctaid_y() {
    unsigned n; asm("{{NCTAID_Y}}" : "=r"(n)); return n;
}

__global__ void __cluster_dims__({{CLUSTER_X}}, {{CLUSTER_Y}}, 1) rank_geometry_kernel(unsigned *ranks, unsigned *err) {
    if (threadIdx.x == 0) {
        cg::cluster_group cluster = cg::this_cluster();
        unsigned cg_rank = cluster.block_rank();
        unsigned cg_size = cluster.num_blocks();
        unsigned ptx_rank = ptx_block_rank();
        unsigned ptx_size = ptx_cluster_size();
		unsigned cta_x = ptx_cluster_ctaid_x();
		unsigned cta_y = ptx_cluster_ctaid_y();
		unsigned ncta_x = ptx_cluster_nctaid_x();
		unsigned ncta_y = ptx_cluster_nctaid_y();

        if (cg_size != {{EXPECTED_SIZE}} || ptx_size != {{EXPECTED_SIZE}}) {
            atomicExch(err, 1);
            return;
        }
		if (ncta_x != {{CLUSTER_X}} || ncta_y != {{CLUSTER_Y}} || cta_x >= ncta_x || cta_y >= ncta_y) { atomicExch(err, 4); return; }
        if (cg_rank != ptx_rank) {
            atomicExch(err, 2);
            return;
        }
        if (ptx_rank != cta_x + cta_y * ncta_x) {
            atomicExch(err, 3);
            return;
        }
        atomicAdd(&ranks[ptx_rank], 1);
    }
}

int main() {
    const int cluster_x = {{CLUSTER_X}};
    const int cluster_y = {{CLUSTER_Y}};
    const int cluster_size = {{EXPECTED_SIZE}};
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

    rank_geometry_kernel<<<dim3(cluster_x, cluster_y, 1), 32>>>(d_ranks, d_err);
    cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        fprintf(stderr, "rank_geometry_kernel launch failed: %s\n", cudaGetErrorString(launch));
        return 8;
    }

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
	if clusterSize < 4 || clusterSize > 8 || clusterSize%2 != 0 {
		return ""
	}
	clusterX := 2
	clusterY := clusterSize / clusterX
	src := strings.ReplaceAll(rankGeometryTemplate, "{{BLOCK_RANK}}", strings.ReplaceAll(EmitBlockRank("%0"), "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{CLUSTER_SIZE}}", strings.ReplaceAll(EmitClusterSize("%0"), "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{EXPECTED_SIZE}}", strconv.Itoa(clusterSize))
	cta, _ := EmitClusterCTAID("%0", DimX)
	ctaY, _ := EmitClusterCTAID("%0", DimY)
	nctaX, _ := EmitClusterNCTAID("%0", DimX)
	nctaY, _ := EmitClusterNCTAID("%0", DimY)
	src = strings.ReplaceAll(src, "{{CTAID_X}}", strings.ReplaceAll(cta, "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{CTAID_Y}}", strings.ReplaceAll(ctaY, "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{NCTAID_X}}", strings.ReplaceAll(nctaX, "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{NCTAID_Y}}", strings.ReplaceAll(nctaY, "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{CLUSTER_Y}}", strconv.Itoa(clusterY))
	src = strings.ReplaceAll(src, "{{CLUSTER_X}}", strconv.Itoa(clusterX))
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

func TestGeometryEmittersHaveLiteralRegisterSpellings(t *testing.T) {
	if got := EmitClusterSize("%r1"); got != "mov.u32 %r1, %cluster_nctarank;" {
		t.Fatalf("EmitClusterSize = %q", got)
	}
	for _, tc := range []struct {
		name string
		fn   func(string, Dim) (string, error)
		reg  string
	}{
		{"CTAID", EmitClusterCTAID, "%cluster_ctaid."},
		{"NCTAID", EmitClusterNCTAID, "%cluster_nctaid."},
	} {
		for _, axis := range []struct {
			dim  Dim
			name string
		}{{DimX, "x"}, {DimY, "y"}, {DimZ, "z"}} {
			got, err := tc.fn("%r7", axis.dim)
			if err != nil {
				t.Fatal(err)
			}
			want := "mov.u32 %r7, " + tc.reg + axis.name + ";"
			if got != want {
				t.Errorf("EmitCluster%s(%s) = %q, want %q", tc.name, axis.name, got, want)
			}
		}
	}
}

func TestExpectedReduction(t *testing.T) {
	for _, tc := range []struct {
		size int
		want uint32
	}{{2, 114}, {4, 628}, {8, 2856}} {
		if got := ExpectedReduction(tc.size); got != tc.want {
			t.Errorf("ExpectedReduction(%d) = %d, want %d", tc.size, got, tc.want)
		}
	}
}

func TestRankGeometrySourceUsesRequestedClusterShape(t *testing.T) {
	src := rankGeometrySource(4)
	for _, want := range []string{
		"__cluster_dims__(2, 2, 1)",
		"dim3(cluster_x, cluster_y, 1)",
		"cudaGetLastError()",
		"cta_x + cta_y * ncta_x",
		"ptx_cluster_ctaid_y()",
		"unsigned cta_y = ptx_cluster_ctaid_y();",
		"if (ncta_x != 2 || ncta_y != 2 || cta_x >= ncta_x || cta_y >= ncta_y)",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("rank geometry source lacks %q", want)
		}
	}
	if strings.Contains(src, `launch failed: %s\\n`) {
		t.Fatal("rank geometry launch error prints a literal backslash-n")
	}
	if rankGeometrySource(6) == "" || !strings.Contains(rankGeometrySource(6), "const int cluster_y = 3;") {
		t.Fatal("rank geometry source ignored the requested size")
	}
	if rankGeometrySource(3) != "" {
		t.Fatal("rank geometry source accepted a shape it cannot represent")
	}
}
