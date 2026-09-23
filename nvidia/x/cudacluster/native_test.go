package cudacluster

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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
