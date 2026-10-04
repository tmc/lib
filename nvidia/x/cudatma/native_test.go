package cudatma

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatest"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// tmaTarget returns the architecture the fixtures are generated and compiled
// for. CUDA_TMA_NATIVE_ARCH names another target [FixtureSource] supports.
func tmaTarget() string {
	if target := os.Getenv("CUDA_TMA_NATIVE_ARCH"); target != "" {
		return target
	}
	return TargetSM121a
}

func TestCompileTMAFixture(t *testing.T) {
	if os.Getenv("CUDA_TMA_COMPILE") != "1" {
		t.Skip("compile qualification pending: CUDA_TMA_COMPILE=1 required")
	}
	nvcc, _ := cudatest.NVCC(t)
	target := tmaTarget()
	cudatest.SkipUnlessTarget(t, nvcc, target)
	s, e := FixtureSource(target, 32)
	if e != nil {
		t.Fatal(e)
	}
	h, _ := os.UserHomeDir()
	d, e := os.MkdirTemp(filepath.Join(h, "tmp"), "cudatma-compile-")
	if e != nil {
		t.Fatal(e)
	}
	defer os.RemoveAll(d)
	f := filepath.Join(d, "f.cu")
	os.WriteFile(f, []byte(s), 0600)
	if o, e := exec.Command(nvcc, "-arch="+target, "-c", f, "-o", filepath.Join(d, "f.o")).CombinedOutput(); e != nil {
		t.Fatalf("compile: %v\n%s", e, o)
	}
}
func TestNativeTMAReference(t *testing.T) {
	if os.Getenv("CUDA_TMA_NATIVE") != "1" {
		t.Skip("native qualification pending: CUDA_TMA_NATIVE=1 required")
	}
	// See cudashared/native_test.go: the lock exports the holder's pid, not "1".
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("CUDA_TMA_NATIVE=1 requires running under the GPU lock")
	}
	nvcc, _ := cudatest.NVCC(t)
	target := tmaTarget()
	cudatest.SkipUnlessTarget(t, nvcc, target)
	s, _ := FixtureSource(target, 32)
	s += `#include <cuda_runtime.h>
int main(){float h[32],o[32];for(int i=0;i<32;i++)h[i]=i;float*d=0,*q=0;if(cudaMalloc(&d,sizeof h)||cudaMalloc(&q,sizeof o))return 2;if(cudaMemcpy(d,h,sizeof h,cudaMemcpyHostToDevice))return 3;tma_fixture<<<1,32,sizeof h>>>(d,q);if(cudaDeviceSynchronize())return 4;if(cudaMemcpy(o,q,sizeof o,cudaMemcpyDeviceToHost))return 5;for(int i=0;i<32;i++)if(o[i]!=h[i])return 6;return 0;}`
	h, _ := os.UserHomeDir()
	d, e := os.MkdirTemp(filepath.Join(h, "tmp"), "cudatma-native-")
	if e != nil {
		t.Fatal(e)
	}
	defer os.RemoveAll(d)
	f := filepath.Join(d, "f.cu")
	os.WriteFile(f, []byte(s), 0600)
	x := filepath.Join(d, "f")
	if o, e := exec.Command(nvcc, "-arch="+target, f, "-o", x).CombinedOutput(); e != nil {
		t.Fatalf("compile: %v\n%s", e, o)
	}
	ctx, c := context.WithTimeout(context.Background(), 30*time.Second)
	defer c()
	if o, e := exec.CommandContext(ctx, x).CombinedOutput(); e != nil {
		t.Fatalf("native: %v\n%s", e, o)
	}
}

// nvccOrSkip returns an nvcc to compile with, skipping when the host has
// none. The gates below need a compiler but no device, so they run on any
// CUDA host rather than behind an opt-in variable. That matters here: the
// fixture they check did not compile at all for as long as they were
// skipped by default.
func nvccOrSkip(t *testing.T) string {
	t.Helper()
	if _, err := cudatool.LookTool("nvcc", ""); err != nil {
		t.Skipf("no nvcc on this host: %v", err)
	}
	nvcc, _ := cudatest.NVCC(t)
	return nvcc
}

// compileFixture compiles src for target and returns the output of nvcc with
// the given extra arguments, failing the test if it does not compile.
func compileFixture(t *testing.T, nvcc, target, src string, args ...string) string {
	t.Helper()
	home, _ := os.UserHomeDir()
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudatma-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	file := filepath.Join(dir, "tma.cu")
	if err := os.WriteFile(file, []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "tma.out")
	cmd := append([]string{"-arch=" + target}, args...)
	cmd = append(cmd, file, "-o", out)
	if o, err := exec.Command(nvcc, cmd...).CombinedOutput(); err != nil {
		t.Fatalf("nvcc %v: %v\n%s", cmd, err, o)
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestCompileTMAInstructionFixture requires ptxas to accept the fixture.
//
// It is the only thing that does. TestGeneratedTMAProtocol checks that four
// instruction mnemonics appear in the text, which a source ptxas rejects
// passes just as happily: the fixture named mbarrier.try_wait.parity with a
// 32-bit destination for a predicate, and ptxas refused it with "Arguments
// mismatch" while that test stayed green.
func TestCompileTMAInstructionFixture(t *testing.T) {
	nvcc := nvccOrSkip(t)
	target := tmaTarget()
	cudatest.SkipUnlessTarget(t, nvcc, target)
	src, err := TMAFixtureSource(target)
	if err != nil {
		t.Fatal(err)
	}
	compileFixture(t, nvcc, target, src, "-c")
}

// TestTMAFixturePassesTheMapInParamSpace requires the emitted PTX to hand
// cp.async.bulk.tensor an address in parameter space.
//
// This is the defect a compile gate cannot see. A CUtensorMap parameter that
// is not __grid_constant__ compiles cleanly, but the compiler copies all 128
// bytes onto the stack and the instruction receives a generic address of that
// copy rather than the descriptor.
//
// The decisive signal is the address-space conversion: cvta.param for the
// correct form against cvta.local for the copy. Those two are what a peer's
// independent three-kernel comparison on the same toolkit isolated, and they
// hold regardless of what the kernel does with the descriptor.
//
// The vector-load assertion is the weak one of the three and is specific to
// this fixture. How many ld.param.v2.b64 loads appear depends on how much of
// the descriptor the kernel reads: a kernel that only takes its address
// copies nothing and emits none, so their absence is not evidence of the
// correct form. They are asserted here because this fixture hands the whole
// descriptor to cp.async.bulk.tensor, which makes the copy real and its
// absence meaningful -- a property of the fixture, not of the defect.
func TestTMAFixturePassesTheMapInParamSpace(t *testing.T) {
	nvcc := nvccOrSkip(t)
	target := tmaTarget()
	cudatest.SkipUnlessTarget(t, nvcc, target)
	src, err := TMAFixtureSource(target)
	if err != nil {
		t.Fatal(err)
	}
	ptx := compileFixture(t, nvcc, target, src, "-ptx")
	if !strings.Contains(ptx, "cvta.param") {
		t.Errorf("PTX has no cvta.param: the descriptor is not being read from parameter space")
	}
	if n := strings.Count(ptx, "ld.param.v2.b64"); n > 0 {
		t.Errorf("PTX copies the descriptor out of parameter space (%d vector loads); the parameter is not __grid_constant__", n)
	}
	if strings.Contains(ptx, "cvta.local") {
		t.Errorf("PTX converts a local address: the descriptor was copied to the stack")
	}
}
