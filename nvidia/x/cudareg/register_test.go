package cudareg

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func fixture(t *testing.T) Schedule {
	t.Helper()
	s, _, err := GenerateFixture()
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestRegisterTargetLegality(t *testing.T) {
	if _, err := TargetFor("sm_100"); err == nil {
		t.Fatal("accepted unsuffixed target")
	}
	s := fixture(t)
	s.Groups["producer"] = Budget{Group: s.Groups["consumer"].Group, CompilerAllocated: 64, PeakLive: 32, Cap: 64, Dynamic: 41}
	if err := s.Validate(); err == nil {
		t.Fatal("accepted illegal count")
	}
}
func TestRegisterPoolConservation(t *testing.T) {
	s := fixture(t)
	s.CTARegisterPool = 1000
	if err := s.Validate(); err == nil {
		t.Fatal("accepted pool overflow")
	}
}
func TestRegisterPoolExcess(t *testing.T) {
	s := fixture(t)
	s.CTARegisterPool++
	if err := s.Validate(); err == nil {
		t.Fatal("accepted unaccounted register reserve")
	}
}
func TestRegisterTransitionDirection(t *testing.T) {
	s := fixture(t)
	s.Steps[0].Action = Increase
	if err := s.Validate(); err == nil {
		t.Fatal("accepted inverted decrease")
	}
}
func TestRegisterUniformity(t *testing.T) {
	s := fixture(t)
	s.Steps[0].Participants = 127
	if err := s.Validate(); err == nil {
		t.Fatal("accepted partial warpgroup")
	}
}
func TestRegisterIncreaseProgress(t *testing.T) {
	s := fixture(t)
	s.CTARegisterPool = 16384
	s.Steps = []Transition{{Role: "consumer", Action: Increase, Count: 96, Participants: 128, Synchronized: true, Initialized: true}}
	if err := s.Validate(); err == nil {
		t.Fatal("accepted growth before donor release")
	}
}
func TestAllocatedVsLiveRegisters(t *testing.T) {
	s := fixture(t)
	if s.Groups["producer"].CompilerAllocated == s.Groups["producer"].PeakLive {
		t.Fatal("facts were conflated")
	}
	s.Groups["producer"] = Budget{Group: s.Groups["consumer"].Group, CompilerAllocated: 64, PeakLive: 65, Cap: 64, Dynamic: 64}
	if err := s.Validate(); err == nil {
		t.Fatal("accepted live peak over cap")
	}
}
func TestNativeRegisterTransitionReference(t *testing.T) {
	if os.Getenv("CUDA_REGISTER_NATIVE") != "1" {
		t.Skip("native qualification opt-in: set CUDA_REGISTER_NATIVE=1")
	}
	// See cudashared/native_test.go: the lock exports the holder's pid, not "1".
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("CUDA_REGISTER_NATIVE=1 requires running under the GPU lock")
	}
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Fatal("CUDA_REGISTER_NATIVE=1 requires nvcc: ", err)
	}
	if os.Getenv("CUDA_REGISTER_NATIVE_ARCH") != "sm_90a" {
		t.Fatal("CUDA_REGISTER_NATIVE=1 requires CUDA_REGISTER_NATIVE_ARCH=sm_90a")
	}
	_, dynamic, err := GenerateFixture()
	if err != nil {
		t.Fatal(err)
	}
	src := `#include <cuda_runtime.h>
#include <cstdio>
` + dynamic + "\n" + FixedBudgetSource() + `
int main() {
	unsigned h[256], a[256], b[256];
	for (int i = 0; i < 256; i++) h[i] = (unsigned)i;
  unsigned *in, *out1, *out2;
  if (cudaMalloc(&in, sizeof h) || cudaMalloc(&out1, sizeof a) || cudaMalloc(&out2, sizeof b)) return 2;
  if (cudaMemcpy(in, h, sizeof h, cudaMemcpyHostToDevice)) return 3;
  cudaFuncAttributes attrs{};
  if (cudaFuncGetAttributes(&attrs, cudareg_fixture) != cudaSuccess) return 7;
  if (attrs.numRegs != 64) return 8;
  cudareg_fixture<<<1,256>>>(in, out1);
	cudareg_fixed<<<1,256>>>(in, out2);
  if (cudaDeviceSynchronize()) return 4;
  if (cudaMemcpy(a, out1, sizeof a, cudaMemcpyDeviceToHost) || cudaMemcpy(b, out2, sizeof b, cudaMemcpyDeviceToHost)) return 5;
	for (int i = 0; i < 256; i++) if (a[i] != b[i]) return 6;
  cudaFree(in); cudaFree(out1); cudaFree(out2);
  return 0;
}`
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudareg-native-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	cu := filepath.Join(dir, "fixture.cu")
	exe := filepath.Join(dir, "fixture")
	if err := os.WriteFile(cu, []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(nvcc, "-arch=sm_90a", "-maxrregcount=64", cu, "-o", exe)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("nvcc compile: %v\n%s", err, out)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, exe).CombinedOutput(); err != nil {
		t.Fatalf("native fixture: %v\n%s", err, out)
	}
}

func TestCompileRegisterFixture(t *testing.T) {
	if os.Getenv("CUDA_REGISTER_COMPILE") != "1" {
		t.Skip("compile qualification pending: CUDA_REGISTER_COMPILE=1 required")
	}
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Fatalf("compile gate enabled but nvcc is unavailable: %v", err)
	}
	_, dynamic, err := GenerateFixture()
	if err != nil {
		t.Fatal(err)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudareg-compile-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	cu := filepath.Join(dir, "fixture.cu")
	if err := os.WriteFile(cu, []byte(dynamic), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(nvcc, "-arch=sm_90a", "-maxrregcount=64", "-c", cu, "-o", filepath.Join(dir, "fixture.o")).CombinedOutput(); err != nil {
		t.Fatalf("nvcc compile: %v\n%s", err, out)
	}
}

func TestReferenceMatchesFixedBudget(t *testing.T) {
	in := []uint32{0, 1, 17, 0xffffffff}
	if got, want := Reference(in), FixedBudgetReference(in); !reflect.DeepEqual(got, want) {
		t.Fatalf("register transition output = %v, fixed budget = %v", got, want)
	}
	if !reflect.DeepEqual(Reference(in), []uint32{1, 2, 18, 0}) {
		t.Fatal("reference mutated input")
	}
}

func TestGeneratedFixtureContainsReferenceInstructions(t *testing.T) {
	_, src, err := GenerateFixture()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"setmaxnreg.dec.sync.aligned.u32 32", "setmaxnreg.inc.sync.aligned.u32 96"} {
		if !strings.Contains(src, want) {
			t.Fatalf("fixture missing %q", want)
		}
	}
	for _, want := range []string{"tid < 128", "tid >= 128", "__syncthreads()"} {
		if !strings.Contains(src, want) {
			t.Fatalf("fixture missing role/synchronization marker %q", want)
		}
	}
	if strings.Contains(FixedBudgetSource(), "setmaxnreg") {
		t.Fatal("fixed budget baseline unexpectedly resizes registers")
	}
}
