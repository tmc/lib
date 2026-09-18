package cudashared

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

// stagedCopyTarget returns the architecture the staged-copy fixture is compiled
// for. CUDA_STAGED_COPY_NATIVE_ARCH names another.
func stagedCopyTarget() string {
	if target := os.Getenv("CUDA_STAGED_COPY_NATIVE_ARCH"); target != "" {
		return target
	}
	return "sm_100a"
}

func TestNativeStagedCopy(t *testing.T) {
	if os.Getenv("CUDA_STAGED_COPY_NATIVE") != "1" {
		t.Skip("native qualification pending: CUDA_STAGED_COPY_NATIVE=1 required")
	}
	// gpu-lock exports GPU_LOCK_HELD=$$, the holder's pid, so it is never the
	// literal "1". Testing for "1" made this gate satisfiable only by someone
	// who set the variable by hand -- that is, by the one caller not holding
	// the lock. Test for a value, as cudatrial does.
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("native gate enabled but no GPU_LOCK_HELD: run under gpu-lock")
	}
	nvcc, _ := cudatest.NVCC(t)
	target := stagedCopyTarget()
	plan, err := NewPlan(Limits{StaticMax: 1024, DynamicMax: 1024, TotalMax: 1024, Source: "bounded fixture"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := plan.Add(Request{Name: "padding", Size: 16, Alignment: 16, Dynamic: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := plan.Add(Request{Name: "stage0", Size: 65 * 4, Alignment: 16, Dynamic: true}); err != nil {
		t.Fatal(err)
	}
	src, launchBytes, err := StagedCopyFixture(plan, target, 65)
	if err != nil {
		t.Fatal(err)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudashared-native-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	file := filepath.Join(dir, "staged.cu")
	src += fmt.Sprintf(`
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
  float h[65] = {}, out[65] = {};
  for (int i=0; i<65; i++) h[i] = (float)i;
  float *d=0, *o=0;
  if (cudaMalloc(&d, sizeof h) != cudaSuccess || cudaMalloc(&o, sizeof out) != cudaSuccess) return 2;
  if (cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice) != cudaSuccess) return 3;
  staged_copy<<<3, 32, %d>>>(d, o);
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;
  if (cudaMemcpy(out, o, sizeof out, cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
  for (int i=0; i<65; i++) if (out[i] != h[i]) return 6;
  cudaFree(d); cudaFree(o); return 0;
}
`, launchBytes)
	if err := os.WriteFile(file, []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(nvcc, "-std=c++17", "-arch="+target, file, "-o", filepath.Join(dir, "staged"))
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("nvcc staged fixture link: %v\n%s", err, out)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, filepath.Join(dir, "staged")).CombinedOutput(); err != nil {
		t.Fatalf("native staged copy: %v\n%s", err, out)
	}
}

func TestCompileStagedCopy(t *testing.T) {
	if os.Getenv("CUDA_STAGED_COPY_COMPILE") != "1" {
		t.Skip("compile qualification pending: CUDA_STAGED_COPY_COMPILE=1 required")
	}
	nvcc, _ := cudatest.NVCC(t)
	target := stagedCopyTarget()
	src, err := StagedCopyFixtureSource(target, 32)
	if err != nil {
		t.Fatal(err)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudashared-compile-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	file := filepath.Join(dir, "staged.cu")
	if err := os.WriteFile(file, []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(nvcc, "-std=c++17", "-arch="+target, "-c", file, "-o", filepath.Join(dir, "staged.o")).CombinedOutput(); err != nil {
		t.Fatalf("nvcc staged fixture compile: %v\n%s", err, out)
	}
}
