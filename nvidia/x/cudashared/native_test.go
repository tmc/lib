package cudashared

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestNativeStagedCopy(t *testing.T) {
	if os.Getenv("CUDA_STAGED_COPY_NATIVE") != "1" {
		t.Skip("native qualification pending: CUDA_STAGED_COPY_NATIVE=1 required")
	}
	if os.Getenv("GPU_LOCK_HELD") != "1" {
		t.Fatalf("native gate enabled but GPU_LOCK_HELD=1 is absent")
	}
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Fatalf("native gate enabled but nvcc is unavailable: %v", err)
	}
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
	src, launchBytes, err := StagedCopyFixture(plan, "sm_100a", 65)
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
	cmd := exec.Command(nvcc, "-std=c++17", "-arch=sm_100a", file, "-o", filepath.Join(dir, "staged"))
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
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Fatalf("compile gate enabled but nvcc is unavailable: %v", err)
	}
	src, err := StagedCopyFixtureSource("sm_100a", 32)
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
	if out, err := exec.Command(nvcc, "-std=c++17", "-arch=sm_100a", "-c", file, "-o", filepath.Join(dir, "staged.o")).CombinedOutput(); err != nil {
		t.Fatalf("nvcc staged fixture compile: %v\n%s", err, out)
	}
}
