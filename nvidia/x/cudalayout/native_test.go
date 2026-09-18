package cudalayout

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

func layoutNativeTarget(t *testing.T) string {
	t.Helper()
	target := os.Getenv("CUDA_LAYOUT_NATIVE_ARCH")
	if target == "" {
		target = TargetSM100a
	}
	if target != TargetSM100a && target != TargetSM121a {
		t.Fatalf("unsupported CUDA_LAYOUT_NATIVE_ARCH=%q", target)
	}
	return target
}

func layoutScratch(t *testing.T, name string) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	parent := filepath.Join(home, "tmp")
	if err := os.MkdirAll(parent, 0o755); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(parent, "cudalayout-"+name+"-")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("retained fixture: %s", dir)
	return dir
}

func TestCompileLayoutFixture(t *testing.T) {
	if os.Getenv("CUDA_LAYOUT_COMPILE") != "1" {
		t.Skip("compile qualification requires CUDA_LAYOUT_COMPILE=1")
	}
	nvcc, _ := cudatest.NVCC(t)
	target := layoutNativeTarget(t)
	tile, err := NewTile(8, 8, 2, WithStride(16), WithSwizzle(Swizzle32B))
	if err != nil {
		t.Fatal(err)
	}
	src, err := FixtureSourceForTarget(tile, target)
	if err != nil {
		t.Fatal(err)
	}
	dir := layoutScratch(t, "compile")
	path := filepath.Join(dir, "layout_fixture.cu")
	if err := os.WriteFile(path, []byte(src), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch="+target, "-c", path, "-o", filepath.Join(dir, "layout_fixture.o"))
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("nvcc compile: %v\n%s", err, out)
	}
}

func TestNativeLayoutFixture(t *testing.T) {
	if os.Getenv("CUDA_LAYOUT_NATIVE") != "1" {
		t.Skip("native qualification requires CUDA_LAYOUT_NATIVE=1")
	}
	// See cudashared/native_test.go: gpu-lock exports the holder's pid, not "1".
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("CUDA_LAYOUT_NATIVE=1 requires running under gpu-lock")
	}
	if testing.Short() {
		t.Skip("native layout fixture disabled in short mode")
	}
	nvcc, _ := cudatest.NVCC(t)
	target := layoutNativeTarget(t)
	tile, err := NewTile(8, 8, 2, WithStride(16), WithSwizzle(Swizzle32B))
	if err != nil {
		t.Fatal(err)
	}
	src, err := FixtureSourceForTarget(tile, target)
	if err != nil {
		t.Fatal(err)
	}
	src += `
#include <cuda_runtime.h>
#include <stdio.h>
#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 2; } } while (0)
int main() {
  int device = 0; cudaDeviceProp prop{}; int runtime = 0, driver = 0;
  CHECK(cudaGetDevice(&device)); CHECK(cudaGetDeviceProperties(&prop, device));
  CHECK(cudaRuntimeGetVersion(&runtime)); CHECK(cudaDriverGetVersion(&driver));
  fprintf(stderr, "device=%s compute=%d.%d runtime=%d driver=%d\n", prop.name, prop.major, prop.minor, runtime, driver);
  half h_in[64], h_out[136];
  for (int i = 0; i < 64; i++) h_in[i] = __float2half((float)i);
  for (int i = 0; i < 128; i++) h_out[i] = __float2half(-7.0f);
  for (int i = 128; i < 136; i++) h_out[i] = __float2half(-99.0f);
  half *d_in = 0, *d_out = 0;
  CHECK(cudaMalloc((void**)&d_in, sizeof(h_in))); CHECK(cudaMalloc((void**)&d_out, sizeof(h_out)));
  CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_out, h_out, sizeof(h_out), cudaMemcpyHostToDevice));
  layout_copy<<<1,64>>>(d_in, d_out); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
  for (int i = 0; i < 136; i++) printf("%d ", (int)__half2float(h_out[i]));
  return 0;
}
`
	dir := layoutScratch(t, "native")
	path := filepath.Join(dir, "layout_fixture.cu")
	if err := os.WriteFile(path, []byte(src), 0o600); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(dir, "layout_fixture")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch="+target, path, "-o", exe).CombinedOutput(); err != nil {
		t.Fatalf("nvcc compile: %v\n%s", err, out)
	}
	run := exec.CommandContext(ctx, exe)
	var stdout, stderr bytes.Buffer
	run.Stdout, run.Stderr = &stdout, &stderr
	err = run.Run()
	if err != nil {
		t.Fatalf("native execution: %v\nstdout=%s\nstderr=%s", err, stdout.Bytes(), stderr.Bytes())
	}
	t.Logf("native device: %s", strings.TrimSpace(stderr.String()))
	fields := strings.Fields(stdout.String())
	if len(fields) != 136 {
		t.Fatalf("native output elements=%d, want 136", len(fields))
	}
	got := make([]int, len(fields))
	for i, field := range fields {
		got[i], err = strconv.Atoi(field)
		if err != nil {
			t.Fatal(err)
		}
	}
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			physical := r*16 + (c ^ (r % 16))
			if got[physical] != r*8+c {
				t.Fatalf("physical[%d] = %d, want %d for logical (%d,%d)", physical, got[physical], r*8+c, r, c)
			}
		}
	}
	for r := 0; r < 8; r++ {
		for c := 0; c < 16; c++ {
			physical := r*16 + c
			occupied := false
			for logical := 0; logical < 8; logical++ {
				if logical^(r%16) == c {
					occupied = true
					break
				}
			}
			if !occupied && got[physical] != -7 {
				t.Fatalf("padding physical[%d] = %d, want -7", physical, got[physical])
			}
		}
	}
	for i := 128; i < 136; i++ {
		if got[i] != -99 {
			t.Fatalf("guard[%d] = %d, want -99", i, got[i])
		}
	}
}
