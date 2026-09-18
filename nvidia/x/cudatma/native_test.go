package cudatma

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestCompileTMAFixture(t *testing.T) {
	if os.Getenv("CUDA_TMA_COMPILE") != "1" {
		t.Skip("compile qualification pending: CUDA_TMA_COMPILE=1 required")
	}
	nvcc, e := exec.LookPath("nvcc")
	if e != nil {
		t.Fatal(e)
	}
	s, e := FixtureSource(TargetSM121a, 32)
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
	if o, e := exec.Command(nvcc, "-arch=sm_121a", "-c", f, "-o", filepath.Join(d, "f.o")).CombinedOutput(); e != nil {
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
	nvcc, e := exec.LookPath("nvcc")
	if e != nil {
		t.Fatal(e)
	}
	s, _ := FixtureSource(TargetSM121a, 32)
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
	if o, e := exec.Command(nvcc, "-arch=sm_121a", f, "-o", x).CombinedOutput(); e != nil {
		t.Fatalf("compile: %v\n%s", e, o)
	}
	ctx, c := context.WithTimeout(context.Background(), 30*time.Second)
	defer c()
	if o, e := exec.CommandContext(ctx, x).CombinedOutput(); e != nil {
		t.Fatalf("native: %v\n%s", e, o)
	}
}

func TestCompileTMAInstructionFixture(t *testing.T) {
	if os.Getenv("CUDA_TMA_INSTRUCTION_COMPILE") != "1" {
		t.Skip("TMA instruction compile pending: CUDA_TMA_INSTRUCTION_COMPILE=1 required")
	}
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Fatalf("compile gate enabled but nvcc unavailable: %v", err)
	}
	src, err := TMAFixtureSource(TargetSM121a)
	if err != nil {
		t.Fatal(err)
	}
	home, _ := os.UserHomeDir()
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudatma-instruction-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	file := filepath.Join(dir, "tma.cu")
	if err := os.WriteFile(file, []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(nvcc, "-arch=sm_121a", "-c", file, "-o", filepath.Join(dir, "tma.o")).CombinedOutput(); err != nil {
		t.Fatalf("TMA instruction compile: %v\n%s", err, out)
	}
}
