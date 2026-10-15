package cudashared

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprepack"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// This is a separately authorized CUDA gate, never a host oracle. A nonempty
// GPU_LOCK_HELD is only an invocation precondition, not proof of lock custody.
func TestNativeStagedQ8Copy(t *testing.T) {
	if os.Getenv("CUDA_STAGED_Q8_NATIVE") != "1" {
		t.Skip("native Q8 qualification pending: explicit hardware authorization required")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("native Q8 gate requires the admitted GPU lease wrapper")
	}
	nvcc, _ := cudatest.NVCC(t)
	raw := make([]byte, 6*34)
	for i := range raw {
		raw[i] = byte(i*37 + 19)
	}
	tensor, err := cudaprepack.RepackQ8_0(raw, 2, 96)
	if err != nil {
		t.Fatal(err)
	}
	p := q8CopyPlan(t, int64(len(raw)))
	f, err := StagedQ8CopyFixture(p, tensor)
	if err != nil {
		t.Fatal(err)
	}
	bytes := func(b []byte) string {
		s := ""
		for _, v := range b {
			s += fmt.Sprintf("%d,", v)
		}
		return s
	}
	source := "#include <cuda_runtime.h>\n#include <stdio.h>\n" + f.Source + fmt.Sprintf(`
int main() {
  unsigned char payload[] = {%s}, scales[] = {%s}, expected[] = {%s};
  unsigned char result[sizeof expected] = {}, *p = 0, *s = 0, *out = 0;
  cudaDeviceProp device;
  if (cudaGetDeviceProperties(&device, 0) != cudaSuccess || device.major != 12 || device.minor != 1) return 1;
  if (cudaMalloc(&p, sizeof payload) != cudaSuccess) return 2;
  int rc = 0;
  do {
    if (cudaMalloc(&s, sizeof scales) != cudaSuccess || cudaMalloc(&out, sizeof expected) != cudaSuccess) { rc = 3; break; }
    if (cudaMemcpy(p, payload, sizeof payload, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(s, scales, sizeof scales, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 4; break; }
    staged_q8_copy<<<%d, %d, %d>>>(p, s, out);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) { rc = 5; break; }
    if (cudaMemcpy(result, out, sizeof result, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 6; break; }
    for (size_t i = 0; i < sizeof expected; i++) if (result[i] != expected[i]) { rc = 7; break; }
  } while (0);
  cudaFree(out); cudaFree(s); cudaFree(p);
  return rc;
}
`, bytes(tensor.Payload), bytes(tensor.Scales), bytes(raw), f.GridBlocks, f.BlockThreads, f.DynamicBytes)
	dir := t.TempDir()
	file, binary := filepath.Join(dir, "staged-q8.cu"), filepath.Join(dir, "staged-q8")
	if err := os.WriteFile(file, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch=sm_121a", file, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile Q8 fixture: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("native Q8 fixture: %v\n%s", err, out)
	}
}
