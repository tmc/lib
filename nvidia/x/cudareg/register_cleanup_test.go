package cudareg

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This shim tests the actual generated host control flow, not CUDA execution.
const registerCleanupShim = `
#include <cstdio>
#include <cstdlib>
#include <cstring>
using cudaError_t = int;
const int cudaSuccess = 0;
const int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
struct cudaFuncAttributes { int numRegs; };
static unsigned *allocated[3];
static int mallocCalls, memcpyCalls;
static bool testCase(const char *name) {
  const char *value = std::getenv("CUDAREGISTER_CLEANUP_CASE");
  return value && std::strcmp(value, name) == 0;
}
static int cudaMalloc(unsigned **ptr, size_t size) {
  int index = mallocCalls++;
  if ((index == 0 && testCase("malloc1")) ||
      (index == 1 && testCase("malloc2")) ||
      (index == 2 && testCase("malloc3"))) return 42;
  allocated[index] = *ptr = (unsigned *)std::malloc(size);
  return *ptr ? 0 : 42;
}
static int cudaMemcpy(void *dst, const void *src, size_t size, int) {
  int index = ++memcpyCalls;
  if (index == 1 && (testCase("copyin") || testCase("copyin_free"))) return 43;
  if (index == 2 && testCase("copyout")) return 43;
  std::memcpy(dst, src, size);
  return 0;
}
static void cudareg_fixture(unsigned *in, unsigned *out) {
  for (int i = 0; i < 256; i++) out[i] = in[i] + 1;
  if (testCase("mismatch")) out[0]++;
}
static void cudareg_fixed(unsigned *in, unsigned *out) {
  for (int i = 0; i < 256; i++) out[i] = in[i] + 1;
}
static int cudaFuncGetAttributes(cudaFuncAttributes *attrs, void (*)(unsigned *, unsigned *)) {
  if (testCase("attributes")) return 44;
  attrs->numRegs = testCase("numregs") ? 65 : 64;
  return 0;
}
static int cudaDeviceSynchronize() {
  return testCase("synchronize") ? 45 : 0;
}
static int cudaFree(unsigned *ptr) {
  int index = 0;
  while (index < 3 && allocated[index] != ptr) index++;
  if (index == 3) return 46;
  std::printf("free:%d\n", index + 1);
  if (testCase("free_all") || (index == 1 && (testCase("free") || testCase("copyin_free")))) return 47;
  std::free(ptr);
  return 0;
}
`

func TestRegisterFixtureCleanup(t *testing.T) {
	cxx, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler required for cleanup fault injection")
	}
	// Keep the producer's allocations, primary status and comparisons intact.
	// Only CUDA kernel launch syntax is replaced at the CPU shim boundary.
	src := registerNativeSource("")
	start := strings.Index(src, "static int runFixture(")
	if start < 0 {
		t.Fatal("native producer entry point missing")
	}
	src = registerCleanupShim + strings.ReplaceAll(src[start:], "<<<1,256>>>", "")
	dir := t.TempDir()
	cpp, exe := filepath.Join(dir, "fixture.cpp"), filepath.Join(dir, "fixture")
	if err := os.WriteFile(cpp, []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, cxx, "-std=c++11", cpp, "-o", exe).CombinedOutput(); err != nil {
		t.Fatalf("compile cleanup shim: %v\n%s", err, out)
	}
	for _, tt := range []struct {
		name       string
		wantStatus int
		wantFrees  string
		wantError  string
	}{
		{"success", 0, "free:3\nfree:2\nfree:1\n", ""},
		{"malloc1", 2, "", ""},
		{"malloc2", 2, "free:1\n", ""},
		{"malloc3", 2, "free:2\nfree:1\n", ""},
		{"copyin", 3, "free:3\nfree:2\nfree:1\n", ""},
		{"attributes", 7, "free:3\nfree:2\nfree:1\n", ""},
		{"numregs", 8, "free:3\nfree:2\nfree:1\n", ""},
		{"synchronize", 4, "free:3\nfree:2\nfree:1\n", ""},
		{"copyout", 5, "free:3\nfree:2\nfree:1\n", ""},
		{"mismatch", 6, "free:3\nfree:2\nfree:1\n", ""},
		{"free", 9, "free:3\nfree:2\nfree:1\n", "cudaFree(out1): 47"},
		{"copyin_free", 3, "free:3\nfree:2\nfree:1\n", "cudaFree(out1): 47"},
		{"free_all", 9, "free:3\nfree:2\nfree:1\n", "cudaFree(out2): 47\ncudaFree(out1): 47\ncudaFree(in): 47"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, exe)
			cmd.Env = append(os.Environ(), "CUDAREGISTER_CLEANUP_CASE="+tt.name)
			var stdout, stderr bytes.Buffer
			cmd.Stdout, cmd.Stderr = &stdout, &stderr
			err := cmd.Run()
			status := 0
			if err != nil {
				var exit *exec.ExitError
				if !errors.As(err, &exit) {
					t.Fatalf("execute cleanup shim: %v", err)
				}
				status = exit.ExitCode()
			}
			if status != tt.wantStatus {
				t.Errorf("status = %d, want %d; stderr=%s", status, tt.wantStatus, &stderr)
			}
			if stdout.String() != tt.wantFrees {
				t.Errorf("free calls = %q, want %q", stdout.String(), tt.wantFrees)
			}
			if got := strings.TrimSpace(stderr.String()); got != tt.wantError {
				t.Errorf("cleanup errors = %q, want %q", got, tt.wantError)
			}
		})
	}
}
