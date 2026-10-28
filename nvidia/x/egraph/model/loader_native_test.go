package model

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// modelNativeEnv opens the native gate: CUDA_EGRAPH_MODEL_NATIVE=1 under an
// inherited gpu-lock. CUDA_EGRAPH_MODEL_NATIVE_ARCH selects the nvcc -arch
// (default sm_121a).
const modelNativeEnv = "CUDA_EGRAPH_MODEL_NATIVE"

const (
	nativeRepackRows     = 16
	nativeRepackRowBytes = 256
)

// buildLoaderNativeHarness returns a CUDA program that runs the emitted
// 128-byte swizzle repack kernel and compares its output byte for byte with
// what RepackSwizzled computes on the host. It prints LOADER_NATIVE_OK and
// mutant_rejected=1 (an unswizzled copy must not match).
func buildLoaderNativeHarness(t *testing.T) string {
	t.Helper()
	const rows, rowBytes = nativeRepackRows, nativeRepackRowBytes
	kernel, err := SwizzleRepackCUDA("loader_repack_sw128", 128, 128, rows)
	if err != nil {
		t.Fatal(err)
	}
	src := make([]byte, rows*rowBytes)
	for i := range src {
		src[i] = byte(i*131 + 7)
	}
	want, err := RepackSwizzled(src, rows, rowBytes, 128, 128, rows)
	if err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	sb.WriteString("#include <cstdio>\n#include <cstring>\n\n")
	sb.WriteString(kernel)
	fmt.Fprintf(&sb, "\nstatic const unsigned char want[%d] = {", len(want))
	for i, b := range want {
		if i%32 == 0 {
			sb.WriteString("\n\t")
		}
		fmt.Fprintf(&sb, "%d,", b)
	}
	fmt.Fprintf(&sb, `
};

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("cuda error %%s at %%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

int main() {
	const int rows = %d, row_bytes = %d, n = rows * row_bytes;
	static unsigned char src[%d], got[%d];
	for (int i = 0; i < n; i++) src[i] = (unsigned char)(i * 131 + 7);
	unsigned char *dsrc, *ddst;
	CK(cudaMalloc(&dsrc, n));
	CK(cudaMalloc(&ddst, n));
	CK(cudaMemcpy(dsrc, src, n, cudaMemcpyHostToDevice));
	CK(cudaMemset(ddst, 0, n));
	loader_repack_sw128<<<4, 64>>>(dsrc, ddst, rows, row_bytes);
	CK(cudaGetLastError());
	CK(cudaMemcpy(got, ddst, n, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n; i++) {
		if (got[i] != want[i]) { printf("loader_repack_sw128 mismatch at %%d: %%d want %%d\n", i, got[i], want[i]); return 1; }
	}
	printf("loader_repack_sw128 ok bytes=%%d\n", n);
	if (memcmp(src, want, n) == 0) { printf("mutant_rejected=0\n"); return 1; }
	printf("mutant_rejected=1\nLOADER_NATIVE_OK\n");
	return 0;
}
`, rows, rowBytes, rows*rowBytes, rows*rowBytes)
	return sb.String()
}

func TestLoaderNativeHarnessStructure(t *testing.T) {
	src := buildLoaderNativeHarness(t)
	for _, want := range []string{"loader_repack_sw128<<<", "LOADER_NATIVE_OK", "mutant_rejected=1", "static const unsigned char want[4096]"} {
		if !strings.Contains(src, want) {
			t.Errorf("harness missing %q", want)
		}
	}
}

// TestLoaderNative runs the repack kernel on the device. Run it with
//
//	CUDA_EGRAPH_MODEL_NATIVE=1 gpu-lock go test ./x/egraph/model -run TestLoaderNative -v
func TestLoaderNative(t *testing.T) {
	if os.Getenv(modelNativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_EGRAPH_MODEL_NATIVE=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	target := os.Getenv("CUDA_EGRAPH_MODEL_NATIVE_ARCH")
	if target == "" {
		target = "sm_121a"
	}
	dir := t.TempDir()
	cuPath := filepath.Join(dir, "loader_native.cu")
	binPath := filepath.Join(dir, "loader_native")
	if err := os.WriteFile(cuPath, []byte(buildLoaderNativeHarness(t)), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-O2", "-arch="+target, cuPath, "-o", binPath).CombinedOutput(); err != nil {
		t.Fatalf("nvcc (%s, CUDA %s): %v\n%s", target, cudaVer, err, out)
	}
	out, err := exec.CommandContext(ctx, binPath).CombinedOutput()
	if err != nil {
		t.Fatalf("loader native (%s): %v\n%s", target, err, out)
	}
	if !strings.Contains(string(out), "LOADER_NATIVE_OK") || !strings.Contains(string(out), "mutant_rejected=1") {
		t.Fatalf("loader native output missing markers:\n%s", out)
	}
	t.Logf("loader native passed (arch=%s, CUDA=%s):\n%s", target, cudaVer, strings.TrimSpace(string(out)))
}
