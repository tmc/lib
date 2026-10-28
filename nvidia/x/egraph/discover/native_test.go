package discover

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

const discoverNativeEnv = "CUDA_EGRAPH_DISCOVER_NATIVE"

type discoverNativeGateClosedError struct{}

func (discoverNativeGateClosedError) Error() string {
	return "discover: native gate is closed: CUDA_EGRAPH_DISCOVER_NATIVE=1 unset, no device examined"
}

func (discoverNativeGateClosedError) Unwrap() error { return cudaerrors.ErrHardwarePending }

func checkDiscoverNativeGate(getenv func(string) string) error {
	if getenv(discoverNativeEnv) != "1" {
		return discoverNativeGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("discover: native gate open without inherited gpu lock: %w", err)
	}
	return nil
}

type nativeIdiom struct {
	name     string
	spec     SpecFunc
	baseline Program
}

func nativeIdioms() []nativeIdiom {
	return []nativeIdiom{
		{"mux_parity", muxParitySpec, muxParityBaseline()},
		{"byte_plane", bytePlaneSpec, bytePlaneBaseline()},
		{"rotate_byte", rotateSpec, rotateBaseline()},
	}
}

// buildDiscoverNativeHarness returns a CUDA program that, for each idiom,
// evaluates the baseline, the discovered program and its first mutant on
// the oracle's 64 boundary and first 960 random vectors, compares each to
// the host spec's outputs, and times dependent chains of the baseline and
// the discovered program with clock64.
func buildDiscoverNativeHarness(t *testing.T) string {
	t.Helper()
	var b strings.Builder
	b.WriteString("#include <cstdio>\n#include <cstdint>\n#include <cuda_runtime.h>\n\n")
	b.WriteString("#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf(\"cuda error %s at %d\\n\", cudaGetErrorString(e), __LINE__); return 1; } } while (0)\n\n")
	var main strings.Builder
	main.WriteString("int main() {\n    int fail = 0;\n")
	const n = 1024
	for _, id := range nativeIdioms() {
		idiom, err := RunMCTSDiscovery(id.name, id.spec, id.baseline, MCTSConfig{Seed: 1})
		if err != nil {
			t.Fatalf("%s: %v", id.name, err)
		}
		ms := Mutants(idiom.Program)
		o, err := NewOracle(id.baseline.NumInputs, id.spec, n-64, 1)
		if err != nil {
			t.Fatal(err)
		}
		var mutant Program
		for _, m := range ms {
			if o.Verify(id.baseline, m) != nil {
				mutant = m
				break
			}
		}
		k := id.baseline.NumInputs
		vecs := append(append([][]uint32{}, o.boundary...), o.random...)
		b.WriteString(id.baseline.CUDA(id.name + "_baseline"))
		b.WriteString(idiom.Program.CUDA(id.name + "_discovered"))
		b.WriteString(mutant.CUDA(id.name + "_mutant"))
		fmt.Fprintf(&b, "__device__ const uint32_t %s_in[%d][%d] = {\n", id.name, n, k)
		var want []string
		for _, v := range vecs {
			var ws []string
			for _, w := range v {
				ws = append(ws, fmt.Sprintf("0x%08xu", w))
			}
			fmt.Fprintf(&b, "    {%s},\n", strings.Join(ws, ", "))
			want = append(want, fmt.Sprintf("0x%08xu", id.spec(v)))
		}
		b.WriteString("};\n")
		fmt.Fprintf(&b, "__device__ const uint32_t %s_want[%d] = {%s};\n", id.name, n, strings.Join(want, ", "))
		args := func(prefix string) string {
			var a []string
			for j := range k {
				a = append(a, fmt.Sprintf("%s[%d]", prefix, j))
			}
			return strings.Join(a, ", ")
		}
		fmt.Fprintf(&b, `__global__ void %[1]s_check(unsigned *bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= %[2]d) return;
    const uint32_t *in = %[1]s_in[i];
    if (%[1]s_baseline(%[3]s) != %[1]s_want[i]) atomicAdd(&bad[0], 1u);
    if (%[1]s_discovered(%[3]s) != %[1]s_want[i]) atomicAdd(&bad[1], 1u);
    if (%[1]s_mutant(%[3]s) != %[1]s_want[i]) atomicAdd(&bad[2], 1u);
}
`, id.name, n, args("in"))
		// Chain the result back into input 0 so each call depends on the last.
		fmt.Fprintf(&b, `__global__ void %[1]s_time(long long *cyc, uint32_t *sink) {
    uint32_t r[%[2]d];
    for (int j = 0; j < %[2]d; j++) r[j] = %[1]s_in[threadIdx.x][j];
    long long t0 = clock64();
    for (int it = 0; it < 4096; it++) r[0] = %[1]s_baseline(%[3]s);
    long long t1 = clock64();
    uint32_t s = r[0];
    for (int j = 0; j < %[2]d; j++) r[j] = %[1]s_in[threadIdx.x][j];
    long long t2 = clock64();
    for (int it = 0; it < 4096; it++) r[0] = %[1]s_discovered(%[3]s);
    long long t3 = clock64();
    sink[threadIdx.x] = s ^ r[0];
    if (threadIdx.x == 0) { cyc[0] = t1 - t0; cyc[1] = t3 - t2; }
}
`, id.name, k, args("r"))
		fmt.Fprintf(&main, `    {
        unsigned *bad; long long *cyc; uint32_t *sink;
        CHECK(cudaMallocManaged(&bad, 3 * sizeof(unsigned)));
        CHECK(cudaMallocManaged(&cyc, 2 * sizeof(long long)));
        CHECK(cudaMallocManaged(&sink, 32 * sizeof(uint32_t)));
        bad[0] = bad[1] = bad[2] = 0;
        %[1]s_check<<<%[2]d / 256, 256>>>(bad);
        %[1]s_time<<<1, 32>>>(cyc, sink);
        CHECK(cudaDeviceSynchronize());
        printf("%[1]s baseline_bad=%%u discovered_bad=%%u mutant_bad=%%u baseline_cycles_per_iter=%%.2f discovered_cycles_per_iter=%%.2f\n",
            bad[0], bad[1], bad[2], cyc[0] / 4096.0, cyc[1] / 4096.0);
        if (bad[0] != 0 || bad[1] != 0 || bad[2] == 0) fail = 1;
        cudaFree(bad); cudaFree(cyc); cudaFree(sink);
    }
`, id.name, n)
	}
	main.WriteString("    if (fail) { printf(\"DISCOVER_NATIVE_FAIL\\n\"); return 1; }\n")
	main.WriteString("    printf(\"DISCOVER_NATIVE_OK mutant_rejected=1\\n\");\n    return 0;\n}\n")
	b.WriteString(main.String())
	return b.String()
}

func TestDiscoverNativeClosedGateControl(t *testing.T) {
	if os.Getenv(discoverNativeEnv) == "1" {
		t.Skip("closed-gate control applies when CUDA_EGRAPH_DISCOVER_NATIVE != 1")
	}
	if err := checkDiscoverNativeGate(os.Getenv); !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkDiscoverNativeGate(closed) = %v, want ErrHardwarePending", err)
	}
	presence := func(k string) string {
		switch k {
		case discoverNativeEnv, "GPU_LOCK_HELD":
			return "1"
		}
		return ""
	}
	if err := checkDiscoverNativeGate(presence); err == nil || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkDiscoverNativeGate(GPU_LOCK_HELD=1) = %v, want lock error", err)
	}
	src := buildDiscoverNativeHarness(t)
	for _, want := range []string{"mux_parity_discovered", "lop3.b32", "prmt.b32", "byte_plane_mutant", "rotate_byte_time", "DISCOVER_NATIVE_OK"} {
		if !strings.Contains(src, want) {
			t.Errorf("native harness missing %q", want)
		}
	}
}

// TestDiscoverNativeQualification compiles the discovered idioms with nvcc
// and checks them and a mutant on the GPU against the host spec. It reports
// clock64 cycles per dependent call; it does not assert a speedup, since
// ptxas may already fuse the baseline's logic into LOP3 or PRMT.
func TestDiscoverNativeQualification(t *testing.T) {
	if os.Getenv(discoverNativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_EGRAPH_DISCOVER_NATIVE=1 required")
	}
	if err := checkDiscoverNativeGate(os.Getenv); err != nil {
		t.Fatalf("%v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	arch := os.Getenv("CUDA_EGRAPH_DISCOVER_NATIVE_ARCH")
	if arch == "" {
		arch = "sm_121a"
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(base, "discover-native-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	cu := filepath.Join(dir, "discover_native.cu")
	bin := filepath.Join(dir, "discover_native")
	if err := os.WriteFile(cu, []byte(buildDiscoverNativeHarness(t)), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-O2", "-arch="+arch, "-Xptxas", "-v", cu, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("nvcc (%s, CUDA %s): %v\n%s", arch, cudaVer, err, out)
	} else if regexp.MustCompile(`[1-9][0-9]* bytes spill (stores|loads)`).Match(out) {
		t.Errorf("ptxas reports spills:\n%s", out)
	}
	rctx, rcancel := context.WithTimeout(t.Context(), time.Minute)
	defer rcancel()
	out, err := exec.CommandContext(rctx, bin).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "DISCOVER_NATIVE_OK") {
		t.Fatalf("native run (%s, CUDA %s): %v\n%s", arch, cudaVer, err, out)
	}
	t.Logf("arch=%s CUDA=%s\n%s", arch, cudaVer, out)
}
