package frontend

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// frontendNativeEnv opens the native gate: CUDA_EGRAPH_FRONTEND_NATIVE=1
// under an inherited gpu-lock. CUDA_EGRAPH_FRONTEND_NATIVE_ARCH selects
// the nvcc -arch (default sm_121a).
const frontendNativeEnv = "CUDA_EGRAPH_FRONTEND_NATIVE"

// genBF16 is the deterministic input generator the native harness
// repeats in C: a hash of the element index mapped to [-scale, scale] and
// rounded to BF16 (nearest even).
func genBF16(i, seed uint32, scale float32) uint16 {
	u := i*2654435761 + seed*97
	f := float32(int32(u%2001)-1000) * (scale / 1000)
	bits := math.Float32bits(f)
	bits += 0x7fff + (bits >> 16 & 1)
	return uint16(bits >> 16)
}

// buildFrontendNativeHarness returns a CUDA program that runs res's plan on
// generated inputs and compares the output with hopper.Eval of the
// optimized term. It prints FRONTEND_NATIVE_OK and mutant_rejected=1.
func buildFrontendNativeHarness(t *testing.T, res ImportResult) string {
	t.Helper()
	src, err := res.Plan.Source()
	if err != nil {
		t.Fatal(err)
	}
	env := make(map[string]hopper.Value)
	var b strings.Builder
	b.WriteString("#include <cstdio>\n#include <cstdint>\n#include <cmath>\n#include <cstring>\n#include <vector>\n")
	b.WriteString(src)
	b.WriteString(`
static uint16_t gen_bf16(uint32_t i, uint32_t seed, float scale) {
	uint32_t u = i * 2654435761u + seed * 97u;
	float f = (float)((int32_t)(u % 2001u) - 1000) * (scale / 1000.0f);
	uint32_t bits;
	memcpy(&bits, &f, 4);
	bits += 0x7fffu + ((bits >> 16) & 1u);
	return (uint16_t)(bits >> 16);
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("cuda error %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

int main() {
`)
	var args []string
	for i, in := range res.Plan.Inputs {
		if in.DType != hopper.BF16 {
			t.Fatalf("input %s is %v, harness generates BF16", in.Name, in.DType)
		}
		n := size(in.Shape)
		scale := float32(0.05)
		if len(in.Shape) == 1 {
			scale = 1
		}
		v := make([]float64, n)
		for j := range v {
			v[j] = float64(math.Float32frombits(uint32(genBF16(uint32(j), uint32(i+1), scale)) << 16))
		}
		env[in.Name] = hopper.Value{Shape: in.Shape, Data: v}
		fmt.Fprintf(&b, "\tstd::vector<uint16_t> h_%s(%d);\n\tfor (int j = 0; j < %d; j++) h_%s[j] = gen_bf16(j, %d, %#gf);\n", in.Name, n, n, in.Name, i+1, scale)
		fmt.Fprintf(&b, "\tuint16_t* %s; CK(cudaMalloc(&%s, %d)); CK(cudaMemcpy(%s, h_%s.data(), %d, cudaMemcpyHostToDevice));\n", in.Name, in.Name, 2*n, in.Name, in.Name, 2*n)
		args = append(args, in.Name)
	}
	for _, tmp := range append(append([]hopper.Tensor(nil), res.Plan.Temps...), res.Plan.Output) {
		ct := cType(tmp.DType)
		fmt.Fprintf(&b, "\t%s* %s; CK(cudaMalloc(&%s, %d));\n", ct, tmp.Name, tmp.Name, size(tmp.Shape)*tmp.DType.Bits()/8)
		args = append(args, tmp.Name)
	}
	want, err := hopper.Eval(res.Expr, env)
	if err != nil {
		t.Fatal(err)
	}
	// The mutant is the root gemm without its residual epilogue.
	mut := *res.Expr
	if mut.Op != hopper.OpGEMM {
		t.Fatalf("root is %s, want gemm", mut.Op)
	}
	mut.Children = append([]*egraph.Expr{{Op: "fn:$0"}}, res.Expr.Children[1:3]...)
	mutant, err := hopper.Eval(&mut, env)
	if err != nil {
		t.Fatal(err)
	}
	out := res.Plan.Output
	fmt.Fprintf(&b, "\t%s_run(%s, 0);\n\tCK(cudaGetLastError());\n\tCK(cudaDeviceSynchronize());\n", res.Plan.Name, strings.Join(args, ", "))
	fmt.Fprintf(&b, "\tstatic float got[%d];\n\tCK(cudaMemcpy(got, %s, sizeof got, cudaMemcpyDeviceToHost));\n", len(want.Data), out.Name)
	fmt.Fprintf(&b, "\tstatic const float want[%d] = {", len(want.Data))
	for i, v := range want.Data {
		if i%8 == 0 {
			b.WriteString("\n\t\t")
		}
		fmt.Fprintf(&b, "%.9gf, ", v)
	}
	fmt.Fprintf(&b, "\n\t};\n\tstatic const float mutant[%d] = {", len(mutant.Data))
	for i, v := range mutant.Data {
		if i%8 == 0 {
			b.WriteString("\n\t\t")
		}
		fmt.Fprintf(&b, "%.9gf, ", v)
	}
	fmt.Fprintf(&b, `
	};
	int bad = 0, mutant_bad = 0;
	for (int i = 0; i < %d; i++) {
		float tol = 2e-3f * (1.0f + fabsf(want[i]));
		if (!(fabsf(got[i] - want[i]) <= tol)) { if (bad++ < 4) printf("out[%%d] = %%g, want %%g\n", i, got[i], want[i]); }
		if (!(fabsf(got[i] - mutant[i]) <= tol)) mutant_bad++;
	}
	printf("%s plan kernels=%d elements=%d mismatches=%%d\n", bad);
	if (bad) return 1;
	printf("mutant_rejected=%%d\n", mutant_bad > 0);
	if (!mutant_bad) return 1;
	printf("FRONTEND_NATIVE_OK\n");
	return 0;
}
`, len(want.Data), res.Plan.Name, len(res.Plan.Kernels), len(want.Data))
	return b.String()
}

func TestFrontendNativeHarnessStructure(t *testing.T) {
	src := buildFrontendNativeHarness(t, importAll(t)["onnx"])
	for _, want := range []string{"block_run(", "FRONTEND_NATIVE_OK", "mutant_rejected=", "gen_bf16(j, 1, 0.0500000f)"} {
		if !strings.Contains(src, want) {
			t.Errorf("harness missing %q", want)
		}
	}
}

// TestFrontendNative compiles each frontend's plan with nvcc and runs it on
// the device against hopper.Eval. Run it with
//
//	CUDA_EGRAPH_FRONTEND_NATIVE=1 gpu-lock go test ./x/egraph/frontend -run TestFrontendNative -v
func TestFrontendNative(t *testing.T) {
	if os.Getenv(frontendNativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_EGRAPH_FRONTEND_NATIVE=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	target := os.Getenv("CUDA_EGRAPH_FRONTEND_NATIVE_ARCH")
	if target == "" {
		target = "sm_121a"
	}
	for name, res := range importAll(t) {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			cu, bin := filepath.Join(dir, "block.cu"), filepath.Join(dir, "block")
			if err := os.WriteFile(cu, []byte(buildFrontendNativeHarness(t, res)), 0o600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Minute)
			defer cancel()
			if out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-O2", "-arch="+target, cu, "-o", bin).CombinedOutput(); err != nil {
				t.Fatalf("nvcc (%s, CUDA %s): %v\n%s", target, cudaVer, err, out)
			}
			out, err := exec.CommandContext(ctx, bin).CombinedOutput()
			if err != nil || !strings.Contains(string(out), "FRONTEND_NATIVE_OK") {
				t.Fatalf("native %s (%s): %v\n%s", name, target, err, out)
			}
			t.Logf("native %s passed (arch=%s, CUDA=%s):\n%s", name, target, cudaVer, strings.TrimSpace(string(out)))
		})
	}
}
