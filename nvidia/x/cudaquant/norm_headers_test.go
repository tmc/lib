package cudaquant

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// This checks the emitted source with only a CUDA header shim and simulated
// NVRTC predefined types/functions. It does not qualify NVRTC or GPU execution.
func TestNormSourceHeaders(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("requires host clang++")
	}
	for _, tc := range []struct {
		name, source string
	}{
		{"fused", RMSNormQ8_1Source()},
		{"unfused", RMSNormQ8_1UnfusedSource()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			files := map[string]string{
				"cuda_fp16.h": "#pragma once\nusing __half = _Float16;\n__half __float2half_rn(float);\n",
				"norm.cc": normHeaderBuiltins + tc.source + `
static_assert(sizeof(Q8_1) == 36 && alignof(Q8_1) == 2, "q8_1 size and alignment");
static_assert(__builtin_offsetof(Q8_1, d) == 0, "q8_1 d offset");
static_assert(__builtin_offsetof(Q8_1, s) == 2, "q8_1 s offset");
static_assert(__builtin_offsetof(Q8_1, q) == 4, "q8_1 payload offset");
constexpr Q8_1 negative = {0, 0, {-127}};
static_assert(negative.q[0] == -127, "q8_1 signed payload");
`,
			}
			for name, source := range files {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0600); err != nil {
					t.Fatal(err)
				}
			}
			// Plain char may be unsigned. Q8 bytes must remain explicitly signed.
			cmd := exec.Command(compiler, "-std=c++17", "-fsyntax-only", "-funsigned-char",
				"-nostdinc", "-nostdinc++", "-I", dir, filepath.Join(dir, "norm.cc"))
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("compile without host headers: %v\n%s", err, out)
			}
		})
	}
}

// NVRTC supplies these declarations before compiling user source. The shim
// deliberately does not supply stdint.h, stddef.h or math.h.
const normHeaderBuiltins = `
#define __global__
using size_t = __SIZE_TYPE__;
static_assert(sizeof(size_t) == 8, "64-bit device indexing");
struct dim { unsigned x; };
extern dim blockIdx, threadIdx;
double sqrt(double);
float fabsf(float);
float fmaxf(float, float);
float roundf(float);
`
