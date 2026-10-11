package cudamoe

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// Only declared CUDA header shims and simulated NVRTC builtins are visible.
// This checks the source's include contract, not physical WMMA behavior.
func TestPrefillSourceHeaders(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("requires host clang++")
	}
	for _, tc := range []struct {
		name, source string
	}{
		{"scalar", prefillW256ScalarSource},
		{"wmma", prefillW256Source},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			files := map[string]string{
				"cuda_fp16.h": "#pragma once\nusing half = _Float16;\nfloat __half2float(half);\n",
				"mma.h":       prefillHeaderWMMA,
				"prefill.cc": `
#define __global__
using size_t = __SIZE_TYPE__;
static_assert(sizeof(size_t) == 8, "64-bit device indexing");
struct dim { unsigned x, y, z; };
extern dim blockDim, blockIdx, threadIdx;
` + tc.source,
			}
			for name, source := range files {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0600); err != nil {
					t.Fatal(err)
				}
			}
			cmd := exec.Command(compiler, "-std=c++17", "-fsyntax-only", "-nostdinc", "-nostdinc++",
				"-I", dir, filepath.Join(dir, "prefill.cc"))
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("compile without host headers: %v\n%s", err, out)
			}
		})
	}
}

const prefillHeaderWMMA = `
namespace nvcuda { namespace wmma {
struct matrix_a {}; struct matrix_b {}; struct accumulator {}; struct row_major {};
constexpr int mem_row_major = 0;
template<class Role, int M, int N, int K, class T, class Layout = void>
struct fragment {};
template<class F> void fill_fragment(F&, float);
template<class F, class T> void load_matrix_sync(F&, const T*, unsigned);
template<class D, class A, class B, class C> void mma_sync(D&, const A&, const B&, const C&);
template<class F> void store_matrix_sync(float*, const F&, unsigned, int);
}}
`
