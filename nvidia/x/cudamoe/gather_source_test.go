package cudamoe

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Execute the emitted indexing on the CPU, one CUDA thread at a time. Resetting
// the output between threads exposes overlapping writers that final-value
// comparisons miss. This does not qualify CUDA compilation or GPU execution.
func TestGatherSourceOwnership(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("host generated-source check requires clang++")
	}
	for _, tc := range []struct {
		name, old, replacement string
		wantFailure            bool
	}{
		{name: "source"},
		{name: "overlapping_writers", old: "if (c == group)", replacement: "", wantFailure: true},
		{name: "unchecked_output_alignment", old: " | (unsigned long long)dst", replacement: "", wantFailure: true},
		{name: "permuted_experts", old: "int e = route[t];", replacement: "int e = route[(t+1) % tokens];", wantFailure: true},
		{name: "padded_storage", old: "out[t * width + c]", replacement: "out[i]", wantFailure: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			stub := `#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#define __global__
__attribute__((noinline)) void checkVectorAddresses(const void* dst, const void* src) {
  if (((uintptr_t)dst | (uintptr_t)src) & 15) abort();
}
struct alignas(16) uint4 {
  unsigned x,y,z,w;
  uint4& operator=(const uint4& src) {
    checkVectorAddresses(this, &src);
    x=src.x; y=src.y; z=src.z; w=src.w;
    return *this;
  }
};
struct dim { unsigned x; };
extern dim blockIdx, blockDim, threadIdx;
`
			if err := os.WriteFile(filepath.Join(dir, "cuda_runtime.h"), []byte(stub), 0600); err != nil {
				t.Fatal(err)
			}
			var source strings.Builder
			for _, tile := range []int{16, 32} {
				s, err := GatherSource(tile)
				if err != nil {
					t.Fatal(err)
				}
				if tc.old != "" {
					if !strings.Contains(s, tc.old) {
						t.Fatal("mutation target missing")
					}
					s = strings.ReplaceAll(s, tc.old, tc.replacement)
				}
				if tile == 32 {
					s = strings.ReplaceAll(s, "#include <cuda_runtime.h>", "")
					s = strings.ReplaceAll(s, "moe_gather", "moe_gather32")
				}
				source.WriteString(s)
				source.WriteByte('\n')
			}
			source.WriteString(strings.ReplaceAll(GatherScalarSource(), "#include <cuda_runtime.h>", ""))
			source.WriteString(gatherHostHarness)
			path := filepath.Join(dir, "gather.cc")
			if err := os.WriteFile(path, []byte(source.String()), 0600); err != nil {
				t.Fatal(err)
			}
			binary := filepath.Join(dir, "gather")
			if out, err := exec.Command(compiler, "-std=c++17", "-O1", "-fno-strict-aliasing", "-fsanitize=undefined", "-fno-sanitize-recover=all", "-I", dir, path, "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("compile: %v\n%s", err, out)
			}
			out, err := exec.Command(binary).CombinedOutput()
			if (err != nil) != tc.wantFailure {
				t.Fatalf("run: %v (want failure %v)\n%s", err, tc.wantFailure, out)
			}
		})
	}
}

const gatherHostHarness = `
#include <cstdio>
#include <algorithm>
dim blockIdx, blockDim, threadIdx;
int main() {
  alignas(16) float input[512], output[512];
  int route[] = {2, 0, 1, -1, 3};
  for (int i=0; i<512; ++i) input[i] = (float)(i+1);
  using Kernel = void(*)(const float*,float*,const int*,int,int,int);
  Kernel kernels[] = {moe_gather, moe_gather32, moe_gather_scalar};
  for (int choice=0; choice<3; ++choice)
  for (int width=1; width<=35; ++width)
  for (int inOffset=0; inOffset<4; ++inOffset)
  for (int outOffset=0; outOffset<4; ++outOffset) {
    int writes[512] = {};
    Kernel kernel=kernels[choice];
    int tile=choice==0 ? 16 : 32;
    if (choice==2) tile=width;
    int slots=5*((width+tile-1)/tile)*tile;
    // Include extra threads and boundaries not aligned to vector groups.
    blockDim.x = 7;
    for (int lane=0; lane<slots+9; ++lane) {
      std::fill(output, output+512, -1.0f);
      blockIdx.x = lane/7; threadIdx.x = lane%7;
      kernel(input+inOffset, output+outOffset, route, 5, width, 3);
      for (int j=0; j<512; ++j) {
        if (output[j] == -1.0f) continue;
        int index = j-outOffset;
        if (index < 0 || index >= 3*width) return 2;
        int token=index/width, col=index%width;
        if (output[j] != input[inOffset+route[token]*width+col]) return 3;
        ++writes[j];
      }
    }
    for (int j=0; j<512; ++j) {
      int expected = j>=outOffset && j<outOffset+3*width ? 1 : 0;
      if (writes[j] != expected) {
        std::fprintf(stderr,"width=%d offsets=%d/%d index=%d writes=%d expected=%d\n",
          width,inOffset,outOffset,j,writes[j],expected);
        return 4;
      }
    }
  }
  return 0;
}
`
