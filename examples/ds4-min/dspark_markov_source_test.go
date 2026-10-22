package main

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// Only the actual F32 row-selection body executes here. Fake half declarations
// compile the unused branches; no F16/Q8_0 or physical CUDA claim follows.
func TestDSparkMarkovSourcePreviousProposal(t *testing.T) {
	source, err := os.ReadFile("cuda_dspark_head.cu")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	start := strings.Index(text, "extern \"C\" __global__ void ds4_dspark_markov_row(")
	end := strings.Index(text, "extern \"C\" __global__ void ds4_dspark_markov_argmax(")
	if start < 0 || end <= start {
		t.Fatal("Markov row kernel missing")
	}
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("clang++ unavailable: CPU kernel-body gate unmeasured")
	}
	for _, mutant := range []bool{false, true} {
		body := text[start:end]
		if mutant {
			changed := strings.Replace(body, "row == 0 ? initial : (unsigned int)proposal[row - 1]", "initial", 1)
			if changed == body {
				t.Fatal("Markov mutation did not apply")
			}
			body = changed
		}
		program := `#include <cstdio>
#define __global__
using __half = unsigned short;
static float __half2float(__half v) { return (float)v; }
struct { unsigned int x; } threadIdx={0}, blockIdx={0}, blockDim={256};
` + body + `
int main() {
  float weights[] = {0.1f,0.2f,0.3f,1.1f,1.2f,1.3f,2.1f,2.2f,2.3f,3.1f,3.2f,3.3f};
  int proposals[] = {3,1};
  for (unsigned int row=0; row<3; ++row) {
    float state[3]={};
    for (unsigned int d=0; d<3; ++d) {
      threadIdx.x=d;
      ds4_dspark_markov_row((unsigned char*)weights,state,proposals,2,row,3,0);
    }
    printf("%.9g %.9g %.9g\n",state[0],state[1],state[2]);
  }
}
`
		dir := t.TempDir()
		path, binary := filepath.Join(dir, "markov.cc"), filepath.Join(dir, "markov")
		if err := os.WriteFile(path, []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
		if out, err := exec.Command(compiler, "-std=c++17", path, "-o", binary).CombinedOutput(); err != nil {
			t.Fatalf("compile: %v\n%s", err, out)
		}
		out, err := exec.Command(binary).Output()
		if err != nil {
			t.Fatal(err)
		}
		fields := strings.Fields(string(out))
		if len(fields) != 9 {
			t.Fatalf("output fields: %q", out)
		}
		mismatches := 0
		for row, token := range []int{2, 3, 1} {
			for d := 0; d < 3; d++ {
				got, err := strconv.ParseFloat(fields[row*3+d], 64)
				if err != nil {
					t.Fatal(err)
				}
				want := float64(token) + float64(d+1)/10
				if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 1e-6 {
					mismatches++
				}
			}
		}
		if mutant && mismatches == 0 || !mutant && mismatches != 0 {
			t.Fatalf("mutant=%t mismatches=%d", mutant, mismatches)
		}
	}
}
