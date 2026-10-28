package cudamega_test

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

type pairBody struct{}

func (pairBody) Inputs() int { return 2 }
func (pairBody) Source(inputs []string, output string) (string, error) {
	return fmt.Sprintf("%s = %s*7u + %s;", output, inputs[0], inputs[1]), nil
}
func (pairBody) Evaluate(inputs []uint32) uint32 { return inputs[0]*7 + inputs[1] }

func TestMultiInputSource(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "a", Scope: cudataskgraph.ScopeDevice, Extent: 5},
		{ID: "b", Scope: cudataskgraph.ScopeDevice, Extent: 3},
		{ID: "c", Rank: 1, Deps: []string{"a", "b"}, Inputs: []string{"b", "a"}, Scope: cudataskgraph.ScopeDevice, Extent: 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	p, err := cudamega.NewProgram(g, map[string]cudamega.Operation{"a": {Kind: cudamega.Add, Scalar: 1}, "b": {Kind: cudamega.Multiply, Scalar: 3}, "c": {Body: pairBody{}}}, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 32, MaxElements: 5, Policy: cudamega.CountedPolicy}, cudamega.SM80)
	if err != nil {
		t.Fatal(err)
	}
	source, err := p.Source()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"op_2(unsigned x0, unsigned x1)",
		"result = x0*7u + x1;",
		"const unsigned *src_2_0 = scratch + (5ull + (size_t)0u*n);",
		"const unsigned *src_2_1 = scratch + (0ull + (size_t)0u*n);",
		"scratch[(8ull + (size_t)0u*n) + e] = op_2(src_2_0[e], src_2_1[e]);",
		"e < 3u;",
		"if ((size_t)(n) < (size_t)(5u))",
		"scratch[(8ull + (size_t)0u*n) + e] = op_2(x0[e], x1[e]);",
	} {
		if !strings.Contains(source, want) {
			t.Errorf("missing binding %q", want)
		}
	}
	if strings.Contains(source, "scratch[(size_t)2*n") {
		t.Fatal("consumer output still uses the input extent as a stride")
	}
	// Compile and run the actual ordinary emitted body as ordinary C++. This
	// checks its operand order and ragged indexing; it is not CUDA execution.
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Fatal("host source oracle requires clang++")
	}
	prefix := source[:strings.Index(source, "extern \"C\" __global__ void dag_worker")]
	prefix = strings.Replace(prefix, "#include <cuda_runtime.h>", "", 1)
	code := `#include <cstdint>
#include <cstddef>
#define __device__
#define __global__
struct Dim { unsigned x,y,z; };
Dim blockIdx{0,0,0}, blockDim{1,1,1}, threadIdx{0,0,0};
` + prefix + `
int main() {
  unsigned input[5]={1,2,3,4,5}, guarded[13];
  for (unsigned &x: guarded) x=0xdeadbeefu;
  unsigned *scratch=guarded+1;
  for (unsigned task=0; task<3; ++task)
    for (blockIdx.x=0; blockIdx.x<7; ++blockIdx.x)
      ordinary_task(input,scratch,5,task);
  unsigned want[11]={2,3,4,5,6,3,6,9,23,45,67};
  for (unsigned i=0; i<11; ++i) if (scratch[i]!=want[i]) return 1;
  if (guarded[0]!=0xdeadbeefu || guarded[12]!=0xdeadbeefu) return 2;
  for (unsigned &x: guarded) x=0xdeadbeefu;
  blockIdx.x=0;
  ordinary_task(input,scratch,4,0);
  ordinary_task(input,scratch,6,0);
  for (unsigned x: guarded) if (x!=0xdeadbeefu) return 3;
}
`
	dir := t.TempDir()
	path, binary := filepath.Join(dir, "ordinary.cc"), filepath.Join(dir, "ordinary")
	if err := os.WriteFile(path, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	if output, err := exec.Command(compiler, "-std=c++11", path, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("ordinary source compilation: %v\n%s", err, output)
	}
	if output, err := exec.Command(binary).CombinedOutput(); err != nil {
		t.Fatalf("ordinary emitted arithmetic/indexing: %v\n%s", err, output)
	}
}
