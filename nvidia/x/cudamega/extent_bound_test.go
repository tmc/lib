package cudamega_test

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
)

// This runs the emitted entry refusals as ordinary C++, not as GPU qualification.
func TestScalarExtentEntryRefusal(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("clang++ not installed")
	}
	grid := cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128, MaxElements: 4}
	source, err := cudamega.ProgramSource(programGraph(t, 16), programOps(), grid, goldenTarget)
	if err != nil {
		t.Fatal(err)
	}
	entry := source[strings.Index(source, "extern \"C\" __global__ void dag_worker"):]
	entry = entry[:strings.Index(entry, "__shared__")]
	stub := `#include <cstdint>
#define __global__
constexpr unsigned grid_ctas=1;
constexpr unsigned grid_threads=128;
struct Dim { unsigned x,y,z; };
Dim gridDim{1,1,1}, blockDim{128,1,1}, threadIdx{0,0,0};
unsigned atomicCAS(unsigned *p, unsigned old, unsigned value) { unsigned prev=*p; if(prev==old)*p=value; return prev; }
`
	code := stub + entry + "}\n" + fmt.Sprintf(`
int main() {
 unsigned abort=0; unsigned input=0, scratch=0, completed=0; unsigned long long order=0;
 struct Case { unsigned n; unsigned long long epoch; unsigned want; };
 Case cases[]={{0,1,%d},{1,1,0},{3,1,0},{4,1,0},{5,1,%d},{UINT32_MAX,1,%d},{4,0,%d}};
 for(auto c: cases) { abort=0; dag_worker(&input,&scratch,&completed,&order,&abort,c.n,c.epoch); if(abort!=c.want) return 1; }
 abort=0; gridDim.x=2; dag_worker(&input,&scratch,&completed,&order,&abort,4,1); if(abort!=%d) return 2;
 abort=0; gridDim.x=1; blockDim.x=256; dag_worker(&input,&scratch,&completed,&order,&abort,4,1); if(abort!=2) return 4;
 abort=99; blockDim.x=128; dag_worker(&input,&scratch,&completed,&order,&abort,5,1); if(abort!=99) return 3;
 return 0;
}
`, cudamega.AbortBadExtent, cudamega.AbortBadExtent, cudamega.AbortBadExtent, cudamega.AbortBadExtent, cudamega.AbortBadLaunch)
	dir := t.TempDir()
	path := filepath.Join(dir, "entry.cc")
	binary := filepath.Join(dir, "entry")
	if err := os.WriteFile(path, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(compiler, "-std=c++11", path, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile entry: %v\n%s", err, out)
	}
	if out, err := exec.Command(binary).CombinedOutput(); err != nil {
		t.Fatalf("entry refusals: %v\n%s", err, out)
	}
}

func TestScalarExtentZeroLeavesCallerBoundUnset(t *testing.T) {
	source, err := cudamega.ProgramSource(programGraph(t, 16), programOps(), cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128}, goldenTarget)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(source, "if (n >") {
		t.Fatal("zero MaxElements inserted a caller bound")
	}
}

func ExampleGrid() {
	grid := cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128, MaxElements: 4}
	fmt.Println(grid.Validate(), grid.MaxElements)
	// Output: <nil> 4
}
