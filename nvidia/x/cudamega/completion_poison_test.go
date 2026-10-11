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

func TestCompletionCountRefusal(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("clang++ not installed")
	}
	source, err := cudamega.ProgramSource(programGraph(t, 16), programOps(), cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128}, goldenTarget)
	if err != nil {
		t.Fatal(err)
	}
	var guards strings.Builder
	for _, line := range strings.Split(source, "\n") {
		if strings.Contains(line, "if (published") {
			guards.WriteString(line)
			guards.WriteByte('\n')
		}
	}
	code := `#include <cstdint>
constexpr unsigned producers_0=1;
unsigned atomicCAS(unsigned *p,unsigned old,unsigned value) { unsigned prev=*p; if(prev==old)*p=value; return prev; }
int main() {
 struct Case { unsigned published,ready,abort; };
 Case cases[]={{0,0,0},{1,1,0},{2,0,5},{UINT32_MAX,0,5}};
 for(auto c: cases) {
 unsigned published=c.published, task_ready=1, abort=0; unsigned *abort_code=&abort;
` + guards.String() + fmt.Sprintf(`
 if(task_ready!=c.ready || abort!=c.abort) return 1;
 }
 if(%d!=5) return 2;
 return 0;
}
`, cudamega.AbortBadCompletion)
	dir := t.TempDir()
	path := filepath.Join(dir, "completion.cc")
	binary := filepath.Join(dir, "completion")
	if err := os.WriteFile(path, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(compiler, "-std=c++11", path, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile completion guards: %v\n%s", err, out)
	}
	if out, err := exec.Command(binary).CombinedOutput(); err != nil {
		t.Fatalf("completion guards: %v\n%s", err, out)
	}
}
