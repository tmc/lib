package cudamega_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

type extentBody struct{}

func (extentBody) Inputs() int { return 1 }
func (extentBody) Source(in []string, out string) (string, error) {
	return out + " = " + in[0] + ";", nil
}
func (extentBody) Evaluate(in []uint32) uint32 { return in[0] }

func TestProgramRequiresMaximum(t *testing.T) {
	g, _ := cudataskgraph.New([]cudataskgraph.Task{{ID: "a", Scope: cudataskgraph.ScopeDevice}})
	for _, op := range []cudamega.Operation{{Kind: cudamega.Add}, {Kind: cudamega.Multiply}, {Body: extentBody{}}} {
		grid := cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32}
		ops := map[string]cudamega.Operation{"a": op}
		if p, err := cudamega.NewProgram(g, ops, grid, cudamega.SM80); err == nil || p != nil || !strings.Contains(err.Error(), "nonzero maximum") {
			t.Fatalf("undeclared bound: p=%v err=%v", p, err)
		}
		grid.MaxElements = 4
		p, err := cudamega.NewProgram(g, ops, grid, cudamega.SM80)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := p.Invocation(4); err != nil {
			t.Fatal(err)
		}
		if _, err := p.Invocation(5); err == nil {
			t.Fatal("invocation exceeds maximum")
		}
		if _, err := p.Storage(5); err == nil {
			t.Fatal("storage exceeds maximum")
		}
		src, err := p.Source()
		if err != nil {
			t.Fatal(err)
		}
		if strings.Count(src, "if (n > 4u)") != 2 {
			t.Fatal("ordinary/persistent bound mismatch")
		}
	}
}

// Run only the emitted ordinary entry's extent guard as host C++. This is
// not device or synchronization qualification. The stub payload gives an
// independent canary oracle for a bound removed from that actual entry.
func TestOrdinaryDeclaredMaximum(t *testing.T) {
	g, _ := cudataskgraph.New([]cudataskgraph.Task{{ID: "a", Scope: cudataskgraph.ScopeDevice}})
	src, err := cudamega.ProgramSource(g, map[string]cudamega.Operation{"a": {Kind: cudamega.Add, Scalar: 3}}, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32, MaxElements: 4}, cudamega.SM80)
	if err != nil {
		t.Fatal(err)
	}
	start := strings.Index(src, "extern \"C\" __global__ void ordinary_task")
	if start < 0 {
		t.Fatal("ordinary entry missing")
	}
	code := `#include <cstddef>
#define __global__
struct Dim { unsigned x; };
Dim blockIdx{0},blockDim{32},threadIdx{0};
int dependency(unsigned) {return -1;}
unsigned dispatch(unsigned,unsigned x){return x+3;}
` + src[start:] + `
int main(){
 unsigned in[5]={1,2,3,4,5},out[5]={99,99,99,99,99};
 for(threadIdx.x=0;threadIdx.x<5;threadIdx.x++) ordinary_task(in,out,5,0);
 for(unsigned x:out) if(x!=99) return 1;
 for(threadIdx.x=0;threadIdx.x<5;threadIdx.x++) ordinary_task(in,out,4,0);
 for(unsigned i=0;i<4;i++) if(out[i]!=in[i]+3) return 2;
 if(out[4]!=99) return 3;
}
`
	dir := t.TempDir()
	cc, bin := filepath.Join(dir, "bound.cc"), filepath.Join(dir, "bound")
	if err := os.WriteFile(cc, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(compiler, "-std=c++11", cc, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("host ordinary compile: %v %s", err, out)
	}
	if out, err := exec.Command(bin).CombinedOutput(); err != nil {
		t.Fatalf("host ordinary extent guard: %v %s", err, out)
	}
}
