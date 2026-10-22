package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestDSparkDraftDiscardSource(t *testing.T) {
	data, err := os.ReadFile("dspark_cuda_draft_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	body := dsparkSourceFunction(t, string(data), "func (d *cudaDSparkDrafter) PreparedSeed(") + "\n" + dsparkSourceFunction(t, string(data), "func (d *cudaDSparkDrafter) DiscardDrafts(")
	for _, mutant := range []bool{false, true} {
		source := body
		if mutant {
			source = strings.Replace(source, "d.pending, d.drafted = nil, false", "d.pending, d.drafted = nil, false\n d.history,d.hash,d.prepared=nil,\"\",false", 1)
			if source == body {
				t.Fatal("preparation deletion mutation missing")
			}
		}
		program := `package main
import("fmt";"strings";"testing")
type fakeRuntime struct{pos int}
type fakeTarget struct{runtime *fakeRuntime;epoch uint64}
func(s *fakeTarget)speculativeEpoch()(uint64,error){return s.epoch,nil}
type fakeForward struct{target *fakeTarget;closed,failed bool}
type cudaDSparkDrafter struct{forward *fakeForward;history,pending []int;hash string;epoch uint64;prepared,drafted bool}
` + source + `
func TestDiscard(t *testing.T){
 for _,pending:=range [][]int{nil,{99}}{
  d:=&cudaDSparkDrafter{forward:&fakeForward{target:&fakeTarget{runtime:&fakeRuntime{pos:3},epoch:7}},history:[]int{1,2,3},hash:strings.Repeat("a",64),epoch:7,prepared:true,pending:pending,drafted:true}
  if err:=d.DiscardDrafts();err!=nil{t.Fatal(err)}
  if token,pos,err:=d.PreparedSeed();err!=nil||token!=3||pos!=3{t.Fatal("free decline lost preparation",err)}
  if d.drafted||len(d.pending)!=0{t.Fatal("lookahead retained")}
  d.forward.target.epoch++
  if err:=d.DiscardDrafts();err!=nil||d.prepared||len(d.history)!=0||d.hash!=""{t.Fatal("stale preparation retained",err)}
 }
}
`
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "lifecycle_test.go"), []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("go", "test", "lifecycle_test.go", "-count=1")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if !mutant && err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		if mutant && (err == nil || !strings.Contains(string(out), "free decline lost preparation")) {
			t.Fatalf("mutant not caught: %v\n%s", err, out)
		}
	}
}
