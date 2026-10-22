package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Run the actual proposal admission prefix without any CUDA operations.
func TestDSparkProposalAdmissionSource(t *testing.T) {
	data, err := os.ReadFile("dspark_cuda_forward_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	body := dsparkSourceFunction(t, string(data), "func (s *cudaDSparkForward) proposal(")
	end := strings.Index(body, "\n\tdefer func()")
	if end < 0 {
		t.Fatal("proposal admission boundary missing")
	}
	body = body[:end] + "\n return nil, nil, nil\n}\n"
	fixture := `package fixture
import("fmt";"math";"testing")
var Flash=struct{Vocab int}{129280}
var calls int
type model struct{block int}
type weights struct{closed bool;model *model}
type runtime struct{}
type target struct{closed bool;runtime *runtime;dspark *int}
type cudaDSparkForward struct{closed,failed bool;target *target;weights *weights}
func dsparkRequestOpen(*runtime)error{calls++;return nil}
func TestAdmission(t *testing.T){
 n:=1;s:=&cudaDSparkForward{target:&target{runtime:&runtime{},dspark:&n},weights:&weights{model:&model{block:5}}}
 for _,c:=range []struct{seed,count int;threshold float32}{ {-1,1,.5},{Flash.Vocab,1,.5},{0,0,.5},{0,6,.5},{0,1,-.1},{0,1,1.1},{0,1,float32(math.NaN())},{0,1,float32(math.Inf(1))},{0,1,float32(math.Inf(-1))}}{
 calls=0;if _,_,err:=s.proposal(c.seed,c.count,c.threshold);err==nil||calls!=0||s.failed{t.Fatal("rejected metadata reached runtime",c,calls,err)}
 }
 calls=0;if _,_,err:=s.proposal(0,5,1);err!=nil||calls!=1{t.Fatal("valid admission rejected",err,calls)}
}
`
	for _, mutation := range []string{"baseline", "seed", "threshold"} {
		candidate := body
		if mutation != "baseline" {
			needle := " || seed < 0 || seed >= Flash.Vocab"
			if mutation == "threshold" {
				needle = " || threshold < 0 || threshold > 1 || math.IsNaN(float64(threshold))"
			}
			if strings.Count(candidate, needle) != 1 {
				t.Fatal("admission mutation missing", mutation)
			}
			candidate = strings.Replace(candidate, needle, "", 1)
		}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "admission_test.go"), []byte(fixture+candidate), 0600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("go", "test", "-count=1", "admission_test.go")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if mutation == "baseline" && err != nil || mutation != "baseline" && (err == nil || !strings.Contains(string(out), "--- FAIL: TestAdmission")) {
			t.Fatalf("%s: %v\n%s", mutation, err, out)
		}
	}
}
