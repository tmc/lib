package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func powerFunctionSource(t *testing.T, path, name string) string {
	t.Helper()
	source, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, source, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == name {
			return string(source[fset.Position(fn.Pos()).Offset:fset.Position(fn.End()).Offset])
		}
	}
	t.Fatalf("missing method %s", name)
	return ""
}

// Actual admission methods run against stub owners. This is a host admission
// test, not a grouped-device qualification or numerical reference.
func TestPowerGroupedAdmissionSource(t *testing.T) {
	cuda := strings.ReplaceAll(powerFunctionSource(t, "cuda_speculative_linux.go", "VerifyBatch"), "dsv4.ErrClosed", "errClosed")
	metal := powerFunctionSource(t, "metal_grouped_darwin.go", "admitMetalGroup")
	policy, err := os.ReadFile("power_control.go")
	if err != nil {
		t.Fatal(err)
	}
	stub := `package main
import("errors";"fmt";"testing";"time")
var errClosed=errors.New("closed")
func powerSleep(time.Duration){}
func powerSleepSupported()error{return nil}
type cudaRuntime struct{power *powerControl}
type snapshot struct{}
type cudaRequestState struct{closed bool;runtime *cudaRuntime;snapshots,evals int}
func(s *cudaRequestState)Snapshot()(*snapshot,error){s.snapshots++;return &snapshot{},nil}
type cudaStateAdapter struct{}
func(a *cudaStateAdapter)Eval(s *cudaRequestState,t,p int)error{s.evals++;return nil}
type cudaSpeculativeVerifier struct{closed,batched bool;adapter *cudaStateAdapter;state *cudaRequestState;failed error;frontier *snapshot;seed,pos,groups int;drafts []int}
func(v *cudaSpeculativeVerifier)closeFrontier(error)error{return nil}
func(v *cudaSpeculativeVerifier)verifyGrouped(d []int,p int)([]int,error){v.groups++;return d,nil}
func(v *cudaSpeculativeVerifier)CurrentLogits()[]float32{return []float32{1}}
func(v *cudaSpeculativeVerifier)abort(e error)error{return e}
func(v *cudaSpeculativeVerifier)retainRows([][]float32)error{return nil}
func validateCUDASpeculativeBatch(int,[]int,int)error{return nil}
func cudaSpeculativeVocabSize(*cudaRequestState)(uint64,error){return 128,nil}
func validateCUDASpeculativeBatchVocab(int,[]int,uint64)error{return nil}
func cudaSpeculativeArgmax([]float32)int{return 0}
func checkFiniteLogits(string,[]float32)error{return nil}
const ds4RawCap=128
var Flash=struct{Vocab int}{128}
type indexState struct{n int}
type layerState struct{indexer indexState}
type metalRuntime struct{power *powerControl;streaming,collectors,steering any;failed error;speculativeEpoch uint64;ctx,pos int;layers [43]int;state struct{layers [43]layerState}}
func TestPowerGroupedAdmission(t *testing.T){
for _,throttle:=range []bool{false,true}{
var p *powerControl;if throttle{p=newPowerControl(50,43)}
s:=&cudaRequestState{runtime:&cudaRuntime{power:p}}
v:=&cudaSpeculativeVerifier{adapter:&cudaStateAdapter{},state:s,batched:true,pos:11,seed:7,drafts:[]int{9}}
_,err:=v.VerifyBatch(3,[]int{4,5},2)
if throttle {if err==nil || s.snapshots!=0 || s.evals!=0 || v.groups!=0 || v.pos!=11 || v.seed!=7 || len(v.drafts)!=1 || v.drafts[0]!=9 {t.Fatal("throttled CUDA grouped admission changed frontier")}}else if err!=nil || s.snapshots!=1 || v.groups!=1{t.Fatal("ordinary CUDA grouped baseline")}
r:=&metalRuntime{power:p,ctx:16,pos:1};err=admitMetalGroup(r,[]int{4,5},1)
if throttle && err==nil{t.Fatal("throttled Metal grouped admission accepted")};if !throttle && err!=nil{t.Fatal("ordinary Metal grouped baseline",err)}
}}
`
	dir := t.TempDir()
	for _, mutation := range []string{"baseline", "cuda", "metal"} {
		c, m := cuda, metal
		if mutation == "cuda" {
			start := strings.Index(c, "\tif v.batched && len(drafts) > 1 && v.state.runtime != nil {")
			if start < 0 {
				t.Fatal("CUDA admission mutation start")
			}
			offset := strings.Index(c[start:], "\n\tif err := validateCUDASpeculativeBatch")
			if offset < 0 {
				t.Fatal("CUDA admission mutation end")
			}
			c = c[:start] + c[start+offset:]
		}
		if mutation == "metal" {
			start := strings.Index(m, "\tif r != nil {")
			if start < 0 {
				t.Fatal("Metal admission mutation start")
			}
			offset := strings.Index(m[start:], "\n\tif r == nil")
			if offset < 0 {
				t.Fatal("Metal admission mutation end")
			}
			m = m[:start] + m[start+offset:]
		}
		if err := os.WriteFile(filepath.Join(dir, "policy.go"), policy, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "admission_test.go"), []byte(stub+"\n"+c+"\n"+m), 0600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("go", "test", "-timeout=60s", "-count=1", "policy.go", "admission_test.go")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if mutation == "baseline" {
			if err != nil {
				t.Fatalf("actual admission baseline: %v\n%s", err, out)
			}
			continue
		}
		exit, ok := err.(*exec.ExitError)
		if !ok || exit.ExitCode() != 1 || !strings.Contains(string(out), "--- FAIL: TestPowerGroupedAdmission") || !strings.Contains(string(out), "throttled ") || strings.Contains(string(out), "[build failed]") || strings.Contains(string(out), "panic:") {
			t.Fatalf("admission mutant %s did not fail intended assertion: %v\n%s", mutation, err, out)
		}
	}
}
