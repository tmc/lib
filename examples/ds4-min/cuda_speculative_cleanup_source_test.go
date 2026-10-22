package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Execute production cleanup and replay bodies with faulting metadata owners.
// This is not a device-copy or target-arithmetic qualification.
func TestCUDASpeculativeCleanupSource(t *testing.T) {
	data, err := os.ReadFile("cuda_speculative_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	function := func(name string) string {
		t.Helper()
		start := strings.Index(string(data), "func (v *cudaSpeculativeVerifier) "+name+"(")
		if start < 0 {
			t.Fatal("missing method", name)
		}
		body := string(data[start:])
		end := strings.Index(body, "\n}\n")
		if end < 0 {
			t.Fatal("missing method boundary", name)
		}
		return body[:end+3]
	}
	body := function("closeFrontier") + function("Close") + function("abort") + function("CommitPrefix") + function("VerifyBatch") + function("CurrentLogits") + function("Snapshot") + function("Restore")
	power, err := os.ReadFile("power_control.go")
	if err != nil {
		t.Fatal(err)
	}
	body += dsparkSourceFunction(t, string(power), "func admitGroupedPower(")
	body = strings.ReplaceAll(body, "dsv4.Snapshot", "*snap")
	for _, mutation := range []string{"", "retain", "restore", "commit-admission", "prior-close", "explicit-close", "logits-failure", "snapshot-failure", "restore-failure"} {
		source := body
		switch mutation {
		case "retain":
			source = strings.Replace(source, "return &backendCleanupOwner", "v.frontier = nil; return &backendCleanupOwner", 1)
		case "restore":
			source = strings.Replace(source, "fmt.Errorf(\"restore speculative frontier: %w\", restoreErr)", "fmt.Errorf(\"restore speculative frontier: %v\", restoreErr)", 1)
		case "commit-admission":
			source = strings.Replace(source, "if v.failed != nil {", "if false {", 1)
		case "prior-close":
			source = strings.Replace(source, "if err := v.closeFrontier(nil); err != nil {", "if err := error(nil); err != nil {", 1)
		case "explicit-close":
			source = strings.Replace(source, "v.closed = true", "v.closed = false", 1)
		case "logits-failure":
			source = strings.Replace(source, "v.closed || v.failed != nil ||", "v.closed ||", 1)
		case "snapshot-failure", "restore-failure":
			name := "Snapshot"
			if mutation == "restore-failure" {
				name = "Restore"
			}
			original := strings.ReplaceAll(function(name), "dsv4.Snapshot", "*snap")
			source = strings.Replace(source, original, strings.Replace(original, "if v.failed != nil {", "if false {", 1), 1)
		}
		program := `package main
import("errors";"fmt";"io";"testing")
var dsv4=struct{ErrClosed,ErrDesynchronized,ErrUnimplemented error}{errors.New("closed"),errors.New("desync"),errors.New("unimplemented")}
type snap struct{fail error;calls int}
func(s *snap)Close()error{s.calls++;return s.fail}
type powerControl struct{}
type runtime struct{power *powerControl}
type state struct{restoreErr error;restores,snapshots int;closed bool;generation uint64;runtime *runtime}
func(s *state)Restore(*snap)error{s.restores++;return s.restoreErr}
func(s *state)Snapshot()(*snap,error){s.snapshots++;return &snap{},nil}
type adapter struct{err error;evals int;closed bool}
func(a *adapter)Eval(*state,int,int)error{a.evals++;return a.err}
func(a *adapter)Logits(*state)[]float32{return []float32{1,0}}
type cudaSpeculativeVerifier struct{frontier *snap;state *state;adapter *adapter;drafts []int;rows [][]float32;rowGeneration uint64;pos,seed int;batched,closed bool;failed error}
func(v *cudaSpeculativeVerifier)retainRows(rows [][]float32)error{v.rows=rows;v.rowGeneration=v.state.generation;return nil}
func(v *cudaSpeculativeVerifier)validateRowFrontier()error{return nil}
func checkFiniteLogits(string,[]float32)error{return nil}
func(v *cudaSpeculativeVerifier)verifyGrouped([]int,int)([]int,error){return nil,errors.New("unexpected grouped dispatch")}
func validateCUDASpeculativeBatch(int,[]int,int)error{return nil}
func cudaSpeculativeVocabSize(*state)(uint64,error){return 2,nil}
func validateCUDASpeculativeBatchVocab(int,[]int,uint64)error{return nil}
func cudaSpeculativeArgmax([]float32)int{return 0}
type backendCleanupOwner struct{cause error;backend io.Closer}
func(e *backendCleanupOwner)Error()string{return e.cause.Error()}
func(e *backendCleanupOwner)Unwrap()error{return e.cause}
func(e *backendCleanupOwner)Close()error{return e.backend.Close()}
` + source + `
func TestCleanup(t *testing.T){
 cause:=errors.New("primary");free:=errors.New("free");restore:=errors.New("restore")
 f:=&snap{fail:free};v:=&cudaSpeculativeVerifier{frontier:f,state:&state{},adapter:&adapter{},drafts:[]int{4}}
 err:=v.closeFrontier(cause)
 if !errors.Is(err,cause)||!errors.Is(err,free)||!errors.Is(err,dsv4.ErrClosed)||v.frontier!=f{t.Fatal("lost cleanup frontier or cause")}
 if v.CurrentLogits()!=nil{t.Fatal("failed verifier exposed logits")}
 if _,err:=v.Snapshot();!errors.Is(err,free)||v.state.snapshots!=0{t.Fatal("failed verifier snapshot admitted")}
 if err:=v.Restore(f);!errors.Is(err,free)||v.state.restores!=0{t.Fatal("failed verifier restore admitted")}
 var owner io.Closer;if !errors.As(err,&owner){t.Fatal("no retry owner")};f.fail=nil
 if err:=owner.Close();err!=nil||f.calls!=2||v.frontier!=nil{t.Fatal("retry did not close retained frontier")}
 if v.failed==nil{t.Fatal("cleanup retry reopened execution")}
 f=&snap{fail:free};v=&cudaSpeculativeVerifier{frontier:f,state:&state{},adapter:&adapter{}}
 _,err=v.VerifyBatch(0,[]int{0},0)
 if !errors.Is(err,free)||v.frontier!=f||v.state.snapshots!=0||v.adapter.evals!=0{t.Fatal("prior close failure admitted target work")}
 f=&snap{};v=&cudaSpeculativeVerifier{frontier:f,state:&state{restoreErr:restore},adapter:&adapter{},drafts:[]int{4}}
 err=v.abort(cause)
 if !errors.Is(err,cause)||!errors.Is(err,restore)||!errors.Is(err,dsv4.ErrDesynchronized)||v.failed==nil{t.Fatal("restore cause or poison lost")}
 f=&snap{fail:free};v=&cudaSpeculativeVerifier{frontier:f,state:&state{},adapter:&adapter{},drafts:[]int{4}}
 err=v.CommitPrefix(1)
 if !errors.Is(err,free)||v.adapter.evals!=1||v.frontier!=f||v.failed==nil{t.Fatal("commit lost cleanup owner")}
 if err:=v.CommitPrefix(1);!errors.Is(err,dsv4.ErrClosed)||v.adapter.evals!=1{t.Fatal("failed verifier replayed")}
 f.fail=nil;if err:=v.Close();err!=nil{t.Fatal(err)}
 v=&cudaSpeculativeVerifier{state:&state{},adapter:&adapter{}}
 if err:=v.Close();err!=nil{t.Fatal(err)}
 if v.CurrentLogits()!=nil{t.Fatal("explicit close exposed logits")}
 if _,err:=v.Snapshot();!errors.Is(err,dsv4.ErrClosed){t.Fatal("explicit close admitted snapshot")}
 if err:=v.Restore(&snap{});!errors.Is(err,dsv4.ErrClosed){t.Fatal("explicit close admitted restore")}
 if _,err:=v.VerifyBatch(0,[]int{0},0);!errors.Is(err,dsv4.ErrClosed){t.Fatal("explicit close admitted verify")}
 if err:=v.Close();err!=nil{t.Fatal("idempotent close",err)}
}
`
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "cleanup_test.go"), []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("go", "test", "-v", "cleanup_test.go", "-count=1")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		t.Logf("mutation=%q rc=%v\n%s", mutation, err, out)
		if mutation == "" && err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		if mutation != "" && (err == nil || !strings.Contains(string(out), "--- FAIL: TestCleanup")) {
			t.Fatalf("mutation uncaught: %v\n%s", err, out)
		}
	}
}
