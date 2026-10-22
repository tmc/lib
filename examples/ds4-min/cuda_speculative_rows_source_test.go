package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Exercise actual target-row ownership and lifecycle methods without CUDA.
func TestCUDASpeculativeRowsSource(t *testing.T) {
	data, err := os.ReadFile("cuda_speculative_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	var body string
	for _, name := range []string{"CurrentLogits", "BoundTo", "RowLogits", "validateRowFrontier", "VerifyBatch", "CommitPrefix", "retainRows", "verifyGrouped", "Restore", "closeFrontier", "Close"} {
		body += dsparkSourceFunction(t, string(data), "func (v *cudaSpeculativeVerifier) "+name+"(") + "\n"
	}
	for _, name := range []string{"cudaSpeculativeRowTops", "cudaSpeculativeArgmax"} {
		body += dsparkSourceFunction(t, string(data), "func "+name+"(") + "\n"
	}
	power, err := os.ReadFile("power_control.go")
	if err != nil {
		t.Fatal(err)
	}
	body += dsparkSourceFunction(t, string(power), "func admitGroupedPower(") + "\n"
	body = strings.ReplaceAll(body, "dsv4.State", "*state")
	body = strings.ReplaceAll(body, "dsv4.Snapshot", "*snapshot")
	fixture := `package fixture
import("errors";"fmt";"io";"testing")
var dsv4=struct{ErrClosed,ErrUnimplemented error}{errors.New("closed"),errors.New("unimplemented")}
const cudaSpeculativeMaxBatch=16
func checkFiniteLogits(string,[]float32)error{return nil}
func validateCUDASpeculativeBatch(int,[]int,int)error{return nil}
func cudaSpeculativeVocabSize(*state)(uint64,error){return 8,nil}
func validateCUDASpeculativeBatchVocab(int,[]int,uint64)error{return nil}
type features struct{}
func(*features)beginGrouped([]int,int)error{return nil}
func(*features)invalidate(){}
func(*features)sealGrouped()error{return nil}
type powerControl struct{}
type runtime struct{pos,ctx int;dspark *features;rows [][]float32;power *powerControl;readouts int}
func(r *runtime)evalBatchReadout(d []int,p int,_ bool,rows *[][]float32)error{r.readouts++;*rows=r.rows;r.pos=p+len(d);return nil}
type state struct{closed bool;owner *adapter;runtime *runtime;dspark *features;generation,epoch uint64;restores,snapshots int}
func(s *state)speculativeEpoch()(uint64,error){return s.epoch,nil}
func(s *state)advanceSpeculativeEpoch()error{s.generation++;return nil}
func(s *state)Restore(*snapshot)error{s.restores++;s.runtime.pos=7;return nil}
func(s *state)Snapshot()(*snapshot,error){s.snapshots++;return &snapshot{},nil}
type adapter struct{closed bool;values []float32;evals int}
func(a *adapter)Logits(*state)[]float32{return a.values}
func(a *adapter)Eval(s *state,_ int,p int)error{a.evals++;s.runtime.pos=p+1;return nil}
type snapshot struct{}
func(*snapshot)Close()error{return nil}
type backendCleanupOwner struct{cause error;backend io.Closer}
func(e *backendCleanupOwner)Error()string{return e.cause.Error()}
type cudaSpeculativeVerifier struct{adapter *adapter;state *state;frontier *snapshot;rows [][]float32;rowGeneration,rowEpoch uint64;pos,seed int;drafts []int;closed,batched bool;failed error}
func(v *cudaSpeculativeVerifier)abort(err error)error{return v.closeFrontier(err)}
func verifier()*cudaSpeculativeVerifier{a:=&adapter{values:[]float32{1,2}};s:=&state{owner:a,runtime:&runtime{pos:9},generation:4};return &cudaSpeculativeVerifier{adapter:a,state:s,frontier:&snapshot{},pos:7,drafts:[]int{1,2}}}
func TestRows(t *testing.T){
 v:=verifier();if !v.BoundTo(v.state)||v.BoundTo(&state{}){t.Fatal("foreign identity admitted")}
 current:=v.CurrentLogits();current[0]=8;if v.adapter.values[0]!=1{t.Fatal("current logits alias")}
 source:=[][]float32{{3,4},{5,6}};v.retainRows(source);source[0][0]=99
 row,err:=v.RowLogits(0);if err!=nil||row[0]!=3{t.Fatal("retained row alias",row,err)};row[0]=98;row,err=v.RowLogits(0);if err!=nil||row[0]!=3{t.Fatal("returned row alias",row,err)}
 if _,err=v.RowLogits(-1);err==nil{t.Fatal("negative row admitted")};if _,err=v.RowLogits(2);err==nil{t.Fatal("missing row admitted")}
 v.state.generation++;if _,err=v.RowLogits(0);err==nil{t.Fatal("stale generation admitted")};v.state.generation--
 v.state.epoch++;if _,err=v.RowLogits(0);err==nil{t.Fatal("foreign lifecycle admitted")};v.state.epoch--
 v.state.runtime.pos++;if _,err=v.RowLogits(0);err==nil{t.Fatal("foreign frontier admitted")};v.state.runtime.pos--
 if err=v.Restore(&snapshot{});err!=nil||v.rows!=nil{t.Fatal("restore retained rows",err)}
 v=verifier();v.retainRows(source);if err=v.closeFrontier(nil);err!=nil||v.rows!=nil{t.Fatal("finalization retained rows",err)}
 v=verifier();v.retainRows(source);if err=v.Close();err!=nil||v.BoundTo(v.state)||v.rows!=nil{t.Fatal("close retained capability",err)}
 v=verifier();v.retainRows(source);v.state.closed=true;if v.BoundTo(v.state)||v.CurrentLogits()!=nil{t.Fatal("closed target exposed capability")};if _,err=v.RowLogits(0);err==nil{t.Fatal("closed target exposed rows")}
 v=verifier();v.retainRows(source);v.adapter.closed=true;if v.BoundTo(v.state)||v.CurrentLogits()!=nil{t.Fatal("closed backend exposed capability")};if _,err=v.RowLogits(0);err==nil{t.Fatal("closed backend exposed rows")}
}
func TestGroupedRows(t *testing.T){
 v:=verifier();v.state.runtime.pos=7;v.state.runtime.ctx=16;v.state.runtime.rows=[][]float32{{3,4},{5,6}}
 tops,err:=v.verifyGrouped(v.drafts,7);if err!=nil||len(tops)!=2{t.Fatal("grouped target failed",tops,err)}
 row,err:=v.RowLogits(0);if err!=nil||len(row)!=2||row[0]!=3{t.Fatal("grouped target row missing or shifted",row,err)}
 v.state.runtime.rows[0][0]=88;row,err=v.RowLogits(0);if err!=nil||row[0]!=3{t.Fatal("grouped row not detached",row,err)}
}
func TestGroupedPowerAdmission(t *testing.T){
 v:=verifier();v.batched=true;v.state.runtime.pos=7;v.state.runtime.ctx=16;v.state.runtime.power=&powerControl{}
 frontier:=v.frontier
 if _,err:=v.VerifyBatch(0,[]int{1,2},7);err==nil||err.Error()!="power throttling is not qualified for grouped speculative execution"{t.Fatal("grouped power admitted",err)}
 if v.frontier!=frontier||v.state.snapshots!=0||v.state.restores!=0||v.adapter.evals!=0||v.state.runtime.readouts!=0||v.state.runtime.pos!=7{t.Fatal("grouped power rejection changed frontier")}
 for _,tc:=range []struct{grouped bool;drafts []int}{{false,[]int{1,2}},{true,[]int{1}}}{
  v=verifier();v.frontier=nil;v.batched=tc.grouped;v.state.runtime.pos=7;v.state.runtime.ctx=16;v.state.runtime.power=&powerControl{}
  if _,err:=v.VerifyBatch(0,tc.drafts,7);err!=nil||v.state.snapshots!=1||v.adapter.evals!=len(tc.drafts)||v.state.runtime.readouts!=0{t.Fatal("scalar power rejected",tc.grouped,err)}
 }
}
func TestCommitAdmission(t *testing.T){
 for _,kind:=range []string{"epoch","generation","position","owner","closed"}{
  v:=verifier();v.retainRows([][]float32{{3,4},{5,6}})
  switch kind{case "epoch":v.state.epoch++;case "generation":v.state.generation++;case "position":v.state.runtime.pos++;case "owner":v.state.owner=&adapter{};case "closed":v.adapter.closed=true}
  before:=v.state.runtime.pos
  if err:=v.CommitPrefix(0);err==nil||v.state.restores!=0||v.state.runtime.pos!=before||v.frontier==nil{t.Fatal("foreign transaction overwritten",kind,err)}
 }
 v:=verifier();v.retainRows([][]float32{{3,4},{5,6}})
 if err:=v.CommitPrefix(0);err!=nil||v.state.restores!=1||v.frontier!=nil||v.rows!=nil{t.Fatal("valid empty prefix failed",err)}
}
func TestCommitVerifiedFrontiers(t *testing.T){
 for _,grouped:=range []bool{false,true}{for _,accepted:=range []int{0,2}{
  v:=verifier();v.frontier=nil;v.state.runtime.pos=7;v.state.runtime.ctx=16
  rows:=[][]float32{{3,4},{5,6}}
  v.batched=grouped;v.state.runtime.rows=rows
  if _,err:=v.VerifyBatch(0,[]int{1,2},7);err!=nil{t.Fatal(err)}
  if err:=v.CommitPrefix(accepted);err!=nil||v.state.runtime.pos!=7+accepted||v.state.restores!=1||v.frontier!=nil{t.Fatal("valid verified prefix refused",grouped,accepted,err)}
 }}
 v:=verifier();v.retainRows([][]float32{{3,4},{5,6}});v.state.epoch++
 if err:=v.CommitPrefix(0);err==nil{t.Fatal("stale epoch admitted")}
 if err:=v.Close();err!=nil||v.state.restores!=0||v.frontier!=nil{t.Fatal("stale cleanup restored foreign state",err)}
}
`
	for _, mutation := range []string{"baseline", "retained-alias", "returned-alias", "generation", "epoch", "restore-clear", "finalize-clear", "grouped-retention", "commit-preflight", "grouped-power"} {
		candidate := body
		if mutation != "baseline" {
			needle, replacement := "", ""
			switch mutation {
			case "grouped-power":
				needle, replacement = "if err := admitGroupedPower(v.state.runtime.power); err != nil {", "if err := error(nil); err != nil {"
			case "commit-preflight":
				needle, replacement = "if err := v.validateRowFrontier(); err != nil {\n\t\treturn err\n\t}\n\tv.rows = nil", "v.rows = nil"
			case "retained-alias":
				needle, replacement = "v.rows[i] = append([]float32(nil), row...)", "v.rows[i] = row"
			case "returned-alias":
				needle, replacement = "return append([]float32(nil), v.rows[i]...), nil", "return v.rows[i], nil"
			case "generation":
				needle = " || v.state.generation != v.rowGeneration"
			case "epoch":
				needle, replacement = " || epoch != v.rowEpoch", " || false && epoch != v.rowEpoch"
			case "restore-clear":
				needle, replacement = "v.rows = nil\n\treturn v.state.Restore(snapshot)", "return v.state.Restore(snapshot)"
			case "finalize-clear":
				needle, replacement = "func (v *cudaSpeculativeVerifier) closeFrontier(cause error) error {\n\tv.rows = nil", "func (v *cudaSpeculativeVerifier) closeFrontier(cause error) error {"
			case "grouped-retention":
				needle, replacement = "if err := v.retainRows(rows); err != nil {", "if err := error(nil); err != nil {"
			}
			count := 1
			if mutation == "grouped-retention" {
				count = 2
			}
			if strings.Count(candidate, needle) != count {
				t.Fatal("mutation missing", mutation)
			}
			candidate = strings.Replace(candidate, needle, replacement, count)
		}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "rows_test.go"), []byte(fixture+candidate), 0600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("go", "test", "-count=1", "rows_test.go")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if mutation == "grouped-power" && (err == nil || !strings.Contains(string(out), "--- FAIL: TestGroupedPowerAdmission")) {
			t.Fatalf("grouped power mutation uncaught: %v\n%s", err, out)
		}
		if mutation == "baseline" && err != nil || mutation != "baseline" && (err == nil || !strings.Contains(string(out), "--- FAIL: Test")) {
			t.Fatalf("%s: %v\n%s", mutation, err, out)
		}
	}
}
