package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Run actual request bodies with counting owners. This qualifies admission and
// retained cleanup ownership only; no native operation is performed.
func dsparkOwnerSourceGate(t *testing.T, program, want string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "owner_test.go"), []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "test", "-count=1", "owner_test.go")
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if want == "" {
		if err != nil {
			t.Fatalf("actual body failed: %v\n%s", err, output)
		}
	} else if err == nil || !strings.Contains(string(output), want) {
		t.Fatalf("mutant did not fail on %q: %v\n%s", want, err, output)
	}
}

func TestCUDAClosedAdapterBatchAdmission(t *testing.T) {
	data, err := os.ReadFile("cuda_state_adapter_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	actual := dsparkSourceFunction(t, string(data), "func (a *cudaStateAdapter) EvalBatch(")
	actual = strings.ReplaceAll(actual, "dsv4.State", "any")
	fixture := `package main
import("errors";"fmt";"testing")
var dsv4=struct{ErrClosed,ErrUnimplemented error}{errors.New("closed"),errors.New("unimplemented")}
type powerControl struct{prefill bool}
type cudaRuntime struct{power *powerControl}
type cudaStateAdapter struct{closed bool;runtime *cudaRuntime}
type cudaRequestState struct{owner *cudaStateAdapter;closed bool;runtime *cudaRuntime;evals int;evalErr error}
func(s *cudaRequestState)eval(int,int)error{s.evals++;if s.runtime.power!=nil&&!s.runtime.power.prefill{return errors.New("batch eval missing prefill power")};return s.evalErr}
func TestAdmission(t *testing.T){
 for _,powered:=range []bool{false,true}{for _,previous:=range []bool{false,true}{
 for _,tokens:=range [][]int{nil,{1,2}}{
  a:=&cudaStateAdapter{closed:true,runtime:&cudaRuntime{}};s:=&cudaRequestState{owner:a,runtime:a.runtime}
  if powered{a.runtime.power=&powerControl{prefill:previous}}
  if err:=a.EvalBatch(s,tokens,0);!errors.Is(err,dsv4.ErrClosed)||s.evals!=0{t.Fatal("closed adapter reached batch",tokens,err)}
  if powered&&a.runtime.power.prefill!=previous{t.Fatal("closed adapter changed power")}
  a.closed=false;if err:=a.EvalBatch(s,tokens,0);err!=nil||s.evals!=len(tokens){t.Fatal("open batch refused",tokens,err)}
  if powered&&a.runtime.power.prefill!=previous{t.Fatal("batch power not restored")}
 }}}
}
func TestPowerFailure(t *testing.T){
 failure:=errors.New("eval failure")
 for _,previous:=range []bool{false,true}{
  a:=&cudaStateAdapter{runtime:&cudaRuntime{power:&powerControl{prefill:previous}}};s:=&cudaRequestState{owner:a,runtime:a.runtime,evalErr:failure}
  if err:=a.EvalBatch(s,[]int{1,2},0);!errors.Is(err,failure)||s.evals!=1{t.Fatal("eval failure lost or batch continued",err)}
  if a.runtime.power.prefill!=previous{t.Fatal("batch power not restored after failure")}
 }
}
`
	dsparkOwnerSourceGate(t, fixture+actual, "")
	mutant := strings.Replace(actual, " || a.closed", "", 1)
	if mutant == actual {
		t.Fatal("closed adapter mutation missing")
	}
	dsparkOwnerSourceGate(t, fixture+mutant, "closed adapter reached batch")
	for _, tc := range []struct{ name, old, new, failure string }{
		{"power-prefill", "s.runtime.power.prefill = true", "s.runtime.power.prefill = previous", "open batch refused"},
		{"power-restore", "defer func() { s.runtime.power.prefill = previous }()", "_ = previous", "batch power not restored"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if strings.Count(actual, tc.old) != 1 {
				t.Fatal("power mutation site not unique")
			}
			source := strings.Replace(actual, tc.old, tc.new, 1)
			dsparkOwnerSourceGate(t, fixture+source, tc.failure)
		})
	}
}

func TestDSparkFeatureCloseTerminalRequest(t *testing.T) {
	data, err := os.ReadFile("cuda_state_adapter_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	var actual string
	for _, name := range []string{"Close", "finishClose", "Reset", "Load"} {
		actual += dsparkSourceFunction(t, string(data), "func (s *cudaRequestState) "+name+"(")
	}
	fixture := `package main
import("errors";"io";"strings";"testing")
var dsv4=struct{ErrClosed,ErrDesynchronized error}{errors.New("closed"),errors.New("desynchronized")}
var releaseError=errors.New("feature sync failed")
type cudaRuntime struct{failed error;resets,loads,closes int}
func(r *cudaRuntime)Reset()error{if errors.Is(r.failed,dsv4.ErrClosed){return r.failed};r.resets++;return nil}
func(r *cudaRuntime)Close()error{r.closes++;return nil}
type cudaDSparkFeatures struct{closed bool;closes int}
func(f *cudaDSparkFeatures)close()error{f.closed=true;f.closes++;if f.closes==1{return releaseError};return nil}
type cudaRequestState struct{runtime *cudaRuntime;closed bool;dspark *cudaDSparkFeatures}
func(s *cudaRequestState)invalidateDSpark(){}
func(s *cudaRequestState)advanceSpeculativeEpoch()error{return nil}
func(s *cudaRequestState)completeObservedCacheReplacement(err error)error{return err}
func loadCUDAState(s *cudaRequestState,r io.Reader)error{s.runtime.loads++;return nil}
func(s *cudaRequestState)loadSupport(io.Reader)error{panic("closed request reached support loader")}
func TestTerminal(t *testing.T){
 r:=&cudaRuntime{};f:=&cudaDSparkFeatures{};s:=&cudaRequestState{runtime:r,dspark:f}
 err:=s.Close()
 if !errors.Is(err,releaseError){t.Fatal("feature release error lost",err)}
 if s.dspark!=f||s.runtime!=r||s.closed||r.closes!=0{t.Fatal("pending native owners lost")}
 if !errors.Is(r.failed,dsv4.ErrClosed)||!errors.Is(r.failed,dsv4.ErrDesynchronized){t.Fatal("feature-close failure leaves native request admission open")}
 if err:=s.Reset();!errors.Is(err,dsv4.ErrClosed)||r.resets!=0{t.Fatal("reset reached pending native owner",err)}
 if err:=s.Load(strings.NewReader("valid"));!errors.Is(err,dsv4.ErrClosed)||r.loads!=0{t.Fatal("load reached pending native owner",err)}
 if err:=s.Close();err!=nil||!s.closed||s.runtime!=nil||s.dspark!=nil||r.closes!=1||f.closes!=2{t.Fatal("cleanup retry lost pending owner",err)}
 if err:=s.Close();err!=nil||r.closes!=1||f.closes!=2{t.Fatal("completed cleanup repeated",err)}
}
`
	for _, mutation := range []string{"baseline", "terminal-disposition"} {
		t.Run(mutation, func(t *testing.T) {
			source, want := actual, ""
			if mutation != "baseline" {
				source = strings.Replace(source, "return s.finishClose(err)", "return err", 1)
				if source == actual {
					t.Fatal("feature close mutation missing")
				}
				want = "feature-close failure leaves native request admission open"
			}
			dsparkOwnerSourceGate(t, fixture+source, want)
		})
	}
}

func TestDSparkGroupedContextPreservesFrontier(t *testing.T) {
	data, err := os.ReadFile("dspark_cuda_grouped_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	actual := dsparkSourceFunction(t, string(data), "func (f *cudaDSparkFeatures) beginGrouped(")
	fixture := `package main
import("fmt";"testing")
var Flash=struct{Vocab int}{128}
type fakeStatus struct{writes int}
func(s *fakeStatus)CopyFromHost([]uint32)error{s.writes++;return nil}
type fakeJournal struct{rows []int;frontier int}
type fakeHistory struct{journal *fakeJournal}
type cudaRuntime struct{pos,ctx int;failed error;dspark *cudaDSparkFeatures}
type cudaRequestState struct{generation uint64}
type cudaDSparkGroupedCapture struct{start int;tokens []int;captured []bool;generation uint64}
type cudaDSparkFeatures struct{r *cudaRuntime;request *cudaRequestState;position,token int;published bool;generation uint64;captured []bool;status fakeStatus;history *fakeHistory;grouped *cudaDSparkGroupedCapture;targets []int}
func(f *cudaDSparkFeatures)bound()bool{return f!=nil}
func(f *cudaDSparkFeatures)invalidate(){f.published=false}
func dsparkRequestOpen(*cudaRuntime)error{return nil}
func TestBounds(t *testing.T){
 for _,position:=range []int{-1,5,4,3}{
  f:=&cudaDSparkFeatures{r:&cudaRuntime{pos:position,ctx:4},request:&cudaRequestState{generation:7},position:position-1,token:11,published:true,generation:6,captured:[]bool{true,true},history:&fakeHistory{journal:&fakeJournal{rows:make([]int,8),frontier:position}},targets:[]int{1,2}}
  if err:=f.beginGrouped([]int{12,13},position);err==nil{t.Fatalf("out-of-context grouped begin admitted: position=%d",position)}
  if !f.published||f.position!=position-1||f.token!=11||f.generation!=6||!f.captured[0]||!f.captured[1]||f.status.writes!=0||f.grouped!=nil{t.Fatal("grouped preflight changed frontier")}
 }
 f:=&cudaDSparkFeatures{r:&cudaRuntime{pos:2,ctx:4},request:&cudaRequestState{generation:7},published:true,history:&fakeHistory{journal:&fakeJournal{rows:make([]int,8),frontier:2}},targets:[]int{1,2}}
 if err:=f.beginGrouped([]int{12,13},2);err!=nil||f.published||f.grouped==nil||f.grouped.start!=2||f.status.writes!=1{t.Fatal("valid final grouped admission refused",err)}
}
`
	for _, mutation := range []string{"baseline", "context-count"} {
		t.Run(mutation, func(t *testing.T) {
			source, want := actual, ""
			if mutation != "baseline" {
				source = strings.Replace(source, "len(tokens) > f.r.ctx-position || ", "", 1)
				if source == actual {
					t.Fatal("grouped context mutation missing")
				}
				want = "out-of-context grouped begin admitted"
			}
			dsparkOwnerSourceGate(t, fixture+source, want)
		})
	}
}
