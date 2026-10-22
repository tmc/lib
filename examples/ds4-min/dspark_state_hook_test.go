package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func dsparkSourceFunction(t *testing.T, source, name string) string {
	t.Helper()
	start := strings.Index(source, name)
	if start < 0 {
		t.Fatalf("source function missing: %s", name)
	}
	rest := source[start:]
	end := strings.Index(rest, "\n}\n")
	if end < 0 {
		t.Fatalf("source function unterminated: %s", name)
	}
	return rest[:end+3]
}

// Execute actual adapter hooks against fake target operations. This does not
// qualify device copies, successful physical rollback, or snapshot bytes.
func TestDSparkStateTransitionSource(t *testing.T) {
	adapter, err := os.ReadFile("cuda_state_adapter_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	features, err := os.ReadFile("dspark_cuda_features_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	var source string
	for _, name := range []string{"func (s *cudaRequestState) eval(", "func (s *cudaRequestState) Reset(", "func (s *cudaRequestState) Load("} {
		source += dsparkSourceFunction(t, string(adapter), name) + "\n"
	}
	for _, name := range []string{"func (s *cudaRequestState) invalidateDSpark(", "func (s *cudaRequestState) finishDSparkEval("} {
		source += dsparkSourceFunction(t, string(features), name) + "\n"
	}
	for _, mutation := range []string{"baseline", "post-seal", "reset", "load", "load-epoch", "support-dispatch"} {
		body := source
		switch mutation {
		case "post-seal":
			body = strings.Replace(body, "return s.finishDSparkEval()", "return nil", 1)
		case "reset":
			body = strings.Replace(body, "s.invalidateDSpark()\n\treturn s.advanceSpeculativeEpoch()", "return s.advanceSpeculativeEpoch()", 1)
		case "load":
			body = strings.Replace(body, "s.invalidateDSpark()\n\treturn s.completeObservedCacheReplacement(nil)", "return s.completeObservedCacheReplacement(nil)", 1)
		case "load-epoch":
			body = strings.Replace(body, "return s.completeObservedCacheReplacement(nil)", "return nil", 1)
		case "support-dispatch":
			body = strings.Replace(body, "return s.loadSupport(r)", "return nil", 1)
		}
		if mutation != "baseline" && body == source {
			t.Fatalf("mutation %s did not apply", mutation)
		}
		program := `package main
import("errors";"fmt";"io";"strings";"testing")
var dsv4=struct{ErrClosed,ErrDesynchronized error}{errors.New("closed"),errors.New("desynchronized")}
var operationError=errors.New("operation failed")
var supportResult=errors.New("support loader result")
type fakeHistory struct{epoch int}
func(h *fakeHistory)reset()error{h.epoch++;return nil}
type cudaDSparkFeatures struct{history *fakeHistory;published bool;beginError,sealError error;sealed int}
func(f *cudaDSparkFeatures)begin(int,int)error{return f.beginError}
func(f *cudaDSparkFeatures)seal()error{f.sealed++;return f.sealError}
func(f *cudaDSparkFeatures)invalidate(){f.published=false}
type cudaRuntime struct{pos int;failed error;dspark *cudaDSparkFeatures;resetError error}
func(r *cudaRuntime)Eval(int,int)error{if r.failed!=nil{return r.failed};r.pos++;return nil}
func(r *cudaRuntime)Reset()error{if r.resetError!=nil{return r.resetError};r.pos=0;return nil}
type cudaRequestState struct{runtime *cudaRuntime;closed bool;dspark *cudaDSparkFeatures;generation,lifecycleEpoch uint64;supportLoads int}
// Support restore owns a separate transactional source gate. Here only the
// actual adapter's dispatch and error propagation are under test.
func(s *cudaRequestState)loadSupport(io.Reader)error{s.supportLoads++;return supportResult}
func(s *cudaRequestState)advanceSpeculativeEpoch()error{s.lifecycleEpoch++;return nil}
func(s *cudaRequestState)completeObservedCacheReplacement(err error)error{if err!=nil{return err};return s.advanceSpeculativeEpoch()}
func loadCUDAState(s *cudaRequestState,r io.Reader)error{b,_:=io.ReadAll(r);if string(b)!="valid"{return operationError};s.runtime.pos=3;return nil}
` + body + `
func fixture()*cudaRequestState{f:=&cudaDSparkFeatures{history:&fakeHistory{},published:true};return &cudaRequestState{runtime:&cudaRuntime{},dspark:f,generation:2,lifecycleEpoch:5}}
func TestHooks(t *testing.T){
 s:=fixture();s.dspark.sealError=operationError
 if err:=s.eval(1,0);!errors.Is(err,dsv4.ErrDesynchronized)||!errors.Is(err,operationError){t.Fatal("seal failure not poisoned",err)}
 if s.runtime.pos!=1||s.runtime.failed==nil||s.runtime.dspark!=nil||s.dspark.published{t.Fatal("advanced target remains reusable")}
 s=fixture();if err:=s.eval(1,0);err!=nil||s.generation!=3||s.lifecycleEpoch!=6||s.dspark.sealed!=1||s.runtime.dspark!=nil{t.Fatal("successful seal not called",err)}
 s=fixture();s.dspark.beginError=operationError
 if err:=s.eval(1,0);!errors.Is(err,operationError)||s.runtime.pos!=0||s.generation!=2||s.lifecycleEpoch!=5{t.Fatal("rejected begin advanced target",err)}
 s=fixture();s.runtime.resetError=operationError
 if err:=s.Reset();!errors.Is(err,operationError)||!s.dspark.published||s.generation!=2||s.lifecycleEpoch!=5{t.Fatal("rejected reset mutated history",err)}
 s.runtime.resetError=nil
 if err:=s.Reset();err!=nil||s.dspark.published||s.generation!=3||s.lifecycleEpoch!=6||s.dspark.history.epoch!=1{t.Fatal("reset did not invalidate",err)}
 s=fixture()
 if err:=s.Load(strings.NewReader("support"));!errors.Is(err,supportResult)||s.supportLoads!=1||!s.dspark.published||s.generation!=2||s.lifecycleEpoch!=5{t.Fatal("support load dispatch lost",err)}
 // No attached support means the legacy loader owns cache replacement. Its
 // successful load must advance the speculative epoch; failure must not.
 s.dspark=nil
 if err:=s.Load(strings.NewReader("bad"));!errors.Is(err,operationError)||s.generation!=2||s.lifecycleEpoch!=5{t.Fatal("rejected load mutated history",err)}
 if err:=s.Load(strings.NewReader("valid"));err!=nil||s.generation!=3{t.Fatal("load did not invalidate",err)}
 if s.lifecycleEpoch!=6||s.runtime.pos!=3{t.Fatal("load did not advance epoch")}
}
`
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "hooks_test.go"), []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("go", "test", "hooks_test.go", "-count=1")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if mutation == "baseline" && err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		want := map[string]string{"post-seal": "seal failure not poisoned", "reset": "reset did not invalidate", "load": "load did not invalidate", "load-epoch": "load did not advance epoch", "support-dispatch": "support load dispatch lost"}[mutation]
		if mutation != "baseline" && (err == nil || !strings.Contains(string(out), want)) {
			t.Fatalf("%s mutant not caught: %v\n%s", mutation, err, out)
		}
	}
}
