package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// cudamem can return a live allocation together with a context-restoration
// error. Execute the actual caller admission fragments with that result.
func TestDSparkPartialAllocationCleanup(t *testing.T) {
	runDSparkPartialAllocationCleanup(t, "weights")
}

func runDSparkPartialAllocationCleanup(t *testing.T, kind string) {
	t.Helper()
	weights, err := os.ReadFile("dspark_cuda_weights_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(weights)
	start := strings.Index(text, "\t\tmemory, e := cudamem.Alloc[byte]")
	if start < 0 {
		t.Fatal("weights allocation missing")
	}
	rest := text[start:]
	end := strings.Index(rest, "\n\t\tfor start :=")
	if end < 0 {
		t.Fatal("weights allocation boundary missing")
	}
	weightBody := strings.ReplaceAll(rest[:end], "cudamem.Alloc[byte]", "fakeAlloc")
	historyBody := "_ = f; _ = values; _ = j; return nil, nil"
	if kind == "history" {
		history, err := os.ReadFile("dspark_cuda_history_linux.go")
		if err != nil {
			t.Fatal(err)
		}
		text = string(history)
		start = strings.Index(text, "\tstorage, err := cudamem.Alloc[float32]")
		if start < 0 {
			t.Fatal("history allocation missing")
		}
		rest = text[start:]
		end = strings.Index(rest, "\n}\n")
		if end < 0 {
			t.Fatal("history allocation boundary missing")
		}
		historyBody = strings.ReplaceAll(rest[:end], "cudamem.Alloc[float32]", "fakeAlloc")
	}
	cleanup, err := os.ReadFile("backend_cleanup.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, mutant := range []bool{false, true} {
		wbody, hbody := weightBody, historyBody
		if mutant && kind == "weights" {
			changed := strings.Replace(wbody, "w.storage = append(w.storage, memory)", "_ = memory", 1)
			if changed == wbody {
				t.Fatal("weights retention mutation missing")
			}
			wbody = changed
		}
		if mutant && kind == "history" {
			changed := strings.Replace(hbody, "finishBackendCleanup(err, h)", "err", 1)
			if changed == hbody {
				t.Fatal("history retention mutation missing")
			}
			hbody = changed
		}
		program := `package main
import("errors";"fmt";"io";"testing")
var allocationError=errors.New("allocation context restore failed")
var releaseError=errors.New("free failed")
var resource *fakeResource
type fakeResource struct{ frees int }
type fakeSlice struct{ resource *fakeResource }
func(s fakeSlice)Ptr()uintptr{if s.resource==nil{return 0};return 1}
func(s fakeSlice)Free()error{if s.resource==nil{return nil};s.resource.frees++;if s.resource.frees==1{return releaseError};return nil}
func fakeAlloc(int,int)(fakeSlice,error){return fakeSlice{resource},allocationError}
type fakeOwner struct{storage []fakeSlice}
func(w *fakeOwner)Close()error{for _,s:=range w.storage{if err:=s.Free();err!=nil{return err}};w.storage=nil;return nil}
func weightsConstructor(w *fakeOwner)(result *fakeOwner,err error){
 defer func(){if err!=nil{err=finishBackendCleanup(err,w)}}()
 device,bytes,name:=0,1,"tensor"
` + wbody + `
 return w,nil
}
type fakeRuntime struct{device int}
type cudaDSparkFeatures struct{r *fakeRuntime;targets []int}
type cudaDSparkHistory struct{owner *cudaDSparkFeatures;values fakeSlice;journal int;width int}
func(h *cudaDSparkHistory)Close()error{if err:=h.values.Free();err!=nil{return err};h.values=fakeSlice{};return nil}
var Flash=struct{Embed int}{1}
func historyConstructor()(*cudaDSparkHistory,error){f:=&cudaDSparkFeatures{r:&fakeRuntime{},targets:[]int{1}};values,j:=1,1
` + hbody + `
}
func TestPartial(t *testing.T){resource=&fakeResource{}
 var err error
 if "` + kind + `"=="weights"{_,err=weightsConstructor(&fakeOwner{})}else{_,err=historyConstructor()}
 if !errors.Is(err,allocationError)||!errors.Is(err,releaseError){t.Fatal("allocation cleanup missing",err)}
 var owner io.Closer;if !errors.As(err,&owner){t.Fatal("partial owner missing")}
 if err=owner.Close();err!=nil{t.Fatal(err)}
 if resource.frees!=2{t.Fatal("partial allocation not retried")}
 if err=owner.Close();err!=nil||resource.frees!=2{t.Fatal("successful partial free repeated")}
}
`
		dir := t.TempDir()
		for name, data := range map[string][]byte{"backend_cleanup.go": cleanup, "partial_test.go": []byte(program)} {
			if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		cmd := exec.Command("go", "test", "backend_cleanup.go", "partial_test.go", "-count=1")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if !mutant && err != nil {
			t.Fatalf("%s baseline: %v\n%s", kind, err, out)
		}
		if mutant && (err == nil || !strings.Contains(string(out), "allocation cleanup missing")) {
			t.Fatalf("%s mutant not caught: %v\n%s", kind, err, out)
		}
	}
}
