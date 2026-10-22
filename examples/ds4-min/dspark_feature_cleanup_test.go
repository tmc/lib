package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The constructor defer and Close bodies are copied verbatim except replacing
// device slice types with fake handles. No actual allocation or NVRTC runs.
func TestDSparkFeatureConstructorCleanup(t *testing.T) {
	source, err := os.ReadFile("dspark_cuda_features_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	start := strings.Index(text, "\tdefer func() {")
	if start < 0 {
		t.Fatal("feature constructor cleanup missing")
	}
	rest := text[start:]
	end := strings.Index(rest, "\n\tf.features, err =")
	if end < 0 {
		t.Fatal("feature constructor boundary missing")
	}
	deferred := rest[:end]
	start = strings.Index(text, "func (f *cudaDSparkFeatures) close() error {")
	if start < 0 {
		t.Fatal("feature close missing")
	}
	closeSource := text[start:]
	closeSource = strings.ReplaceAll(closeSource, "cudamem.Slice[float32]", "fakeSlice")
	closeSource = strings.ReplaceAll(closeSource, "cudamem.Slice[uint32]", "fakeSlice")
	cleanup, err := os.ReadFile("backend_cleanup.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, mutation := range []string{"baseline", "owner", "module"} {
		body := deferred
		closeBody := closeSource
		if mutation == "owner" {
			body = strings.Replace(body, "finishBackendCleanup(err, f)", "errors.Join(err, f.Close())", 1)
			if body == deferred {
				t.Fatal("feature owner mutation missing")
			}
		}
		if mutation == "module" {
			closeBody = strings.Replace(closeSource, "f.module = 0", "_ = f.module", 1)
			if closeBody == closeSource {
				t.Fatal("module sentinel mutation missing")
			}
		}
		program := `package main
import("errors";"io";"testing")
var cause=errors.New("feature constructor failed")
var syncError=errors.New("sync failed")
var freeError=errors.New("status free failed")
var restoreError=errors.New("context restoration failed")
var unloads int
type fakeResource struct{ calls int; fail bool }
type fakeSlice struct{resource *fakeResource}
func(s fakeSlice)Ptr()uintptr{if s.resource==nil{return 0};return 1}
func(s fakeSlice)Free()error{if s.resource==nil{return nil};s.resource.calls++;if s.resource.fail&&s.resource.calls==1{return freeError};return nil}
type fakeDevice struct{restoreFail bool}
func(d *fakeDevice)WithContext(f func()error)error{if err:=f();err!=nil{return err};if d.restoreFail{d.restoreFail=false;return restoreError};return nil}
type fakeModule int
func(fakeModule)Unload()error{unloads++;return nil}
type fakeRuntime struct{syncs int;device *fakeDevice}
func(r *fakeRuntime)stream()*fakeRuntime{return r}
func syncStream(r *fakeRuntime)error{r.syncs++;if r.syncs==1{return syncError};return nil}
type fakeHistory struct{}
func(*fakeHistory)close()error{return nil}
type fakeRequest struct{dspark *cudaDSparkFeatures}
type cudaDSparkFeatures struct{closed bool;r *fakeRuntime;request *fakeRequest;module fakeModule;features,status fakeSlice;history *fakeHistory}
func(*cudaDSparkFeatures)invalidate(){}
` + closeBody + `
func failedConstructor(f *cudaDSparkFeatures)(err error){
` + body + `
 return cause
}
func TestRetainedFeatures(t *testing.T){
 a,b:=&fakeResource{},&fakeResource{fail:true};request:=&fakeRequest{}
 f:=&cudaDSparkFeatures{r:&fakeRuntime{device:&fakeDevice{}},request:request,features:fakeSlice{a},status:fakeSlice{b}}
 request.dspark=f;err:=failedConstructor(f)
 if !errors.Is(err,cause)||!errors.Is(err,syncError){t.Fatal("constructor/sync error missing",err)}
 var owner io.Closer
 if !errors.As(err,&owner){t.Fatal("cleanup owner missing")}
 if err=owner.Close();!errors.Is(err,freeError){t.Fatal("free failure missing",err)}
 if request.dspark!=f||a.calls!=1||b.calls!=1{t.Fatal("premature detach or release")}
 if err=owner.Close();err!=nil{t.Fatal(err)}
 if request.dspark!=nil||a.calls!=1||b.calls!=2{t.Fatal("retry lost handle or repeated free")}
 if err=owner.Close();err!=nil||a.calls!=1||b.calls!=2{t.Fatal("completed close repeated")}
}
func TestModuleRestore(t *testing.T){
 unloads=0
 f:=&cudaDSparkFeatures{r:&fakeRuntime{syncs:1,device:&fakeDevice{restoreFail:true}},module:7}
 if err:=f.Close();!errors.Is(err,restoreError){t.Fatal("context restore error missing",err)}
 if f.module!=0||unloads!=1{t.Fatal("unloaded module retained")}
 if err:=f.Close();err!=nil||unloads!=1{t.Fatal("unloaded module retried",err)}
}
`
		dir := t.TempDir()
		for name, data := range map[string][]byte{"backend_cleanup.go": cleanup, "feature_test.go": []byte(program)} {
			if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		cmd := exec.Command("go", "test", "backend_cleanup.go", "feature_test.go", "-count=1")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if mutation == "baseline" && err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		want := map[string]string{"owner": "cleanup owner missing", "module": "unloaded module retained"}[mutation]
		if mutation != "baseline" && (err == nil || !strings.Contains(string(out), want)) {
			t.Fatalf("mutant not caught: %v\n%s", err, out)
		}
	}
}
