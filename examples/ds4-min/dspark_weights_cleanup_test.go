package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Exercise the actual weights Close body with a successful module unload
// followed by failed context restoration. Neither case may retry that module.
func TestDSparkWeightsUnloadRestore(t *testing.T) {
	data, err := os.ReadFile("dspark_cuda_weights_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	start := strings.Index(string(data), "func (w *cudaDSparkWeights) close() error {")
	if start < 0 {
		t.Fatal("weights close missing")
	}
	body := string(data[start:])
	end := strings.Index(body, "\n}\n")
	if end < 0 {
		t.Fatal("weights close boundary missing")
	}
	body = body[:end+3]
	body = strings.ReplaceAll(body, "driver.CtxSynchronize", "fakeSync")
	body = strings.ReplaceAll(body, "driver.Module", "fakeModule")
	body = strings.ReplaceAll(body, "cudamem.Slice[byte]", "fakeSlice")
	lifetime, err := os.ReadFile("dspark_lifetime.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, mutation := range []string{"baseline", "active", "pending"} {
		source := body
		if mutation == "active" {
			source = strings.Replace(source, "w.module, w.matrix = 0, 0", "_ = w.module", 1)
		}
		if mutation == "pending" {
			source = strings.Replace(source, "module = 0", "_ = module", 1)
		}
		if mutation != "baseline" && source == body {
			t.Fatal("module sentinel mutation missing", mutation)
		}
		program := `package main
import("errors";"sync";"testing")
var restoreError=errors.New("context restoration failed")
var unloads,frees int
type fakeModule int
func(fakeModule)Unload()error{unloads++;return nil}
type fakeSlice struct{}
func(fakeSlice)Free()error{frees++;return nil}
type fakeDevice struct{fail bool}
func(d *fakeDevice)WithContext(f func()error)error{if err:=f();err!=nil{return err};if d.fail&&unloads>0{d.fail=false;return restoreError};return nil}
func fakeSync()error{return nil}
type cudaDSparkWeights struct{mu sync.Mutex;closed bool;storage []fakeSlice;module fakeModule;matrix int;moduleCleanup []fakeModule;device *fakeDevice;tensors map[string]int;model any}
` + source + `
func TestRestore(t *testing.T){
 for _,pending:=range []bool{false,true}{
  unloads,frees=0,0;w:=&cudaDSparkWeights{device:&fakeDevice{fail:true},storage:[]fakeSlice{{}}}
  if pending{w.moduleCleanup=[]fakeModule{7}}else{w.module=7}
  if err:=w.close();!errors.Is(err,restoreError){t.Fatal("context restoration error lost",err)}
  if w.module!=0||len(w.moduleCleanup)!=0||unloads!=1{t.Fatal("freed module retained")}
  if err:=w.close();err!=nil||unloads!=1||frees!=1{t.Fatal("freed module retried",err)}
  if err:=w.close();err!=nil||unloads!=1||frees!=1{t.Fatal("completed resources retried",err)}
 }
}
`
		dir := t.TempDir()
		for name, contents := range map[string][]byte{"dspark_lifetime.go": lifetime, "weights_test.go": []byte(program)} {
			if err := os.WriteFile(filepath.Join(dir, name), contents, 0600); err != nil {
				t.Fatal(err)
			}
		}
		cmd := exec.Command("go", "test", "dspark_lifetime.go", "weights_test.go", "-count=1")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if mutation == "baseline" && err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		if mutation != "baseline" && (err == nil || !strings.Contains(string(out), "freed module retained")) {
			t.Fatalf("%s mutant not caught: %v\n%s", mutation, err, out)
		}
	}
}
