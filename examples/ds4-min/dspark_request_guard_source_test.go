package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestDSparkRequestGuardSource(t *testing.T) {
	data, err := os.ReadFile("dspark_cuda_dense_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	start := strings.Index(string(data), "func dsparkRequestOpen(")
	if start < 0 {
		t.Fatal("request guard missing")
	}
	body := string(data[start:])
	end := strings.Index(body, "\n}\n")
	if end < 0 {
		t.Fatal("request guard boundary missing")
	}
	body = body[:end+3]
	for _, mutant := range []bool{false, true} {
		source := body
		if mutant {
			source = strings.Replace(source, "return r.checkOpen()", "return nil", 1)
			if source == body {
				t.Fatal("closing guard mutation missing")
			}
		}
		program := `package main
import("errors";"testing")
var dsv4=struct{ErrClosed error}{errors.New("closed")}
type fakeFoundation struct{closing bool}
type cudaRuntime struct{cudaFoundation *fakeFoundation;failed error}
func(r *cudaRuntime)checkOpen()error{if r.cudaFoundation.closing{return dsv4.ErrClosed};return nil}
` + source + `
func TestGuard(t *testing.T){
 if err:=dsparkRequestOpen(nil);!errors.Is(err,dsv4.ErrClosed){t.Fatal("nil admitted")}
 if err:=dsparkRequestOpen(&cudaRuntime{});!errors.Is(err,dsv4.ErrClosed){t.Fatal("no foundation admitted")}
 r:=&cudaRuntime{cudaFoundation:&fakeFoundation{closing:true}}
 if err:=dsparkRequestOpen(r);!errors.Is(err,dsv4.ErrClosed){t.Fatal("closing foundation admitted")}
 cause:=errors.New("poisoned");r.failed=cause
 if err:=dsparkRequestOpen(r);!errors.Is(err,cause){t.Fatal("failure lost")}
 r.failed=nil;r.cudaFoundation.closing=false
 if err:=dsparkRequestOpen(r);err!=nil{t.Fatal("open rejected",err)}
}
`
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "guard_test.go"), []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("go", "test", "guard_test.go", "-count=1", "-v")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		t.Logf("mutant=%t ordinary_runner_error=%v\n%s", mutant, err, out)
		if !mutant && err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		if mutant && (err == nil || !strings.Contains(string(out), "closing foundation admitted")) {
			t.Fatalf("mutant not caught: %v\n%s", err, out)
		}
	}
}
