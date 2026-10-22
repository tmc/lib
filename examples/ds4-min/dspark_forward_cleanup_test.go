package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Execute the constructor's actual deferred cleanup against fake resources.
// This is a host ownership gate, not device synchronization qualification.
func TestDSparkForwardConstructorCleanup(t *testing.T) {
	source, err := os.ReadFile("dspark_cuda_forward_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	start := strings.Index(string(source), "\tdefer func() {")
	if start < 0 {
		t.Fatal("constructor cleanup missing")
	}
	rest := string(source[start:])
	end := strings.Index(rest, "\n\ts.arena =")
	if end < 0 {
		t.Fatal("constructor cleanup boundary missing")
	}
	deferred := rest[:end]
	closeStart := strings.Index(string(source), "func (s *cudaDSparkForward) close() error {")
	if closeStart < 0 {
		t.Fatal("forward cleanup missing")
	}
	closeSource := string(source[closeStart:])
	cleanup, err := os.ReadFile("backend_cleanup.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, mutant := range []bool{false, true} {
		body := deferred
		if mutant {
			body = strings.Replace(body, "finishBackendCleanup(err, s)", "errors.Join(err, s.Close())", 1)
			if body == deferred {
				t.Fatal("owner deletion mutation missing")
			}
		}
		dir := t.TempDir()
		program := `package main
import("errors";"io";"testing")
var primary=errors.New("FFN construction failed")
var syncFailure=errors.New("stream sync failed")
var freeFailure=errors.New("arena free failed")
type fakeRuntime struct { syncs int; failSync bool }
func(r *fakeRuntime)stream()*fakeRuntime{return r}
func syncStream(r *fakeRuntime)error{r.syncs++;if r.failSync&&r.syncs==1{return syncFailure};return nil}
type fakeRequest struct{runtime *fakeRuntime}
type fakeArena struct { frees int; released bool }
func(a *fakeArena)Free()error{a.frees++;if a.frees==1{return freeFailure};a.released=true;return nil}
type fakeFFN struct{}
func(*fakeFFN)close()error{return nil}
type cudaDSparkForward struct { target *fakeRequest; arena *fakeArena; ffn *fakeFFN; closed bool }
` + closeSource + `
func failedConstructor(s *cudaDSparkForward)(err error){
` + body + `
 return primary
}
func TestRetainedConstructor(t *testing.T){
 for _, failSync:=range []bool{false,true}{
  r:=&fakeRuntime{failSync:failSync};a:=&fakeArena{}
  s:=&cudaDSparkForward{target:&fakeRequest{r},arena:a}; err:=failedConstructor(s)
  if !errors.Is(err,primary){t.Fatal("primary missing")}
  var owner io.Closer
  if !errors.As(err,&owner){t.Fatal("cleanup owner missing")}
  if failSync {
   if !errors.Is(err,syncFailure){t.Fatal("sync failure missing")}
   if err=owner.Close(); !errors.Is(err,freeFailure){t.Fatal("free retry missing",err)}
  } else if !errors.Is(err,freeFailure){t.Fatal("free failure missing")}
  if err=owner.Close();err!=nil{t.Fatal(err)}
  if !a.released || a.frees!=2 || s.arena!=nil || !s.closed{t.Fatal("resource ownership lost")}
  syncs,frees:=r.syncs,a.frees
  if err=owner.Close();err!=nil || r.syncs!=syncs || a.frees!=frees{t.Fatal("successful release repeated")}
 }
}
`
		for name, data := range map[string][]byte{"backend_cleanup.go": cleanup, "forward_test.go": []byte(program)} {
			if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		cmd := exec.Command("go", "test", "backend_cleanup.go", "forward_test.go", "-count=1")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if !mutant && err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		if mutant && (err == nil || !strings.Contains(string(out), "cleanup owner missing")) {
			t.Fatalf("mutant not caught: %v\n%s", err, out)
		}
	}
}
