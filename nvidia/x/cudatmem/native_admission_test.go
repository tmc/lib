package cudatmem

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// tmemNativeAdmission reuses the inherited custody protocol before native
// tool discovery or execution. Disabled native mode inspects only its opt-in;
// in particular it does not impose custody on compile-only controls.
func tmemNativeAdmission(env func(string) string) (bool, error) {
	if env == nil {
		return false, fmt.Errorf("%w: no environment", cudalock.ErrNotInherited)
	}
	if env("CUDA_TMEM_TEST") != "1" {
		return false, nil
	}
	if env("GPU_LOCK_DISABLED") != "" || env("CUDATEST_NO_LOCK") != "" {
		return false, fmt.Errorf("%w: native lock bypass is set", cudalock.ErrNotInherited)
	}
	if _, err := cudalock.Inherited(env); err != nil {
		return false, err
	}
	return true, nil
}

func TestNativeAdmissionDisabled(t *testing.T) {
	for _, value := range []string{"", "0", "true", " 1", "1 "} {
		calls := 0
		enabled, err := tmemNativeAdmission(func(key string) string {
			calls++
			if key != "CUDA_TMEM_TEST" {
				t.Fatalf("disabled native inspected %s", key)
			}
			return value
		})
		if enabled || err != nil || calls != 1 {
			t.Fatalf("disabled %q = %v, %v, calls %d", value, enabled, err, calls)
		}
	}
	if enabled, err := tmemNativeAdmission(nil); enabled || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("nil environment = %v, %v", enabled, err)
	}
}

func nativeAdmissionOwner(t *testing.T, root, name, host string, pid int) string {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	owner := cudalock.Owner{PID: pid, Host: host, Label: "constructed-cpu-control", Epoch: time.Now().Unix(), At: time.Now()}
	if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(owner.String()), 0600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestNativeAdmissionCustody(t *testing.T) {
	// Task-local owner records exercise the existing verifier without acquiring
	// the host GPU lock, discovering tools, or dispatching a native fixture.
	root := t.TempDir()
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	host = strings.SplitN(host, ".", 2)[0]
	parent := strconv.Itoa(os.Getppid())
	live := nativeAdmissionOwner(t, root, "parent", host, os.Getppid())
	forged := nativeAdmissionOwner(t, root, "foreign-host", "invalid-fixture-host", os.Getppid())
	malformed := filepath.Join(root, "malformed-owner")
	if err := os.Mkdir(malformed, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(malformed, "owner"), []byte("pid=invalid\n"), 0600); err != nil {
		t.Fatal(err)
	}
	values := map[string]string{"CUDA_TMEM_TEST": "1", "GPU_LOCK_HELD": parent, "GPU_LOCKDIR": live}
	env := func(key string) string { return values[key] }
	if enabled, err := tmemNativeAdmission(env); !enabled || err != nil {
		t.Fatalf("constructed parent control = %v, %v", enabled, err)
	}
	for _, tc := range []struct{ name, held, dir string }{
		{"missing", "", live}, {"hand-set-one", "1", live}, {"arbitrary-nonempty", "held", live},
		{"malformed-negative", "-1", live}, {"malformed-zero", "0", live}, {"malformed-fraction", "1.5", live},
		{"absent-directory", parent, filepath.Join(root, "absent")},
		{"missing-owner", parent, root}, {"forged-foreign-host", parent, forged},
		{"mismatched-owner", strconv.Itoa(os.Getpid()), live}, {"malformed-owner", parent, malformed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values["GPU_LOCK_HELD"], values["GPU_LOCKDIR"] = tc.held, tc.dir
			enabled, err := tmemNativeAdmission(env)
			if enabled || !errors.Is(err, cudalock.ErrNotInherited) {
				t.Fatalf("unverified custody admitted: %v, %v", enabled, err)
			}
		})
	}
	values["GPU_LOCK_HELD"], values["GPU_LOCKDIR"] = parent, live
	for _, key := range []string{"GPU_LOCK_DISABLED", "CUDATEST_NO_LOCK"} {
		for _, value := range []string{"0", "1", "false"} {
			t.Run(key+"="+value, func(t *testing.T) {
				values[key] = value
				enabled, err := tmemNativeAdmission(env)
				if enabled || !errors.Is(err, cudalock.ErrNotInherited) {
					t.Fatalf("bypass admitted: %v, %v", enabled, err)
				}
				delete(values, key)
			})
		}
	}
	if enabled, err := tmemNativeAdmission(env); !enabled || err != nil {
		t.Fatalf("valid control after refusals = %v, %v", enabled, err)
	}
}

// checkNativeAdmission checks the actual fixture's complete admission prefix.
// The oracle parses source; it never calls a fixture or discovers a compiler.
func checkNativeAdmission(source string) error {
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "tmem_test.go", source, 0)
	if err != nil {
		return err
	}
	var entry *ast.FuncDecl
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "TestNativeTMEMReference" {
			entry = fn
		}
	}
	if entry == nil || entry.Body == nil {
		return errors.New("native fixture missing")
	}
	wantSet := token.NewFileSet()
	wantFile, err := parser.ParseFile(wantSet, "", `package cudatmem
func f(t *testing.T){
f, err := GenerateFixture(SM100A)
if err != nil { t.Fatal(err) }
if err := ValidateFixture(f); err != nil { t.Fatal(err) }
if os.Getenv("CUDA_TMEM_TEST") != "1" && os.Getenv("CUDA_TMEM_COMPILE") != "1" {
 t.Skip("native fixture requires CUDA_TMEM_COMPILE=1 or CUDA_TMEM_TEST=1")
}
if _, err := tmemNativeAdmission(os.Getenv); err != nil { t.Fatal(err) }
}`, 0)
	if err != nil {
		return err
	}
	want := wantFile.Decls[0].(*ast.FuncDecl).Body.List
	if len(entry.Body.List) <= len(want) {
		return errors.New("native admission or tool boundary missing")
	}
	for i, statement := range want {
		var gotText, wantText bytes.Buffer
		if err := format.Node(&gotText, set, entry.Body.List[i]); err != nil {
			return err
		}
		if err := format.Node(&wantText, wantSet, statement); err != nil {
			return err
		}
		if strings.Join(strings.Fields(gotText.String()), " ") != strings.Join(strings.Fields(wantText.String()), " ") {
			return fmt.Errorf("native admission statement %d differs", i)
		}
	}
	// The next operation is the existing tool boundary, with no intervening body.
	var next bytes.Buffer
	if err := format.Node(&next, set, entry.Body.List[len(want)]); err != nil {
		return err
	}
	if strings.Join(strings.Fields(next.String()), " ") != "nvcc, _ := cudatest.NVCC(t)" {
		return errors.New("native tool boundary changed")
	}
	return nil
}

func TestNativeAdmissionCallSite(t *testing.T) {
	raw, err := os.ReadFile("tmem_test.go")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	if err := checkNativeAdmission(source); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, old, new string }{
		{"wrong environment", "tmemNativeAdmission(os.Getenv)", "tmemNativeAdmission(nil)"},
		{"bypass helper", "tmemNativeAdmission(os.Getenv)", "otherAdmission(os.Getenv)"},
		{"continue on error", "if _, err := tmemNativeAdmission(os.Getenv); err != nil {\n\t\tt.Fatal(err)", "if _, err := tmemNativeAdmission(os.Getenv); err != nil {\n\t\tt.Log(err)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Limit edits to the native function so unrelated host checks cannot kill it.
			start := strings.Index(source, "func TestNativeTMEMReference(")
			if start < 0 {
				t.Fatal("native fixture missing")
			}
			tail := strings.Replace(source[start:], tc.old, tc.new, 1)
			mutated := source[:start] + tail
			if mutated == source {
				t.Fatal("call-site omission did not change source")
			}
			if err := checkNativeAdmission(mutated); err == nil {
				t.Fatal("accepted missing native admission binding")
			}
		})
	}
}

// This is a constructed protocol control, not custody of a physical GPU.
func TestNativeAdmissionCanonicalParent(t *testing.T) {
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	dir := nativeAdmissionOwner(t, t.TempDir(), "parent", strings.SplitN(host, ".", 2)[0], os.Getppid())
	values := map[string]string{"CUDA_TMEM_TEST": "1", "GPU_LOCK_HELD": strconv.Itoa(os.Getppid()), "GPU_LOCKDIR": dir}
	enabled, err := tmemNativeAdmission(func(key string) string { return values[key] })
	if !enabled || err != nil {
		t.Fatalf("constructed parent admission = %v, %v", enabled, err)
	}
}
