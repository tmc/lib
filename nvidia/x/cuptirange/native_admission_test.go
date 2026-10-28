package cuptirange_test

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

// rangeCaptureNativeAdmission verifies inherited custody; it never acquires a GPU lease,
// discovers tools, reads PTX or initializes a device.
func rangeCaptureNativeAdmission(env func(string) string) (bool, error) {
	if env == nil {
		return false, fmt.Errorf("%w: no environment", cudalock.ErrNotInherited)
	}
	if env("CUPTI_RANGE_TEST") != "1" {
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

func TestRangeCaptureNativeAdmissionRefusesUnverifiedCustody(t *testing.T) {
	root := t.TempDir()
	for _, tc := range []struct{ name, held string }{
		{"missing", ""},
		{"malformed", "not-a-pid"},
		{"hand-set-one", "1"},
		{"unowned-live-pid", strconv.Itoa(os.Getpid())},
		{"unknown-owner", "2147483647"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values := map[string]string{"CUPTI_RANGE_TEST": "1", "GPU_LOCK_HELD": tc.held, "GPU_LOCKDIR": filepath.Join(root, tc.name)}
			enabled, err := rangeCaptureNativeAdmission(func(key string) string { return values[key] })
			if enabled || !errors.Is(err, cudalock.ErrNotInherited) {
				t.Fatalf("admission = %v, %v; want false, ErrNotInherited", enabled, err)
			}
		})
	}
	if enabled, err := rangeCaptureNativeAdmission(nil); enabled || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("nil environment = %v, %v", enabled, err)
	}
}

func TestRangeCaptureNativeAdmissionDisabled(t *testing.T) {
	for _, value := range []string{"", "0", "true"} {
		enabled, err := rangeCaptureNativeAdmission(func(key string) string {
			if key != "CUPTI_RANGE_TEST" {
				t.Fatalf("disabled fixture inspected %s", key)
			}
			return value
		})
		if enabled || err != nil {
			t.Fatalf("disabled(%q) = %v, %v", value, enabled, err)
		}
	}
}

func TestRangeCaptureNativeAdmissionRefusesBypassBeforeOwnerLookup(t *testing.T) {
	for _, flag := range []string{"GPU_LOCK_DISABLED", "CUDATEST_NO_LOCK"} {
		for _, value := range []string{"1", "0"} {
			t.Run(flag+"="+value, func(t *testing.T) {
				enabled, err := rangeCaptureNativeAdmission(func(key string) string {
					switch key {
					case "CUPTI_RANGE_TEST":
						return "1"
					case "GPU_LOCK_DISABLED", "CUDATEST_NO_LOCK":
						if key == flag {
							return value
						}
						return ""
					default:
						t.Fatalf("bypass admission inspected %s", key)
						return ""
					}
				})
				if enabled || !errors.Is(err, cudalock.ErrNotInherited) {
					t.Fatalf("admission = %v, %v", enabled, err)
				}
			})
		}
	}
}

func TestRangeCaptureNativeAdmissionCanonicalParent(t *testing.T) {
	// This is a private synthetic owner record for the actual live parent,
	// not an acquisition of the host GPU namespace or a native lease claim.
	dir := filepath.Join(t.TempDir(), "lock")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	owner := cudalock.Owner{PID: os.Getppid(), Host: strings.SplitN(host, ".", 2)[0], Label: "synthetic-parent", Epoch: time.Now().Unix(), At: time.Now()}
	file := filepath.Join(dir, "owner")
	if err := os.WriteFile(file, []byte(owner.String()), 0600); err != nil {
		t.Fatal(err)
	}
	values := map[string]string{"CUPTI_RANGE_TEST": "1", "GPU_LOCK_HELD": strconv.Itoa(owner.PID), "GPU_LOCKDIR": dir}
	env := func(key string) string { return values[key] }
	if enabled, err := rangeCaptureNativeAdmission(env); !enabled || err != nil {
		t.Fatalf("canonical parent = %v, %v", enabled, err)
	}
	values["GPU_LOCK_HELD"] = strconv.Itoa(os.Getpid())
	if enabled, err := rangeCaptureNativeAdmission(env); enabled || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("mismatched owner = %v, %v", enabled, err)
	}
	values["GPU_LOCK_HELD"] = strconv.Itoa(owner.PID)
	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}
	if enabled, err := rangeCaptureNativeAdmission(env); enabled || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("removed owner = %v, %v", enabled, err)
	}
}

// Inspect the real tagged entry without compiling or invoking its GPU body.
// Comparing the complete prefix catches omission, moving it past any file/tool
// call, ignoring its error and passing a substituted environment lookup.
func TestRangeCaptureNativeAdmissionPrecedesInputs(t *testing.T) {
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "native_linux_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var entry *ast.FuncDecl
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "TestNativeRangeCapture" {
			entry = fn
		}
	}
	if entry == nil || entry.Body == nil {
		t.Fatal("native entry missing")
	}
	wantSet := token.NewFileSet()
	wantFile, err := parser.ParseFile(wantSet, "", `package p
func f() {
	enabled, err := rangeCaptureNativeAdmission(os.Getenv)
	if err != nil { t.Fatal(err) }
	if !enabled { t.Skip("CUPTI_RANGE_TEST=1 required") }
}`, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := wantFile.Decls[0].(*ast.FuncDecl).Body.List
	if len(entry.Body.List) < len(want) {
		t.Fatal("admission prefix missing")
	}
	for i, statement := range want {
		var gotText, wantText bytes.Buffer
		if err := format.Node(&gotText, set, entry.Body.List[i]); err != nil {
			t.Fatal(err)
		}
		if err := format.Node(&wantText, wantSet, statement); err != nil {
			t.Fatal(err)
		}
		if strings.Join(strings.Fields(gotText.String()), " ") != strings.Join(strings.Fields(wantText.String()), " ") {
			t.Fatalf("statement %d before native inputs:\ngot: %s\nwant: %s", i, gotText.String(), wantText.String())
		}
	}
}
