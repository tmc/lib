package cudareduce

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

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// distillNativeAdmission verifies existing custody. It never acquires a lock,
// discovers a tool or initializes a device.
func distillNativeAdmission(env func(string) string) error {
	if env == nil {
		return fmt.Errorf("%w: no environment", cudalock.ErrNotInherited)
	}
	if env("GPU_LOCK_DISABLED") != "" || env("CUDATEST_NO_LOCK") != "" {
		return fmt.Errorf("%w: native lock bypass is set", cudalock.ErrNotInherited)
	}
	_, err := cudalock.Inherited(env)
	return err
}

func TestSparseTeacherNativeAdmissionRefusesUnverifiedCustody(t *testing.T) {
	root := t.TempDir()
	for _, tc := range []struct{ name, held string }{
		{"missing", ""},
		{"malformed", "not-a-pid"},
		{"hand-set-one", "1"},
		{"unowned-live-pid", strconv.Itoa(os.Getpid())},
		{"unowned-pid", "2147483647"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values := map[string]string{"GPU_LOCK_HELD": tc.held, "GPU_LOCKDIR": filepath.Join(root, tc.name)}
			if err := distillNativeAdmission(func(key string) string { return values[key] }); !errors.Is(err, cudalock.ErrNotInherited) {
				t.Fatalf("admission = %v; want ErrNotInherited", err)
			}
		})
	}
	if err := distillNativeAdmission(nil); !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("nil environment: %v", err)
	}
}

func TestSparseTeacherNativeAdmissionRefusesBypassBeforeOwnerLookup(t *testing.T) {
	for _, flag := range []string{"GPU_LOCK_DISABLED", "CUDATEST_NO_LOCK"} {
		t.Run(flag, func(t *testing.T) {
			err := distillNativeAdmission(func(key string) string {
				if key != "GPU_LOCK_DISABLED" && key != "CUDATEST_NO_LOCK" {
					t.Fatalf("bypass admission inspected %s", key)
				}
				if key == flag {
					return "0"
				} // Nonempty, not a boolean interpretation.
				return ""
			})
			if !errors.Is(err, cudalock.ErrNotInherited) {
				t.Fatalf("admission = %v; want ErrNotInherited", err)
			}
		})
	}
}

// Parse the actual native entry without compiling or invoking its body.
// Matching the complete admission prefix covers both refusal and call order.
func TestSparseTeacherNativeAdmissionPrecedesVendorEntry(t *testing.T) {
	fset := token.NewFileSet()
	tree, err := parser.ParseFile(fset, "distill_native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	funcs := make(map[string]*ast.FuncDecl)
	for _, decl := range tree.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			funcs[fn.Name.Name] = fn
		}
	}
	for _, tc := range []struct{ name, prefix string }{
		{"TestSparseTeacher_Native", `if os.Getenv("DISTILL_NATIVE") != "1" { t.Skip("coordinator native dispatch not requested") }
if err := distillNativeAdmission(os.Getenv); err != nil { t.Fatal(err) }`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fn := funcs[tc.name]
			if fn == nil || fn.Body == nil {
				t.Fatal("native entry missing")
			}
			wantSet := token.NewFileSet()
			want, err := parser.ParseFile(wantSet, "", "package p\nfunc f() {\n"+tc.prefix+"\n}", 0)
			if err != nil {
				t.Fatal(err)
			}
			expected := want.Decls[0].(*ast.FuncDecl).Body.List
			if len(fn.Body.List) < len(expected) {
				t.Fatal("admission prefix missing")
			}
			for i, stmt := range expected {
				var got, want bytes.Buffer
				if err := format.Node(&got, fset, fn.Body.List[i]); err != nil {
					t.Fatal(err)
				}
				if err := format.Node(&want, wantSet, stmt); err != nil {
					t.Fatal(err)
				}
				if strings.Join(strings.Fields(got.String()), " ") != strings.Join(strings.Fields(want.String()), " ") {
					t.Fatalf("statement %d before vendor entry:\ngot: %s\nwant: %s", i, got.String(), want.String())
				}
			}
		})
	}
}
