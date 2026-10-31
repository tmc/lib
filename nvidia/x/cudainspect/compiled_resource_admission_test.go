package cudainspect

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// compiledResourceAdmission verifies inherited custody before the fixture can
// read a packet, emit source, inspect an image, or enter a vendor library.
// It does not acquire a lock or establish native qualification.
func compiledResourceAdmission(env func(string) string) (bool, error) {
	if env == nil {
		return false, fmt.Errorf("%w: no environment", cudalock.ErrNotInherited)
	}
	if env("GPU_LOCK_DISABLED") != "" || env("CUDATEST_NO_LOCK") != "" {
		return false, fmt.Errorf("%w: compiled-resource fixture lock bypass is set", cudalock.ErrNotInherited)
	}
	if env("GPU_LOCK_HELD") == "" {
		return false, nil
	}
	if _, err := cudalock.Inherited(env); err != nil {
		return false, err
	}
	return true, nil
}

func TestCompiledResourceAdmissionAbsent(t *testing.T) {
	enabled, err := compiledResourceAdmission(func(key string) string {
		switch key {
		case "GPU_LOCK_DISABLED", "CUDATEST_NO_LOCK", "GPU_LOCK_HELD":
			return ""
		default:
			t.Fatalf("absent opt-in inspected %q", key)
			return ""
		}
	})
	if enabled || err != nil {
		t.Fatalf("absent admission = %v, %v; want false,nil", enabled, err)
	}
}

func TestCompiledResourceAdmissionRefuses(t *testing.T) {
	if enabled, err := compiledResourceAdmission(nil); enabled || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("nil admission = %v, %v", enabled, err)
	}
	for _, tc := range []struct {
		name   string
		held   string
		bypass string
		value  string
	}{
		{name: "malformed", held: "not-a-holder"},
		{name: "negative", held: "-1"},
		{name: "zero", held: "0"},
		{name: "legacy-one", held: "1"},
		{name: "unknown-owner", held: strconv.Itoa(os.Getppid())},
		{name: "gpu-disabled-one", held: "1", bypass: "GPU_LOCK_DISABLED", value: "1"},
		{name: "gpu-disabled-zero", held: "1", bypass: "GPU_LOCK_DISABLED", value: "0"},
		{name: "test-no-lock-one", held: "1", bypass: "CUDATEST_NO_LOCK", value: "1"},
		{name: "test-no-lock-zero", held: "1", bypass: "CUDATEST_NO_LOCK", value: "0"},
		{name: "bypass-without-held", bypass: "CUDATEST_NO_LOCK", value: "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values := map[string]string{
				"GPU_LOCK_HELD": tc.held,
				"GPU_LOCKDIR":   filepath.Join(t.TempDir(), "unknown-lock"),
			}
			if tc.bypass != "" {
				values[tc.bypass] = tc.value
			}
			enabled, err := compiledResourceAdmission(func(key string) string { return values[key] })
			if enabled || !errors.Is(err, cudalock.ErrNotInherited) {
				t.Fatalf("admission = %v, %v; want false,ErrNotInherited", enabled, err)
			}
		})
	}
}

func TestCompiledResourceAdmissionPrecedesPacket(t *testing.T) {
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "compiled_resource_native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var entry *ast.FuncDecl
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "runCompiledMLP" {
			entry = fn
		}
	}
	if entry == nil || entry.Body == nil {
		t.Fatal("compiled-resource fixture missing")
	}
	wantSet := token.NewFileSet()
	wantFile, err := parser.ParseFile(wantSet, "", `package cudainspect
func f(t *testing.T) {
	t.Helper()
	enabled, err := compiledResourceAdmission(os.Getenv)
	if err != nil {
		t.Fatal(err)
	}
	if !enabled {
		t.Skip("SKIPPED, NOT VERIFIED: compiled resource native tests require gpu-lock")
	}
}`, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := wantFile.Decls[0].(*ast.FuncDecl).Body.List
	if len(entry.Body.List) < len(want) {
		t.Fatal("compiled-resource admission prefix missing")
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
			t.Fatalf("admission statement %d: got %s; want %s", i, gotText.String(), wantText.String())
		}
	}
}

// TestCompiledResourceAdmissionInheritedParent uses an actual task-private
// filesystem namespace. It is not an installed GPU mutex or native receipt.
func TestCompiledResourceAdmissionInheritedParent(t *testing.T) {
	if os.Getenv("COMPILED_RESOURCE_HELPER_CHILD") == "1" {
		enabled, err := compiledResourceAdmission(os.Getenv)
		if !enabled || err != nil {
			t.Fatalf("private inherited owner = %v, %v; want true,nil", enabled, err)
		}
		return
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	h, err := cudalock.Acquire(ctx, cudalock.WithDir(filepath.Join(t.TempDir(), "private-lock")))
	if err != nil {
		t.Fatal(err)
	}
	released := false
	defer func() {
		if !released {
			if err := h.Release(); err != nil {
				t.Error(err)
			}
		}
	}()
	if h.Owner().PID != os.Getpid() {
		t.Fatalf("private owner pid = %d; want current parent %d", h.Owner().PID, os.Getpid())
	}
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCompiledResourceAdmissionInheritedParent$", "-test.v")
	cmd.WaitDelay = 2 * time.Second
	cmd.Env = append(os.Environ(),
		"COMPILED_RESOURCE_HELPER_CHILD=1",
		"GPU_LOCK_HELD="+strconv.Itoa(h.Owner().PID),
		"GPU_LOCKDIR="+h.Dir(),
		"GPU_LOCK_DISABLED=",
		"CUDATEST_NO_LOCK=",
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("helper-only inherited-owner child: %v\n%s", err, out)
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	released = true
	values := map[string]string{"GPU_LOCK_HELD": strconv.Itoa(h.Owner().PID), "GPU_LOCKDIR": h.Dir()}
	if enabled, err := compiledResourceAdmission(func(key string) string { return values[key] }); enabled || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("released private holder = %v, %v; want false,ErrNotInherited", enabled, err)
	}
}
