package cudatma

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

// tmaNativeAdmission checks inherited custody before the fixture can emit,
// compile, execute, or discover tools. It does not acquire a lock, infer UUID
// or device identity, or establish native qualification.
func tmaNativeAdmission(env func(string) string) (bool, error) {
	if env == nil {
		return false, fmt.Errorf("%w: no environment", cudalock.ErrNotInherited)
	}
	if env("CUDA_TMA_NATIVE") != "1" {
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
	for _, value := range []string{"", "0", "true"} {
		calls := 0
		enabled, err := tmaNativeAdmission(func(key string) string {
			calls++
			if key != "CUDA_TMA_NATIVE" {
				t.Fatalf("disabled fixture inspected %s", key)
			}
			return value
		})
		if enabled || err != nil || calls != 1 {
			t.Fatalf("admission(%q) = %v, %v, calls %d", value, enabled, err, calls)
		}
	}
}

func TestNativeAdmissionRefusesUnverifiedCustody(t *testing.T) {
	root := t.TempDir()
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	shortHost := strings.SplitN(host, ".", 2)[0]

	liveDir := filepath.Join(root, "live")
	if err := os.Mkdir(liveDir, 0700); err != nil {
		t.Fatal(err)
	}
	liveOwner := cudalock.Owner{
		PID:   os.Getppid(),
		Host:  shortHost,
		Label: "live-parent",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}
	if err := os.WriteFile(filepath.Join(liveDir, "owner"), []byte(liveOwner.String()), 0600); err != nil {
		t.Fatal(err)
	}

	forgedDir := filepath.Join(root, "forged")
	if err := os.Mkdir(forgedDir, 0700); err != nil {
		t.Fatal(err)
	}
	forgedOwner := cudalock.Owner{
		PID:   os.Getpid(),
		Host:  "invalid-native-fixture-host",
		Label: "not-a-holder",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}
	if err := os.WriteFile(filepath.Join(forgedDir, "owner"), []byte(forgedOwner.String()), 0600); err != nil {
		t.Fatal(err)
	}

	deadCmd := exec.Command(os.Args[0], "-test.run=^$")
	if err := deadCmd.Run(); err != nil {
		t.Fatal(err)
	}
	deadPID := deadCmd.Process.Pid
	deadDir := filepath.Join(root, "dead")
	if err := os.Mkdir(deadDir, 0700); err != nil {
		t.Fatal(err)
	}
	deadOwner := cudalock.Owner{
		PID:   deadPID,
		Host:  shortHost,
		Label: "dead-holder",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}
	if err := os.WriteFile(filepath.Join(deadDir, "owner"), []byte(deadOwner.String()), 0600); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	releasedHandle, err := cudalock.Acquire(ctx, cudalock.WithDir(filepath.Join(root, "released")))
	if err != nil {
		t.Fatal(err)
	}
	releasedPID := strconv.Itoa(releasedHandle.Owner().PID)
	releasedDir := releasedHandle.Dir()
	if err := releasedHandle.Release(); err != nil {
		t.Fatal(err)
	}

	selfPID := strconv.Itoa(os.Getpid())
	parentPID := strconv.Itoa(os.Getppid())

	for _, tc := range []struct {
		name      string
		held, dir string
		bypassKey string
		bypassVal string
	}{
		{"missing", "", liveDir, "", ""},
		{"malformed-word", "not-a-pid", liveDir, "", ""},
		{"malformed-neg", "-1", liveDir, "", ""},
		{"malformed-zero", "0", liveDir, "", ""},
		{"hand-set-one", "1", liveDir, "", ""},
		{"absent-lockdir", parentPID, filepath.Join(root, "absent"), "", ""},
		{"forged-foreign-host", selfPID, forgedDir, "", ""},
		{"mismatched-pid", selfPID, liveDir, "", ""},
		{"stale-dead-holder", strconv.Itoa(deadPID), deadDir, "", ""},
		{"released-holder", releasedPID, releasedDir, "", ""},
		{"bypass-gpu-lock-disabled-1", parentPID, liveDir, "GPU_LOCK_DISABLED", "1"},
		{"bypass-gpu-lock-disabled-0", parentPID, liveDir, "GPU_LOCK_DISABLED", "0"},
		{"bypass-cudatest-no-lock-1", parentPID, liveDir, "CUDATEST_NO_LOCK", "1"},
		{"bypass-cudatest-no-lock-0", parentPID, liveDir, "CUDATEST_NO_LOCK", "0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values := map[string]string{
				"CUDA_TMA_NATIVE": "1",
				"GPU_LOCK_HELD":   tc.held,
				"GPU_LOCKDIR":     tc.dir,
			}
			if tc.bypassKey != "" {
				values[tc.bypassKey] = tc.bypassVal
			}
			enabled, err := tmaNativeAdmission(func(key string) string { return values[key] })
			if enabled || !errors.Is(err, cudalock.ErrNotInherited) {
				t.Fatalf("admission = %v, %v; want false, ErrNotInherited", enabled, err)
			}
		})
	}
	if enabled, err := tmaNativeAdmission(nil); enabled || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("nil environment = %v, %v; want false, ErrNotInherited", enabled, err)
	}
}

func TestNativeAdmissionCanonicalParent(t *testing.T) {
	// This checks a task-local owner record for the live parent process in
	// t.TempDir(); it does not acquire the host GPU lock or establish native qualification.
	dir := filepath.Join(t.TempDir(), "lock")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	owner := cudalock.Owner{
		PID:   os.Getppid(),
		Host:  strings.SplitN(host, ".", 2)[0],
		Label: "synthetic-parent",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}
	ownerFile := filepath.Join(dir, "owner")
	if err := os.WriteFile(ownerFile, []byte(owner.String()), 0600); err != nil {
		t.Fatal(err)
	}
	values := map[string]string{
		"CUDA_TMA_NATIVE": "1",
		"GPU_LOCK_HELD":   strconv.Itoa(owner.PID),
		"GPU_LOCKDIR":     dir,
	}
	env := func(key string) string { return values[key] }
	if enabled, err := tmaNativeAdmission(env); !enabled || err != nil {
		t.Fatalf("canonical parent = %v, %v; want true, nil", enabled, err)
	}
	values["GPU_LOCK_HELD"] = strconv.Itoa(os.Getpid())
	if enabled, err := tmaNativeAdmission(env); enabled || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("mismatched owner = %v, %v; want false, ErrNotInherited", enabled, err)
	}
	values["GPU_LOCK_HELD"] = strconv.Itoa(owner.PID)
	if err := os.Remove(ownerFile); err != nil {
		t.Fatal(err)
	}
	if enabled, err := tmaNativeAdmission(env); enabled || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("removed owner = %v, %v; want false, ErrNotInherited", enabled, err)
	}
}

// TestNativeAdmissionPrecedesFixtureAndTools parses native_test.go without
// invoking its body, verifying that TestNativeTMAReference checks
// tmaNativeAdmission(os.Getenv) before tmaTarget(), FixtureSource, or cudatest.NVCC.
func TestNativeAdmissionPrecedesFixtureAndTools(t *testing.T) {
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var entry *ast.FuncDecl
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "TestNativeTMAReference" {
			entry = fn
		}
	}
	if entry == nil || entry.Body == nil {
		t.Fatal("native fixture missing")
	}
	wantSet := token.NewFileSet()
	wantFile, err := parser.ParseFile(wantSet, "", `package cudatma
func f(t *testing.T) {
	enabled, err := tmaNativeAdmission(os.Getenv)
	if err != nil {
		t.Fatal(err)
	}
	if !enabled {
		t.Skip("native qualification pending: CUDA_TMA_NATIVE=1 required")
	}
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
			t.Fatalf("statement %d before fixture/tools:\ngot: %s\nwant: %s", i, gotText.String(), wantText.String())
		}
	}
}

// TestNativeTMAReferenceRefusesBogusAndStaleCustodyBeforeFixture verifies that
// the child TestNativeTMAReference refuses hand-set GPU_LOCK_HELD=1 and a
// released/stale lock directory before reaching FixtureSource or any compiler
// or device path.
func TestNativeTMAReferenceRefusesBogusAndStaleCustodyBeforeFixture(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	h, err := cudalock.Acquire(ctx, cudalock.WithDir(filepath.Join(t.TempDir(), "lock")))
	if err != nil {
		t.Fatal(err)
	}
	stalePID := strconv.Itoa(h.Owner().PID)
	staleDir := h.Dir()
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		held string
		dir  string
	}{
		{"hand-set-one", "1", staleDir},
		{"released-stale-lock", stalePID, staleDir},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestNativeTMAReference$", "-test.v")
			cmd.Env = append(os.Environ(),
				"CUDA_TMA_NATIVE=1",
				"CUDA_TMA_NATIVE_ARCH=invalid_fixture_target",
				"GPU_LOCK_HELD="+tc.held,
				"GPU_LOCKDIR="+tc.dir,
				"GPU_LOCK_DISABLED=",
				"CUDATEST_NO_LOCK=",
			)
			out, err := cmd.CombinedOutput()
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
				t.Fatalf("refused custody exit = %v, want 1\n%s", err, out)
			}
			text := string(out)
			if !strings.Contains(text, cudalock.ErrNotInherited.Error()) {
				t.Fatalf("missing canonical custody refusal %q:\n%s", cudalock.ErrNotInherited, text)
			}
			if strings.Contains(text, "emit fixture for invalid_fixture_target") {
				t.Fatalf("fixture emission ran despite unverified custody:\n%s", text)
			}
		})
	}
}
