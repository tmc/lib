package cusparse

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestErrorStringsWithoutLibrary(t *testing.T) {
	// Isolate the loader's cached failure from other tests in this package.
	if os.Getenv("CUSPARSE_ERROR_STRINGS_CHILD") != "1" {
		exe, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		// Let the test runner print goroutine stacks before the parent kills it.
		cmd := exec.CommandContext(ctx, exe, "-test.run=^TestErrorStringsWithoutLibrary$", "-test.v", "-test.timeout=9s")
		cmd.Env = append(os.Environ(), "CUSPARSE_ERROR_STRINGS_CHILD=1")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("missing-library subprocess: %v (context: %v)\n%s", err, ctx.Err(), out)
		}
		return
	}

	if err := Close(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := Close(); err != nil {
			t.Error(err)
		}
	})
	path := filepath.Join(t.TempDir(), "missing-cusparse")
	t.Setenv("CUSPARSE_LIB_PATH", path)
	// Load explicitly resets any failed load attempt from another test.
	if err := Load(path); err == nil {
		t.Fatal("loaded nonexistent library")
	}
	for _, tt := range []struct {
		name string
		call func(Result) (string, error)
	}{
		{"name", GetErrorName}, {"description", GetErrorString},
	} {
		t.Run(tt.name, func(t *testing.T) {
			text, err := tt.call(ErrInvalidValue)
			var loadError *LibraryError
			if text != "" || !errors.As(err, &loadError) {
				t.Fatalf("missing-library result = %q, %v", text, err)
			}
		})
	}
}

func TestNativeErrorStrings(t *testing.T) {
	if os.Getenv("CUSPARSE_PHYSICAL_TEST") != "1" {
		t.Skip("set CUSPARSE_PHYSICAL_TEST=1 for native validation")
	}
	if err := Close(); err != nil {
		t.Fatal(err)
	}
	if err := Load(""); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := Close(); err != nil {
			t.Error(err)
		}
	})
	for _, tt := range []struct {
		status Result
		name   string
	}{
		{Success, "CUSPARSE_STATUS_SUCCESS"}, {ErrInvalidValue, "CUSPARSE_STATUS_INVALID_VALUE"},
	} {
		name, err := GetErrorName(tt.status)
		if err != nil || name != tt.name {
			t.Fatalf("native name = %q, %v; want %q", name, err, tt.name)
		}
		description, err := GetErrorString(tt.status)
		if err != nil || description == "" {
			t.Fatalf("native description = %q, %v", description, err)
		}
		t.Logf("%s: %s", name, description)
	}
}
