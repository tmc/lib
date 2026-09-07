//go:build linux

package nccl

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestLoadOrderNative(t *testing.T) {
	if path := os.Getenv("NCCL_ORDER_STUB"); path != "" {
		t.Setenv("CODEGEN_SKIP_BINDINGS", "")
		t.Setenv("NCCL_LIB_PATH", path)
		for _, adopt := range []string{"", path} {
			if err := Close(); err != nil {
				t.Fatal(err)
			}
			if err := checkLib(); err != nil {
				t.Fatal(err)
			}
			conflict := func(want bool) {
				t.Helper()
				var e *LibraryConflictError
				if err := Load(path + ".other"); !errors.As(err, &e) {
					t.Fatalf("conflict=%v", err)
				}
				if e.Implicit != want {
					t.Fatalf("Implicit=%v, want %v", e.Implicit, want)
				}
			}
			conflict(true)
			conflict(true)
			if err := Load(adopt); err != nil {
				t.Fatal(err)
			}
			conflict(false)
		}
		if err := Close(); err != nil {
			t.Fatal(err)
		}
		if err := Load(path); err != nil {
			t.Fatal(err)
		}
		if !explicitLoad {
			t.Fatal("fresh explicit load not recorded")
		}
		if err := Close(); err != nil {
			t.Fatal(err)
		}
		t.Log("lazy conflicts, empty/same-path adoption, rejected calls, close reset, fresh explicit load passed")
		return
	}
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("requires dlopen host")
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("requires C compiler")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(home, "tmp")
	if err := os.MkdirAll(base, 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(base, "nccl-load-order-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	src := filepath.Join(dir, "stub.c")
	if err := os.WriteFile(src, []byte("int nccl_test_stub(void) { return 42; }\n"), 0600); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "stub.so")
	flag := "-shared"
	if runtime.GOOS == "darwin" {
		flag = "-dynamiclib"
	}
	if b, err := exec.Command(cc, flag, "-fPIC", src, "-o", path).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, b)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe, "-test.run=^TestLoadOrderNative$", "-test.v")
	cmd.Env = append(os.Environ(), "NCCL_ORDER_STUB="+path, "NCCL_LIB_PATH="+path, "CODEGEN_SKIP_BINDINGS=")
	b, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("loader ordering: %v\n%s", err, b)
	}
	t.Log(string(b))
}
