package main

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func TestCommandReleasesLock(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("requires POSIX shell")
	}
	binary := filepath.Join(t.TempDir(), "cudalock")
	cmd := exec.Command("go", "build", "-o", binary, ".")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build cudalock: %v\n%s", err, out)
	}
	for _, code := range []int{0, 7} {
		t.Run(strconv.Itoa(code), func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "lock")
			// The child must observe the held lock before exiting. Testing only
			// its absence afterward could pass if acquisition were removed.
			cmd := exec.Command(binary, "-dir", dir, "--", "sh", "-c",
				`test -f "$1/owner" || exit 99; exit "$2"`, "sh", dir, strconv.Itoa(code))
			for _, env := range os.Environ() {
				if !strings.HasPrefix(env, "GPU_LOCK_") && !strings.HasPrefix(env, "GPU_LOCKDIR=") {
					cmd.Env = append(cmd.Env, env)
				}
			}
			out, err := cmd.CombinedOutput()
			got := 0
			if err != nil {
				var exit *exec.ExitError
				if !errors.As(err, &exit) {
					t.Fatalf("run cudalock: %v\n%s", err, out)
				}
				got = exit.ExitCode()
			}
			if got != code {
				t.Fatalf("exit status = %d, want %d\n%s", got, code, out)
			}
			if _, err := os.Stat(dir); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("lock remains after exit: %v\n%s", err, out)
			}
		})
	}
}
