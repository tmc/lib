//go:build darwin || linux

package hosttest

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

func compileFixture(t *testing.T, mutant bool) string {
	t.Helper()
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Fatal("host resolver check requires a C compiler:", err)
	}
	version, err := exec.Command(cc, "--version").CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("compiler=%s version=%s", cc, version)
	source, err := os.ReadFile("../interpose.c")
	if err != nil {
		t.Fatal(err)
	}
	if mutant {
		guard := "    if (resolving) return 3; /* CUDA_ERROR_NOT_INITIALIZED: fail closed. */\n"
		if strings.Count(string(source), guard) != 1 {
			t.Fatal("expected one entry guard")
		}
		source = []byte(strings.Replace(string(source), guard, "", 1))
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "interpose-source.c"), source, 0600); err != nil {
		t.Fatal(err)
	}
	fixture, err := os.ReadFile("testdata/resolver.c")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "resolver.c")
	if err := os.WriteFile(path, fixture, 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "resolver")
	args := []string{"-std=c11", "-pthread", path, "-o", binary}
	t.Logf("compile: %s %v", cc, args)
	if out, err := exec.Command(cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	return binary
}

func runFixture(t *testing.T, binary, mode string) (bool, error, string, time.Duration) {
	t.Helper()
	cmd := exec.Command(binary, mode)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	start := time.Now()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()
	select {
	case err := <-done:
		return false, err, out.String(), time.Since(start)
	case <-timer.C:
		if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
			t.Fatal("kill fixture group:", err)
		}
		err := <-done
		return true, err, out.String(), time.Since(start)
	}
}

func TestResolverHostContract(t *testing.T) {
	binary := compileFixture(t, false)
	for _, tt := range []struct{ name, mode, want string }{
		{"pass_through", "0", "result=77 recursive=-1 resolutions=1 launches=1"},
		{"missing_symbol", "1", "result=3 recursive=-1 resolutions=1 launches=0"},
		{"reentrant_lookup", "2", "result=77 recursive=3 resolutions=1 launches=1"},
		{"concurrent_first_use", "3", "result=77 recursive=-1 resolutions=1 launches=32"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			timedOut, err, out, elapsed := runFixture(t, binary, tt.mode)
			t.Logf("elapsed=%s timeout=%v result=%v output=%s", elapsed, timedOut, err, out)
			if timedOut || err != nil || strings.TrimSpace(out) != tt.want {
				t.Fatal("resolver contract failed")
			}
		})
	}
}

func TestResolverEntryGuardOmission(t *testing.T) {
	binary := compileFixture(t, true)
	timedOut, err, out, elapsed := runFixture(t, binary, "2")
	t.Logf("platform=%s elapsed=%s timeout=%v result=%v output=%s", runtime.GOOS, elapsed, timedOut, err, out)
	if runtime.GOOS == "darwin" {
		// Darwin terminates recursive pthread_once before the budget expires.
		// That failure is neither a deadline timeout nor proof of deadlock.
		exit, ok := err.(*exec.ExitError)
		if !ok || timedOut || exit.ProcessState.Sys().(syscall.WaitStatus).Signal() != syscall.SIGKILL {
			t.Fatal("omission did not reproduce the observed Darwin signal failure")
		}
	} else if !timedOut || err == nil {
		t.Fatal("omission did not reach the bounded Linux reentry deadline")
	}
}
