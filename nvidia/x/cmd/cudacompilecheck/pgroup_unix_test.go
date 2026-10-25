//go:build unix

package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// TestRunToolKillsGroup checks that a timeout kills the tool's children, not
// just the tool: the grandchild here holds the output pipe and would
// otherwise outlive the kill.
func TestRunToolKillsGroup(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pid")
	start := time.Now()
	_, err := runTool(context.Background(), 300*time.Millisecond,
		[]string{"sh", "-c", "sleep 30 & echo $! > " + pidFile + "; wait"})
	if err == nil || !strings.Contains(err.Error(), "timed out after 300ms (killed its process group)") {
		t.Fatalf("err = %v", err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("returned after %v", d)
	}
	assertDead(t, pidFile)
}

// TestRunToolGroupFallback forces setpgid to fail, as it does where the
// caller may not create process groups, and checks that the tool still
// runs, that the fallback is announced once, and that a timeout says the
// kill did not reach the children.
func TestRunToolGroupFallback(t *testing.T) {
	var warn bytes.Buffer
	defer func(g int, w io.Writer) { processGroup, groupWarning, groupWarningOnce = g, w, sync.Once{} }(processGroup, groupWarning)
	// Group 1 belongs to init's session; joining it is EPERM.
	processGroup, groupWarning, groupWarningOnce = 1, &warn, sync.Once{}

	for range 2 {
		out, err := runTool(context.Background(), 10*time.Second, []string{"sh", "-c", "echo ok"})
		if err != nil || string(out) != "ok\n" {
			t.Fatalf("runTool = %q, %v", out, err)
		}
	}
	if got := strings.Count(warn.String(), "cannot create a process group"); got != 1 {
		t.Errorf("warned %d times:\n%s", got, warn.String())
	}
	if !strings.Contains(warn.String(), "a timeout kills only sh, not its children") {
		t.Errorf("warning = %q", warn.String())
	}

	pidFile := filepath.Join(t.TempDir(), "pid")
	_, err := runTool(context.Background(), 300*time.Millisecond,
		[]string{"sh", "-c", "sleep 30 & echo $! > " + pidFile + "; wait"})
	if err == nil || !strings.Contains(err.Error(), "killed sh only; its children may still be running") {
		t.Errorf("err = %v", err)
	}
	// The report is true: the grandchild did survive. Clean it up.
	if pid := readPID(t, pidFile); pid > 0 {
		if syscall.Kill(pid, 0) != nil {
			t.Errorf("grandchild %d exited; expected it to survive a kill that cannot reach it", pid)
		}
		syscall.Kill(pid, syscall.SIGKILL)
	}
}

func readPID(t *testing.T, path string) int {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		t.Fatal(err)
	}
	return pid
}

func assertDead(t *testing.T, pidFile string) {
	t.Helper()
	pid := readPID(t, pidFile)
	for i := 0; i < 50; i++ {
		if syscall.Kill(pid, 0) != nil {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	syscall.Kill(pid, syscall.SIGKILL)
	t.Errorf("grandchild %d survived the timeout", pid)
}
