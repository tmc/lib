//go:build unix

package cudasanitizer

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestExitedParentKillsDescendant(t *testing.T) {
	for _, tc := range []struct {
		name    string
		exit    int
		timeout time.Duration
		noPipes bool
	}{
		{"clean parent", 0, 0, false},
		{"failed parent", 3, 0, false},
		{"context expires after parent", 0, 100 * time.Millisecond, false},
		{"clean parent without inherited pipes", 0, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseConfig(t)
			cfg.Timeout = tc.timeout
			cfg.KillDelay = 400 * time.Millisecond
			pidFile := filepath.Join(t.TempDir(), "child.pid")
			// Always clean up this test's child, including on the unfixed runner.
			var pid int
			defer func() {
				if pid > 1 {
					_ = syscall.Kill(pid, syscall.SIGKILL)
				}
			}()
			noPipes := ""
			if tc.noPipes {
				noPipes = "1"
			}
			res := run(t, cfg, fixtureEnv(fakeChildNoPipes+"="+noPipes, fakeReport+"="+cleanReport, fakeSpawn+"=1", fakeChildPID+"="+pidFile, fakeExit+"="+strconv.Itoa(tc.exit)))
			data, err := os.ReadFile(pidFile)
			if err != nil {
				t.Fatal(err)
			}
			pid, err = strconv.Atoi(string(data))
			if err != nil || pid <= 1 {
				t.Fatalf("invalid fixture PID %q: %v", data, err)
			}
			if res.ExitCode != tc.exit {
				t.Errorf("recorded parent exit = %d, want %d", res.ExitCode, tc.exit)
			}
			if tc.timeout > 0 && !errors.Is(res.Err, ErrTimeout) {
				t.Errorf("timeout error = %v", res.Err)
			}
			deadline := time.Now().Add(time.Second)
			for {
				err := syscall.Kill(pid, 0)
				if errors.Is(err, syscall.ESRCH) {
					pid = 0
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				// An orphan can remain a zombie until init reaps it; it is terminated.
				out, err := exec.Command("ps", "-o", "stat=", "-p", strconv.Itoa(pid)).Output()
				if err == nil && strings.HasPrefix(strings.TrimSpace(string(out)), "Z") {
					pid = 0
					break
				}
				if time.Now().After(deadline) {
					t.Fatalf("child %d still running after parent exit; outcome=%v error=%v", pid, res.Outcome, res.Err)
				}
				time.Sleep(10 * time.Millisecond)
			}
		})
	}
}

func TestToolVersionKillsDescendant(t *testing.T) {
	tool := fixtureTool(t)
	for _, timeout := range []bool{false, true} {
		t.Run(strconv.FormatBool(timeout), func(t *testing.T) {
			pidFile := filepath.Join(t.TempDir(), "pid")
			t.Setenv(fakeSpawn, "1")
			t.Setenv(fakeChildPID, pidFile)
			t.Setenv(fakeChildNoPipes, "1")
			t.Setenv(fakeStdout, "fixture version")
			if timeout {
				t.Setenv(fakeSleep, "5s")
			}
			var pid int
			defer func() {
				if pid > 1 {
					_ = syscall.Kill(pid, syscall.SIGKILL)
				}
			}()
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			version, err := ToolVersion(ctx, tool)
			data, readErr := os.ReadFile(pidFile)
			if readErr != nil {
				t.Fatal(readErr)
			}
			pid, readErr = strconv.Atoi(string(data))
			if readErr != nil || pid <= 1 {
				t.Fatalf("invalid child PID %q", data)
			}
			if timeout {
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Errorf("timeout error = %v", err)
				}
			} else if err != nil || version != "fixture version" {
				t.Errorf("version = %q, %v", version, err)
			}
			deadline := time.Now().Add(time.Second)
			for {
				err := syscall.Kill(pid, 0)
				if errors.Is(err, syscall.ESRCH) {
					pid = 0
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				out, err := exec.Command("ps", "-o", "stat=", "-p", strconv.Itoa(pid)).Output()
				if err == nil && strings.HasPrefix(strings.TrimSpace(string(out)), "Z") {
					pid = 0
					break
				}
				if time.Now().After(deadline) {
					t.Fatalf("version query left child %d running", pid)
				}
				time.Sleep(10 * time.Millisecond)
			}
		})
	}
}
