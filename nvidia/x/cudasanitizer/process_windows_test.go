package cudasanitizer

import (
	"errors"
	"os"
	"os/exec"
	"strconv"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

func TestWindowsDescendantCleanup(t *testing.T) {
	for _, tc := range []struct {
		name        string
		parentSleep string
		timeout     time.Duration
		noPipes     bool
	}{
		{"parent running", "5m", time.Second, false},
		{"parent exited", "", 0, false},
		{"parent exited before timeout", "", 2 * time.Second, false},
		{"parent exited without inherited pipes", "", 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseConfig(t)
			cfg.Timeout = tc.timeout
			cfg.KillDelay = 500 * time.Millisecond
			if tc.parentSleep == "" && tc.timeout > 0 {
				cfg.KillDelay = 4 * time.Second
			}
			file, err := os.CreateTemp("", "sanitizer-child-*.pid")
			if err != nil {
				t.Fatal(err)
			}
			pidFile := file.Name()
			if err := file.Close(); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := os.Remove(pidFile); err != nil {
					t.Error(err)
				}
			})
			var child windows.Handle
			defer func() {
				if child != 0 {
					_ = windows.TerminateProcess(child, 1)
					windows.CloseHandle(child)
				}
			}()
			noPipes := ""
			if tc.noPipes {
				noPipes = "1"
			}
			res := run(t, cfg, fixtureEnv(fakeReport+"="+cleanReport, fakeSpawn+"=1", fakeChildPID+"="+pidFile, fakeChildNoPipes+"="+noPipes, fakeSleep+"="+tc.parentSleep))
			data, err := os.ReadFile(pidFile)
			if err != nil {
				t.Fatalf("fixture did not record a child: %v; run=%s", err, res.Details())
			}
			pid, err := strconv.ParseUint(string(data), 10, 32)
			if err != nil || pid == 0 {
				t.Fatalf("invalid child PID %q: %v", data, err)
			}
			child, err = windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_TERMINATE, false, uint32(pid))
			if err == windows.ERROR_INVALID_PARAMETER {
				// The process has terminated and its object has already been removed.
				child = 0
			} else if err != nil {
				t.Fatal(err)
			} else {
				state, err := windows.WaitForSingleObject(child, 1000)
				if err != nil || state != windows.WAIT_OBJECT_0 {
					t.Fatalf("child %d still active: wait=%d err=%v; run=%s", pid, state, err, res.Details())
				}
			}
			if tc.parentSleep == "" && res.ExitCode != 0 {
				t.Errorf("parent exit status=%d, want completed exit 0", res.ExitCode)
			}
			if tc.timeout > 0 && !errors.Is(res.Err, ErrTimeout) {
				t.Fatalf("timeout classified as %v: %v", res.Outcome, res.Err)
			}
			t.Logf("child %d terminated; outcome=%v", pid, res.Outcome)
		})
	}
}

func TestWindowsSetupFailureDoesNotRunProcess(t *testing.T) {
	file, err := os.CreateTemp("", "sanitizer-launch-*.marker")
	if err != nil {
		t.Fatal(err)
	}
	marker := file.Name()
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(marker); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(marker)
	cmd := exec.Command(fixtureTool(t))
	cmd.Env = fixtureEnv(fakeArgvFile + "=" + marker)
	fault := errors.New("injected resume failure")
	group, err := startGroupWithResume(cmd, func(pid uint32) error {
		if pid == 0 {
			t.Error("no suspended process")
		}
		return fault
	})
	if group != nil || !errors.Is(err, fault) {
		t.Fatalf("setup returned group=%v, err=%v", group, err)
	}
	if cmd.ProcessState == nil || cmd.ProcessState.ExitCode() == 0 {
		t.Fatalf("suspended process was not terminated: %v", cmd.ProcessState)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("process ran before setup completed: marker error=%v", err)
	}
	t.Log("injected setup failure terminated the suspended process without executing its program")
}
