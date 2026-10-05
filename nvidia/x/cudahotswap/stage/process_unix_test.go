//go:build linux || darwin

package stage

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

func TestExitedCompilerDoesNotRetainDescendant(t *testing.T) {
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "child.pid")
	compiler := filepath.Join(dir, "compiler")
	body := "#!/bin/sh\nsleep 30 &\necho $! > '" + pidFile + "'\nexit 0\n"
	if err := os.WriteFile(compiler, []byte(body), 0700); err != nil {
		t.Fatal(err)
	}
	builder, err := NewBuilder(filepath.Join(dir, "build"), compiler)
	if err != nil {
		t.Fatal(err)
	}
	_, buildErr := builder.Build(context.Background(), Source{Code: hostStage})
	data, err := os.ReadFile(pidFile)
	if err != nil {
		t.Fatal(err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || pid <= 0 {
		t.Fatalf("child pid %q: %v", data, err)
	}
	t.Cleanup(func() { _ = syscall.Kill(pid, syscall.SIGKILL) })
	if !errors.Is(buildErr, exec.ErrWaitDelay) {
		t.Fatalf("Build = %v; want WaitDelay", buildErr)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		state, psErr := exec.CommandContext(ctx, "ps", "-o", "stat=", "-p", strconv.Itoa(pid)).Output()
		cancel()
		if psErr != nil {
			var exit *exec.ExitError
			if errors.As(psErr, &exit) && exit.ExitCode() == 1 && len(state) == 0 && len(exit.Stderr) == 0 {
				return
			}
			t.Fatalf("inspect compiler descendant: %v", psErr)
		}
		if strings.HasPrefix(strings.TrimSpace(string(state)), "Z") {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("compiler descendant %d remains live: %q", pid, state)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestCompilerOutputRefusesNonRegularOrOversized(t *testing.T) {
	dir := t.TempDir()
	for _, kind := range []string{"fifo", "symlink", "oversized"} {
		t.Run(kind, func(t *testing.T) {
			path := filepath.Join(dir, kind)
			switch kind {
			case "fifo":
				if err := syscall.Mkfifo(path, 0600); err != nil {
					t.Fatal(err)
				}
			case "symlink":
				target := filepath.Join(dir, "target")
				if err := os.WriteFile(target, []byte("image"), 0600); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(target, path); err != nil {
					t.Fatal(err)
				}
			case "oversized":
				f, err := os.Create(path)
				if err != nil {
					t.Fatal(err)
				}
				err = f.Truncate(maxCompilerImage + 1)
				closeErr := f.Close()
				if err != nil || closeErr != nil {
					t.Fatal(errors.Join(err, closeErr))
				}
			}
			done := make(chan error, 1)
			go func() { _, err := readCompilerImage(path); done <- err }()
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("compiler output was admitted")
				}
			case <-time.After(400 * time.Millisecond):
				// Retire only this test's FIFO reader before reporting a valid mutant.
				if kind == "fifo" {
					fd, err := syscall.Open(path, syscall.O_RDWR|syscall.O_NONBLOCK, 0)
					if err == nil {
						_, _ = syscall.Write(fd, []byte("image"))
						_ = syscall.Close(fd)
					}
				}
				select {
				case <-done:
				case <-time.After(time.Second):
					t.Fatal("compiler reader did not retire")
				}
				t.Fatal("compiler output read blocked")
			}
		})
	}
}
