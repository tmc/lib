package cudasass

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestInstalledCancellation observes exec of nvdisasm before canceling. A fixture
// that finishes before observation is inconclusive and fails this opt-in gate.
func TestInstalledCancellation(t *testing.T) {
	input := os.Getenv("CUDASASS_TEST_CUBIN")
	if input == "" {
		t.Skip("set CUDASASS_TEST_CUBIN to a real cubin")
	}
	data, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	tool := os.Getenv("NVDISASM")
	if tool == "" {
		tool = "nvdisasm"
	}
	tool, err = exec.LookPath(tool)
	if err != nil {
		t.Fatal(err)
	}
	tool, err = filepath.Abs(tool)
	if err != nil {
		t.Fatal(err)
	}
	real, err := filepath.EvalSymlinks(tool)
	if err != nil {
		t.Fatal(err)
	}
	versionContext, stopVersion := context.WithTimeout(context.Background(), 5*time.Second)
	version, err := (Disassembler{Path: real}).Version(versionContext)
	stopVersion()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("version %s: %s%s", version.Path, version.Stdout, version.Stderr)
	dir := t.TempDir()
	marker := filepath.Join(dir, "pid")
	launcher := filepath.Join(dir, "observe")
	t.Setenv("CUDASASS_OBSERVED_PID", marker)
	t.Setenv("CUDASASS_OBSERVED_TOOL", tool)
	if err := os.WriteFile(launcher, []byte("#!/bin/sh\nprintf '%s' \"$$\" > \"$CUDASASS_OBSERVED_PID\"\nexec \"$CUDASASS_OBSERVED_TOOL\" \"$@\"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	scratch := t.TempDir()
	type result struct {
		out Output
		err error
	}
	done := make(chan result, 1)
	go func() {
		out, err := (Disassembler{Path: launcher, TempDir: scratch}).Disassemble(ctx, data, Options{})
		done <- result{out, err}
	}()
	var pid int
	observed := false
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	var got result
wait:
	for {
		select {
		case got = <-done:
			break wait
		case <-ticker.C:
			if observed {
				continue
			}
			b, err := os.ReadFile(marker)
			if err != nil {
				continue
			}
			pid, err = strconv.Atoi(strings.TrimSpace(string(b)))
			if err != nil || pid <= 1 {
				continue
			}
			exe, err := os.Readlink("/proc/" + strconv.Itoa(pid) + "/exe")
			if err == nil && exe == real {
				observed = true
				cancel()
			}
		}
	}
	if !observed {
		t.Fatalf("real nvdisasm startup not observed: %v", got.err)
	}
	if !errors.Is(got.err, context.Canceled) {
		t.Fatalf("cancellation = %v, want context.Canceled", got.err)
	}
	if _, err := os.Stat("/proc/" + strconv.Itoa(pid)); !os.IsNotExist(err) {
		t.Fatalf("tool PID %d still exists after return: %v", pid, err)
	}
	entries, err := os.ReadDir(scratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("scratch contains %d leaked entries", len(entries))
	}
	t.Logf("observed %s PID %d before cancellation; process reaped and scratch removed; args %q stderr %s", real, pid, got.out.Args, got.out.Stderr)
}
