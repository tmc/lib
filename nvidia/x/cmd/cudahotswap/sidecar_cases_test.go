package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func sidecarNativeCases(t *testing.T, run func(*testing.T, bool)) {
	for _, fused := range []bool{false, true} {
		if !t.Run(fmt.Sprintf("fused_%t", fused), func(t *testing.T) { run(t, fused) }) {
			return
		}
	}
}

func TestSidecarCasesFailureChild(t *testing.T) {
	dir := os.Getenv("SIDECAR_CPU_FAILURE_CHILD")
	if dir == "" {
		return
	}
	sidecarNativeCases(t, func(t *testing.T, fused bool) {
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprint(fused)), []byte("entered"), 0600); err != nil {
			t.Fatal(err)
		}
		if !fused {
			t.Fatal("owned CPU first-case failure")
		}
	})
}

func TestSidecarCasesStopAfterFailureCPU(t *testing.T) {
	dir := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestSidecarCasesFailureChild$")
	command.Env = append(os.Environ(), "SIDECAR_CPU_FAILURE_CHILD="+dir)
	output, err := command.CombinedOutput()
	var exit *exec.ExitError
	if err == nil {
		t.Fatalf("failing child passed: %s", output)
	}
	if !errors.As(err, &exit) || exit.ExitCode() != 1 {
		t.Fatalf("unexpected child exit: %v %s", err, output)
	}
	if _, err := os.Stat(filepath.Join(dir, "false")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "true")); !os.IsNotExist(err) {
		t.Fatalf("continued after failed case: %v", err)
	}
}
