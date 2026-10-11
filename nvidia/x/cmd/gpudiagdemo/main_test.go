package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiagdemo"
)

func TestCMD_Usage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	err := run([]string{"--help"}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("expected no error for --help, got: %v", err)
	}
	if !strings.Contains(stdout.String(), "gpudiagdemo:") {
		t.Fatalf("expected usage message, got:\n%s", stdout.String())
	}
}

func TestCMD_NoArgs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	err := run([]string{}, &stdout, &stderr)
	if err == nil {
		t.Fatal("expected error with no arguments")
	}
}

func TestCMD_RunAndVerify(t *testing.T) {
	tempDir := t.TempDir()
	outDir := filepath.Join(tempDir, "artifacts")

	var stdout, stderr bytes.Buffer
	args := []string{
		"run",
		"-n", "6",
		"-c", "2",
		"-host-delay", "5ms",
		"-device-delay", "100us",
		"-out-dir", outDir,
	}

	if err := run(args, &stdout, &stderr); err != nil {
		t.Fatalf("cmd run failed: %v\nstderr: %s", err, stderr.String())
	}

	outStr := stdout.String()
	if !strings.Contains(outStr, "=== Demonstration Summary ===") {
		t.Fatalf("expected summary header, got:\n%s", outStr)
	}
	if !strings.Contains(outStr, "host_submission_delay") {
		t.Fatalf("expected host_submission_delay in output, got:\n%s", outStr)
	}

	// Verify the generated artifacts using the verify subcommand
	stdout.Reset()
	stderr.Reset()
	verifyArgs := []string{"verify", "-dir", outDir}
	if err := run(verifyArgs, &stdout, &stderr); err != nil {
		t.Fatalf("cmd verify failed: %v\nstderr: %s", err, stderr.String())
	}

	if !strings.Contains(stdout.String(), "verified successfully") {
		t.Fatalf("expected verification success, got: %s", stdout.String())
	}
}

func TestCMD_JSONOutput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{
		"run",
		"-n", "4",
		"-c", "2",
		"-host-delay", "5ms",
		"-device-delay", "100us",
		"-json",
	}

	if err := run(args, &stdout, &stderr); err != nil {
		t.Fatalf("cmd run -json failed: %v\nstderr: %s", err, stderr.String())
	}

	var res cudadiagdemo.DemonstrationResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to parse json output: %v\noutput: %s", err, stdout.String())
	}

	if res.HostDelayDiagnosis.Mechanism != "host_submission_delay" {
		t.Fatalf("expected mechanism host_submission_delay, got %q", res.HostDelayDiagnosis.Mechanism)
	}
}

func TestCMD_VerifyMissingDir(t *testing.T) {
	var stdout, stderr bytes.Buffer
	nonExistentDir := filepath.Join(os.TempDir(), "non-existent-dir-gpudiagdemo")
	err := run([]string{"verify", "-dir", nonExistentDir}, &stdout, &stderr)
	if err == nil {
		t.Fatal("expected error when verifying non-existent directory")
	}
}

func TestNativeInvalidBounds(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if err := run([]string{"run", "-native", "-elements", "0"}, &stdout, &stderr); err == nil {
		t.Fatal("native invalid bounds accepted")
	}
	if stdout.Len() != 0 {
		t.Fatal("invalid trial produced qualification receipt")
	}
}
