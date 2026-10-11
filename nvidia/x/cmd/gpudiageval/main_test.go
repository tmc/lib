package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCMD_GPUDiagEval_RunVerifyClaim(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "gpudiageval-test-*")
	if err != nil {
		t.Fatalf("MkdirTemp: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// 1. Test "run" command
	var outBuf bytes.Buffer
	var errBuf bytes.Buffer
	args := []string{"run", "-phase", "pilot", "-out-dir", tempDir}
	if err := run(args, &outBuf, &errBuf); err != nil {
		t.Fatalf("run failed: %v, stderr: %s", err, errBuf.String())
	}

	outStr := outBuf.String()
	if !strings.Contains(outStr, "CUDA SOTA EVALUATION REPORT") {
		t.Errorf("expected summary header, got:\n%s", outStr)
	}
	if !strings.Contains(outStr, "zero_incorrect_exact_joins") {
		t.Errorf("expected zero_incorrect_exact_joins in summary, got:\n%s", outStr)
	}

	// 2. Test "verify" command
	outBuf.Reset()
	errBuf.Reset()
	verifyArgs := []string{"verify", "-dir", tempDir}
	if err := run(verifyArgs, &outBuf, &errBuf); err != nil {
		t.Fatalf("verify failed: %v, stderr: %s", err, errBuf.String())
	}
	verifyStr := outBuf.String()
	if !strings.Contains(verifyStr, "Artifact verification PASSED") {
		t.Errorf("expected verification pass, got:\n%s", verifyStr)
	}

	// 3. Test "claim" command
	outBuf.Reset()
	errBuf.Reset()
	claimArgs := []string{"claim", "-report", filepath.Join(tempDir, "evaluation_report.json")}
	if err := run(claimArgs, &outBuf, &errBuf); err == nil {
		t.Fatal("synthetic pilot unexpectedly produced qualified claim")
	}
	if strings.Contains(outBuf.String(), "Claim:") {
		t.Fatal("refused claim was printed")
	}

	// 4. Test "run" with JSON output
	outBuf.Reset()
	errBuf.Reset()
	jsonArgs := []string{"run", "-phase", "pilot", "-json"}
	if err := run(jsonArgs, &outBuf, &errBuf); err != nil {
		t.Fatalf("run -json failed: %v", err)
	}
	jsonStr := outBuf.String()
	if !strings.Contains(jsonStr, `"protocol":`) || !strings.Contains(jsonStr, `"gates":`) {
		t.Errorf("expected JSON output, got:\n%s", jsonStr)
	}

	// 5. Test error cases
	if err := run([]string{}, &outBuf, &errBuf); err == nil {
		t.Errorf("expected error with empty args")
	}
	if err := run([]string{"unknown-cmd"}, &outBuf, &errBuf); err == nil {
		t.Errorf("expected error with unknown command")
	}
}
