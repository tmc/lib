package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"rsc.io/script"
	"rsc.io/script/scripttest"
)

func TestVerifyCapture(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		qualified bool
		wantParts []string
	}{
		{
			name:      "complete qualified receipt",
			input:     `{"capture_success":true,"run_completeness":"complete","target_exit_code":0,"collector_finalized":true,"unqualified_generations":0,"activity_records":42,"dropped_records":0}`,
			qualified: true,
		},
		{
			name:      "complete status envelope",
			input:     `{"status":{"capture_success":true,"run_completeness":"complete","target_exit_code":0,"collector_finalized":true,"activity_records":12}}`,
			qualified: true,
		},
		{
			name:      "partial run with compiler child unfinalized",
			input:     `{"capture_success":true,"run_completeness":"partial","target_exit_code":0,"collector_finalized":true,"unqualified_generations":1,"activity_records":8}`,
			qualified: false,
			wantParts: []string{`run_completeness is "partial", want "complete"`, "unqualified_generations is 1, want 0"},
		},
		{
			name:      "unfinalized collector",
			input:     `{"capture_success":false,"run_completeness":"failed","target_exit_code":0,"collector_finalized":false,"unqualified_generations":0,"activity_records":5}`,
			qualified: false,
			wantParts: []string{"capture_success is false", `run_completeness is "failed", want "complete"`, "collector_finalized is false"},
		},
		{
			name:      "empty capture zero activity records",
			input:     `{"capture_success":true,"run_completeness":"empty","target_exit_code":0,"collector_finalized":true,"unqualified_generations":0,"activity_records":0}`,
			qualified: false,
			wantParts: []string{`run_completeness is "empty", want "complete"`, "activity_records is 0, want > 0"},
		},
		{
			name:      "nonzero target exit code",
			input:     `{"capture_success":true,"run_completeness":"complete","target_exit_code":1,"collector_finalized":true,"unqualified_generations":0,"activity_records":10}`,
			qualified: false,
			wantParts: []string{"target_exit_code is 1, want 0"},
		},
		{
			name: "A2 log with CUPTI_ERROR_UNKNOWN after device loss",
			input: "/opt/sglang/lib/python3.12/site-packages/torch/cuda/__init__.py:1174: UserWarning: Can't initialize NVML\n" +
				`{"capture_success":false,"target_exit_code":1,"collector_finalized":false,"activity_records":0,"run_completeness":"failed","unqualified_generations":1,"capture_error":"cupti_unavailable: cuptiActivityRegisterCallbacks: CUPTI_ERROR_UNKNOWN"}` + "\n" +
				"target exited with status 1\n",
			qualified: false,
			wantParts: []string{"terminal CUDA error", "CUPTI_ERROR_UNKNOWN", "capture_success is false", "target_exit_code is 1, want 0"},
		},
		{
			name:      "A5 receipt with illegal memory access in stdout",
			input:     `{"capture_success":true,"run_completeness":"complete","target_exit_code":0,"collector_finalized":true,"unqualified_generations":0,"activity_records":4,"stdout":"torch.AcceleratorError: CUDA error: an illegal memory access was encountered\n"}`,
			qualified: false,
			wantParts: []string{"terminal CUDA error", "an illegal memory access"},
		},
		{
			name:      "worker receipt without capture_success after GPU loss",
			input:     `{"status":"HOLD","mode":"capture","error":"RuntimeError: one H200 CUDA device required: CUDA unavailable"}`,
			qualified: false,
			wantParts: []string{"terminal CUDA error", "CUDA unavailable", "missing capture completeness receipt"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := VerifyCapture([]byte(tt.input))
			if got.Qualified != tt.qualified {
				t.Fatalf("VerifyCapture() qualified = %v (reason=%q), want %v", got.Qualified, got.Reason, tt.qualified)
			}
			if tt.qualified && got.Status != "PASS" {
				t.Errorf("Status = %q, want PASS", got.Status)
			}
			if !tt.qualified && got.Status != "HOLD" {
				t.Errorf("Status = %q, want HOLD", got.Status)
			}
			for _, want := range tt.wantParts {
				if !strings.Contains(got.Reason, want) {
					t.Errorf("Reason = %q, want substring %q", got.Reason, want)
				}
			}
		})
	}
}

func TestVerifyCaptureCommand(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.json")
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(good, []byte(`{"capture_success":true,"run_completeness":"complete","target_exit_code":0,"collector_finalized":true,"unqualified_generations":0,"activity_records":16}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(bad, []byte(`{"capture_success":true,"run_completeness":"partial","target_exit_code":0,"collector_finalized":true,"unqualified_generations":1,"activity_records":8}`), 0o600); err != nil {
		t.Fatal(err)
	}

	var out bytes.Buffer
	if err := runVerifyCaptureCommand([]string{"-json", good}, &out); err != nil {
		t.Fatalf("verify-capture good: %v", err)
	}
	var rep VerifyCaptureReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("decode JSON report: %v", err)
	}
	if !rep.Qualified || rep.Status != "PASS" || len(rep.Captures) != 1 {
		t.Fatalf("unexpected report: %+v", rep)
	}

	out.Reset()
	err := runVerifyCaptureCommand([]string{good, bad}, &out)
	if err == nil {
		t.Fatal("verify-capture with partial capture succeeded, want error")
	}
	if !strings.Contains(out.String(), "good.json: PASS") || !strings.Contains(out.String(), "bad.json: HOLD") {
		t.Errorf("unexpected text output:\n%s", out.String())
	}
}

func TestVerifyCaptureScript(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the CLI")
	}
	bin := t.TempDir()
	build := exec.Command("go", "build", "-o", filepath.Join(bin, "cudaprof"), ".")
	build.Env = append(os.Environ(), "GOWORK=off")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build cudaprof: %v\n%s", err, out)
	}
	engine := &script.Engine{
		Cmds:  scripttest.DefaultCmds(),
		Conds: scripttest.DefaultConds(),
	}
	env := []string{
		"PATH=" + bin + string(os.PathListSeparator) + os.Getenv("PATH"),
		"HOME=" + os.Getenv("HOME"),
		"GOWORK=off",
	}
	scripttest.Test(t, context.Background(), engine, env, "testdata/script/verify_capture.txt")
}
