package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func TestRegressCommandRegistration(t *testing.T) {
	c, ok := lookupCommand("regress")
	if !ok || c.usage == "" || c.run == nil {
		t.Fatalf("command 'regress' not registered properly")
	}

	err := c.run([]string{"regress"})
	if err == nil || !strings.Contains(err.Error(), "usage: cudaprof regress") {
		t.Fatalf("expected usage error, got %v", err)
	}
}

func TestRegressCLI_PlanAndCheckAndExport(t *testing.T) {
	dir := t.TempDir()

	inc := cudaregress.Incident{
		ID:       "inc-cli-test",
		Admitted: true,
		Workload: cudaworkload.WorkloadIdentity{
			WorkloadName: "test_workload",
			RunID:        "run-1",
		},
		Diagnosis: cudadiag.DiagnosisResult{
			Admitted:  true,
			Mechanism: cudadiag.MechanismDeviceBottleneck,
		},
		PrimaryFinding: tracecontract.Finding{
			ID:    "finding-cli-01",
			Class: tracecontract.FindingObserved,
			Admission: tracecontract.Admission{
				Admitted: true,
				Status:   "admitted",
			},
		},
		ObservedEnvironment: cudaregress.Environment{
			Model:             "NVIDIA GB10",
			ComputeCapability: "sm_121a",
			Driver:            "550.54.14",
			CUDA:              "12.8",
		},
		CreatedAt: time.Now(),
	}

	incBytes, err := json.Marshal(inc)
	if err != nil {
		t.Fatal(err)
	}
	incPath := filepath.Join(dir, "incident.json")
	if err := os.WriteFile(incPath, incBytes, 0644); err != nil {
		t.Fatal(err)
	}

	// 1. Run 'regress plan'
	planPath := filepath.Join(dir, "plan.json")
	var planOut bytes.Buffer
	err = runRegressPlanCommand([]string{"-incident", incPath, "-o", planPath, "-rounds", "3"}, &planOut)
	if err != nil {
		t.Fatalf("regress plan failed: %v", err)
	}

	// Verify plan file exists and is valid JSON
	planData, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("failed to read generated plan: %v", err)
	}
	var res struct {
		Plan    cudaregress.ExperimentPlan `json:"plan"`
		Receipt cudaregress.PlanReceipt    `json:"receipt"`
	}
	if err := json.Unmarshal(planData, &res); err != nil {
		t.Fatalf("invalid generated plan JSON: %v", err)
	}
	if res.Plan.PlanID == "" {
		t.Errorf("plan ID is empty")
	}
	envPath := filepath.Join(dir, "observed-environment.json")
	writeJSONFile(t, envPath, observedEnvironmentFile{
		Version: 1, Environment: inc.ObservedEnvironment, Provenance: "supplied CLI fixture observation",
	})

	// 2. Run 'regress check'
	var checkOut bytes.Buffer
	err = runRegressCheckCommand([]string{"-plan", planPath, "-environment", envPath}, &checkOut)
	if err != nil {
		t.Fatalf("regress check failed: %v", err)
	}
	wantCheck := fmt.Sprintf("Plan %s: reconstructable (validated artifacts; caller-supplied environment, provenance %q)\n", res.Plan.PlanID, "supplied CLI fixture observation")
	if checkOut.String() != wantCheck {
		t.Errorf("check output = %q, want %q", checkOut.String(), wantCheck)
	}
	// Replay must read that same producer file before attempting the spec. The
	// missing spec prevents runner creation or any command/device execution.
	var replayOut bytes.Buffer
	err = runRegressReplayCommand([]string{"-plan", planPath, "-environment", envPath, "-spec", filepath.Join(dir, "missing-spec.json"), "-o", filepath.Join(dir, "unused-replay-output")}, &replayOut)
	if !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), "open execution spec") || replayOut.Len() != 0 {
		t.Fatalf("replay did not accept actual producer plan before spec boundary: %v, %q", err, replayOut.String())
	}

	// 3. Run 'regress export'
	exportDir := filepath.Join(dir, "export_bundle")
	var exportOut bytes.Buffer
	err = runRegressExportCommand([]string{"-plan", planPath, "-o", exportDir}, &exportOut)
	if err != nil {
		t.Fatalf("regress export failed: %v", err)
	}

	for _, fname := range []string{"plan.json", "manifest.json", "README.md"} {
		if _, err := os.Stat(filepath.Join(exportDir, fname)); err != nil {
			t.Errorf("missing exported file %s: %v", fname, err)
		}
	}
	exported, err := readRegressPlanFile(filepath.Join(exportDir, "plan.json"))
	if err != nil || exported.PlanID != res.Plan.PlanID || exported.PlanDigest != res.Plan.PlanDigest {
		t.Fatalf("export lost validated producer plan: plan=%+v error=%v", exported, err)
	}
}
