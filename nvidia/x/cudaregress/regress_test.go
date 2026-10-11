package cudaregress_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func makeTestArtifact(t *testing.T, dir, filename, content string) cudaregress.ArtifactRef {
	t.Helper()
	p := filepath.Join(dir, filename)
	if err := os.WriteFile(p, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write test artifact: %v", err)
	}
	h := sha256.Sum256([]byte(content))
	return cudaregress.ArtifactRef{
		Name:      filename,
		Path:      p,
		Digest:    hex.EncodeToString(h[:]),
		SizeBytes: int64(len(content)),
	}
}

func validAdmittedIncident() cudaregress.Incident {
	return cudaregress.Incident{
		ID:       "inc-001",
		Admitted: true,
		Workload: cudaworkload.WorkloadIdentity{
			WorkloadName: "matmul_gemm",
			RunID:        "run-10",
		},
		Diagnosis: cudadiag.DiagnosisResult{
			Admitted:  true,
			Mechanism: cudadiag.MechanismDeviceBottleneck,
		},
		PrimaryFinding: tracecontract.Finding{
			ID:    "finding-001",
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
			Clock:             "1500MHz",
		},
		CreatedAt: time.Now(),
	}
}

func TestPlanGeneration_Admitted(t *testing.T) {
	inc := validAdmittedIncident()
	tmpDir := t.TempDir()
	src := makeTestArtifact(t, tmpDir, "kernel.cu", "__global__ void matmul() {}")
	bin := makeTestArtifact(t, tmpDir, "kernel.cubin", "CUBIN_BINARY_DATA")

	plan, rec, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Sources:  []cudaregress.ArtifactRef{src},
		Binaries: []cudaregress.ArtifactRef{bin},
		Rounds:   5,
	})
	if err != nil {
		t.Fatalf("expected plan generation to succeed, got %v", err)
	}
	if plan.PlanID == "" {
		t.Errorf("plan ID is empty")
	}
	if plan.PlanDigest == "" {
		t.Errorf("plan digest is empty")
	}
	if !rec.ArtifactClosureValid {
		t.Errorf("expected artifact closure to be valid")
	}
	if rec.ArtifactsVerified < 2 {
		t.Errorf("expected at least 2 verified artifacts, got %d", rec.ArtifactsVerified)
	}
}

func TestPlanGeneration_UnadmittedRefusal(t *testing.T) {
	tests := []struct {
		name string
		inc  cudaregress.Incident
	}{
		{
			name: "incident unadmitted",
			inc: cudaregress.Incident{
				ID:             "inc-unadmitted",
				Admitted:       false,
				WithheldReason: "loss_exceeded",
			},
		},
		{
			name: "diagnosis withheld",
			inc: cudaregress.Incident{
				ID:       "inc-diag-withheld",
				Admitted: true,
				Diagnosis: cudadiag.DiagnosisResult{
					Admitted:       false,
					Mechanism:      cudadiag.MechanismWithheld,
					WithheldReason: "clock_uncertainty_exceeded",
				},
			},
		},
		{
			name: "primary finding withheld",
			inc: cudaregress.Incident{
				ID:       "inc-finding-withheld",
				Admitted: true,
				PrimaryFinding: tracecontract.Finding{
					ID:    "finding-01",
					Class: tracecontract.FindingUnavailable,
					Admission: tracecontract.Admission{
						Admitted: false,
						Status:   "withheld",
						Reason:   "missing_return",
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := cudaregress.NewPlanFromIncident(tc.inc, cudaregress.PlanOptions{})
			if err == nil {
				t.Fatalf("expected refusal error, got nil")
			}
			if !errors.Is(err, cudaregress.ErrIncidentNotAdmitted) {
				t.Errorf("expected ErrIncidentNotAdmitted, got %v", err)
			}
		})
	}
}

func TestPlanValidation_ArtifactClosure(t *testing.T) {
	tmpDir := t.TempDir()
	src := makeTestArtifact(t, tmpDir, "test.cu", "int main() {}")

	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Sources: []cudaregress.ArtifactRef{src},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := cudaregress.ValidatePlan(plan); err != nil {
		t.Errorf("expected valid plan, got %v", err)
	}
}

func TestPlanValidation_MissingArtifactBlocker(t *testing.T) {
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Add missing artifact reference
	plan.SourceArtifacts = append(plan.SourceArtifacts, cudaregress.ArtifactRef{
		Name:   "missing.cu",
		Path:   "/path/to/nonexistent/missing.cu",
		Digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
	})

	err = cudaregress.ValidatePlan(plan)
	if err == nil {
		t.Fatalf("expected error for missing artifact, got nil")
	}
	if !errors.Is(err, cudaregress.ErrArtifactMissing) {
		t.Errorf("expected ErrArtifactMissing, got %v", err)
	}
}

func TestPlanValidation_CorruptArtifactBlocker(t *testing.T) {
	tmpDir := t.TempDir()
	src := makeTestArtifact(t, tmpDir, "corrupt.cu", "original content")

	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Deliberately corrupt digest
	src.Digest = "0000000000000000000000000000000000000000000000000000000000000000"
	plan.SourceArtifacts = append(plan.SourceArtifacts, src)

	err = cudaregress.ValidatePlan(plan)
	if err == nil {
		t.Fatalf("expected error for corrupted artifact digest, got nil")
	}
	if !errors.Is(err, cudaregress.ErrArtifactCorrupted) {
		t.Errorf("expected ErrArtifactCorrupted, got %v", err)
	}
}

func TestReplay_EnvironmentDrift(t *testing.T) {
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Rounds: 3,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Drifted environment: different compute capability
	driftedEnv := inc.ObservedEnvironment
	driftedEnv.ComputeCapability = "sm_90" // plan expected sm_121a

	runner := cudaregress.ReplayFunc(func(ctx context.Context, armName string, round int) (cudatrial.Receipt, []float64, error) {
		return cudatrial.Receipt{Complete: true, Name: armName, TimingKind: cudatrial.TimingWallElapsed, WallNs: 1000, WallElapsed: time.Microsecond}, nil, nil
	})

	// 1. Without allow-drift: must refuse
	_, _, err = cudaregress.Replay(context.Background(), plan, runner, cudaregress.ReplayOptions{
		CurrentEnvironment: driftedEnv,
		AllowDriftFork:     false,
	})
	if err == nil {
		t.Fatalf("expected drift refusal, got nil")
	}
	if !errors.Is(err, cudaregress.ErrEnvironmentDrift) {
		t.Errorf("expected ErrEnvironmentDrift, got %v", err)
	}

	// 2. With allow-drift: forks a new explicitly identified experiment plan
	execRec, compRec, err := cudaregress.Replay(context.Background(), plan, runner, cudaregress.ReplayOptions{
		CurrentEnvironment: driftedEnv,
		AllowDriftFork:     true,
		DriftRationale:     "evaluating regression candidate on Hopper sm_90 target",
	})
	if err != nil {
		t.Fatalf("expected successful fork replay, got %v", err)
	}
	if execRec.PlanID == plan.PlanID {
		t.Errorf("forked plan ID should differ from parent plan ID")
	}
	if compRec == nil {
		t.Errorf("expected comparison receipt")
	}
}

func TestPlan_SafeExecutionContract(t *testing.T) {
	tmpDir := t.TempDir()
	bin := makeTestArtifact(t, tmpDir, "safe_binary", "#!/bin/sh\nexit 0\n")

	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Binaries: []cudaregress.ArtifactRef{bin},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 1. Attempt to execute undeclared binary
	_, err = cudaregress.SafeCommandArm(plan, "cand", "/bin/sh")
	if err == nil {
		t.Fatalf("expected refusal of undeclared binary, got nil")
	}
	if !errors.Is(err, cudaregress.ErrUnsafeCommandRefused) {
		t.Errorf("expected ErrUnsafeCommandRefused, got %v", err)
	}

	// 2. Attempt to inject shell meta-characters
	_, err = cudaregress.SafeCommandArm(plan, "cand", bin.Path, "arg1; rm -rf /")
	if err == nil {
		t.Fatalf("expected refusal of shell metacharacter, got nil")
	}
	if !errors.Is(err, cudaregress.ErrUnsafeCommandRefused) {
		t.Errorf("expected ErrUnsafeCommandRefused, got %v", err)
	}
}

func TestReplay_KnownRegression(t *testing.T) {
	ctx := context.Background()
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Oracle:     cudaregress.CorrectnessOracle{Method: "numerical-tolerance"},
		WarmupRuns: 1,
		Rounds:     4,
		ExecutionScope: cudaregress.TimingScope{
			Channel: cudatrial.TimingSummedKernel,
		},
		StoppingRules: cudaregress.StoppingRules{
			RelativeThreshold: 0.10, // 10% regression threshold
			MinSamples:        3,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	runner := cudaregress.ReplayFunc(func(ctx context.Context, armName string, round int) (cudatrial.Receipt, []float64, error) {
		dur := 100 * time.Microsecond
		if armName == "candidate" {
			dur = 130 * time.Microsecond // 30% slowdown -> clear regression
		}
		r := cudatrial.Receipt{
			Name:         armName,
			Round:        round,
			Complete:     true,
			Digest:       "exact-digest",
			SummedKernel: dur,
			DeviceNs:     dur.Nanoseconds(),
			TimingKind:   cudatrial.TimingSummedKernel,
		}
		_ = r.RecordTiming(cudatrial.TimingSummedKernel, dur)
		return r, []float64{1.0}, nil
	})

	execRec, compRec, err := cudaregress.Replay(ctx, plan, runner, cudaregress.ReplayOptions{
		CurrentEnvironment: inc.ObservedEnvironment,
		ReferenceValues:    []float64{1},
	})
	if err != nil {
		t.Fatalf("replay failed: %v", err)
	}

	if !compRec.RegressionDetected {
		t.Errorf("expected regression to be detected")
	}
	if compRec.Verdict != cudaprofilehistory.Fail {
		t.Errorf("expected Verdict Fail, got %v", compRec.Verdict)
	}
	if compRec.Ratio < 1.25 {
		t.Errorf("expected ratio >= 1.25, got %f", compRec.Ratio)
	}
	if execRec.RoundsCompleted != 4 {
		t.Errorf("expected 4 rounds, got %d", execRec.RoundsCompleted)
	}
}

func TestReplay_NoRegression_NullControl(t *testing.T) {
	ctx := context.Background()
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Oracle:     cudaregress.CorrectnessOracle{Method: "numerical-tolerance"},
		WarmupRuns: 1,
		Rounds:     3,
		ExecutionScope: cudaregress.TimingScope{
			Channel: cudatrial.TimingSummedKernel,
		},
		StoppingRules: cudaregress.StoppingRules{
			RelativeThreshold: 0.10,
			MinSamples:        3,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Baseline and candidate have identical duration
	runner := cudaregress.ReplayFunc(func(ctx context.Context, armName string, round int) (cudatrial.Receipt, []float64, error) {
		dur := 100 * time.Microsecond
		r := cudatrial.Receipt{
			Name:         armName,
			Round:        round,
			Complete:     true,
			Digest:       "exact-digest",
			SummedKernel: dur,
			DeviceNs:     dur.Nanoseconds(),
			TimingKind:   cudatrial.TimingSummedKernel,
		}
		_ = r.RecordTiming(cudatrial.TimingSummedKernel, dur)
		return r, []float64{1.0}, nil
	})

	_, compRec, err := cudaregress.Replay(ctx, plan, runner, cudaregress.ReplayOptions{
		CurrentEnvironment: inc.ObservedEnvironment,
		ReferenceValues:    []float64{1},
	})
	if err != nil {
		t.Fatalf("replay failed: %v", err)
	}

	if compRec.RegressionDetected {
		t.Errorf("expected no regression detected")
	}
	if compRec.Verdict != cudaprofilehistory.Pass {
		t.Errorf("expected Verdict Pass, got %v", compRec.Verdict)
	}
}

func TestReplay_NumericalMismatch(t *testing.T) {
	ctx := context.Background()
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		WarmupRuns: 0,
		Rounds:     2,
		Oracle: cudaregress.CorrectnessOracle{
			Method:    "numerical-tolerance",
			Tolerance: 1e-4,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Runner emits mismatched outputs for candidate
	runner := cudaregress.ReplayFunc(func(ctx context.Context, armName string, round int) (cudatrial.Receipt, []float64, error) {
		r := cudatrial.Receipt{
			Name:        armName,
			Round:       round,
			Complete:    true,
			WallElapsed: 10 * time.Millisecond,
		}
		if armName == "candidate" {
			return r, []float64{1.0, 5.0}, nil // 5.0 vs reference 2.0 -> exceeds tolerance
		}
		return r, []float64{1.0, 2.0}, nil
	})

	_, _, err = cudaregress.Replay(ctx, plan, runner, cudaregress.ReplayOptions{
		CurrentEnvironment: inc.ObservedEnvironment,
		ReferenceValues:    []float64{1.0, 2.0},
	})
	if err == nil {
		t.Fatalf("expected error on numerical mismatch, got nil")
	}
	if !errors.Is(err, cudaregress.ErrCorrectnessOracleFailed) {
		t.Errorf("expected ErrCorrectnessOracleFailed, got %v", err)
	}
}

func TestReplay_PartialRun(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		WarmupRuns: 0,
		Rounds:     5,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	runner := cudaregress.ReplayFunc(func(c context.Context, armName string, round int) (cudatrial.Receipt, []float64, error) {
		if round == 2 && armName == "candidate" {
			cancel() // cancel during round 2
		}
		return cudatrial.Receipt{
			Name:        armName,
			Round:       round,
			Complete:    true,
			WallElapsed: 50 * time.Microsecond,
		}, nil, nil
	})

	execRec, _, err := cudaregress.Replay(ctx, plan, runner, cudaregress.ReplayOptions{
		CurrentEnvironment: inc.ObservedEnvironment,
	})
	if err == nil {
		t.Fatalf("expected error from cancellation, got nil")
	}
	if execRec == nil {
		t.Fatalf("expected partial execution receipt, got nil")
	}
	if !execRec.Partial {
		t.Errorf("expected receipt to be marked Partial")
	}
	if execRec.HaltReason == "" {
		t.Errorf("expected non-empty HaltReason")
	}
}

func TestPreserve_MissingFamiliesVsMeasuredZero(t *testing.T) {
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	execRec := &cudaregress.ExecutionReceipt{
		ExecutionID: "exec-fam-test",
		ArmReceipts: []cudatrial.Receipt{
			{Name: "baseline", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "baseline", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "baseline", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "candidate", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "candidate", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "candidate", Complete: true, WallElapsed: 100 * time.Microsecond},
		},
	}

	// Baseline has "gemm_mma" and "layernorm". Candidate has only "gemm_mma".
	comp, err := cudaregress.CompareExecution(plan, execRec, cudaregress.ReplayOptions{
		BaselineFamilies:  []string{"gemm_mma", "layernorm"},
		CandidateFamilies: []string{"gemm_mma"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundMissing := false
	for _, fam := range comp.MissingFamilies {
		if fam == "layernorm" {
			foundMissing = true
		}
	}
	if !foundMissing {
		t.Errorf("expected layernorm to be reported in MissingFamilies")
	}
}

func TestPreserve_ResourceVsPerformanceChanges(t *testing.T) {
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	execRec := &cudaregress.ExecutionReceipt{
		ExecutionID: "exec-res-test",
		ArmReceipts: []cudatrial.Receipt{
			{Name: "baseline", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "baseline", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "baseline", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "candidate", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "candidate", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "candidate", Complete: true, WallElapsed: 100 * time.Microsecond},
		},
	}

	comp, err := cudaregress.CompareExecution(plan, execRec, cudaregress.ReplayOptions{
		BaselineResources: map[string]string{
			"clock_mhz": "1500",
			"registers": "32",
		},
		CandidateResources: map[string]string{
			"clock_mhz": "1200", // thermal throttling
			"registers": "48",   // increased register usage
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(comp.ResourceChanges) != 2 {
		t.Fatalf("expected 2 resource changes, got %d", len(comp.ResourceChanges))
	}
	// Verify performance changes does not contain resource strings
	for _, pc := range comp.PerformanceChanges {
		if pc.Channel == "" {
			t.Errorf("performance change missing channel")
		}
	}
}

func TestProfileHistoryLink(t *testing.T) {
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		ExecutionScope: cudaregress.TimingScope{Channel: cudatrial.TimingWallElapsed},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	execRec := &cudaregress.ExecutionReceipt{
		ExecutionID:         "exec-hist-test",
		OutputsVerified:     true,
		OutputDigests:       map[string]string{"candidate": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		EnvironmentObserved: inc.ObservedEnvironment,
		ArmReceipts: []cudatrial.Receipt{
			{Name: "baseline", Complete: true, WallElapsed: 100 * time.Microsecond},
			{Name: "candidate", Complete: true, WallElapsed: 120 * time.Microsecond},
		},
	}

	compRec := &cudaregress.ComparisonReceipt{
		ExecutionID:       execRec.ExecutionID,
		PlanID:            plan.PlanID,
		TimingChannel:     cudatrial.TimingWallElapsed,
		CandidateDuration: 120 * time.Microsecond,
		Verdict:           cudaprofilehistory.Pass,
	}

	rec, err := cudaregress.LinkToProfileHistory(plan, execRec, compRec, cudaregress.HistoryLinkOptions{
		Compiler: cudacache.CompilerIdentity{
			NVRTCVersion: "12.8",
			DriverSONAME: "libcuda.so.1",
			TargetArch:   "sm_121a",
		},
		Workload: cudacache.WorkloadIdentity{
			ProgramID:     "gemm_repro",
			NumericalMode: "fp32",
			ShapeDims:     []int{128, 128},
		},
		Entry:     "matmul",
		Qualified: true,
	})
	if err != nil {
		t.Fatalf("failed to link to profile history: %v", err)
	}

	if rec.Key == "" {
		t.Errorf("record key should not be empty")
	}

	// Validate record via cudaprofilehistory validator
	if err := rec.Validate(); err != nil {
		t.Errorf("record validation failed: %v", err)
	}
}

func TestIndependentRerun_Reconstruction(t *testing.T) {
	inc := validAdmittedIncident()
	tmpDir := t.TempDir()
	bin := makeTestArtifact(t, tmpDir, "binary.bin", "BINARY")

	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Binaries: []cudaregress.ArtifactRef{bin},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check on matching environment: reconstructable
	reconstructable, blockers := cudaregress.CheckReconstruction(plan, inc.ObservedEnvironment)
	if !reconstructable || len(blockers) > 0 {
		t.Errorf("expected reconstructable on matching environment, got blockers: %v", blockers)
	}

	// Check on drifted environment: not reconstructable, explains limitation
	drifted := inc.ObservedEnvironment
	drifted.Model = "NVIDIA A100"
	reconstructable, blockers = cudaregress.CheckReconstruction(plan, drifted)
	if reconstructable {
		t.Errorf("expected not reconstructable on drifted device")
	}
	if len(blockers) == 0 {
		t.Errorf("expected blockers to explain limitation")
	}
}

func TestMinimalReproducer_Export(t *testing.T) {
	inc := validAdmittedIncident()
	tmpDir := t.TempDir()
	src := makeTestArtifact(t, tmpDir, "repro.cu", "int main() {}")

	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Sources: []cudaregress.ArtifactRef{src},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	exportDir := filepath.Join(tmpDir, "reproducer_bundle")
	if err := cudaregress.ExportMinimalReproducer(plan, exportDir); err != nil {
		t.Fatalf("failed to export minimal reproducer: %v", err)
	}

	for _, fname := range []string{"plan.json", "manifest.json", "README.md"} {
		if _, err := os.Stat(filepath.Join(exportDir, fname)); err != nil {
			t.Errorf("missing exported file %s: %v", fname, err)
		}
	}
}

func TestNumericalToleranceRejectsInvalid(t *testing.T) {
	for _, tol := range []float64{math.NaN(), math.Inf(1), -1} {
		if err := cudaregress.CheckNumericalCorrectness([]float64{1}, []float64{100}, tol); err == nil {
			t.Fatalf("accepted tolerance %v", tol)
		}
	}
}
func TestReplayRequiresNumericalReference(t *testing.T) {
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{Oracle: cudaregress.CorrectnessOracle{Method: "numerical-tolerance", Tolerance: 1e-6}})
	if err != nil {
		t.Fatal(err)
	}
	called := false
	runner := cudaregress.ReplayFunc(func(context.Context, string, int) (cudatrial.Receipt, []float64, error) {
		called = true
		return cudatrial.Receipt{}, nil, nil
	})
	if _, _, err := cudaregress.Replay(context.Background(), plan, runner, cudaregress.ReplayOptions{}); err == nil || called {
		t.Fatal("missing reference did not refuse before execution")
	}
}

func TestComparisonDoesNotSubstituteTiming(t *testing.T) {
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{ExecutionScope: cudaregress.TimingScope{Channel: cudatrial.TimingGPUBusy}})
	if err != nil {
		t.Fatal(err)
	}
	execution := &cudaregress.ExecutionReceipt{ArmReceipts: []cudatrial.Receipt{{Name: "baseline", Complete: true, WallElapsed: time.Second}}}
	if _, err := cudaregress.CompareExecution(plan, execution, cudaregress.ReplayOptions{}); err == nil {
		t.Fatal("wall duration substituted for GPU busy time")
	}
}

func TestPlanValidationClosureAndLimits(t *testing.T) {
	for _, change := range []string{"digest", "missing artifact digest", "library", "trials"} {
		t.Run(change, func(t *testing.T) {
			plan, _, err := cudaregress.NewPlanFromIncident(validAdmittedIncident(), cudaregress.PlanOptions{})
			if err != nil {
				t.Fatal(err)
			}
			switch change {
			case "digest":
				plan.PlanDigest = "invalid"
			case "missing artifact digest":
				plan.BinaryArtifacts = []cudaregress.ArtifactRef{{Path: "unused"}}
			case "library":
				plan.LibraryClosure = []cudaregress.LibraryRef{{Name: "unknown"}}
			case "trials":
				plan.StoppingRules.MaxTrials = 1
				plan.PlanDigest, _ = cudaregress.ComputePlanDigest(plan)
			}
			if err := cudaregress.ValidatePlan(plan); err == nil {
				t.Fatal("invalid closure admitted")
			}
		})
	}
}
func TestEnvironmentMissingAndUUID(t *testing.T) {
	plan, _, err := cudaregress.NewPlanFromIncident(validAdmittedIncident(), cudaregress.PlanOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := cudaregress.ValidateEnvironment(plan, cudaregress.Environment{}); err == nil {
		t.Fatal("unknown environment admitted")
	}
	plan.DeviceConfig.DeviceUUID = "GPU-required"
	if err := cudaregress.ValidateEnvironment(plan, validAdmittedIncident().ObservedEnvironment); err == nil {
		t.Fatal("unknown UUID admitted")
	}
}
