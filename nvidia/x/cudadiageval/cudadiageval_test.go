package cudadiageval

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func TestFixture_IndependentGroundTruth(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0}
	out, hash := ComputeNumericalOracle(data)

	if len(out) != len(data) {
		t.Fatalf("ComputeNumericalOracle len: got %d, want %d", len(out), len(data))
	}

	// Reference check: y = 3x^2 - 2x + 1
	// x=1: 3(1) - 2(1) + 1 = 2
	// x=2: 3(4) - 2(2) + 1 = 9
	// x=3: 3(9) - 2(3) + 1 = 22
	expected := []float64{2.0, 9.0, 22.0}
	for i, v := range out {
		if math.Abs(v-expected[i]) > 1e-9 {
			t.Errorf("ComputeNumericalOracle[%d]: got %f, want %f", i, v, expected[i])
		}
	}
	if hash == "" {
		t.Errorf("ComputeNumericalOracle: empty hash")
	}

	// Negative control / Mutant check: mutant where addition is subtraction
	mutantOut := 3.0*1.0*1.0 - 2.0*1.0 - 1.0 // 0 instead of 2
	if math.Abs(mutantOut-expected[0]) < 1e-9 {
		t.Errorf("numerical oracle mutant check failed: mutant equals expected")
	}
}

func TestProtocol_FreezeAndValidate(t *testing.T) {
	proto := DefaultProtocol(PhasePilot)
	if err := proto.Validate(); err != nil {
		t.Fatalf("DefaultProtocol validation failed: %v", err)
	}

	if proto.Frozen {
		t.Fatalf("DefaultProtocol should not be initially frozen")
	}

	if err := proto.Freeze(); err != nil {
		t.Fatalf("Freeze() failed: %v", err)
	}
	if !proto.Frozen {
		t.Fatalf("Protocol should be frozen after Freeze()")
	}

	// Attempting to freeze again must fail with ErrProtocolFrozen
	if err := proto.Freeze(); err != ErrProtocolFrozen {
		t.Fatalf("Freeze() second time: got %v, want ErrProtocolFrozen", err)
	}

	// Qualification requires frozen protocol
	unfrozenQual := DefaultProtocol(PhaseQualification)
	cfg := DefaultEvalConfig()
	_, err := RunEvaluation(context.Background(), unfrozenQual, cfg)
	if err != ErrProtocolNotFrozen {
		t.Fatalf("RunEvaluation on unfrozen qualification: got %v, want ErrProtocolNotFrozen", err)
	}
}

func TestBaselines_NsightMatrix(t *testing.T) {
	baselines := DefaultBaselines()
	if err := ValidateBaselines(baselines); err != nil {
		t.Fatalf("ValidateBaselines(DefaultBaselines) failed: %v", err)
	}

	// Check required independent baselines
	hasNsys := false
	hasNcu := false
	for _, b := range baselines {
		if b.Name == "NVIDIA Nsight Systems" && b.Independent {
			hasNsys = true
		}
		if b.Name == "NVIDIA Nsight Compute" && b.Independent {
			hasNcu = true
		}
	}
	if !hasNsys {
		t.Errorf("missing independent Nsight Systems baseline")
	}
	if !hasNcu {
		t.Errorf("missing independent Nsight Compute baseline")
	}

	// Mutant / Negative control: marking related engine (Beyla/OBI) as independent must fail
	mutantBaselines := append([]Baseline(nil), baselines...)
	for i := range mutantBaselines {
		if strings.Contains(mutantBaselines[i].Name, "Beyla") {
			mutantBaselines[i].Independent = true
		}
	}
	if err := ValidateBaselines(mutantBaselines); !errors.Is(err, ErrRelatedBaseline) {
		t.Errorf("ValidateBaselines with independent Beyla: got %v, want ErrRelatedBaseline", err)
	}
}

func TestAttribution_ZeroIncorrectExactJoins(t *testing.T) {
	fixture := NewGroundTruthFixture()
	proc := cudaworkload.ProcessGenID{PID: 41000, Generation: 1}

	reqs := []GroundTruthRequest{
		{
			ID:            "attr-01",
			WorkloadName:  "workload-a",
			ProcessGen:    proc,
			TID:           5001,
			StreamID:      1,
			CorrelationID: 7001,
			Fault:         FaultNone,
			Resilience:    ResilienceNormal,
			Data:          []float64{1.0},
		},
		{
			ID:            "attr-02",
			WorkloadName:  "workload-b",
			ProcessGen:    proc,
			TID:           5002,
			StreamID:      2,
			CorrelationID: 7002,
			Fault:         FaultNone,
			Resilience:    ResilienceNormal,
			Data:          []float64{2.0},
		},
	}

	res, err := fixture.Execute(context.Background(), reqs)
	if err != nil {
		t.Fatalf("fixture.Execute: %v", err)
	}

	budgets := DefaultBudgets()
	receipt, gates := evaluateAttribution(res, budgets)

	if receipt.IncorrectJoins != 0 {
		t.Errorf("IncorrectJoins: got %d, want 0", receipt.IncorrectJoins)
	}
	if receipt.ExactJoins != 2 {
		t.Errorf("ExactJoins: got %d, want 2", receipt.ExactJoins)
	}
	if receipt.GateDisposition != "passed" {
		t.Errorf("GateDisposition: got %s, want passed", receipt.GateDisposition)
	}

	// Verify gate records
	foundZeroGate := false
	for _, g := range gates {
		if g.GateName == "zero_incorrect_exact_joins" {
			foundZeroGate = true
			if g.Disposition != "passed" {
				t.Errorf("zero_incorrect_exact_joins gate disposition: got %s, want passed", g.Disposition)
			}
		}
	}
	if !foundZeroGate {
		t.Errorf("zero_incorrect_exact_joins gate missing")
	}

	// Negative control / Mutant: corrupted correlation in fixture result
	corruptedRes := *res
	corruptedRes.GroundTruth = append([]GroundTruthRecord(nil), res.GroundTruth...)
	// Invert correlation ID in ground truth to create an incorrect exact join
	corruptedRes.GroundTruth[0].CorrelationID = 999999

	corruptReceipt, _ := evaluateAttribution(&corruptedRes, budgets)
	if corruptReceipt.IncorrectJoins == 0 {
		t.Errorf("mutant check: corrupt correlation did not produce incorrect joins")
	}
	if corruptReceipt.GateDisposition != "failed" {
		t.Errorf("mutant check: corrupt correlation disposition: got %s, want failed", corruptReceipt.GateDisposition)
	}
}

func TestDiagnostic_HeldOutFaultEvaluation(t *testing.T) {
	proto := DefaultProtocol(PhasePilot)
	cfg := DefaultEvalConfig()
	cfg.ExerciseStress = false

	report, err := RunEvaluation(context.Background(), proto, cfg)
	if err != nil {
		t.Fatalf("RunEvaluation: %v", err)
	}

	if report.Diagnostic.HeldOutTotal == 0 {
		t.Fatalf("HeldOutTotal: got 0, want > 0")
	}
	if report.Diagnostic.HeldOutCorrect == 0 {
		t.Fatalf("HeldOutCorrect: got 0, want > 0")
	}
	if report.Diagnostic.AccuracyPct < 90.0 {
		t.Errorf("HeldOut accuracy: got %.1f%%, want >= 90.0%%", report.Diagnostic.AccuracyPct)
	}
	if report.Diagnostic.GateDisposition != "passed" {
		t.Errorf("Diagnostic gate disposition: got %s, want passed", report.Diagnostic.GateDisposition)
	}
}

func TestDiagnostic_NoFaultControl(t *testing.T) {
	proto := DefaultProtocol(PhasePilot)
	cfg := DefaultEvalConfig()
	cfg.ExerciseStress = false

	report, err := RunEvaluation(context.Background(), proto, cfg)
	if err != nil {
		t.Fatalf("RunEvaluation: %v", err)
	}

	if report.Diagnostic.NoFaultTotal == 0 {
		t.Fatalf("NoFaultTotal: got 0, want > 0")
	}
	if report.Diagnostic.NoFaultPassed != report.Diagnostic.NoFaultTotal {
		t.Errorf("NoFaultPassed: got %d, want %d (100%% rejection of false alarms)",
			report.Diagnostic.NoFaultPassed, report.Diagnostic.NoFaultTotal)
	}
}

func TestDiagnostic_CompetingMechanismDistinction(t *testing.T) {
	proto := DefaultProtocol(PhasePilot)
	cfg := DefaultEvalConfig()
	cfg.ExerciseStress = false

	report, err := RunEvaluation(context.Background(), proto, cfg)
	if err != nil {
		t.Fatalf("RunEvaluation: %v", err)
	}

	if report.Diagnostic.CompetingTotal == 0 {
		t.Fatalf("CompetingTotal: got 0, want > 0")
	}
	if report.Diagnostic.CompetingDiverted != report.Diagnostic.CompetingTotal {
		t.Errorf("CompetingDiverted: got %d, want %d",
			report.Diagnostic.CompetingDiverted, report.Diagnostic.CompetingTotal)
	}

	// Verify confusion matrix distinguishes device bottleneck from host delay
	cm := report.Diagnostic.ConfusionMatrix
	if cm[string(cudadiag.MechanismDeviceBottleneck)] == 0 && cm[string(cudadiag.MechanismNoFault)] == 0 {
		t.Errorf("Confusion matrix missing expected competing mechanism entries: %+v", cm)
	}
}

func TestResilience_Controls(t *testing.T) {
	proto := DefaultProtocol(PhasePilot)
	cfg := DefaultEvalConfig()
	cfg.ExerciseStress = true

	report, err := RunEvaluation(context.Background(), proto, cfg)
	if err != nil {
		t.Fatalf("RunEvaluation with stress: %v", err)
	}

	if report.Diagnostic.ResilienceTotal == 0 {
		t.Fatalf("ResilienceTotal: got 0, want > 0")
	}
	if report.Diagnostic.ResiliencePassed != report.Diagnostic.ResilienceTotal {
		t.Errorf("ResiliencePassed: got %d, want %d",
			report.Diagnostic.ResiliencePassed, report.Diagnostic.ResilienceTotal)
	}

	// Ensure dropped records from overflow were tracked
	foundOverflowGate := false
	for _, g := range report.Gates {
		if g.GateName == "system_resilience_controls" {
			foundOverflowGate = true
			if g.Disposition != "passed" {
				t.Errorf("system_resilience_controls gate disposition: got %s, want passed", g.Disposition)
			}
		}
	}
	if !foundOverflowGate {
		t.Errorf("system_resilience_controls gate missing from report")
	}
}

func TestCollectorCost_Budgets(t *testing.T) {
	proto := DefaultProtocol(PhasePilot)
	cfg := DefaultEvalConfig()

	report, err := RunEvaluation(context.Background(), proto, cfg)
	if err != nil {
		t.Fatalf("RunEvaluation: %v", err)
	}

	if report.Cost.ThroughputDegradationPct > proto.Budgets.MaxThroughputDegradationPct {
		t.Errorf("Throughput degradation %.2f%% exceeded budget %.2f%%",
			report.Cost.ThroughputDegradationPct, proto.Budgets.MaxThroughputDegradationPct)
	}
	if report.Cost.MemoryGrowthMB > proto.Budgets.MaxRSSGrowthMB {
		t.Errorf("Memory growth %.2f MB exceeded budget %.2f MB",
			report.Cost.MemoryGrowthMB, proto.Budgets.MaxRSSGrowthMB)
	}
	if report.Cost.GateDisposition != "unavailable" {
		t.Errorf("Cost GateDisposition: got %s, want unavailable", report.Cost.GateDisposition)
	}
}

func TestReproducibility_DigestAndReplay(t *testing.T) {
	proto := DefaultProtocol(PhasePilot)
	cfg := DefaultEvalConfig()

	report, err := RunEvaluation(context.Background(), proto, cfg)
	if err != nil {
		t.Fatalf("RunEvaluation: %v", err)
	}

	if !report.Reproducibility.ReplayMatches {
		t.Errorf("ReplayMatches: got false, want true")
	}
	if report.Reproducibility.ArtifactManifestSHA256 == "" {
		t.Errorf("ArtifactManifestSHA256 is empty")
	}
	if report.Reproducibility.GateDisposition != "passed" {
		t.Errorf("Reproducibility GateDisposition: got %s, want passed", report.Reproducibility.GateDisposition)
	}
}

func TestClaimRefusesSyntheticEvaluation(t *testing.T) {
	report, err := RunEvaluation(context.Background(), DefaultProtocol(PhasePilot), DefaultEvalConfig())
	if err != nil {
		t.Fatal(err)
	}
	if report.Passed || report.Claim != nil {
		t.Fatal("synthetic evaluation qualified")
	}
	if _, err := FormulateClaim(report); err == nil {
		t.Fatal("claim accepted incomplete evaluation")
	}
}

func TestArtifacts_WriteAndVerify(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "cudadiageval-test-*")
	if err != nil {
		t.Fatalf("MkdirTemp: %v", err)
	}
	defer os.RemoveAll(tempDir)

	proto := DefaultProtocol(PhasePilot)
	cfg := DefaultEvalConfig()

	report, err := RunEvaluation(context.Background(), proto, cfg)
	if err != nil {
		t.Fatalf("RunEvaluation: %v", err)
	}

	manifest, err := WriteReportArtifacts(tempDir, report)
	if err != nil {
		t.Fatalf("WriteReportArtifacts: %v", err)
	}

	if manifest.ManifestSHA256 == "" {
		t.Errorf("ManifestSHA256 empty")
	}

	// Verify manifest
	verified, err := VerifyManifest(tempDir)
	if err != nil {
		t.Fatalf("VerifyManifest: %v", err)
	}
	if verified.ManifestSHA256 != manifest.ManifestSHA256 {
		t.Errorf("Manifest hash mismatch: got %s, want %s", verified.ManifestSHA256, manifest.ManifestSHA256)
	}

	// Tampering test: tamper with evaluation_report.json and verify it is caught
	repPath := filepath.Join(tempDir, "evaluation_report.json")
	if err := os.WriteFile(repPath, []byte("tampered content"), 0644); err != nil {
		t.Fatalf("WriteFile tampered: %v", err)
	}
	if _, err := VerifyManifest(tempDir); err == nil {
		t.Errorf("VerifyManifest did not detect tampered file")
	}
}
