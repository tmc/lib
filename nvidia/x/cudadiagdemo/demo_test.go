package cudadiagdemo

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func TestDemonstration_EndToEnd(t *testing.T) {
	tempDir := t.TempDir()
	outDir := filepath.Join(tempDir, "artifacts")

	cfg := DemonstrationConfig{
		NumRequests:       8,
		Concurrency:       2,
		HostDelay:         10 * time.Millisecond,
		DeviceDelay:       250 * time.Microsecond,
		OutputDir:         outDir,
		ExerciseCorruptEv: true,
		ExercisePressure:  true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := RunDemonstration(ctx, cfg)
	if err != nil {
		t.Fatalf("RunDemonstration failed: %v", err)
	}

	// 1. Check baseline diagnosis
	if res.BaselineDiagnosis.Mechanism != cudadiag.MechanismNoFault {
		t.Fatalf("expected baseline mechanism %q, got %q", cudadiag.MechanismNoFault, res.BaselineDiagnosis.Mechanism)
	}
	if !res.BaselineDiagnosis.Admitted {
		t.Fatalf("expected baseline diagnosis to be admitted, reason: %s", res.BaselineDiagnosis.WithheldReason)
	}

	// 2. Check host delay diagnosis
	if res.HostDelayDiagnosis.Mechanism != cudadiag.MechanismHostSubmissionDelay {
		t.Fatalf("expected host delay mechanism %q, got %q",
			cudadiag.MechanismHostSubmissionDelay, res.HostDelayDiagnosis.Mechanism)
	}
	if !res.HostDelayDiagnosis.Admitted {
		t.Fatalf("expected host delay diagnosis to be admitted, reason: %s", res.HostDelayDiagnosis.WithheldReason)
	}
	if res.HostDelayDiagnosis.PrimaryFinding.Rule != "host_submission_delay" {
		t.Fatalf("expected primary finding rule host_submission_delay, got %s", res.HostDelayDiagnosis.PrimaryFinding.Rule)
	}

	// 3. Check controlled trial replay
	if res.TrialComparison == nil {
		t.Fatal("expected trial comparison receipt to be populated")
	}
	if res.TrialComparison.Verdict != cudaprofilehistory.Pass && res.TrialComparison.Verdict != cudaprofilehistory.Unknown {
		t.Fatalf("expected trial verdict Pass, got %v (reason: %s)",
			res.TrialComparison.Verdict, res.TrialComparison.Reason)
	}
	if res.TrialComparison.RegressionDetected {
		t.Fatal("expected regression not detected under intervention (candidate should improve speed)")
	}
	if res.TrialComparison.Ratio >= 1.0 {
		t.Fatalf("expected speedup ratio < 1.0, got %v", res.TrialComparison.Ratio)
	}

	// 4. Check negative control: device bottleneck
	if res.DeviceBottleneckDiag.Mechanism != cudadiag.MechanismDeviceBottleneck {
		t.Fatalf("expected device bottleneck mechanism %q, got %q",
			cudadiag.MechanismDeviceBottleneck, res.DeviceBottleneckDiag.Mechanism)
	}
	if !res.DeviceBottleneckDiag.Admitted {
		t.Fatalf("expected device bottleneck diagnosis to be admitted, reason: %s", res.DeviceBottleneckDiag.WithheldReason)
	}

	// 5. Check negative control: withheld diagnosis
	if res.WithheldDiagnosis.Mechanism != cudadiag.MechanismWithheld {
		t.Fatalf("expected withheld mechanism %q, got %q",
			cudadiag.MechanismWithheld, res.WithheldDiagnosis.Mechanism)
	}
	if res.WithheldDiagnosis.Admitted {
		t.Fatal("expected corrupt evidence diagnosis to NOT be admitted")
	}

	// 6. Check confusion matrix
	if res.ConfusionMatrix.Accuracy() < 0.90 {
		t.Fatalf("expected confusion matrix accuracy >= 0.90, got %v", res.ConfusionMatrix.Accuracy())
	}

	// 7. Check exported artifact files on disk
	requiredArtifacts := []string{
		"inputs",
		"ground_truth",
		"capture",
		"diagnosis",
		"trial_plan",
		"execution_receipt",
		"comparison_receipt",
		"summary",
	}
	for _, art := range requiredArtifacts {
		path, ok := res.ArtifactFiles[art]
		if !ok {
			t.Fatalf("missing artifact path for %q", art)
		}
		fi, err := os.Stat(path)
		if err != nil {
			t.Fatalf("failed to stat artifact file %s: %v", path, err)
		}
		if fi.Size() == 0 {
			t.Fatalf("artifact file %s is empty", path)
		}
	}
}

func TestDemonstration_ThreadSchedulingMatch(t *testing.T) {
	proc := cudaworkload.ProcessGenID{PID: 55555, Generation: 1}
	correctTID := uint64(1001)
	mismatchedTID := uint64(9999)

	wall := tracecontract.Interval{Start: 1000, End: 200000, Clock: "monotonic"}
	hostCalls := []cudadiag.HostCallEvent{
		{Span: tracecontract.Interval{Start: 1000, End: 60000, Clock: "monotonic"}, Name: "cudaLaunchKernel", HasReturn: true},
	}
	kernels := []cudadiag.KernelEvent{
		{Span: tracecontract.Interval{Start: 70000, End: 120000, Clock: "monotonic"}, Name: "gemm_kernel"},
	}

	thresh := cudadiag.DefaultThresholds()
	thresh.TargetTID = correctTID

	// Case 1: Matching TID
	schedMatch := cudadiag.SchedEvidence{
		Process:      proc,
		TID:          correctTID,
		ProbePresent: true,
		Intervals: []cudadiag.SchedInterval{
			{State: cudadiag.SchedRunnable, Span: tracecontract.Interval{Start: 1000, End: 50000, Clock: "monotonic"}, CPU: 1},
			{State: cudadiag.SchedRunning, Span: tracecontract.Interval{Start: 50000, End: 60000, Clock: "monotonic"}, CPU: 1},
		},
	}
	inMatch := CreateDiagnosticInput(wall, hostCalls, kernels, schedMatch, thresh)
	wl := cudaworkload.WorkloadIdentity{Process: proc, WorkloadName: "test-wl"}
	resMatch := cudadiag.Diagnose(wl, 1, inMatch)

	if !resMatch.Admitted {
		t.Fatalf("expected admitted finding for matching TID, got withheld: %s", resMatch.WithheldReason)
	}
	if resMatch.Mechanism != cudadiag.MechanismHostSubmissionDelay {
		t.Fatalf("expected MechanismHostSubmissionDelay, got %q", resMatch.Mechanism)
	}

	// Case 2: Mismatched TID
	schedMismatch := schedMatch
	schedMismatch.TID = mismatchedTID
	inMismatch := CreateDiagnosticInput(wall, hostCalls, kernels, schedMismatch, thresh)
	resMismatch := cudadiag.Diagnose(wl, 1, inMismatch)

	if resMismatch.Admitted {
		t.Fatal("expected diagnosis to be withheld for mismatched thread TID")
	}
	if resMismatch.Mechanism != cudadiag.MechanismWithheld {
		t.Fatalf("expected MechanismWithheld, got %q", resMismatch.Mechanism)
	}
	if resMismatch.WithheldReason != tracecontract.ReasonIdentityMismatch {
		t.Fatalf("expected withheld reason %s, got %s",
			tracecontract.ReasonIdentityMismatch, resMismatch.WithheldReason)
	}
}

func TestDemonstration_CorruptAndMissingEvidence(t *testing.T) {
	proc := cudaworkload.ProcessGenID{PID: 11111, Generation: 1}
	tid := uint64(1001)
	wall := tracecontract.Interval{Start: 1000, End: 200000, Clock: "monotonic"}
	kernels := []cudadiag.KernelEvent{
		{Span: tracecontract.Interval{Start: 70000, End: 120000, Clock: "monotonic"}, Name: "gemm_kernel"},
	}
	thresh := cudadiag.DefaultThresholds()
	thresh.TargetTID = tid
	wl := cudaworkload.WorkloadIdentity{Process: proc, WorkloadName: "test-wl"}

	// 1. Missing Return
	hostCallsNoReturn := []cudadiag.HostCallEvent{
		{Span: tracecontract.Interval{Start: 1000, End: 60000, Clock: "monotonic"}, Name: "cudaLaunchKernel", HasReturn: false},
	}
	sched := cudadiag.SchedEvidence{
		Process:      proc,
		TID:          tid,
		ProbePresent: true,
		Intervals: []cudadiag.SchedInterval{
			{State: cudadiag.SchedRunning, Span: tracecontract.Interval{Start: 1000, End: 60000, Clock: "monotonic"}, CPU: 1},
		},
	}
	inNoReturn := CreateDiagnosticInput(wall, hostCallsNoReturn, kernels, sched, thresh)
	resNoReturn := cudadiag.Diagnose(wl, 1, inNoReturn)
	if resNoReturn.Admitted {
		t.Fatal("expected missing return to withhold diagnosis")
	}
	if resNoReturn.WithheldReason != tracecontract.ReasonMissingReturn {
		t.Fatalf("expected reason %s, got %s", tracecontract.ReasonMissingReturn, resNoReturn.WithheldReason)
	}

	// 2. Missing Scheduler Probe
	schedNoProbe := sched
	schedNoProbe.ProbePresent = false
	hostCalls := []cudadiag.HostCallEvent{
		{Span: tracecontract.Interval{Start: 1000, End: 60000, Clock: "monotonic"}, Name: "cudaLaunchKernel", HasReturn: true},
	}
	inNoProbe := CreateDiagnosticInput(wall, hostCalls, kernels, schedNoProbe, thresh)
	inNoProbe.Receipt.ProbeCoverage["sched"] = tracecontract.ProbeCoverage{Present: false, Ratio: 0.0}
	resNoProbe := cudadiag.Diagnose(wl, 1, inNoProbe)
	if resNoProbe.Admitted {
		t.Fatal("expected missing scheduler probe to withhold diagnosis")
	}
	if resNoProbe.WithheldReason != tracecontract.ReasonProbeMissing {
		t.Fatalf("expected reason %s, got %s", tracecontract.ReasonProbeMissing, resNoProbe.WithheldReason)
	}

	// 3. Trace Record Loss
	inLoss := CreateDiagnosticInput(wall, hostCalls, kernels, sched, thresh)
	inLoss.Receipt.LostRecords["activity"] = tracecontract.Loss{Dropped: 10, Total: 100}
	resLoss := cudadiag.Diagnose(wl, 1, inLoss)
	if resLoss.Admitted {
		t.Fatal("expected record loss to withhold diagnosis")
	}
	if resLoss.WithheldReason != tracecontract.ReasonLossExceeded {
		t.Fatalf("expected reason %s, got %s", tracecontract.ReasonLossExceeded, resLoss.WithheldReason)
	}
}

func TestDemonstration_ResourcePressure(t *testing.T) {
	// Small ring buffer capacity to exercise FIFO eviction
	joiner := cudaworkload.NewJoiner(cudaworkload.JoinConfig{
		MaxPendingRecords: 4,
	})
	proc := cudaworkload.ProcessGenID{PID: 99999, Generation: 1}

	// Submit 10 host submissions with no device activities
	for i := 1; i <= 10; i++ {
		sub := cudaworkload.HostSubmission{
			Process:             proc,
			OpID:                cudaworkload.OpID(i),
			NativeCorrelationID: uint32(i),
			Symbol:              "gemm",
		}
		joiner.SubmitHost(sub)
	}

	// Check metrics: evicted records are recorded as MissingJoins
	metrics := joiner.Metrics()
	if metrics.MissingJoins == 0 {
		t.Fatal("expected missing joins due to FIFO buffer eviction under pressure")
	}
	if metrics.TotalSubmissions != 10 {
		t.Fatalf("expected 10 total submissions, got %d", metrics.TotalSubmissions)
	}
}
