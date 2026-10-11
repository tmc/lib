package cudadiagdemo

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// RunDemonstration orchestrates the complete end-to-end GPU submission delay
// investigation, diagnosis, and regression trial flow.
func RunDemonstration(ctx context.Context, cfg DemonstrationConfig) (*DemonstrationResult, error) {
	if cfg.NumRequests <= 0 {
		cfg.NumRequests = 16
	}
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 4
	}
	if cfg.HostDelay <= 0 {
		cfg.HostDelay = 15 * time.Millisecond
	}
	if cfg.DeviceDelay <= 0 {
		cfg.DeviceDelay = 300 * time.Microsecond
	}

	startAll := time.Now()

	// 1. Initialize workload attribution and flight recording components
	proc := cudaworkload.ProcessGenID{PID: 42000, Generation: 1}
	mockHook := NewCorrelationHookMock()
	annotator := cudaworkload.NewAnnotator(proc, "gpudiagdemo-workload",
		cudaworkload.WithRunID("run-demo-01"),
		cudaworkload.WithCorrelationHook(mockHook),
	)

	recConfig := cudaflightrecorder.Config{
		Window:   10 * time.Second,
		Capacity: 65536,
	}
	rec, err := cudaflightrecorder.New(recConfig)
	if err != nil {
		return nil, fmt.Errorf("cudadiagdemo: init recorder: %w", err)
	}
	if err := rec.Start(); err != nil {
		return nil, fmt.Errorf("cudadiagdemo: start recorder: %w", err)
	}
	defer rec.Stop()

	bufFact := newInMemoryWriterFactory()
	budgets := cudaflightrecorder.QualificationBudgets()
	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets,
		cudaflightrecorder.WithCaptureWriterFactory(bufFact),
		cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}),
	)
	if err != nil {
		return nil, fmt.Errorf("cudadiagdemo: init coordinator: %w", err)
	}
	if err := coord.Arm(); err != nil {
		return nil, fmt.Errorf("cudadiagdemo: arm coordinator: %w", err)
	}

	joiner := cudaworkload.NewJoiner(cudaworkload.JoinConfig{
		MaxPendingRecords: 8192,
	})

	sf, err := NewServiceFixture(annotator, rec, coord, joiner, WithProcess(proc))
	if err != nil {
		return nil, fmt.Errorf("cudadiagdemo: init fixture: %w", err)
	}

	cm := cudadiag.NewConfusionMatrix()
	thresh := cudadiag.DefaultThresholds()
	thresh.BaselineKernelDuration = 100 * time.Microsecond
	thresh.BaselineHostAPIDuration = 20 * time.Microsecond
	thresh.HostDelayRatio = 2.0
	thresh.DeviceBottleneckRatio = 2.0

	var allReqs []Request
	refInput := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	// -------------------------------------------------------------
	// Stage 1: Baseline Execution (Ordinary Mode)
	// -------------------------------------------------------------
	baselineCount := cfg.NumRequests / 2
	if baselineCount < 2 {
		baselineCount = 2
	}
	var baselineReqs []Request
	for i := 0; i < baselineCount; i++ {
		baselineReqs = append(baselineReqs, Request{
			ID:           fmt.Sprintf("base-%03d", i+1),
			WorkloadName: "inference-baseline",
			Data:         refInput,
			Mode:         ModeBaseline,
		})
	}
	allReqs = append(allReqs, baselineReqs...)

	baseResps, err := sf.BatchProcess(ctx, baselineReqs, cfg.Concurrency)
	if err != nil {
		return nil, fmt.Errorf("cudadiagdemo: baseline batch: %w", err)
	}

	// Verify numerical correctness
	expectedRef := ComputeReferenceOutput(refInput, 2.0, 1.0)
	for _, r := range baseResps {
		if err := CheckNumericalCorrectness(expectedRef, r.Output, 1e-6); err != nil {
			return nil, fmt.Errorf("cudadiagdemo: baseline numerical check: %w", err)
		}
	}

	// Diagnose baseline
	baseOpID := baseResps[0].OpID
	baseTID := baseResps[0].GroundTruth.TID
	wallBase := tracecontract.Interval{Start: 1000, End: 150000, Clock: "monotonic"}
	hostCallsBase := []cudadiag.HostCallEvent{
		{Span: tracecontract.Interval{Start: 2000, End: 15000, Clock: "monotonic"}, Name: "cudaLaunchKernel", HasReturn: true},
	}
	kernelsBase := []cudadiag.KernelEvent{
		{Span: tracecontract.Interval{Start: 20000, End: 80000, Clock: "monotonic"}, Name: "gemm_kernel", StreamID: 1},
	}
	schedBase := cudadiag.SchedEvidence{
		Process:      proc,
		TID:          baseTID,
		ProbePresent: true,
		Intervals: []cudadiag.SchedInterval{
			{State: cudadiag.SchedRunning, Span: tracecontract.Interval{Start: 1000, End: 15000, Clock: "monotonic"}, CPU: 1},
		},
	}
	threshBase := thresh
	threshBase.TargetTID = baseTID

	inBase := CreateDiagnosticInput(wallBase, hostCallsBase, kernelsBase, schedBase, threshBase)
	baseWorkload := cudaworkload.WorkloadIdentity{
		Process:      proc,
		WorkloadName: "inference-baseline",
		RunID:        "run-demo-01",
	}
	resBaseline := cudadiag.Diagnose(baseWorkload, baseOpID, inBase)
	cm.Record(cudadiag.MechanismNoFault, resBaseline.Mechanism, resBaseline.WithheldReason)

	// -------------------------------------------------------------
	// Stage 2: Host Submission Delay (Incident Simulation)
	// -------------------------------------------------------------
	delayCount := cfg.NumRequests - baselineCount
	if delayCount < 2 {
		delayCount = 2
	}
	var delayReqs []Request
	for i := 0; i < delayCount; i++ {
		// Exercise concurrent same-name requests
		delayReqs = append(delayReqs, Request{
			ID:           fmt.Sprintf("delay-%03d", i+1),
			WorkloadName: "inference-concurrent-workload",
			Data:         refInput,
			Mode:         ModeHostSubmissionDelay,
			Delay:        cfg.HostDelay,
		})
	}
	allReqs = append(allReqs, delayReqs...)

	delayResps, err := sf.BatchProcess(ctx, delayReqs, cfg.Concurrency)
	if err != nil {
		return nil, fmt.Errorf("cudadiagdemo: delay batch: %w", err)
	}

	// Trigger bounded incident escalation on latency regression
	triggerReq := cudaflightrecorder.TriggerRequest{
		TenantID:      "tenant-alpha",
		Reason:        "latency_spike_queue_delay",
		MetricName:    "submission_latency_ms",
		MetricValue:   float64(cfg.HostDelay.Milliseconds()),
		Threshold:     5.0,
		DeviceOrdinal: 0,
		Details: map[string]string{
			"workload": "inference-concurrent-workload",
		},
	}
	incReceipt, incErr := coord.Trigger(ctx, triggerReq)
	if incErr != nil {
		// Try trigger with hysteresis satisfied
		incReceipt, incErr = coord.Trigger(ctx, triggerReq)
		if incErr != nil {
			return nil, fmt.Errorf("cudadiagdemo: escalation trigger: %w", incErr)
		}
	}

	// Retrieve captured snapshot TAR bytes
	var captureBytes []byte
	if incReceipt != nil {
		captureBytes = bufFact.Bytes(incReceipt.IncidentID)
	}

	// Diagnose host submission delay
	targetResp := delayResps[0]
	targetOpID := targetResp.OpID
	targetTID := targetResp.GroundTruth.TID
	hostDelayNs := cfg.HostDelay.Nanoseconds()

	wallDelay := tracecontract.Interval{Start: 1000, End: 1000 + hostDelayNs + 200000, Clock: "monotonic"}
	hostCallsDelay := []cudadiag.HostCallEvent{
		{Span: tracecontract.Interval{Start: 1000, End: 1000 + hostDelayNs + 15000, Clock: "monotonic"}, Name: "cudaLaunchKernel", HasReturn: true},
	}
	kernelsDelay := []cudadiag.KernelEvent{
		{Span: tracecontract.Interval{Start: 1000 + hostDelayNs + 20000, End: 1000 + hostDelayNs + 80000, Clock: "monotonic"}, Name: "gemm_kernel", StreamID: 1},
	}
	schedDelay := cudadiag.SchedEvidence{
		Process:      proc,
		TID:          targetTID,
		ProbePresent: true,
		Intervals: []cudadiag.SchedInterval{
			{State: cudadiag.SchedRunnable, Span: tracecontract.Interval{Start: 1000, End: 1000 + hostDelayNs, Clock: "monotonic"}, CPU: 1},
			{State: cudadiag.SchedRunning, Span: tracecontract.Interval{Start: 1000 + hostDelayNs, End: 1000 + hostDelayNs + 15000, Clock: "monotonic"}, CPU: 1},
		},
	}
	threshDelay := thresh
	threshDelay.TargetTID = targetTID

	inDelay := CreateDiagnosticInput(wallDelay, hostCallsDelay, kernelsDelay, schedDelay, threshDelay)
	delayWorkload := cudaworkload.WorkloadIdentity{
		Process:      proc,
		WorkloadName: "inference-concurrent-workload",
		RunID:        "run-demo-01",
	}
	resHostDelay := cudadiag.Diagnose(delayWorkload, targetOpID, inDelay)
	cm.Record(cudadiag.MechanismHostSubmissionDelay, resHostDelay.Mechanism, resHostDelay.WithheldReason)

	// -------------------------------------------------------------
	// Stage 3: Controlled Intervention Trial (cudaregress)
	// -------------------------------------------------------------
	var trialPlan *cudaregress.ExperimentPlan
	var planRec *cudaregress.PlanReceipt
	var trialExec *cudaregress.ExecutionReceipt
	var trialComp *cudaregress.ComparisonReceipt

	if !cfg.SkipIntervention {
		incident := cudaregress.Incident{
			ID:                  incReceipt.IncidentID,
			Admitted:            resHostDelay.Admitted,
			WithheldReason:      resHostDelay.WithheldReason,
			Diagnosis:           resHostDelay,
			PrimaryFinding:      resHostDelay.PrimaryFinding,
			Workload:            delayWorkload,
			ObservedEnvironment: cudaregress.Environment{DeviceIndex: 0, Model: "synthetic host fixture"},
			CreatedAt:           time.Now(),
		}

		planOpts := cudaregress.PlanOptions{
			WarmupRuns:     1,
			Rounds:         3,
			Order:          "ABBA",
			ExecutionScope: cudaregress.TimingScope{Channel: cudatrial.TimingWallElapsed},
			Oracle: cudaregress.CorrectnessOracle{
				Method:    "numerical-tolerance",
				Tolerance: 1e-6,
			},
			StoppingRules: cudaregress.StoppingRules{
				MaxTrials:         10,
				RelativeThreshold: 0.10,
				MinSamples:        3,
				HaltOnDrift:       false,
			},
		}

		p, pr, pErr := FormulatePlan(incident, planOpts)
		if pErr != nil {
			return nil, fmt.Errorf("cudadiagdemo: formulate plan: %w", pErr)
		}
		trialPlan = p
		planRec = pr

		// Replay runner: baseline injects delay, candidate removes delay
		runner := cudaregress.ReplayFunc(func(c context.Context, armName string, round int) (cudatrial.Receipt, []float64, error) {
			mode := ModeHostSubmissionDelay
			delay := cfg.HostDelay
			if armName == "candidate" {
				// Intervention removes host delay
				mode = ModeBaseline
				delay = 0
			}

			startArm := time.Now()
			req := Request{
				ID:           fmt.Sprintf("replay-%s-%d", armName, round),
				WorkloadName: "inference-replay",
				Data:         refInput,
				Mode:         mode,
				Delay:        delay,
			}
			resp, rErr := sf.Process(c, req)
			dur := time.Since(startArm)
			if rErr != nil {
				return cudatrial.Receipt{Complete: false, Err: rErr}, nil, rErr
			}

			rec := cudatrial.Receipt{
				Name:        armName,
				Complete:    true,
				Started:     startArm,
				Finished:    startArm.Add(dur),
				WallElapsed: dur,
				WallNs:      dur.Nanoseconds(),
				TimingKind:  cudatrial.TimingWallElapsed,
			}
			_ = rec.RecordTiming(cudatrial.TimingWallElapsed, dur)
			return rec, resp.Output, nil
		})

		replayOpts := cudaregress.ReplayOptions{
			CurrentEnvironment: incident.ObservedEnvironment,
			ReferenceValues:    expectedRef,
		}

		execRec, compRec, rErr := ExecuteReplay(ctx, trialPlan, runner, replayOpts)
		if rErr != nil {
			return nil, fmt.Errorf("cudadiagdemo: execute replay: %w", rErr)
		}
		trialExec = execRec
		trialComp = compRec
	}

	// -------------------------------------------------------------
	// Stage 4: Negative Control: Device Bottleneck
	// -------------------------------------------------------------
	devReq := Request{
		ID:           "neg-device-01",
		WorkloadName: "inference-device-bottleneck",
		Data:         refInput,
		Mode:         ModeDeviceBottleneck,
		Delay:        cfg.DeviceDelay,
	}
	devResp, err := sf.Process(ctx, devReq)
	if err != nil {
		return nil, fmt.Errorf("cudadiagdemo: device bottleneck request: %w", err)
	}

	wallDev := tracecontract.Interval{Start: 1000, End: 400000, Clock: "monotonic"}
	hostCallsDev := []cudadiag.HostCallEvent{
		{Span: tracecontract.Interval{Start: 1000, End: 15000, Clock: "monotonic"}, Name: "cudaLaunchKernel", HasReturn: true},
	}
	kernelsDev := []cudadiag.KernelEvent{
		{Span: tracecontract.Interval{Start: 20000, End: 20000 + cfg.DeviceDelay.Nanoseconds(), Clock: "monotonic"}, Name: "gemm_kernel", StreamID: 1},
	}
	schedDev := cudadiag.SchedEvidence{
		Process:      proc,
		TID:          devResp.GroundTruth.TID,
		ProbePresent: true,
		Intervals: []cudadiag.SchedInterval{
			{State: cudadiag.SchedRunning, Span: tracecontract.Interval{Start: 1000, End: 15000, Clock: "monotonic"}, CPU: 1},
		},
	}
	threshDev := thresh
	threshDev.TargetTID = devResp.GroundTruth.TID

	inDev := CreateDiagnosticInput(wallDev, hostCallsDev, kernelsDev, schedDev, threshDev)
	devWorkload := cudaworkload.WorkloadIdentity{
		Process:      proc,
		WorkloadName: "inference-device-bottleneck",
		RunID:        "run-demo-01",
	}
	resDevice := cudadiag.Diagnose(devWorkload, devResp.OpID, inDev)
	cm.Record(cudadiag.MechanismDeviceBottleneck, resDevice.Mechanism, resDevice.WithheldReason)

	// -------------------------------------------------------------
	// Stage 5: Negative Control: Missing / Corrupt Evidence Withholding
	// -------------------------------------------------------------
	var resWithheld cudadiag.DiagnosisResult
	if cfg.ExerciseCorruptEv {
		// Simulate missing return event on host call
		hostCallsCorrupt := []cudadiag.HostCallEvent{
			{Span: tracecontract.Interval{Start: 1000, End: 15000, Clock: "monotonic"}, Name: "cudaLaunchKernel", HasReturn: false},
		}
		inCorrupt := CreateDiagnosticInput(wallDelay, hostCallsCorrupt, kernelsDelay, schedDelay, threshDelay)
		resWithheld = cudadiag.Diagnose(delayWorkload, targetOpID, inCorrupt)
		cm.Record(cudadiag.MechanismHostSubmissionDelay, resWithheld.Mechanism, resWithheld.WithheldReason)

		// Also exercise mismatched scheduler thread TID
		schedMismatched := schedDelay
		schedMismatched.TID = 99999
		threshMismatched := threshDelay
		threshMismatched.TargetTID = targetTID // mismatch with sched.TID
		inMismatched := CreateDiagnosticInput(wallDelay, hostCallsDelay, kernelsDelay, schedMismatched, threshMismatched)
		resMismatched := cudadiag.Diagnose(delayWorkload, targetOpID, inMismatched)
		cm.Record(cudadiag.MechanismHostSubmissionDelay, resMismatched.Mechanism, resMismatched.WithheldReason)
	}

	// -------------------------------------------------------------
	// Stage 6: Metrics & Cost Accounting
	// -------------------------------------------------------------
	joinMetrics := sf.joiner.Metrics()
	totalRecords := 0
	if incReceipt != nil {
		totalRecords = incReceipt.Perturbation.RecordsCaptured
	}

	cost := CollectionCostReport{
		TotalSubmissions:       joinMetrics.TotalSubmissions,
		TotalActivities:        joinMetrics.TotalActivities,
		RingBufferCapacity:     recConfig.Capacity,
		RetainedRecords:        totalRecords,
		CaptureBundleBytes:     int64(len(captureBytes)),
		ElapsedCaptureDuration: time.Since(startAll),
	}

	result := &DemonstrationResult{
		Synthetic:            true,
		Qualification:        "partial: generated host/device/scheduler events; native capture and physical GPU intervention not measured",
		AttributionMetrics:   joinMetrics,
		AttributionAccuracy:  joinMetrics.Precision(),
		ConfusionMatrix:      *cm,
		BaselineDiagnosis:    resBaseline,
		HostDelayDiagnosis:   resHostDelay,
		DeviceBottleneckDiag: resDevice,
		WithheldDiagnosis:    resWithheld,
		IncidentReceipt:      incReceipt,
		TrialPlan:            trialPlan,
		PlanReceipt:          planRec,
		TrialExecution:       trialExec,
		TrialComparison:      trialComp,
		CollectionCost:       cost,
	}

	// -------------------------------------------------------------
	// Stage 7: Export Artifact Chain
	// -------------------------------------------------------------
	if cfg.OutputDir != "" {
		logs := sf.GroundTruthLogs()
		files, err := ExportArtifacts(cfg.OutputDir, result, allReqs, logs, captureBytes)
		if err != nil {
			return result, fmt.Errorf("cudadiagdemo: export artifacts: %w", err)
		}
		result.ArtifactFiles = files
	}

	return result, nil
}

type inMemoryWriterFactory struct {
	mu      sync.Mutex
	buffers map[string]*bytes.Buffer
}

func newInMemoryWriterFactory() *inMemoryWriterFactory {
	return &inMemoryWriterFactory{
		buffers: make(map[string]*bytes.Buffer),
	}
}

func (f *inMemoryWriterFactory) CreateWriter(incidentID string, tenantID string) (io.Writer, func() error, error) {
	buf := new(bytes.Buffer)
	f.mu.Lock()
	f.buffers[incidentID] = buf
	f.mu.Unlock()
	return buf, func() error { return nil }, nil
}

func (f *inMemoryWriterFactory) Bytes(incidentID string) []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	if b, ok := f.buffers[incidentID]; ok {
		return bytes.Clone(b.Bytes())
	}
	return nil
}
