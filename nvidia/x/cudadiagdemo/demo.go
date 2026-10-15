package cudadiagdemo

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// RunDemonstration orchestrates the complete end-to-end GPU submission delay
// investigation, diagnosis, and regression trial flow.
func RunDemonstration(ctx context.Context, cfg DemonstrationConfig) (*DemonstrationResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("cudadiagdemo: nil context")
	}
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
	if cfg.NumRequests > 512 || cfg.Concurrency > 64 || cfg.HostDelay > time.Second || cfg.DeviceDelay > time.Second {
		return nil, fmt.Errorf("cudadiagdemo: demonstration bounds exceeded")
	}
	startAll := time.Now()
	proc := cudaworkload.ProcessGenID{PID: uint32(os.Getpid()), Generation: 1}
	annotator := cudaworkload.NewAnnotator(proc, "gpudiagdemo-workload", cudaworkload.WithRunID("portable-service"), cudaworkload.WithCorrelationHook(NewCorrelationHookMock()))
	recConfig := cudaflightrecorder.Config{Window: 10 * time.Second, Capacity: 65536}
	rec, err := cudaflightrecorder.New(recConfig)
	if err != nil {
		return nil, err
	}
	if err := rec.Start(); err != nil {
		return nil, err
	}
	defer rec.Stop()
	factory := newInMemoryWriterFactory()
	budgets := cudaflightrecorder.QualificationBudgets()
	budgets.HysteresisCount = 1 // One declared trigger, no retry of a failed gate.
	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets, cudaflightrecorder.WithCaptureWriterFactory(factory), cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}))
	if err != nil {
		return nil, err
	}
	if err := coord.Arm(); err != nil {
		return nil, err
	}
	joiner := cudaworkload.NewJoiner(cudaworkload.JoinConfig{MaxPendingRecords: 8192})
	sf, err := NewServiceFixture(annotator, rec, coord, joiner, WithProcess(proc))
	if err != nil {
		return nil, err
	}
	result := &DemonstrationResult{Synthetic: true, Qualification: "portable synthetic-associated pre-entry replay; resource admission uses a synthetic sampler; scheduler cause, physical capture, collector off/on cost and native qualification unavailable", DiagnosticInputs: make(map[string]cudadiag.DiagnosticInput), CaptureSHA256: make(map[string]string), CaptureBundles: make(map[string][]byte)}
	cm := cudadiag.NewConfusionMatrix()
	refInput := []float64{1, 2.5, -3, 4.25, 0}
	// The service computes its own output; this reference helper is independent
	// of that production loop and checked again by the existing replay runner.
	expected := ComputeReferenceOutput(refInput, 2, 1)
	var allReqs []Request
	check := func(responses []Response) error {
		for _, response := range responses {
			if err := CheckNumericalCorrectness(expected, response.Output, 1e-6); err != nil {
				return err
			}
		}
		return nil
	}
	batch := func(prefix string, count int, mode Mode, delay time.Duration) ([]Response, error) {
		requests := make([]Request, count)
		for i := range requests {
			requests[i] = Request{ID: fmt.Sprintf("%s-%d", prefix, i), WorkloadName: "inference", Data: refInput, Mode: mode, Delay: delay}
		}
		allReqs = append(allReqs, requests...)
		responses, err := sf.BatchProcess(ctx, requests, cfg.Concurrency)
		if err == nil {
			err = check(responses)
		}
		return responses, err
	}
	baseResponses, err := batch("base", max(2, cfg.NumRequests/2), ModeBaseline, 0)
	if err != nil {
		return result, err
	}
	baseCapture, baseExpect, err := sf.captureTiming(ctx)
	if err != nil {
		return result, err
	}
	// Independent completed baseline/output checks are inputs to the decoder,
	// not claims copied from the captured declaration or injected delay.
	baseExpect.oraclePassed, baseExpect.baselinePassed = true, true
	var baseInputs []cudadiag.DiagnosticInput
	for _, response := range baseResponses {
		in, err := decodeTiming(baseCapture, baseExpect, response, cudadiag.DefaultThresholds())
		if err != nil {
			return result, err
		}
		baseInputs = append(baseInputs, in)
	}
	thresholds := baselineThresholds(baseInputs)
	store := func(name string, capture []byte, expect timingExpectation, response Response) cudadiag.DiagnosisResult {
		in, diagnosis := diagnoseTiming(capture, expect, response, thresholds)
		result.DiagnosticInputs[name], result.CaptureSHA256[name], result.CaptureBundles[name] = in, timingDigest(capture), append([]byte(nil), capture...)
		return diagnosis
	}
	result.BaselineDiagnosis = store("baseline", baseCapture, baseExpect, baseResponses[0])
	cm.Record(cudadiag.MechanismNoFault, result.BaselineDiagnosis.Mechanism, result.BaselineDiagnosis.WithheldReason)
	delayResponses, err := batch("host", max(2, cfg.NumRequests-cfg.NumRequests/2), ModeHostSubmissionDelay, cfg.HostDelay)
	if err != nil {
		return result, err
	}
	hostExpect, err := sf.attachTiming()
	if err != nil {
		return result, err
	}
	hostExpect.oraclePassed, hostExpect.baselinePassed = true, true
	observed := time.Duration(delayResponses[0].Observation.PreEntry.Duration())
	incidentReceipt, err := coord.Trigger(ctx, cudaflightrecorder.TriggerRequest{TenantID: "portable", Reason: "observed pre-entry latency", MetricName: "pre_entry_ns", MetricValue: float64(observed), Threshold: float64(thresholds.BaselinePreEntryDuration), DeviceOrdinal: 0})
	if err != nil {
		return result, err
	}
	result.IncidentReceipt = incidentReceipt
	hostCapture := factory.Bytes(incidentReceipt.IncidentID)
	result.HostDelayDiagnosis = store("host", hostCapture, hostExpect, delayResponses[0])
	cm.Record(cudadiag.MechanismHostSubmissionDelay, result.HostDelayDiagnosis.Mechanism, result.HostDelayDiagnosis.WithheldReason)

	// Keep the existing trial engine. Its closure contains the actual captured
	// bytes, independent input/reference, observer contract and current binary.
	if !cfg.SkipIntervention && result.HostDelayDiagnosis.Admitted && result.HostDelayDiagnosis.Mechanism == cudadiag.MechanismHostSubmissionDelay {
		dir := cfg.OutputDir
		if dir == "" {
			dir, err = os.MkdirTemp("", "gpudiagdemo-closure-")
			if err != nil {
				return result, err
			}
			defer os.RemoveAll(dir)
		} else if err := os.MkdirAll(dir, 0755); err != nil {
			return result, err
		}
		writeRef := func(name string, data []byte, role string) (cudaregress.ArtifactRef, error) {
			path := filepath.Join(dir, name)
			if err := os.WriteFile(path, data, 0644); err != nil {
				return cudaregress.ArtifactRef{}, err
			}
			return cudaregress.ArtifactRef{Name: name, Path: path, Digest: timingDigest(data), SizeBytes: int64(len(data)), Role: role}, nil
		}
		captureRef, err := writeRef("capture.tar", hostCapture, "trace")
		if err != nil {
			return result, err
		}
		inputBytes, err := json.Marshal(refInput)
		if err != nil {
			return result, err
		}
		inputRef, err := writeRef("replay-input.json", inputBytes, "input")
		if err != nil {
			return result, err
		}
		referenceBytes, err := json.Marshal(expected)
		if err != nil {
			return result, err
		}
		referenceRef, err := writeRef("reference.json", referenceBytes, "reference")
		if err != nil {
			return result, err
		}
		contractBytes, err := json.Marshal(sf.source)
		if err != nil {
			return result, err
		}
		contractRef, err := writeRef("timing-observer-contract.json", contractBytes, "source")
		if err != nil {
			return result, err
		}
		executable, err := os.Executable()
		if err != nil {
			return result, err
		}
		inputBinary, err := os.Open(executable)
		if err != nil {
			return result, err
		}
		info, err := inputBinary.Stat()
		if err != nil || info.Size() > 128<<20 {
			inputBinary.Close()
			return result, fmt.Errorf("cudadiagdemo: executing binary missing or exceeds closure limit")
		}
		outputBinary, err := os.CreateTemp(dir, "executing-host-binary-*")
		if err != nil {
			inputBinary.Close()
			return result, err
		}
		binaryPath := outputBinary.Name()
		_, copyErr := io.Copy(outputBinary, inputBinary)
		err = errors.Join(copyErr, inputBinary.Close(), outputBinary.Close())
		if err != nil {
			return result, err
		}
		digest, size, err := cudaregress.ComputeFileDigest(binaryPath)
		if err != nil {
			return result, err
		}
		binaryRef := cudaregress.ArtifactRef{Name: "executing-host-binary", Path: binaryPath, Digest: digest, SizeBytes: size, Role: "binary"}
		incident := cudaregress.Incident{ID: incidentReceipt.IncidentID, Admitted: result.HostDelayDiagnosis.Admitted, Diagnosis: result.HostDelayDiagnosis, PrimaryFinding: result.HostDelayDiagnosis.PrimaryFinding, Workload: result.HostDelayDiagnosis.Workload, CaptureReceiptRef: captureRef, ObservedEnvironment: cudaregress.Environment{Model: "portable synthetic service", Clock: portableClock}, CreatedAt: time.Now()}
		options := cudaregress.PlanOptions{WarmupRuns: 1, Rounds: 3, Order: "ABBA", ExecutionScope: cudaregress.TimingScope{Channel: cudatrial.TimingWallElapsed, Interval: "portable_service"}, Sources: []cudaregress.ArtifactRef{contractRef, captureRef}, Binaries: []cudaregress.ArtifactRef{binaryRef}, Inputs: []cudaregress.InputRef{{Name: inputRef.Name, Path: inputRef.Path, Digest: inputRef.Digest, ShapeDims: []int64{int64(len(refInput))}, NumericalMode: "float64_cpu"}}, Oracle: cudaregress.CorrectnessOracle{Method: "numerical-tolerance", Tolerance: 1e-6, ReferencePath: referenceRef.Path, ReferenceDigest: referenceRef.Digest}, StoppingRules: cudaregress.StoppingRules{MaxTrials: 10, RelativeThreshold: 0.10, MinSamples: 3, HaltOnDrift: false}}
		result.TrialPlan, result.PlanReceipt, err = FormulatePlan(incident, options)
		if err != nil {
			return result, err
		}
		sequence := 0
		type phaseTrial struct {
			diagnosis cudadiag.DiagnosisResult
			output    []float64
			round     int
		}
		var phaseBaseline, phaseCandidate *phaseTrial
		runner := cudaregress.ReplayFunc(func(c context.Context, arm string, round int) (cudatrial.Receipt, []float64, error) {
			sequence++
			request := Request{ID: fmt.Sprintf("replay-%d-%s-%d", sequence, arm, round), WorkloadName: "inference", Data: refInput, Mode: ModeHostSubmissionDelay, Delay: cfg.HostDelay}
			if arm == "candidate" {
				request.Mode, request.Delay = ModeBaseline, 0
			}
			allReqs = append(allReqs, request)
			started := time.Now()
			response, err := sf.Process(c, request)
			finished := time.Now()
			if err != nil {
				return cudatrial.Receipt{Err: err}, nil, err
			}
			if err := check([]Response{response}); err != nil {
				return cudatrial.Receipt{}, nil, err
			}
			capture, expect, err := sf.captureTiming(c)
			if err != nil {
				return cudatrial.Receipt{}, nil, err
			}
			expect.oraclePassed, expect.baselinePassed = true, true
			in, err := decodeTiming(capture, expect, response, thresholds)
			if err != nil {
				return cudatrial.Receipt{}, nil, err
			}
			got := cudadiag.Diagnose(response.Observation.Host.Workload, response.OpID, in)
			if !got.Admitted {
				return cudatrial.Receipt{}, nil, fmt.Errorf("cudadiagdemo: replay timing withheld: %s", got.WithheldReason)
			}
			if round >= 0 {
				pair := &phaseTrial{diagnosis: got, output: append([]float64(nil), response.Output...), round: round}
				if arm == "baseline" {
					phaseBaseline = pair
				} else if arm == "candidate" {
					phaseCandidate = pair
				}
			}
			result.ReplayTimings = append(result.ReplayTimings, response.Observation)
			name := fmt.Sprintf("trial-%d", sequence)
			result.DiagnosticInputs[name], result.CaptureSHA256[name], result.CaptureBundles[name] = in, timingDigest(capture), capture
			receipt := cudatrial.Receipt{Name: arm, Complete: true, Started: started, Finished: finished, WallElapsed: finished.Sub(started), WallNs: finished.Sub(started).Nanoseconds(), TimingKind: cudatrial.TimingWallElapsed}
			_ = receipt.RecordTiming(cudatrial.TimingWallElapsed, finished.Sub(started))
			return receipt, response.Output, nil
		})
		result.TrialExecution, result.TrialComparison, err = ExecuteReplay(ctx, result.TrialPlan, runner, cudaregress.ReplayOptions{CurrentEnvironment: incident.ObservedEnvironment, ReferenceValues: expected})
		if err != nil {
			return result, err
		}
		if phaseBaseline != nil && phaseCandidate != nil && phaseBaseline.round == phaseCandidate.round {
			plan := cudadiag.InterventionPlan{HypothesisID: string(cudadiag.MechanismHostSubmissionDelay), TargetWorkload: incident.Workload, ReferenceOutput: expected, CorrectnessOracle: "independent affine reference", InterventionType: "remove pre-entry application wait", Ordering: "one retained executed pair; full ABBA protocol in TrialExecution", StoppingRules: cudadiag.StoppingRules{MaxTrials: 2, MinEffectSize: 0.1, OracleTolerance: 1e-6}}
			verification, verificationErr := cudadiag.EvaluateIntervention(plan, phaseBaseline.diagnosis, phaseBaseline.output, phaseCandidate.diagnosis, phaseCandidate.output)
			result.PhaseVerification = &verification
			if verificationErr != nil {
				result.PhaseVerificationError = verificationErr.Error()
			}
		}
	}
	deviceResponses, err := batch("device", 1, ModeDeviceBottleneck, cfg.DeviceDelay)
	if err != nil {
		return result, err
	}
	deviceCapture, deviceExpect, err := sf.captureTiming(ctx)
	if err != nil {
		return result, err
	}
	deviceExpect.oraclePassed, deviceExpect.baselinePassed = true, true
	result.DeviceBottleneckDiag = store("device", deviceCapture, deviceExpect, deviceResponses[0])
	cm.Record(cudadiag.MechanismDeviceBottleneck, result.DeviceBottleneckDiag.Mechanism, result.DeviceBottleneckDiag.WithheldReason)
	if cfg.ExerciseCorruptEv {
		bundle, err := cudaflightrecorder.ReadBundle(bytes.NewReader(hostCapture))
		if err != nil {
			return result, err
		}
		delete(bundle.Files, timingFile)
		var corrupted bytes.Buffer
		if err := cudaflightrecorder.WriteBundle(&corrupted, bundle); err != nil {
			return result, err
		}
		result.WithheldDiagnosis = store("missing", corrupted.Bytes(), hostExpect, delayResponses[0])
		cm.Record(cudadiag.MechanismHostSubmissionDelay, result.WithheldDiagnosis.Mechanism, result.WithheldDiagnosis.WithheldReason)
	}
	if cfg.ExercisePressure {
		// Exercise the same bounded timing retention and capture/decoder path.
		sf.mu.Lock()
		sf.timingLimit = 1
		if len(sf.timings) > 1 {
			sf.timingDropped = addTimingLoss(sf.timingDropped, uint64(len(sf.timings)-1))
			sf.timings = sf.timings[:1]
		}
		sf.mu.Unlock()
		capture, expect, err := sf.captureTiming(ctx)
		if err != nil {
			return result, err
		}
		expect.oraclePassed, expect.baselinePassed = true, true
		diagnosis := store("pressure", capture, expect, delayResponses[0])
		result.PressureDiagnosis = &diagnosis
	}
	result.ConfusionMatrix = *cm
	result.AttributionMetrics = sf.joiner.Metrics()
	// Independent request logs score source identity after diagnosis. They are
	// never used to create timing, scheduler states or classification inputs.
	logs := sf.GroundTruthLogs()
	matched, scored := 0, 0
	for _, in := range result.DiagnosticInputs {
		if len(in.PreEntries) != 1 {
			continue
		}
		scored++
		sub := in.PreEntries[0].Submission
		for _, record := range logs {
			if record.OpID != sub.OpID || record.ProcessGen != sub.Process || !sameTimingWorkload(record.Workload, sub.Workload) {
				continue
			}
			for _, request := range allReqs {
				if request.ID == record.RequestID && request.WorkloadName == record.WorkloadName {
					matched++
					break
				}
			}
			break
		}
	}
	if scored != 0 {
		result.AttributionAccuracy = float64(matched) / float64(scored)
	}
	result.CollectionCost = CollectionCostReport{TotalSubmissions: result.AttributionMetrics.TotalSubmissions, TotalActivities: result.AttributionMetrics.TotalActivities, RingBufferCapacity: recConfig.Capacity, RetainedRecords: incidentReceipt.Perturbation.RecordsCaptured, CaptureBundleBytes: int64(len(hostCapture)), ElapsedCaptureDuration: time.Since(startAll), MeasurementStatus: "counts/bytes/service elapsed retained; record latency and collector-off/on cost unmeasured"}
	if cfg.OutputDir != "" {
		result.ArtifactFiles, err = ExportArtifacts(cfg.OutputDir, result, allReqs, logs, hostCapture)
		if err != nil {
			return result, err
		}
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
