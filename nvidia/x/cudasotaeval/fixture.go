package cudasotaeval

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"math"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// GroundTruthRequest describes a workload item submitted to the evaluation fixture.
type GroundTruthRequest struct {
	ID            string
	WorkloadName  string
	ProcessGen    cudaworkload.ProcessGenID
	TID           uint64
	StreamID      uint32
	CorrelationID uint32
	Fault         FaultLabel
	Resilience    ResilienceMode
	Data          []float64
	DelayDuration time.Duration
}

// FixtureRunResult holds ground truth and emitted telemetry from an evaluation run.
type FixtureRunResult struct {
	GroundTruth      []GroundTruthRecord
	HostSubmissions  []cudaworkload.HostSubmission
	DeviceActivities []cudaworkload.DeviceActivity
	DiagnosticInputs map[string]cudadiag.DiagnosticInput
	DroppedRecords   uint64
	TotalSubmissions uint64
	ElapsedDuration  time.Duration
}

// GroundTruthFixture provides an independent, known-answer execution simulator
// for attributing workloads and qualifying diagnostic utility under induced faults.
type GroundTruthFixture struct {
	mu sync.Mutex
}

// NewGroundTruthFixture constructs a new independent ground-truth evaluation fixture.
func NewGroundTruthFixture() *GroundTruthFixture {
	return &GroundTruthFixture{}
}

// ComputeNumericalOracle evaluates an independent IEEE 754 reference transformation:
// y_i = 3.0 * x_i^2 - 2.0 * x_i + 1.0, returning the transformed vector and SHA256 digest.
func ComputeNumericalOracle(data []float64) ([]float64, string) {
	out := make([]float64, len(data))
	h := sha256.New()
	for i, v := range data {
		res := 3.0*v*v - 2.0*v + 1.0
		out[i] = res
		bits := math.Float64bits(res)
		var b [8]byte
		b[0] = byte(bits)
		b[1] = byte(bits >> 8)
		b[2] = byte(bits >> 16)
		b[3] = byte(bits >> 24)
		b[4] = byte(bits >> 32)
		b[5] = byte(bits >> 40)
		b[6] = byte(bits >> 48)
		b[7] = byte(bits >> 56)
		h.Write(b[:])
	}
	return out, hex.EncodeToString(h.Sum(nil))
}

// Execute processes a batch of requests, simulating concurrency, telemetry emission,
// and held-out fault mechanisms while recording exact ground truth.
func (f *GroundTruthFixture) Execute(ctx context.Context, requests []GroundTruthRequest) (*FixtureRunResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	startTime := time.Now()
	res := &FixtureRunResult{
		DiagnosticInputs: make(map[string]cudadiag.DiagnosticInput),
	}

	baseWallTime := int64(1700000000000000000) // Deterministic reference nanosecond baseline

	for idx, req := range requests {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		res.TotalSubmissions++

		// 1. Independent numerical correctness
		_, outChecksum := ComputeNumericalOracle(req.Data)

		// 2. Base timings
		hostPrep := 20 * time.Microsecond
		hostAPI := 25 * time.Microsecond
		kernelDur := 150 * time.Microsecond

		// 3. Inject fault mechanisms
		switch req.Fault {
		case FaultHostSubmissionDelay:
			if req.DelayDuration > 0 {
				hostPrep = req.DelayDuration
			} else {
				hostPrep = 12 * time.Millisecond
			}
		case FaultDeviceBottleneck:
			if req.DelayDuration > 0 {
				kernelDur = req.DelayDuration
			} else {
				kernelDur = 8 * time.Millisecond
			}
		case FaultSyncStall:
			hostAPI = 5 * time.Millisecond
		case FaultCollectiveDelay:
			kernelDur = 10 * time.Millisecond
		case FaultNone:
			// nominal
		}

		totalDur := hostPrep + hostAPI + kernelDur

		reqStartNS := baseWallTime + int64(idx)*100000000
		prepEndNS := reqStartNS + hostPrep.Nanoseconds()
		apiEndNS := prepEndNS + hostAPI.Nanoseconds()
		kernelEndNS := apiEndNS + kernelDur.Nanoseconds()

		gtRecord := GroundTruthRecord{
			RequestID:       req.ID,
			WorkloadName:    req.WorkloadName,
			ProcessGen:      req.ProcessGen,
			TID:             req.TID,
			StreamID:        req.StreamID,
			CorrelationID:   req.CorrelationID,
			Fault:           req.Fault,
			Resilience:      req.Resilience,
			HostPrepDelay:   hostPrep,
			HostAPIDuration: hostAPI,
			KernelDuration:  kernelDur,
			TotalDuration:   totalDur,
			OutputHash:      outChecksum,
		}
		res.GroundTruth = append(res.GroundTruth, gtRecord)

		// 4. Construct telemetry events
		opID := cudaworkload.OpID(idx + 1)
		workload := cudaworkload.WorkloadIdentity{
			Process:      req.ProcessGen,
			WorkloadName: req.WorkloadName,
		}

		hostSub := cudaworkload.HostSubmission{
			Process:             req.ProcessGen,
			OpID:                opID,
			Workload:            workload,
			NativeCorrelationID: req.CorrelationID,
			Provenance:          cudaworkload.ProvenanceCUPTI,
			TID:                 req.TID,
			TimestampNs:         uint64(prepEndNS),
			Symbol:              "kernel_eval_transform",
		}

		devAct := cudaworkload.DeviceActivity{
			Process:               req.ProcessGen,
			NativeCorrelationID:   req.CorrelationID,
			ExternalCorrelationID: uint64(opID),
			Kind:                  "kernel",
			Name:                  "kernel_eval_transform",
			StartNs:               uint64(apiEndNS),
			EndNs:                 uint64(kernelEndNS),
			DeviceID:              0,
		}

		// Handle Resilience Modes
		switch req.Resilience {
		case ResilienceOverflow:
			// Simulate drop of device activity in ring buffer
			res.DroppedRecords++
			res.HostSubmissions = append(res.HostSubmissions, hostSub)
			// Device activity intentionally dropped to simulate overflow
			continue

		case ResilienceRestart:
			// Simulate process generation recycle
			restartedGen := cudaworkload.ProcessGenID{
				PID:        req.ProcessGen.PID,
				Generation: req.ProcessGen.Generation + 1,
			}
			hostSub.Process = restartedGen
			res.HostSubmissions = append(res.HostSubmissions, hostSub)
			res.DeviceActivities = append(res.DeviceActivities, devAct)

		case ResilienceLateAttach:
			// Simulate activity occurring before collector attachment (no correlation recorded)
			devAct.NativeCorrelationID = 0
			devAct.ExternalCorrelationID = 0
			res.HostSubmissions = append(res.HostSubmissions, hostSub)
			res.DeviceActivities = append(res.DeviceActivities, devAct)

		default:
			res.HostSubmissions = append(res.HostSubmissions, hostSub)
			res.DeviceActivities = append(res.DeviceActivities, devAct)
		}

		// 5. Construct DiagnosticInput for held-out evaluation
		wallInterval := tracecontract.Interval{
			Clock: "monotonic",
			Start: reqStartNS,
			End:   kernelEndNS,
		}

		hostCalls := []cudadiag.HostCallEvent{
			{
				Span:      tracecontract.Interval{Start: prepEndNS, End: apiEndNS, Clock: "monotonic"},
				Name:      "cuLaunchKernel",
				HasReturn: true,
			},
		}

		kernels := []cudadiag.KernelEvent{
			{
				Span:      tracecontract.Interval{Start: apiEndNS, End: kernelEndNS, Clock: "monotonic"},
				Name:      "kernel_eval_transform",
				StreamID:  uint64(req.StreamID),
				ContextID: 1,
				DeviceID:  0,
			},
		}

		sched := cudadiag.SchedEvidence{
			Process:      req.ProcessGen,
			TID:          req.TID,
			ProbePresent: true,
			Intervals: []cudadiag.SchedInterval{
				{
					State: cudadiag.SchedRunning,
					Span:  tracecontract.Interval{Start: reqStartNS, End: prepEndNS, Clock: "monotonic"},
					CPU:   0,
				},
			},
		}
		if req.Fault == FaultHostSubmissionDelay {
			sched.Intervals[0].State = cudadiag.SchedRunnable
		}

		thresh := cudadiag.DefaultThresholds()
		thresh.BaselineKernelDuration = 200 * time.Microsecond
		thresh.BaselineHostAPIDuration = 30 * time.Microsecond
		thresh.HostDelayRatio = 2.0
		thresh.DeviceBottleneckRatio = 2.0
		thresh.TargetTID = req.TID

		gates := make(map[string]tracecontract.Gate)
		for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
			gates[name] = tracecontract.Gate{Disposition: "passed", Reason: "ground-truth fixture", Provenance: "cudasotaeval"}
		}
		unc := int64(10)
		uncF := 10.0

		receipt := tracecontract.Receipt{
			Version:            1,
			Command:            "cudasotaeval",
			InputPath:          "fixture://ground_truth",
			InputSHA256:        outChecksum,
			Kind:               "capture",
			ClockDomain:        "monotonic",
			ClockProvenance:    "portable synthetic fixture timestamps and stipulated uncertainty",
			ClockUncertaintyNS: &unc,
			Uncertainty:        &uncF,
			Gates:              gates,
			Backends: []tracecontract.Backend{
				{Name: "portable_synthetic_fixture", Version: "1"},
			},
			ProbeCoverage: map[string]tracecontract.ProbeCoverage{
				"cupti_activity": {Present: true, Ratio: 1.0},
				"sched":          {Present: true, Ratio: 1.0},
			},
			LostRecords: map[string]tracecontract.Loss{
				"activity": {Dropped: res.DroppedRecords, Total: res.TotalSubmissions, Unknown: false},
				"sched":    {Dropped: 0, Total: res.TotalSubmissions, Unknown: false},
			},
		}

		diagInput := cudadiag.DiagnosticInput{
			WallInterval: wallInterval,
			HostCalls:    hostCalls,
			Kernels:      kernels,
			Sched:        sched,
			Receipt:      receipt,
			Thresholds:   thresh,
		}

		res.DiagnosticInputs[req.ID] = diagInput
	}

	res.ElapsedDuration = time.Since(startTime)
	return res, nil
}
