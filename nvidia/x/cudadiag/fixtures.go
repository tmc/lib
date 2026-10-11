package cudadiag

import (
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// InjectionType specifies the synthetic fault injection kind.
type InjectionType string

const (
	// InjectionNone represents an unperturbed baseline control run.
	InjectionNone InjectionType = "none"
	// InjectionHostDelay injects host submission delay / CPU scheduling starvation.
	InjectionHostDelay InjectionType = "host_delay"
	// InjectionDeviceBottleneck injects competing device kernel contention.
	InjectionDeviceBottleneck InjectionType = "device_bottleneck"
	// InjectionAmbiguousSimultaneous injects both host delay and device bottleneck concurrently.
	InjectionAmbiguousSimultaneous InjectionType = "ambiguous_simultaneous"
	// InjectionMissingReturn injects an unreturned host runtime/uprobe call.
	InjectionMissingReturn InjectionType = "missing_return"
	// InjectionMissingScheduler injects missing scheduler probe evidence.
	InjectionMissingScheduler InjectionType = "missing_scheduler"
	// InjectionClockUncertainty injects cross-domain clock uncertainty exceeding threshold.
	InjectionClockUncertainty InjectionType = "clock_uncertainty"
	// InjectionRecordLoss injects trace record loss exceeding threshold.
	InjectionRecordLoss InjectionType = "record_loss"
)

// SyntheticFixture encapsulates a deterministic test scenario, keeping the
// ground-truth injection label strictly decoupled from the input supplied to Diagnose.
type SyntheticFixture struct {
	Injection        InjectionType
	ExpectedOutcome  MechanismType
	ExpectedReason   string
	Workload         cudaworkload.WorkloadIdentity
	OpID             cudaworkload.OpID
	Input            DiagnosticInput
	ControlReference []float64
	TestOutput       []float64
}

// BuildSyntheticFixture creates a deterministic synthetic fixture for testing.
func BuildSyntheticFixture(inj InjectionType) SyntheticFixture {
	proc := cudaworkload.ProcessGenID{PID: 12345, Generation: 1}
	tid := uint64(12345)
	workload := cudaworkload.WorkloadIdentity{
		Process:      proc,
		WorkloadName: "synthetic_gemm",
		RunID:        "run_001",
	}
	opID := cudaworkload.OpID(42)

	unc := int64(10)
	uncF := 10.0
	gates := make(map[string]tracecontract.Gate)
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		gates[name] = tracecontract.Gate{Disposition: "passed", Reason: "qualified fixture", Provenance: "test harness"}
	}

	baseReceipt := tracecontract.Receipt{
		Version:            1,
		Command:            "cudadiag test",
		InputPath:          "test.capture",
		InputSHA256:        strings.Repeat("a", 64),
		Kind:               "capture",
		ClockDomain:        "monotonic",
		ClockProvenance:    "cupti_timestamp",
		ClockUncertaintyNS: &unc,
		Uncertainty:        &uncF,
		Gates:              gates,
		Backends: []tracecontract.Backend{
			{Name: "native", Version: "12.6"},
		},
		ProbeCoverage: map[string]tracecontract.ProbeCoverage{
			"cupti_activity": {Present: true, Ratio: 1.0},
			"sched":          {Present: true, Ratio: 1.0},
		},
		LostRecords: map[string]tracecontract.Loss{
			"activity": {Dropped: 0, Total: 100, Unknown: false},
			"sched":    {Dropped: 0, Total: 50, Unknown: false},
		},
	}

	wall := tracecontract.Interval{Start: 1000, End: 200000, Clock: "monotonic"}
	thresh := DefaultThresholds()
	thresh.TargetTID = tid

	// Default baseline parameters
	hostDuration := 10 * time.Microsecond
	kernelDuration := 80 * time.Microsecond
	runnableDuration := 1 * time.Microsecond
	runningDuration := 9 * time.Microsecond
	waitDuration := 0 * time.Microsecond
	hasReturn := true
	schedProbePresent := true
	schedTID := tid

	expectedOutcome := MechanismNoFault
	expectedReason := ""

	switch inj {
	case InjectionNone:
		expectedOutcome = MechanismNoFault

	case InjectionHostDelay:
		// Host submission delayed by CPU runqueue starvation (runnable = 55us)
		hostDuration = 70 * time.Microsecond
		runnableDuration = 55 * time.Microsecond
		runningDuration = 15 * time.Microsecond
		expectedOutcome = MechanismHostSubmissionDelay

	case InjectionDeviceBottleneck:
		// Device saturated by competing kernels (kernel = 250us > baseline 100us * 2)
		kernelDuration = 250 * time.Microsecond
		wall.End = 300000
		expectedOutcome = MechanismDeviceBottleneck

	case InjectionAmbiguousSimultaneous:
		// Both host submission delayed and device saturated
		hostDuration = 70 * time.Microsecond
		runnableDuration = 55 * time.Microsecond
		runningDuration = 15 * time.Microsecond
		kernelDuration = 250 * time.Microsecond
		wall.End = 350000
		expectedOutcome = MechanismAmbiguousSimultaneous

	case InjectionMissingReturn:
		hasReturn = false
		expectedOutcome = MechanismWithheld
		expectedReason = tracecontract.ReasonMissingReturn

	case InjectionMissingScheduler:
		schedProbePresent = false
		expectedOutcome = MechanismWithheld
		expectedReason = tracecontract.ReasonProbeMissing

	case InjectionClockUncertainty:
		badUnc := int64(500) // Exceeds threshold 50ns
		baseReceipt.ClockUncertaintyNS = &badUnc
		expectedOutcome = MechanismWithheld
		expectedReason = tracecontract.ReasonClockUncertaintyExceeded

	case InjectionRecordLoss:
		baseReceipt.LostRecords["activity"] = tracecontract.Loss{
			Dropped: 10,
			Total:   100,
			Unknown: false,
		}
		expectedOutcome = MechanismWithheld
		expectedReason = tracecontract.ReasonLossExceeded
	}

	hostStart := int64(2000)
	hostEnd := hostStart + hostDuration.Nanoseconds()
	hostCalls := []HostCallEvent{
		{
			Span:      tracecontract.Interval{Start: hostStart, End: hostEnd, Clock: "monotonic"},
			Name:      "cudaLaunchKernel",
			HasReturn: hasReturn,
		},
	}

	kernelStart := hostEnd + 1000 // launches onto device after host returns
	kernelEnd := kernelStart + kernelDuration.Nanoseconds()
	kernels := []KernelEvent{
		{
			Span:      tracecontract.Interval{Start: kernelStart, End: kernelEnd, Clock: "monotonic"},
			Name:      "gemm_kernel",
			StreamID:  1,
			ContextID: 1,
			DeviceID:  0,
		},
	}

	schedIntervals := []SchedInterval{
		{
			State: SchedRunning,
			Span:  tracecontract.Interval{Start: hostStart, End: hostStart + runningDuration.Nanoseconds(), Clock: "monotonic"},
			CPU:   0,
		},
		{
			State: SchedRunnable,
			Span:  tracecontract.Interval{Start: hostStart + runningDuration.Nanoseconds(), End: hostStart + runningDuration.Nanoseconds() + runnableDuration.Nanoseconds(), Clock: "monotonic"},
			CPU:   0,
		},
	}
	if waitDuration > 0 {
		schedIntervals = append(schedIntervals, SchedInterval{
			State: SchedBlocked,
			Span:  tracecontract.Interval{Start: hostStart + runningDuration.Nanoseconds() + runnableDuration.Nanoseconds(), End: hostEnd, Clock: "monotonic"},
			CPU:   0,
		})
	}
	if !schedProbePresent {
		baseReceipt.ProbeCoverage["sched"] = tracecontract.ProbeCoverage{Present: false, Ratio: 0.0}
	}

	schedEvidence := SchedEvidence{
		Process:      proc,
		TID:          schedTID,
		ProbePresent: schedProbePresent,
		Loss:         0,
		UnknownLoss:  false,
		Intervals:    schedIntervals,
	}

	refOut := []float64{1.0, 2.0, 3.0, 4.0}
	testOut := []float64{1.0, 2.0, 3.0, 4.0}

	return SyntheticFixture{
		Injection:       inj,
		ExpectedOutcome: expectedOutcome,
		ExpectedReason:  expectedReason,
		Workload:        workload,
		OpID:            opID,
		Input: DiagnosticInput{
			WallInterval: wall,
			HostCalls:    hostCalls,
			Kernels:      kernels,
			Sched:        schedEvidence,
			Receipt:      baseReceipt,
			Thresholds:   thresh,
		},
		ControlReference: refOut,
		TestOutput:       testOut,
	}
}
