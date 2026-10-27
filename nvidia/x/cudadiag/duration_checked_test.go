package cudadiag_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func durationControl(t *testing.T, length int64) (cudaworkload.WorkloadIdentity, cudadiag.DiagnosticInput) {
	t.Helper()
	process := cudaworkload.ProcessGenID{PID: 10, Generation: 1}
	workload := cudaworkload.WorkloadIdentity{Process: process, WorkloadName: "supplied-duration-control", RunID: "cpu-only-control"}
	span := func(a, b int64) tracecontract.Interval {
		return tracecontract.Interval{Start: a, End: b, Clock: "synthetic"}
	}
	in := cudadiag.DiagnosticInput{WallInterval: span(0, length+20), HostCalls: []cudadiag.HostCallEvent{{Span: span(1, 11), Name: "supplied API", HasReturn: true}}, Thresholds: cudadiag.DefaultThresholds(),
		Sched: cudadiag.SchedEvidence{Process: process, TID: 11, ProbePresent: true, Intervals: []cudadiag.SchedInterval{{State: cudadiag.SchedRunning, Span: span(1, 11)}}}}
	in.Thresholds.TargetTID = 11 // Independently selected synthetic target.
	for i := 0; i < 5; i++ {
		in.Kernels = append(in.Kernels, cudadiag.KernelEvent{Span: span(20, length+20), Name: "supplied kernel", StreamID: uint64(i + 1), ContextID: 1})
	}
	// This is retained constructed CPU-model input, not a native capture.
	payload, err := json.Marshal(struct {
		Wall    tracecontract.Interval
		Calls   []cudadiag.HostCallEvent
		Kernels []cudadiag.KernelEvent
		Sched   cudadiag.SchedEvidence
	}{in.WallInterval, in.HostCalls, in.Kernels, in.Sched})
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(payload)
	zero := int64(0)
	gates := make(map[string]tracecontract.Gate)
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		gates[name] = tracecontract.Gate{Disposition: "passed", Reason: "bounded known-input CPU control; not physical evidence", Provenance: "synthetic control"}
	}
	in.Receipt = tracecontract.Receipt{Version: 1, Command: "tiny supplied timing control", InputPath: "task:duration-input.json", InputSHA256: hex.EncodeToString(sum[:]), Kind: "capture", Backends: []tracecontract.Backend{{Name: "synthetic-control", Version: "1"}}, Gates: gates,
		ClockDomain: "synthetic", ClockProvenance: "single declared synthetic epoch", ClockUncertaintyNS: &zero, CaptureBounds: &tracecontract.CaptureBounds{Start: in.WallInterval.Start, End: in.WallInterval.End, Clock: "synthetic"}, ProbeCoverage: map[string]tracecontract.ProbeCoverage{"sched": {Present: true, Ratio: 1}}, LostRecords: map[string]tracecontract.Loss{"activity": {Total: 5}, "sched": {Total: 1}}}
	if _, err := in.Receipt.JSON(); err != nil {
		t.Fatal("constructed API/scheduler receipt invalid", err)
	}
	t.Logf("supplied CPU input: %s; receipt input SHA256: %s", payload, in.Receipt.InputSHA256)
	return workload, in
}

func TestCheckedDurationBoundaries(t *testing.T) {
	tests := []struct {
		name             string
		intervals        []tracecontract.Interval
		sum, union       time.Duration
		sumErr, unionErr bool
	}{
		{name: "empty"},
		{name: "zero and inverted", intervals: []tracecontract.Interval{{Start: 2, End: 2}, {Start: 3, End: 1}}},
		{name: "maximum", intervals: []tracecontract.Interval{{End: math.MaxInt64}}, sum: math.MaxInt64, union: math.MaxInt64},
		{name: "negative start fits", intervals: []tracecontract.Interval{{Start: -1, End: math.MaxInt64 - 1}}, sum: math.MaxInt64, union: math.MaxInt64},
		{name: "difference overflow", intervals: []tracecontract.Interval{{Start: -1, End: math.MaxInt64}}, sumErr: true, unionErr: true},
		{name: "full signed range", intervals: []tracecontract.Interval{{Start: math.MinInt64, End: math.MaxInt64}}, sumErr: true, unionErr: true},
		{name: "sum overflow but union fits", intervals: []tracecontract.Interval{{End: math.MaxInt64}, {End: 1}}, union: math.MaxInt64, sumErr: true},
		{name: "disjoint union overflow", intervals: []tracecontract.Interval{{Start: math.MinInt64, End: -1}, {Start: 0, End: 1}}, sumErr: true, unionErr: true},
		{name: "merged endpoints overflow", intervals: []tracecontract.Interval{{Start: math.MinInt64, End: -1}, {Start: -1, End: 0}}, sumErr: true, unionErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sum, err := cudadiag.SumDuration(tt.intervals)
			if (err != nil) != tt.sumErr || sum != tt.sum {
				t.Errorf("sum = %d, %v; want %d, error %v", sum, err, tt.sum, tt.sumErr)
			}
			union, err := cudadiag.UnionDuration(tt.intervals)
			if (err != nil) != tt.unionErr || union != tt.union {
				t.Errorf("union = %d, %v; want %d, error %v", union, err, tt.union, tt.unionErr)
			}
		})
	}
}

func TestCheckedDurationHugeOverlap(t *testing.T) {
	workload, in := durationControl(t, 1<<62)
	intervals := make([]tracecontract.Interval, 0, len(in.Kernels))
	for _, k := range in.Kernels {
		intervals = append(intervals, k.Span)
	}
	if d, err := cudadiag.SumDuration(intervals); err == nil || d != 0 {
		t.Errorf("overflow sum = %d, %v; want zero and error", d, err)
	}
	if d, err := cudadiag.UnionDuration(intervals); err != nil || d != 1<<62 {
		t.Errorf("representable union = %d, %v", d, err)
	}
	ta, err := cudadiag.ComputeTimeAccounting(in.WallInterval, in.HostCalls, in.Kernels, in.Sched)
	if err == nil || ta != (cudadiag.TimeAccounting{}) {
		t.Errorf("overflow accounting = %+v, %v; want zero and error", ta, err)
	}
	res := cudadiag.Diagnose(workload, 7, in)
	if res.Admitted || res.TimeAccounting != (cudadiag.TimeAccounting{}) || !strings.Contains(res.PrimaryFinding.Admission.Reason, "invalid time accounting") {
		t.Errorf("overflow diagnosis admitted or published accounting: %+v", res)
	}
	// A representable busy union and matched host duration survive independently.
	in.Receipt.ProbeCoverage["cupti_activity"] = tracecontract.ProbeCoverage{Present: true, Ratio: 1}
	findings := cudadiag.Observe(in)
	var host, busy bool
	for _, f := range findings {
		if f.Rule == "host_api_duration" && f.Value != nil && *f.Value == 10 && f.Admission.Admitted {
			host = true
		}
		if f.Rule == "device_busy_duration" && f.Value != nil && *f.Value == float64(int64(1)<<62) && f.Admission.Admitted {
			busy = true
		}
	}
	if !host || !busy {
		t.Errorf("independent observations lost: host=%v busy=%v findings=%+v", host, busy, findings)
	}
}

func TestCheckedDurationNormalDiagnosis(t *testing.T) {
	workload, in := durationControl(t, 100)
	ta, err := cudadiag.ComputeTimeAccounting(in.WallInterval, in.HostCalls, in.Kernels, in.Sched)
	if err != nil || ta.SummedKernelDuration != 500 || ta.BusyDuration != 100 || ta.HostAPIDuration != 10 || ta.ScheduledDuration != 10 {
		t.Fatalf("normal accounting = %+v, %v", ta, err)
	}
	if ta.SummedKernelDuration <= time.Duration(in.WallInterval.Duration()) {
		t.Fatal("legitimate overlap sum was constrained to wall duration")
	}
	if res := cudadiag.Diagnose(workload, 7, in); !res.Admitted || res.Mechanism != cudadiag.MechanismNoFault {
		t.Fatalf("normal diagnosis changed: %+v", res)
	}
}

func TestCheckedDurationUnavailableUnion(t *testing.T) {
	in := cudadiag.DiagnosticInput{Kernels: []cudadiag.KernelEvent{{Span: tracecontract.Interval{Start: -1, End: math.MaxInt64, Clock: "synthetic"}}}}
	for _, f := range cudadiag.Observe(in) {
		if f.Rule == "device_busy_duration" && (f.Value != nil || f.Class != tracecontract.FindingUnavailable || f.Admission.Admitted) {
			t.Errorf("unrepresentable union published: %+v", f)
		}
	}
}

func TestCheckedDurationScheduler(t *testing.T) {
	for _, state := range []cudadiag.SchedState{cudadiag.SchedRunnable, cudadiag.SchedRunning, cudadiag.SchedBlocked} {
		t.Run(string(state), func(t *testing.T) {
			s := cudadiag.SchedEvidence{Intervals: []cudadiag.SchedInterval{{State: state, Span: tracecontract.Interval{End: math.MaxInt64}}}}
			r, run, w, err := s.Durations()
			if err != nil || r+run+w != math.MaxInt64 {
				t.Fatalf("boundary: %d %d %d %v", r, run, w, err)
			}
			s.Intervals = append(s.Intervals, cudadiag.SchedInterval{State: state, Span: tracecontract.Interval{End: 1}})
			r, run, w, err = s.Durations()
			if err == nil || r != 0 || run != 0 || w != 0 {
				t.Errorf("scheduler overflow: %d %d %d %v", r, run, w, err)
			}
			if ta, err := cudadiag.ComputeTimeAccounting(tracecontract.Interval{}, nil, nil, s); err == nil || ta != (cudadiag.TimeAccounting{}) {
				t.Errorf("scheduler accounting: %+v %v", ta, err)
			}
			s.Intervals = []cudadiag.SchedInterval{{State: state, Span: tracecontract.Interval{Start: -1, End: math.MaxInt64}}}
			if _, _, _, err := s.Durations(); err == nil {
				t.Error("scheduler endpoint overflow accepted")
			}
		})
	}
	s := cudadiag.SchedEvidence{Intervals: []cudadiag.SchedInterval{{State: cudadiag.SchedRunning, Span: tracecontract.Interval{Start: 2, End: 1}}, {State: cudadiag.SchedRunning, Span: tracecontract.Interval{Start: 1, End: 1}}}}
	if r, run, w, err := s.Durations(); err != nil || r != 0 || run != 0 || w != 0 {
		t.Fatalf("empty/inverted scheduler semantics: %d %d %d %v", r, run, w, err)
	}
}
