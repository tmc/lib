package cudadiag_test

import (
	"crypto/sha256"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func preEntryInput() (cudaworkload.WorkloadIdentity, cudadiag.DiagnosticInput) {
	workload := cudaworkload.WorkloadIdentity{Process: cudaworkload.ProcessGenID{PID: 42, Generation: 9}, RunID: "pre-entry-control", WorkloadName: "same-name"}
	span := func(a, b int64) tracecontract.Interval {
		return tracecontract.Interval{Start: a, End: b, Clock: "control"}
	}
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte("independent protocol timing bytes")))
	gates := make(map[string]tracecontract.Gate)
	for _, name := range []string{"collector", "import", "workload", "shape", "stationarity", "correctness"} {
		gates[name] = tracecontract.Gate{Disposition: "passed", Reason: "bounded synthetic protocol control", Provenance: "independent protocol observer"}
	}
	uncertainty := int64(0)
	in := cudadiag.DiagnosticInput{WallInterval: span(0, 1000), HostCalls: []cudadiag.HostCallEvent{{Span: span(400, 410), Name: "launch", HasReturn: true}}, Kernels: []cudadiag.KernelEvent{{Span: span(420, 470), Name: "synthetic"}}, PreEntries: []cudadiag.PreEntryEvent{{Span: span(10, 400), Complete: true, SourceClass: "synthetic", Submission: cudaworkload.HostSubmission{Process: workload.Process, Workload: workload, OpID: 7, TimestampNs: 400, TimestampClock: "control", TimestampEpoch: "one", Symbol: "launch", EvidenceRef: hash}}}, Thresholds: cudadiag.Thresholds{BaselinePreEntryDuration: 100, BaselineHostAPIDuration: 10, BaselineKernelDuration: 100, HostDelayRatio: 2, DeviceBottleneckRatio: 2}, Receipt: tracecontract.Receipt{Version: 1, Command: "pre-entry protocol", InputPath: "retained control", InputSHA256: hash, Kind: "capture", ClockDomain: "control", ClockProvenance: "synthetic epoch one", ClockUncertaintyNS: &uncertainty, CaptureBounds: &tracecontract.CaptureBounds{Start: 0, End: 1000, Clock: "control"}, Gates: gates, Backends: []tracecontract.Backend{{Name: "synthetic-control", Version: "1"}}, ProbeCoverage: map[string]tracecontract.ProbeCoverage{"application_timing": {Present: true, Ratio: 1}, "synthetic_activity": {Present: true, Ratio: 1}}, LostRecords: map[string]tracecontract.Loss{"activity": {Total: 1}, "pre_entry": {Total: 1}}}}
	return workload, in
}

func assertHostOnly(t *testing.T) {
	t.Helper()
	if driver.Loaded() || nvrtc.Loaded() {
		t.Fatal("native driver/compiler loaded by host control")
	}
	t.Cleanup(func() {
		if driver.Loaded() || nvrtc.Loaded() {
			t.Error("native driver/compiler loaded")
		}
	})
}

func TestPreEntryPublicAccountingAndNegative(t *testing.T) {
	assertHostOnly(t)
	for _, tc := range []struct {
		name             string
		start, kernelEnd int64
		want             cudadiag.MechanismType
	}{
		{"host", 10, 470, cudadiag.MechanismHostSubmissionDelay},
		{"ordinary", 390, 470, cudadiag.MechanismNoFault},
		{"measured-zero", 400, 470, cudadiag.MechanismNoFault},
		{"device-only", 390, 900, cudadiag.MechanismDeviceBottleneck},
		{"simultaneous", 10, 900, cudadiag.MechanismAmbiguousSimultaneous},
	} {
		t.Run(tc.name, func(t *testing.T) {
			workload, in := preEntryInput()
			in.PreEntries[0].Span.Start = tc.start
			in.Kernels[0].Span.End = tc.kernelEnd
			got := cudadiag.Diagnose(workload, 7, in)
			if !got.Admitted || got.Mechanism != tc.want {
				t.Fatalf("got %+v", got)
			}
			if got.TimeAccounting.PreEntryDuration == nil || *got.TimeAccounting.PreEntryDuration != time.Duration(400-tc.start) || got.TimeAccounting.HostAPIDuration != 10 || got.TimeAccounting.RunnableDuration != 0 || got.TimeAccounting.WaitDuration != 0 {
				t.Fatalf("phase accounting: %+v", got.TimeAccounting)
			}
			if got.PrimaryFinding.Class != tracecontract.FindingAssociated || got.HostTimingBasis != "pre_entry" {
				t.Fatalf("source/phase upgraded: %+v", got)
			}
			var unavailable bool
			for _, f := range got.Findings {
				if f.Rule == "scheduler_cause" && !f.Admission.Admitted {
					unavailable = true
				}
			}
			if !unavailable {
				t.Fatal("missing unavailable scheduler cause")
			}
			if tc.want == cudadiag.MechanismHostSubmissionDelay && got.PrimaryFinding.Rule != "host_pre_entry_delay" {
				t.Fatal("API rule used for pre-entry")
			}
		})
	}
}

func TestPreEntryAdmissionMutations(t *testing.T) {
	assertHostOnly(t)
	for _, tc := range []struct {
		name   string
		mutate func(*cudadiag.DiagnosticInput)
	}{
		{"incomplete", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].Complete = false }},
		{"return", func(in *cudadiag.DiagnosticInput) { in.HostCalls[0].HasReturn = false }},
		{"entry", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].Span.End-- }},
		{"negative", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].Span.Start = -1 }},
		{"inverted", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].Span.Start = 401 }},
		{"op", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].Submission.OpID++ }},
		{"process", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].Submission.Process.Generation++ }},
		{"run", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].Submission.Workload.RunID = "other" }},
		{"TID", func(in *cudadiag.DiagnosticInput) { in.Thresholds.TargetTID = 99 }},
		{"epoch", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].Submission.TimestampEpoch = "" }},
		{"clock", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].Span.Clock = "other" }},
		{"raw-digest", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].Submission.EvidenceRef = "" }},
		{"unknown-source", func(in *cudadiag.DiagnosticInput) { in.PreEntries[0].SourceClass = "modeled" }},
		{"unknown-loss", func(in *cudadiag.DiagnosticInput) {
			in.Receipt.LostRecords["pre_entry"] = tracecontract.Loss{Unknown: true}
		}},
		{"dropped", func(in *cudadiag.DiagnosticInput) {
			in.Receipt.LostRecords["pre_entry"] = tracecontract.Loss{Dropped: 1}
		}},
		{"bounds", func(in *cudadiag.DiagnosticInput) { in.Receipt.CaptureBounds.Start = 20 }},
		{"uncertainty", func(in *cudadiag.DiagnosticInput) { bad := int64(1000); in.Receipt.ClockUncertaintyNS = &bad }},
		{"duplicate", func(in *cudadiag.DiagnosticInput) { in.PreEntries = append(in.PreEntries, in.PreEntries[0]) }},
		{"no-baseline", func(in *cudadiag.DiagnosticInput) { in.Thresholds.BaselinePreEntryDuration = 0 }},
		{"overflow", func(in *cudadiag.DiagnosticInput) { in.Thresholds.BaselinePreEntryDuration = math.MaxInt64 }},
		{"negative-baseline", func(in *cudadiag.DiagnosticInput) { in.Thresholds.BaselinePreEntryDuration = -1 }},
		{"nan-ratio", func(in *cudadiag.DiagnosticInput) { in.Thresholds.HostDelayRatio = math.NaN() }},
		{"wrong-scheduler", func(in *cudadiag.DiagnosticInput) {
			in.Sched = cudadiag.SchedEvidence{ProbePresent: true, Process: in.PreEntries[0].Submission.Process, TID: 99}
		}},
		{"unprobed-scheduler", func(in *cudadiag.DiagnosticInput) {
			in.Sched.Intervals = []cudadiag.SchedInterval{{State: cudadiag.SchedRunnable, Span: in.PreEntries[0].Span}}
		}},
		{"missing-application-probe", func(in *cudadiag.DiagnosticInput) { delete(in.Receipt.ProbeCoverage, "application_timing") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			workload, in := preEntryInput()
			tc.mutate(&in)
			got := cudadiag.Diagnose(workload, 7, in)
			if got.Admitted || got.Mechanism != cudadiag.MechanismWithheld {
				t.Fatalf("mutation admitted: %+v", got)
			}
		})
	}
}

func TestPreEntryInterventionPhase(t *testing.T) {
	assertHostOnly(t)
	workload, control := preEntryInput()
	_, candidate := preEntryInput()
	candidate.PreEntries[0].Span.Start = 390
	a, b := cudadiag.Diagnose(workload, 7, control), cudadiag.Diagnose(workload, 7, candidate)
	plan := cudadiag.InterventionPlan{HypothesisID: string(cudadiag.MechanismHostSubmissionDelay), TargetWorkload: workload, ReferenceOutput: []float64{3, 6}, CorrectnessOracle: "independent affine reference", InterventionType: "remove application wait", Ordering: "paired", StoppingRules: cudadiag.StoppingRules{MaxTrials: 2, MinEffectSize: 0.5}}
	r, err := cudadiag.EvaluateIntervention(plan, a, []float64{3, 6}, b, []float64{3, 6})
	if err != nil || !r.Verified || r.TestedRule != "host_pre_entry_delay" {
		t.Fatalf("phase verification: %+v %v", r, err)
	}
	for _, name := range []string{"phase", "missing", "api-only", "wrong-output"} {
		t.Run(name, func(t *testing.T) {
			changed := b
			out := []float64{3, 6}
			switch name {
			case "phase":
				changed.HostTimingBasis = ""
			case "missing":
				changed.TimeAccounting.PreEntryDuration = nil
			case "api-only":
				changed = a
				changed.TimeAccounting.HostAPIDuration = 1
			case "wrong-output":
				out[1] = 7
			}
			if got, err := cudadiag.EvaluateIntervention(plan, a, []float64{3, 6}, changed, out); err == nil || got.Verified {
				t.Fatal("unrelated phase/output verified")
			}
		})
	}
}

func TestPreEntryInterventionMissingPhase(t *testing.T) {
	assertHostOnly(t)
	workload, in := preEntryInput()
	control := cudadiag.Diagnose(workload, 7, in)
	if !control.Admitted || control.TimeAccounting.PreEntryDuration == nil || control.PrimaryFinding.Rule != "host_pre_entry_delay" {
		t.Fatalf("pre-entry control: %+v", control)
	}
	plan := cudadiag.InterventionPlan{HypothesisID: string(cudadiag.MechanismHostSubmissionDelay), TargetWorkload: workload, ReferenceOutput: []float64{-5, 5}, CorrectnessOracle: "independent affine reference", InterventionType: "remove application wait", Ordering: "paired", StoppingRules: cudadiag.StoppingRules{MaxTrials: 2, MinEffectSize: 0.5}}
	for _, tc := range []struct {
		name   string
		mutate func(*cudadiag.DiagnosisResult, *cudadiag.DiagnosisResult)
	}{
		{"both-metrics", func(a, b *cudadiag.DiagnosisResult) {}},
		{"metrics-without-rule", func(a, b *cudadiag.DiagnosisResult) {
			a.PrimaryFinding.Rule, b.PrimaryFinding.Rule = "host_submission_delay", "no_fault"
		}},
		{"control-metric", func(a, b *cudadiag.DiagnosisResult) {
			b.TimeAccounting.PreEntryDuration = nil
			a.PrimaryFinding.Rule, b.PrimaryFinding.Rule = "host_submission_delay", "no_fault"
		}},
		{"intervention-metric", func(a, b *cudadiag.DiagnosisResult) {
			a.TimeAccounting.PreEntryDuration = nil
			a.PrimaryFinding.Rule, b.PrimaryFinding.Rule = "host_submission_delay", "no_fault"
		}},
		{"control-rule", func(a, b *cudadiag.DiagnosisResult) {
			a.TimeAccounting.PreEntryDuration, b.TimeAccounting.PreEntryDuration = nil, nil
			b.PrimaryFinding.Rule = "no_fault"
		}},
		{"intervention-rule", func(a, b *cudadiag.DiagnosisResult) {
			a.TimeAccounting.PreEntryDuration, b.TimeAccounting.PreEntryDuration = nil, nil
			a.PrimaryFinding.Rule = "host_submission_delay"
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, b := control, control
			a.HostTimingBasis, b.HostTimingBasis = "", ""
			b.OpID++
			b.TimeAccounting.HostAPIDuration = time.Nanosecond
			tc.mutate(&a, &b)
			got, err := cudadiag.EvaluateIntervention(plan, a, []float64{-5, 5}, b, []float64{-5, 5})
			if err == nil || got.Verified {
				t.Fatalf("missing phase verified an API-only improvement: %+v %v", got, err)
			}
		})
	}
}

func TestPreEntryInterventionLegacyCompatibility(t *testing.T) {
	assertHostOnly(t)
	fixture := cudadiag.BuildSyntheticFixture(cudadiag.InjectionHostDelay)
	control := cudadiag.Diagnose(fixture.Workload, fixture.OpID, fixture.Input)
	fixture.Input.HostCalls[0].Span = tracecontract.Interval{Start: 2000, End: 17000, Clock: "monotonic"}
	fixture.Input.Sched = cudadiag.BuildSyntheticFixture(cudadiag.InjectionNone).Input.Sched
	candidate := cudadiag.Diagnose(fixture.Workload, fixture.OpID+1, fixture.Input)
	if !control.Admitted || !candidate.Admitted || control.HostTimingBasis != "" || candidate.HostTimingBasis != "" || control.TimeAccounting.PreEntryDuration != nil || candidate.TimeAccounting.PreEntryDuration != nil {
		t.Fatalf("legacy evidence changed: control=%+v candidate=%+v", control, candidate)
	}
	plan := cudadiag.InterventionPlan{HypothesisID: string(cudadiag.MechanismHostSubmissionDelay), TargetWorkload: fixture.Workload, ReferenceOutput: []float64{-5, 5}, CorrectnessOracle: "independent affine reference", InterventionType: "reduce API duration", Ordering: "paired", StoppingRules: cudadiag.StoppingRules{MaxTrials: 2, MinEffectSize: 0.5}}
	got, err := cudadiag.EvaluateIntervention(plan, control, []float64{-5, 5}, candidate, []float64{-5, 5})
	if err != nil || !got.Verified || got.TestedRule != "host_submission_delay" {
		t.Fatalf("legacy API intervention rejected: %+v %v", got, err)
	}
}

func ExamplePreEntryEvent() {
	// Constructed replay timing; the API has its own entry-to-return span.
	event := cudadiag.PreEntryEvent{Span: tracecontract.Interval{Start: 10, End: 400, Clock: "replay"}, Complete: true, SourceClass: "synthetic"}
	fmt.Println(event.Span.Duration(), event.SourceClass)
	// Output: 390 synthetic
}
