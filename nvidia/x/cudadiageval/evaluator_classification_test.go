package cudadiageval

import (
	"context"
	"encoding/json"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// withhold makes the diagnostic input for id lose its only activity
// record, so cudadiag withholds the diagnosis.
func withhold(t *testing.T, res *FixtureRunResult, id string) {
	t.Helper()
	in, ok := res.DiagnosticInputs[id]
	if !ok {
		t.Fatalf("no diagnostic input %q", id)
	}
	lost := make(map[string]tracecontract.Loss, len(in.Receipt.LostRecords))
	for k, v := range in.Receipt.LostRecords {
		lost[k] = v
	}
	lost["activity"] = tracecontract.Loss{Dropped: 1, Total: 1}
	in.Receipt.LostRecords = lost
	res.DiagnosticInputs[id] = in
}

// TestCompetingRequiresExactMechanism checks that a competing device
// bottleneck case passes only when diagnosed as exactly
// cudadiag.MechanismDeviceBottleneck. diagnosticFixture has two competing
// cases, d0 and d1.
func TestCompetingRequiresExactMechanism(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(t *testing.T, res *FixtureRunResult)
		passed int
		want   string
	}{
		{"exact", func(*testing.T, *FixtureRunResult) {}, 2, "passed"},
		{"one withheld by loss", func(t *testing.T, res *FixtureRunResult) { withhold(t, res, "d0") }, 1, "failed"},
		{"both empty inputs", func(_ *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["d0"] = cudadiag.DiagnosticInput{}
			res.DiagnosticInputs["d1"] = cudadiag.DiagnosticInput{}
		}, 0, "failed"},
		{"classified no fault", func(_ *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["d0"] = res.DiagnosticInputs["n0"]
		}, 1, "failed"},
		{"classified host delay", func(_ *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["d1"] = res.DiagnosticInputs["h0"]
		}, 1, "failed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := diagnosticFixture(t)
			tt.mutate(t, res)
			r, gates := evaluateDiagnostics(res, DefaultBudgets())
			g := findGate(t, gates, "competing_mechanism_distinction")
			if r.CompetingTotal != 2 || r.CompetingDiverted != tt.passed || g.Disposition != tt.want {
				t.Errorf("competing %d/%d gate=%s (%s), want %d/2 %s", r.CompetingDiverted, r.CompetingTotal, g.Disposition, g.Reason, tt.passed, tt.want)
			}
			if tt.want != "passed" && r.GateDisposition == "passed" {
				t.Errorf("receipt disposition passed with competing gate %s", g.Disposition)
			}
		})
	}
}

// TestWithheldCompetingCasesDoNotPassDistinction is the independent review
// control: withheld competing diagnoses are not correct ones.
func TestWithheldCompetingCasesDoNotPassDistinction(t *testing.T) {
	res := diagnosticFixture(t)
	for _, id := range []string{"d0", "d1"} {
		res.DiagnosticInputs[id] = cudadiag.DiagnosticInput{}
	}
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	g := findGate(t, gates, "competing_mechanism_distinction")
	if g.Disposition == "passed" {
		t.Fatalf("all competing cases were withheld but distinction passed: diverted=%d total=%d gate=%+v", r.CompetingDiverted, r.CompetingTotal, g)
	}
}

// TestCompetingPartition checks that correct, withheld, missing and
// misclassified competing cases are counted separately and sum to the total.
func TestCompetingPartition(t *testing.T) {
	tests := []struct {
		name                                  string
		mutate                                func(t *testing.T, res *FixtureRunResult)
		correct, withheld, missing, incorrect int
		want                                  string
	}{
		{"exact", func(*testing.T, *FixtureRunResult) {}, 2, 0, 0, 0, "passed"},
		{"withheld", func(t *testing.T, res *FixtureRunResult) { withhold(t, res, "d0") }, 1, 1, 0, 0, "failed"},
		{"empty input", func(_ *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["d1"] = cudadiag.DiagnosticInput{}
		}, 1, 1, 0, 0, "failed"},
		{"missing", func(_ *testing.T, res *FixtureRunResult) { delete(res.DiagnosticInputs, "d0") }, 1, 0, 1, 0, "withheld"},
		{"no fault", func(_ *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["d0"] = res.DiagnosticInputs["n0"]
		}, 1, 0, 0, 1, "failed"},
		{"host delay", func(_ *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["d0"] = res.DiagnosticInputs["h0"]
		}, 1, 0, 0, 1, "failed"},
		{"ambiguous simultaneous", func(t *testing.T, res *FixtureRunResult) {
			f := cudadiag.BuildSyntheticFixture(cudadiag.InjectionAmbiguousSimultaneous)
			d := cudadiag.Diagnose(f.Workload, f.OpID, f.Input)
			if !d.Admitted || d.Mechanism != cudadiag.MechanismAmbiguousSimultaneous {
				t.Fatalf("ambiguous control is not admitted: %+v", d)
			}
			for i := range res.GroundTruth {
				if res.GroundTruth[i].RequestID == "d0" {
					res.GroundTruth[i].ProcessGen = f.Workload.Process
					res.GroundTruth[i].WorkloadName = f.Workload.WorkloadName
					res.GroundTruth[i].CorrelationID = uint32(f.OpID)
				}
			}
			res.DiagnosticInputs["d0"] = f.Input
		}, 1, 0, 0, 1, "failed"},
		{"withheld and missing", func(t *testing.T, res *FixtureRunResult) {
			withhold(t, res, "d0")
			delete(res.DiagnosticInputs, "d1")
		}, 0, 1, 1, 0, "failed"},
		{"withheld and misclassified", func(t *testing.T, res *FixtureRunResult) {
			withhold(t, res, "d0")
			res.DiagnosticInputs["d1"] = res.DiagnosticInputs["n1"]
		}, 0, 1, 0, 1, "failed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := diagnosticFixture(t)
			tt.mutate(t, res)
			r, gates := evaluateDiagnostics(res, DefaultBudgets())
			incorrect := r.CompetingTotal - r.CompetingDiverted - r.CompetingWithheld - r.CompetingMissing
			if r.CompetingTotal != 2 || r.CompetingDiverted != tt.correct || r.CompetingWithheld != tt.withheld || r.CompetingMissing != tt.missing || incorrect != tt.incorrect {
				t.Errorf("competing total=%d correct=%d withheld=%d missing=%d incorrect=%d, want 2 %d %d %d %d", r.CompetingTotal, r.CompetingDiverted, r.CompetingWithheld, r.CompetingMissing, incorrect, tt.correct, tt.withheld, tt.missing, tt.incorrect)
			}
			if g := findGate(t, gates, "competing_mechanism_distinction"); g.Disposition != tt.want {
				t.Errorf("gate = %s (%s), want %s", g.Disposition, g.Reason, tt.want)
			}
		})
	}
}

func TestCompetingWithheldJSON(t *testing.T) {
	b, err := json.Marshal(DiagnosticReceipt{CompetingTotal: 2, CompetingWithheld: 1})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"competing_withheld":1`) {
		t.Errorf("receipt %s lacks competing_withheld", b)
	}

	b, err = json.Marshal(DiagnosticReceipt{CompetingTotal: 2})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"competing_withheld":0`) {
		t.Errorf("zero receipt %s lacks competing_withheld", b)
	}
}

// editInput applies edit to the diagnostic input for id. The receipt maps
// and scheduler intervals are copied first, so the edit reaches no other case.
func editInput(t *testing.T, res *FixtureRunResult, id string, edit func(*cudadiag.DiagnosticInput)) {
	t.Helper()
	in, ok := res.DiagnosticInputs[id]
	if !ok {
		t.Fatalf("no diagnostic input %q", id)
	}
	in.Receipt.LostRecords = maps.Clone(in.Receipt.LostRecords)
	in.Receipt.ProbeCoverage = maps.Clone(in.Receipt.ProbeCoverage)
	in.Sched.Intervals = slices.Clone(in.Sched.Intervals)
	edit(&in)
	res.DiagnosticInputs[id] = in
}

// resilienceProc is the process generation resilienceRun submits under.
var resilienceProc = cudaworkload.ProcessGenID{PID: 50000, Generation: 1}

// dropTelemetry removes every host submission and device activity carrying
// the native correlation ID corr.
func dropTelemetry(res *FixtureRunResult, corr uint32) {
	res.HostSubmissions = slices.DeleteFunc(res.HostSubmissions, func(s cudaworkload.HostSubmission) bool { return s.NativeCorrelationID == corr })
	res.DeviceActivities = slices.DeleteFunc(res.DeviceActivities, func(a cudaworkload.DeviceActivity) bool { return a.NativeCorrelationID == corr })
}

// TestResiliencePartition checks each mode's predicate and that passed,
// withheld, missing and remaining resilience cases are counted separately
// and sum to the total. resilienceRun(allModes) gives r0 overflow
// (correlation 5000), r1 restart (5001) and r2 late attach (5002).
func TestResiliencePartition(t *testing.T) {
	restarted := cudaworkload.ProcessGenID{PID: resilienceProc.PID, Generation: resilienceProc.Generation + 1}
	unknownLoss := func(in *cudadiag.DiagnosticInput) {
		in.Receipt.LostRecords["activity"] = tracecontract.Loss{Unknown: true}
	}
	measuredLoss := func(in *cudadiag.DiagnosticInput) {
		in.Receipt.LostRecords["activity"] = tracecontract.Loss{Dropped: 0, Total: 1}
	}
	fullCoverage := func(in *cudadiag.DiagnosticInput) {
		in.Receipt.ProbeCoverage["cupti_activity"] = tracecontract.ProbeCoverage{Present: true, Ratio: 1}
	}
	tests := []struct {
		name                            string
		mutate                          func(t *testing.T, res *FixtureRunResult)
		passed, withheld, missing, rest int
		want                            string
	}{
		{"fixture evidence", func(*testing.T, *FixtureRunResult) {}, 3, 0, 0, 0, "passed"},

		{"overflow admitted", func(_ *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["r0"] = res.DiagnosticInputs["h0"]
		}, 2, 0, 0, 1, "failed"},
		{"overflow withheld without observed loss", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r0", measuredLoss)
		}, 2, 1, 0, 0, "failed"},
		{"overflow withheld for unknown loss", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r0", unknownLoss)
		}, 2, 1, 0, 0, "failed"},
		{"overflow missing", func(_ *testing.T, res *FixtureRunResult) { delete(res.DiagnosticInputs, "r0") }, 2, 0, 1, 0, "withheld"},

		{"restart admitted", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r1", func(in *cudadiag.DiagnosticInput) { in.Sched.Process = resilienceProc })
		}, 2, 0, 0, 1, "failed"},
		{"restart withheld for unknown loss", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r1", func(in *cudadiag.DiagnosticInput) { in.Sched.UnknownLoss = true })
		}, 2, 1, 0, 0, "failed"},
		{"restart withheld for activity loss", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r1", func(in *cudadiag.DiagnosticInput) {
				in.Sched.Process = resilienceProc
				unknownLoss(in)
			})
		}, 2, 1, 0, 0, "failed"},
		{"restart identity reason without observed generation", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r1", func(in *cudadiag.DiagnosticInput) {
				in.Sched.Process = resilienceProc
				in.Sched.Intervals[0].State = "unknown"
			})
		}, 2, 1, 0, 0, "failed"},
		{"restart generation wrapped to zero", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r1", func(in *cudadiag.DiagnosticInput) { in.Sched.Process.Generation = 0 })
		}, 2, 1, 0, 0, "failed"},
		{"restart other process", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r1", func(in *cudadiag.DiagnosticInput) { in.Sched.Process.PID++ })
		}, 2, 1, 0, 0, "failed"},
		{"restart joined exactly", func(_ *testing.T, res *FixtureRunResult) {
			for i, s := range res.HostSubmissions {
				if s.NativeCorrelationID == 5001 {
					res.HostSubmissions[i].Process = resilienceProc
				}
			}
		}, 2, 1, 0, 0, "failed"},
		{"restart joined exactly in new generation", func(_ *testing.T, res *FixtureRunResult) {
			for i, a := range res.DeviceActivities {
				if a.NativeCorrelationID == 5001 {
					res.DeviceActivities[i].Process = restarted
				}
			}
		}, 2, 1, 0, 0, "failed"},
		{"restart telemetry absent", func(_ *testing.T, res *FixtureRunResult) { dropTelemetry(res, 5001) }, 2, 1, 0, 0, "failed"},
		{"restart identity unknown", func(_ *testing.T, res *FixtureRunResult) {
			for i := range res.GroundTruth {
				if res.GroundTruth[i].RequestID == "r1" {
					res.GroundTruth[i].CorrelationID = 0
				}
			}
		}, 2, 1, 0, 0, "failed"},
		{"restart missing", func(_ *testing.T, res *FixtureRunResult) { delete(res.DiagnosticInputs, "r1") }, 2, 0, 1, 0, "withheld"},

		{"late attach gap omitted", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r2", func(in *cudadiag.DiagnosticInput) {
				measuredLoss(in)
				fullCoverage(in)
			})
		}, 2, 0, 0, 1, "failed"},
		{"late attach gap omitted and withheld", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r2", func(in *cudadiag.DiagnosticInput) {
				measuredLoss(in)
				fullCoverage(in)
				in.Sched.Process = restarted
			})
		}, 2, 1, 0, 0, "failed"},
		{"late attach withheld for measured loss without declared gap", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r2", func(in *cudadiag.DiagnosticInput) {
				in.Receipt.LostRecords["activity"] = tracecontract.Loss{Dropped: 1, Total: 1}
				fullCoverage(in)
			})
		}, 2, 1, 0, 0, "failed"},
		{"late attach withheld for identity mismatch", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r2", func(in *cudadiag.DiagnosticInput) { in.Sched.Process = restarted })
		}, 2, 1, 0, 0, "failed"},
		{"late attach coverage gap admitted", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r2", measuredLoss)
		}, 2, 0, 0, 1, "failed"},
		{"late attach withheld for scheduler unknown loss", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r2", func(in *cudadiag.DiagnosticInput) { in.Sched.UnknownLoss = true })
		}, 2, 1, 0, 0, "failed"},
		{"late attach withheld for measured loss with declared gap", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r2", func(in *cudadiag.DiagnosticInput) {
				in.Receipt.LostRecords["activity"] = tracecontract.Loss{Dropped: 1, Total: 1}
			})
		}, 2, 1, 0, 0, "failed"},
		{"late attach unknown loss only", func(t *testing.T, res *FixtureRunResult) {
			editInput(t, res, "r2", fullCoverage)
		}, 3, 0, 0, 0, "passed"},
		{"late attach joined exactly", func(_ *testing.T, res *FixtureRunResult) {
			var op cudaworkload.OpID
			for _, s := range res.HostSubmissions {
				if s.NativeCorrelationID == 5002 {
					op = s.OpID
				}
			}
			for i, a := range res.DeviceActivities {
				if a.NativeCorrelationID == 0 {
					res.DeviceActivities[i].NativeCorrelationID = 5002
					res.DeviceActivities[i].ExternalCorrelationID = uint64(op)
				}
			}
		}, 2, 1, 0, 0, "failed"},
		{"late attach telemetry absent", func(_ *testing.T, res *FixtureRunResult) { dropTelemetry(res, 5002) }, 2, 1, 0, 0, "failed"},
		{"late attach missing", func(_ *testing.T, res *FixtureRunResult) { delete(res.DiagnosticInputs, "r2") }, 2, 0, 1, 0, "withheld"},

		{"one of each", func(t *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["r0"] = res.DiagnosticInputs["h0"]
			editInput(t, res, "r1", func(in *cudadiag.DiagnosticInput) { in.Sched.UnknownLoss = true })
			delete(res.DiagnosticInputs, "r2")
		}, 0, 1, 1, 1, "failed"},
		{"all missing", func(_ *testing.T, res *FixtureRunResult) {
			for _, id := range []string{"r0", "r1", "r2"} {
				delete(res.DiagnosticInputs, id)
			}
		}, 0, 0, 3, 0, "withheld"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := resilienceRun(t, allModes...)
			tt.mutate(t, res)
			r, gates := evaluateDiagnostics(res, DefaultBudgets())
			rest := r.ResilienceTotal - r.ResiliencePassed - r.ResilienceWithheld - r.ResilienceMissing
			if r.ResilienceTotal != 3 || r.ResiliencePassed != tt.passed || r.ResilienceWithheld != tt.withheld || r.ResilienceMissing != tt.missing || rest != tt.rest {
				t.Errorf("resilience total=%d passed=%d withheld=%d missing=%d rest=%d, want 3 %d %d %d %d", r.ResilienceTotal, r.ResiliencePassed, r.ResilienceWithheld, r.ResilienceMissing, rest, tt.passed, tt.withheld, tt.missing, tt.rest)
			}
			g, ok := resilienceGate(gates)
			if !ok || g.Disposition != tt.want {
				t.Errorf("gate = %s (%s) reported %t, want %s", g.Disposition, g.Reason, ok, tt.want)
			}
			if tt.want != "passed" && r.GateDisposition == "passed" {
				t.Errorf("receipt disposition passed with resilience gate %s", g.Disposition)
			}
			// Resilience cases never enter the no-fault partition,
			// whatever their outcome.
			if r.NoFaultTotal != 2 || r.NoFaultPassed != 2 || r.NoFaultWithheld != 0 || r.NoFaultMissing != 0 {
				t.Errorf("no-fault total=%d passed=%d withheld=%d missing=%d, want 2 2 0 0", r.NoFaultTotal, r.NoFaultPassed, r.NoFaultWithheld, r.NoFaultMissing)
			}
		})
	}
}

// TestResilienceRepeatedCorrelationAcrossGenerations checks that a native
// correlation ID reused by another process generation is not attributed to
// a resilience request. A normal request in generation 5 reuses the
// correlation ID of a restart and a late attach request and joins exactly;
// that join names neither of them.
func TestResilienceRepeatedCorrelationAcrossGenerations(t *testing.T) {
	gen := func(g uint64) cudaworkload.ProcessGenID { return cudaworkload.ProcessGenID{PID: 60, Generation: g} }
	reqs := []GroundTruthRequest{
		{ID: "restart", WorkloadName: "work", ProcessGen: gen(1), TID: 6, StreamID: 1, CorrelationID: 77, Fault: FaultNone, Resilience: ResilienceRestart, Data: []float64{1}},
		{ID: "late", WorkloadName: "work", ProcessGen: gen(3), TID: 6, StreamID: 1, CorrelationID: 77, Fault: FaultNone, Resilience: ResilienceLateAttach, Data: []float64{1}},
		{ID: "normal", WorkloadName: "work", ProcessGen: gen(5), TID: 6, StreamID: 1, CorrelationID: 77, Fault: FaultNone, Resilience: ResilienceNormal, Data: []float64{1}},
	}
	res, err := NewGroundTruthFixture().Execute(context.Background(), reqs)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := evaluateAttribution(res, DefaultBudgets())
	if a.ExactJoins != 1 || a.IncorrectJoins != 0 {
		t.Fatalf("attribution exact=%d incorrect=%d, want 1 0", a.ExactJoins, a.IncorrectJoins)
	}
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	if r.ResilienceTotal != 2 || r.ResiliencePassed != 2 || r.ResilienceWithheld != 0 || r.ResilienceMissing != 0 {
		t.Errorf("resilience total=%d passed=%d withheld=%d missing=%d, want 2 2 0 0", r.ResilienceTotal, r.ResiliencePassed, r.ResilienceWithheld, r.ResilienceMissing)
	}
	if g, ok := resilienceGate(gates); !ok || g.Disposition != "passed" {
		t.Errorf("gate = %+v reported %t, want passed", g, ok)
	}
	if r.NoFaultTotal != 1 || r.NoFaultPassed != 1 {
		t.Errorf("no-fault %d/%d, want 1/1", r.NoFaultPassed, r.NoFaultTotal)
	}
}

func TestResilienceCountsJSON(t *testing.T) {
	b, err := json.Marshal(DiagnosticReceipt{})
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{`"resilience_withheld":0`, `"resilience_missing":0`} {
		if !strings.Contains(string(b), key) {
			t.Errorf("zero receipt %s lacks %s", b, key)
		}
	}
	in := DiagnosticReceipt{ResilienceTotal: 3, ResilienceWithheld: 1, ResilienceMissing: 2}
	if b, err = json.Marshal(in); err != nil {
		t.Fatal(err)
	}
	var out DiagnosticReceipt
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if out.ResilienceWithheld != 1 || out.ResilienceMissing != 2 {
		t.Errorf("round trip = %+v from %s", out, b)
	}
}

// TestResilienceAfterOverflow ports the independent review control in
// which an overflow precedes restart and late attach cases. An earlier
// overflow must not change why the later diagnoses are withheld: each is
// withheld for the evidence of its own mode, and all three pass.
func TestResilienceAfterOverflow(t *testing.T) {
	process := cudaworkload.ProcessGenID{PID: 82, Generation: 1}
	requests := []GroundTruthRequest{
		{ID: "overflow", WorkloadName: "work", ProcessGen: process, TID: 8, StreamID: 1, CorrelationID: 100, Fault: FaultHostSubmissionDelay, Resilience: ResilienceOverflow, Data: []float64{1}},
		{ID: "restart", WorkloadName: "work", ProcessGen: process, TID: 8, StreamID: 1, CorrelationID: 101, Fault: FaultNone, Resilience: ResilienceRestart, Data: []float64{1}},
		{ID: "late", WorkloadName: "work", ProcessGen: process, TID: 8, StreamID: 1, CorrelationID: 102, Fault: FaultNone, Resilience: ResilienceLateAttach, Data: []float64{1}},
	}
	res, err := NewGroundTruthFixture().Execute(context.Background(), requests)
	if err != nil {
		t.Fatal(err)
	}
	for i, tt := range []struct {
		id     string
		reason string
	}{
		{"restart", tracecontract.ReasonIdentityMismatch},
		{"late", tracecontract.ReasonUnknownLoss},
	} {
		d := cudadiag.Diagnose(cudaworkload.WorkloadIdentity{Process: process, WorkloadName: "work"}, cudaworkload.OpID(101+i), res.DiagnosticInputs[tt.id])
		if d.Admitted || d.WithheldReason != tt.reason {
			t.Errorf("%s diagnosis = %s admitted=%t reason=%q, want withheld for %q", tt.id, d.Mechanism, d.Admitted, d.WithheldReason, tt.reason)
		}
	}
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	g, _ := resilienceGate(gates)
	if r.ResiliencePassed != 3 || r.ResilienceWithheld != 0 || g.Disposition != "passed" {
		t.Errorf("resilience %d/%d withheld=%d gate=%s, want 3/3 passed", r.ResiliencePassed, r.ResilienceTotal, r.ResilienceWithheld, g.Disposition)
	}
}

// TestEmptyResiliencePopulationHasNoGate records that a run selecting no
// resilience cases reports no resilience gate. RunEvaluation always selects
// three cases when EvalConfig.ExerciseStress is set.
func TestEmptyResiliencePopulationHasNoGate(t *testing.T) {
	r, gates := evaluateDiagnostics(&FixtureRunResult{DiagnosticInputs: map[string]cudadiag.DiagnosticInput{}}, DefaultBudgets())
	if _, ok := resilienceGate(gates); ok || r.ResilienceTotal != 0 {
		t.Fatalf("empty population reports resilience: receipt=%+v gates=%+v", r, gates)
	}
}

func TestRunEvaluationStressSelectsCases(t *testing.T) {
	for _, stress := range []bool{false, true} {
		cfg := DefaultEvalConfig()
		cfg.ExerciseStress = stress
		report, err := RunEvaluation(context.Background(), DefaultProtocol(PhasePilot), cfg)
		if err != nil {
			t.Fatal(err)
		}
		_, reported := resilienceGate(report.Gates)
		want := 0
		if stress {
			want = 3
		}
		if reported != stress || report.Diagnostic.ResilienceTotal != want {
			t.Errorf("stress=%t: resilience total %d gate reported %t, want %d %t", stress, report.Diagnostic.ResilienceTotal, reported, want, stress)
		}
	}
}

// TestResilienceMetJoins checks the attribution half of the restart and
// late attach predicates against join results built by hand, including
// results the joiner itself never emits, such as an exact join across
// process generations.
func TestResilienceMetJoins(t *testing.T) {
	gen := func(g uint64) cudaworkload.ProcessGenID { return cudaworkload.ProcessGenID{PID: 40, Generation: g} }
	host := func(p cudaworkload.ProcessGenID, corr uint32) *cudaworkload.HostSubmission {
		return &cudaworkload.HostSubmission{Process: p, OpID: 1, NativeCorrelationID: corr}
	}
	dev := func(p cudaworkload.ProcessGenID, corr uint32) *cudaworkload.DeviceActivity {
		return &cudaworkload.DeviceActivity{Process: p, NativeCorrelationID: corr}
	}
	raw := func(p cudaworkload.ProcessGenID, corr uint32) cudaworkload.RawIDs {
		return cudaworkload.RawIDs{ProcessGenID: p, NativeCorrelationID: corr}
	}
	missingDev := cudaworkload.JoinResult{Outcome: cudaworkload.OutcomeMissing, DeviceActivity: dev(gen(1), 9), RawIDs: raw(gen(1), 9)}
	missingHost := cudaworkload.JoinResult{Outcome: cudaworkload.OutcomeMissing, HostSubmission: host(gen(1), 9), RawIDs: raw(gen(1), 9)}
	tests := []struct {
		name  string
		truth cudaworkload.ProcessGenID
		corr  uint32
		joins []cudaworkload.JoinResult
		want  bool
	}{
		{"missing device activity", gen(1), 9, []cudaworkload.JoinResult{missingDev}, true},
		{"missing host submission", gen(1), 9, []cudaworkload.JoinResult{missingHost}, true},
		{"ambiguous", gen(1), 9, []cudaworkload.JoinResult{{Outcome: cudaworkload.OutcomeAmbiguous, DeviceActivity: dev(gen(1), 9), RawIDs: raw(gen(1), 9)}}, true},
		{"no join results", gen(1), 9, nil, false},
		{"only another generation reported", gen(1), 9, []cudaworkload.JoinResult{{Outcome: cudaworkload.OutcomeMissing, HostSubmission: host(gen(2), 9), RawIDs: raw(gen(2), 9)}}, false},
		{"only another correlation reported", gen(1), 9, []cudaworkload.JoinResult{{Outcome: cudaworkload.OutcomeMissing, DeviceActivity: dev(gen(1), 8), RawIDs: raw(gen(1), 8)}}, false},
		{"cross-generation exact join", gen(1), 9, []cudaworkload.JoinResult{missingHost, {Outcome: cudaworkload.OutcomeExact, HostSubmission: host(gen(2), 9), DeviceActivity: dev(gen(1), 9), RawIDs: raw(gen(1), 9)}}, false},
		{"cross-generation exact join named by host", gen(2), 9, []cudaworkload.JoinResult{{Outcome: cudaworkload.OutcomeMissing, DeviceActivity: dev(gen(2), 9), RawIDs: raw(gen(2), 9)}, {Outcome: cudaworkload.OutcomeExact, HostSubmission: host(gen(2), 9), DeviceActivity: dev(gen(1), 9), RawIDs: raw(gen(1), 9)}}, false},
		{"exact join", gen(1), 9, []cudaworkload.JoinResult{missingHost, {Outcome: cudaworkload.OutcomeExact, HostSubmission: host(gen(1), 9), DeviceActivity: dev(gen(1), 9), RawIDs: raw(gen(1), 9)}}, false},
		{"associated join", gen(1), 9, []cudaworkload.JoinResult{missingDev, {Outcome: cudaworkload.OutcomeAssociated, HostSubmission: host(gen(1), 9), DeviceActivity: dev(gen(1), 0), RawIDs: raw(gen(1), 0)}}, false},
		{"repeated correlation exact in another generation", gen(1), 9, []cudaworkload.JoinResult{missingDev, {Outcome: cudaworkload.OutcomeExact, HostSubmission: host(gen(7), 9), DeviceActivity: dev(gen(7), 9), RawIDs: raw(gen(7), 9)}}, true},
		{"repeated correlation exact in another process", gen(1), 9, []cudaworkload.JoinResult{missingDev, {Outcome: cudaworkload.OutcomeExact, HostSubmission: host(cudaworkload.ProcessGenID{PID: 41, Generation: 1}, 9), DeviceActivity: dev(cudaworkload.ProcessGenID{PID: 41, Generation: 1}, 9), RawIDs: raw(cudaworkload.ProcessGenID{PID: 41, Generation: 1}, 9)}}, true},
		{"another record expired from the join queue", gen(1), 9, []cudaworkload.JoinResult{missingDev, {Outcome: cudaworkload.OutcomeMissing, Reason: cudaworkload.ReasonExpiredFIFO, HostSubmission: host(gen(1), 8), RawIDs: raw(gen(1), 8)}}, false},
		{"request expired from the join queue", gen(1), 9, []cudaworkload.JoinResult{{Outcome: cudaworkload.OutcomeMissing, Reason: cudaworkload.ReasonExpiredFIFO, HostSubmission: host(gen(1), 9), RawIDs: raw(gen(1), 9)}, missingDev}, false},
		{"zero generation", cudaworkload.ProcessGenID{PID: 40}, 9, []cudaworkload.JoinResult{{Outcome: cudaworkload.OutcomeMissing, RawIDs: raw(cudaworkload.ProcessGenID{PID: 40}, 9)}}, false},
		{"zero correlation", gen(1), 0, []cudaworkload.JoinResult{{Outcome: cudaworkload.OutcomeMissing, RawIDs: raw(gen(1), 0)}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			restart := cudadiag.DiagnosticInput{Sched: cudadiag.SchedEvidence{Process: cudaworkload.ProcessGenID{PID: tt.truth.PID, Generation: tt.truth.Generation + 1}}}
			late := cudadiag.DiagnosticInput{Receipt: tracecontract.Receipt{LostRecords: map[string]tracecontract.Loss{"activity": {Unknown: true}}}}
			for _, c := range []struct {
				mode   ResilienceMode
				in     cudadiag.DiagnosticInput
				reason string
				rule   string
			}{
				{ResilienceRestart, restart, tracecontract.ReasonIdentityMismatch, "scheduler_identity_match"},
				{ResilienceLateAttach, late, tracecontract.ReasonUnknownLoss, "mechanism_diagnosis"},
			} {
				gt := GroundTruthRecord{ProcessGen: tt.truth, CorrelationID: tt.corr, Resilience: c.mode}
				d := cudadiag.DiagnosisResult{Mechanism: cudadiag.MechanismWithheld, WithheldReason: c.reason, PrimaryFinding: tracecontract.Finding{Rule: c.rule}}
				if got := resilienceMet(gt, c.in, d, tt.joins); got != tt.want {
					t.Errorf("%s: resilienceMet = %t, want %t", c.mode, got, tt.want)
				}
			}
		})
	}
}

// TestResilienceMetRequiresWithheld checks that a diagnosis that was
// admitted, or not marked withheld, never meets a resilience predicate,
// whatever other evidence is present.
func TestResilienceMetRequiresWithheld(t *testing.T) {
	proc := cudaworkload.ProcessGenID{PID: 40, Generation: 1}
	in := cudadiag.DiagnosticInput{
		Sched:   cudadiag.SchedEvidence{Process: cudaworkload.ProcessGenID{PID: 40, Generation: 2}},
		Receipt: tracecontract.Receipt{LostRecords: map[string]tracecontract.Loss{"activity": {Dropped: 1, Total: 1, Unknown: true}}},
	}
	joins := []cudaworkload.JoinResult{{Outcome: cudaworkload.OutcomeMissing, RawIDs: cudaworkload.RawIDs{ProcessGenID: proc, NativeCorrelationID: 9}}}
	for _, mode := range []ResilienceMode{ResilienceOverflow, ResilienceRestart, ResilienceLateAttach, ResilienceShutdown} {
		gt := GroundTruthRecord{ProcessGen: proc, CorrelationID: 9, Resilience: mode}
		for _, d := range []cudadiag.DiagnosisResult{
			{Mechanism: cudadiag.MechanismNoFault, Admitted: true},
			{Mechanism: cudadiag.MechanismWithheld, Admitted: true, WithheldReason: tracecontract.ReasonIdentityMismatch},
			{Mechanism: cudadiag.MechanismNoFault, WithheldReason: tracecontract.ReasonUnknownLoss},
		} {
			if resilienceMet(gt, in, d, joins) {
				t.Errorf("%s: diagnosis %+v met the predicate", mode, d)
			}
		}
	}
	// A mode without a predicate never passes, even when withheld for
	// each mode's reason with every mode's evidence present.
	for _, mode := range []ResilienceMode{ResilienceShutdown, "unknown_mode"} {
		gt := GroundTruthRecord{ProcessGen: proc, CorrelationID: 9, Resilience: mode}
		for _, reason := range []string{tracecontract.ReasonIdentityMismatch, tracecontract.ReasonUnknownLoss, tracecontract.ReasonEvidenceUnavailable} {
			d := cudadiag.DiagnosisResult{Mechanism: cudadiag.MechanismWithheld, WithheldReason: reason}
			if resilienceMet(gt, in, d, joins) {
				t.Errorf("%s withheld for %q met a predicate", mode, reason)
			}
		}
	}
}

// TestLateAttachCreditNeedsActivityCause checks that a late attach case
// passes only when cudadiag withheld its diagnosis for the unknown activity
// loss the attach gap causes. Withholding caused by scheduler evidence, or
// by measured activity loss, is another mode's evidence and never passes,
// even when the receipt still declares the attach gap.
func TestLateAttachCreditNeedsActivityCause(t *testing.T) {
	tests := []struct {
		name     string
		edit     func(*cudadiag.DiagnosticInput)
		admitted bool
		rule     string
		reason   string
		passed   int
	}{
		{"attach gap", func(*cudadiag.DiagnosticInput) {}, false, "mechanism_diagnosis", tracecontract.ReasonUnknownLoss, 1},
		{"bounded activity loss", func(in *cudadiag.DiagnosticInput) {
			in.Receipt.LostRecords["activity"] = tracecontract.Loss{Total: 1}
		}, true, "", "", 0},
		{"scheduler unknown loss", func(in *cudadiag.DiagnosticInput) {
			in.Sched.UnknownLoss = true
		}, false, "scheduler_identity_match", tracecontract.ReasonUnknownLoss, 0},
		{"bounded activity loss and scheduler unknown loss", func(in *cudadiag.DiagnosticInput) {
			in.Receipt.LostRecords["activity"] = tracecontract.Loss{Total: 1}
			in.Sched.UnknownLoss = true
		}, false, "scheduler_identity_match", tracecontract.ReasonUnknownLoss, 0},
		{"measured activity loss", func(in *cudadiag.DiagnosticInput) {
			in.Receipt.LostRecords["activity"] = tracecontract.Loss{Dropped: 1, Total: 1}
		}, false, "mechanism_diagnosis", tracecontract.ReasonLossExceeded, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := resilienceRun(t, ResilienceLateAttach)
			editInput(t, res, "r0", tt.edit)
			in := res.DiagnosticInputs["r0"]
			if cov := in.Receipt.ProbeCoverage["cupti_activity"]; !cov.Present || cov.Ratio >= 1 {
				t.Fatalf("setup: attach gap not declared: %+v", cov)
			}
			gt := res.GroundTruth[len(res.GroundTruth)-1]
			d := cudadiag.Diagnose(cudaworkload.WorkloadIdentity{Process: gt.ProcessGen, WorkloadName: gt.WorkloadName}, cudaworkload.OpID(gt.CorrelationID), in)
			if d.Admitted != tt.admitted || !tt.admitted && (d.PrimaryFinding.Rule != tt.rule || d.WithheldReason != tt.reason) {
				t.Fatalf("setup: diagnosis admitted=%t rule=%q reason=%q, want %t %q %q", d.Admitted, d.PrimaryFinding.Rule, d.WithheldReason, tt.admitted, tt.rule, tt.reason)
			}
			r, _ := evaluateDiagnostics(res, DefaultBudgets())
			withheld := 0
			if !tt.admitted {
				withheld = 1 - tt.passed
			}
			if r.ResilienceTotal != 1 || r.ResiliencePassed != tt.passed || r.ResilienceWithheld != withheld {
				t.Errorf("resilience total=%d passed=%d withheld=%d, want 1 %d %d", r.ResilienceTotal, r.ResiliencePassed, r.ResilienceWithheld, tt.passed, withheld)
			}
		})
	}
}
