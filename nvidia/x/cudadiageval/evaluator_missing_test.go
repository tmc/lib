package cudadiageval

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// diagnosticFixture runs four held-out host delay cases (h0-h3), four
// no-fault controls (n0-n3) and two competing device bottleneck cases
// (d0-d1) through the ground-truth fixture.
func diagnosticFixture(t *testing.T) *FixtureRunResult {
	t.Helper()
	var reqs []GroundTruthRequest
	proc := cudaworkload.ProcessGenID{PID: 50000, Generation: 1}
	for i := 0; i < 2; i++ {
		reqs = append(reqs, GroundTruthRequest{ID: fmt.Sprintf("d%d", i), WorkloadName: "workload-inference", ProcessGen: proc, TID: 6002, StreamID: 1, CorrelationID: uint32(4000 + i), Fault: FaultDeviceBottleneck, Resilience: ResilienceNormal, Data: []float64{1.5, 2.5, 3.5, 4.5}})
	}
	for i := 0; i < 4; i++ {
		reqs = append(reqs,
			GroundTruthRequest{ID: fmt.Sprintf("h%d", i), WorkloadName: "workload-inference", ProcessGen: proc, TID: 6000, StreamID: 1, CorrelationID: uint32(2000 + i), Fault: FaultHostSubmissionDelay, Resilience: ResilienceNormal, Data: []float64{1, 2, 3, 4}},
			GroundTruthRequest{ID: fmt.Sprintf("n%d", i), WorkloadName: "workload-inference", ProcessGen: proc, TID: 6001, StreamID: 2, CorrelationID: uint32(3000 + i), Fault: FaultNone, Resilience: ResilienceNormal, Data: []float64{1, 2, 3, 4}},
		)
	}
	res, err := NewGroundTruthFixture().Execute(context.Background(), reqs)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func findGate(t *testing.T, gates []PropertyGate, name string) PropertyGate {
	t.Helper()
	for _, g := range gates {
		if g.GateName == name {
			return g
		}
	}
	t.Fatalf("gate %q not reported", name)
	return PropertyGate{}
}

func TestDiagnosticsFullFixture(t *testing.T) {
	r, gates := evaluateDiagnostics(diagnosticFixture(t), DefaultBudgets())
	if r.HeldOutTotal != 4 || r.HeldOutCorrect != 4 || r.AccuracyPct != 100 || r.NoFaultTotal != 4 || r.NoFaultPassed != 4 || r.CompetingTotal != 2 || r.CompetingDiverted != 2 {
		t.Fatalf("full fixture receipt = %+v", r)
	}
	for _, name := range []string{"held_out_fault_accuracy", "no_fault_rejection", "competing_mechanism_distinction"} {
		if g := findGate(t, gates, name); g.Disposition != "passed" {
			t.Errorf("%s = %s (%s), want passed", name, g.Disposition, g.Reason)
		}
	}
	if r.GateDisposition != "passed" {
		t.Errorf("receipt disposition = %s, want passed", r.GateDisposition)
	}
}

func TestDiagnosticsMissingInputs(t *testing.T) {
	tests := []struct {
		name    string
		missing []string
		gate    string
		total   func(DiagnosticReceipt) int
		passed  func(DiagnosticReceipt) int
		maxPass int
	}{
		{"one held-out", []string{"h1"}, "held_out_fault_accuracy",
			func(r DiagnosticReceipt) int { return r.HeldOutTotal },
			func(r DiagnosticReceipt) int { return r.HeldOutCorrect }, 3},
		{"three held-out", []string{"h1", "h2", "h3"}, "held_out_fault_accuracy",
			func(r DiagnosticReceipt) int { return r.HeldOutTotal },
			func(r DiagnosticReceipt) int { return r.HeldOutCorrect }, 1},
		{"all held-out", []string{"h0", "h1", "h2", "h3"}, "held_out_fault_accuracy",
			func(r DiagnosticReceipt) int { return r.HeldOutTotal },
			func(r DiagnosticReceipt) int { return r.HeldOutCorrect }, 0},
		{"one no-fault", []string{"n2"}, "no_fault_rejection",
			func(r DiagnosticReceipt) int { return r.NoFaultTotal },
			func(r DiagnosticReceipt) int { return r.NoFaultPassed }, 3},
		{"all no-fault", []string{"n0", "n1", "n2", "n3"}, "no_fault_rejection",
			func(r DiagnosticReceipt) int { return r.NoFaultTotal },
			func(r DiagnosticReceipt) int { return r.NoFaultPassed }, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := diagnosticFixture(t)
			full, _ := evaluateDiagnostics(res, DefaultBudgets())
			for _, id := range tt.missing {
				delete(res.DiagnosticInputs, id)
			}
			r, gates := evaluateDiagnostics(res, DefaultBudgets())
			if got := tt.total(r); got != 4 {
				t.Errorf("total = %d, want 4: missing cases must stay in the denominator", got)
			}
			if got := tt.passed(r); got > tt.maxPass {
				t.Errorf("passed = %d, want at most %d", got, tt.maxPass)
			}
			if r.AccuracyPct > full.AccuracyPct {
				t.Errorf("accuracy rose from %.1f to %.1f after withholding inputs", full.AccuracyPct, r.AccuracyPct)
			}
			// Every present case passes, so the gate could still pass
			// if the missing inputs did: withheld, not failed.
			g := findGate(t, gates, tt.gate)
			if g.Disposition != "withheld" {
				t.Errorf("%s = %s with %d inputs missing, want withheld: %+v", tt.gate, g.Disposition, len(tt.missing), g)
			}
			if !strings.Contains(g.Reason, "missing") {
				t.Errorf("%s reason %q does not report missing inputs", tt.gate, g.Reason)
			}
			if r.GateDisposition != "withheld" {
				t.Errorf("receipt disposition = %s with %d inputs missing, want withheld", r.GateDisposition, len(tt.missing))
			}
		})
	}
}

func TestDiagnosticsGatesKeepOwnReasons(t *testing.T) {
	res := diagnosticFixture(t)
	// A host delay input under a no-fault label is a false alarm.
	res.DiagnosticInputs["n0"] = res.DiagnosticInputs["h0"]
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	nf := findGate(t, gates, "no_fault_rejection")
	if nf.Disposition != "failed" || r.NoFaultPassed != 3 {
		t.Fatalf("no-fault control did not fail: passed=%d gate=%+v", r.NoFaultPassed, nf)
	}
	if !strings.Contains(nf.Reason, "no-fault") {
		t.Errorf("no-fault reason = %q", nf.Reason)
	}
	ho := findGate(t, gates, "held_out_fault_accuracy")
	if ho.Disposition != "passed" || strings.Contains(ho.Reason, "no-fault") {
		t.Errorf("held-out gate took no-fault result: %+v", ho)
	}
	if r.GateDisposition != "failed" {
		t.Errorf("receipt disposition = %s, want failed", r.GateDisposition)
	}
}

func TestDiagnosticsMissingCompetingInput(t *testing.T) {
	res := diagnosticFixture(t)
	delete(res.DiagnosticInputs, "d1")
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	g := findGate(t, gates, "competing_mechanism_distinction")
	if r.CompetingTotal != 2 || r.CompetingDiverted > 1 || g.Disposition != "withheld" {
		t.Fatalf("missing competing input: total=%d diverted=%d gate=%+v", r.CompetingTotal, r.CompetingDiverted, g)
	}
	if r.GateDisposition != "withheld" {
		t.Errorf("receipt disposition = %s with competing gate %s, want withheld", r.GateDisposition, g.Disposition)
	}
}

func TestDiagnosticsMissingInputCannotRescueFailure(t *testing.T) {
	res := diagnosticFixture(t)
	// h0 is misdiagnosed and h1 is missing: 2/4 correct, and even a
	// correct h1 would give 75%, below the 95% minimum.
	res.DiagnosticInputs["h0"] = res.DiagnosticInputs["n0"]
	delete(res.DiagnosticInputs, "h1")
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	g := findGate(t, gates, "held_out_fault_accuracy")
	if r.HeldOutTotal != 4 || r.HeldOutCorrect != 2 || g.Disposition != "failed" {
		t.Fatalf("held-out total=%d correct=%d gate=%+v, want 4, 2, failed", r.HeldOutTotal, r.HeldOutCorrect, g)
	}
	if r.GateDisposition != "failed" {
		t.Errorf("receipt disposition = %s, want failed", r.GateDisposition)
	}
}

func TestDiagnosticsEmptyPopulationFails(t *testing.T) {
	res := diagnosticFixture(t)
	for _, id := range []string{"h0", "h1", "h2", "h3"} {
		delete(res.DiagnosticInputs, id)
	}
	res.GroundTruth = slices.DeleteFunc(res.GroundTruth, func(gt GroundTruthRecord) bool { return gt.Fault == FaultHostSubmissionDelay })
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	if g := findGate(t, gates, "held_out_fault_accuracy"); r.HeldOutTotal != 0 || g.Disposition != "failed" {
		t.Fatalf("held-out total=%d gate=%+v, want 0 and failed", r.HeldOutTotal, g)
	}
}

func TestRateGate(t *testing.T) {
	tests := []struct {
		passed, missing, total int
		min                    float64
		want                   string
	}{
		{4, 0, 4, 95, "passed"},
		{29, 0, 100, 29, "passed"}, // exact boundary
		{28, 0, 100, 29, "failed"},
		{3, 1, 4, 75, "withheld"},  // 3/4 already meets the minimum
		{3, 1, 4, 100, "withheld"}, // passes only if the missing case passes
		{0, 4, 4, 95, "withheld"},
		{2, 1, 4, 80, "failed"}, // fails even if the missing case passes
		{0, 0, 0, 0, "failed"},  // empty population
	}
	for _, tt := range tests {
		got, reason := rateGate("rate", tt.passed, tt.missing, tt.total, tt.min)
		if got != tt.want {
			t.Errorf("rateGate(%d, %d, %d, %v) = %s (%s), want %s", tt.passed, tt.missing, tt.total, tt.min, got, reason, tt.want)
		}
	}
}

func TestDiagnosticReceiptMissingCounts(t *testing.T) {
	full, _ := evaluateDiagnostics(diagnosticFixture(t), DefaultBudgets())
	if full.HeldOutMissing != 0 || full.NoFaultMissing != 0 || full.CompetingMissing != 0 {
		t.Errorf("full fixture missing counts = %d %d %d, want zero", full.HeldOutMissing, full.NoFaultMissing, full.CompetingMissing)
	}

	// Held-out: h0 misdiagnosed, h1 missing; the four outcomes partition the total.
	res := diagnosticFixture(t)
	res.DiagnosticInputs["h0"] = res.DiagnosticInputs["n0"]
	delete(res.DiagnosticInputs, "h1")
	delete(res.DiagnosticInputs, "d0")
	r, _ := evaluateDiagnostics(res, DefaultBudgets())
	if r.HeldOutMissing != 1 || r.HeldOutCorrect != 2 || r.HeldOutCorrect+r.HeldOutWithheld+r.HeldOutIncorrect+r.HeldOutMissing != r.HeldOutTotal {
		t.Errorf("held-out total=%d correct=%d withheld=%d incorrect=%d missing=%d", r.HeldOutTotal, r.HeldOutCorrect, r.HeldOutWithheld, r.HeldOutIncorrect, r.HeldOutMissing)
	}
	if r.CompetingMissing != 1 || r.CompetingTotal != 2 || r.CompetingDiverted != 1 {
		t.Errorf("competing total=%d diverted=%d missing=%d, want 2 1 1", r.CompetingTotal, r.CompetingDiverted, r.CompetingMissing)
	}

	// No-fault: a missing input and a real false alarm now differ.
	missing := diagnosticFixture(t)
	delete(missing.DiagnosticInputs, "n0")
	alarm := diagnosticFixture(t)
	alarm.DiagnosticInputs["n0"] = alarm.DiagnosticInputs["h0"]
	rm, gm := evaluateDiagnostics(missing, DefaultBudgets())
	ra, ga := evaluateDiagnostics(alarm, DefaultBudgets())
	if rm.NoFaultMissing != 1 || ra.NoFaultMissing != 0 || rm.NoFaultPassed != 3 || ra.NoFaultPassed != 3 {
		t.Errorf("missing: %d/%d missing=%d; alarm: %d/%d missing=%d", rm.NoFaultPassed, rm.NoFaultTotal, rm.NoFaultMissing, ra.NoFaultPassed, ra.NoFaultTotal, ra.NoFaultMissing)
	}
	if d := findGate(t, gm, "no_fault_rejection").Disposition; d != "withheld" {
		t.Errorf("missing no-fault gate = %s, want withheld", d)
	}
	if d := findGate(t, ga, "no_fault_rejection").Disposition; d != "failed" {
		t.Errorf("false alarm no-fault gate = %s, want failed", d)
	}
}

func TestDiagnosticReceiptMissingJSON(t *testing.T) {
	zero, err := json.Marshal(DiagnosticReceipt{})
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{`"held_out_missing":0`, `"no_fault_missing":0`, `"competing_missing":0`} {
		if !strings.Contains(string(zero), key) {
			t.Errorf("zero receipt %s lacks %s", zero, key)
		}
	}
	in := DiagnosticReceipt{HeldOutTotal: 4, HeldOutMissing: 1, NoFaultTotal: 4, NoFaultMissing: 2, CompetingTotal: 2, CompetingMissing: 2}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out DiagnosticReceipt
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if out.HeldOutMissing != 1 || out.NoFaultMissing != 2 || out.CompetingMissing != 2 {
		t.Errorf("round trip = %+v from %s", out, b)
	}
}
