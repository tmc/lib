package cudadiageval

import (
	"context"
	"encoding/json"
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
}

// TestResilienceWithheldCannotPass checks that a restart or late attach
// case whose diagnosis is withheld does not pass the resilience gate.
// resilienceRun(allModes) gives r0 overflow, r1 restart, r2 late attach.
func TestResilienceWithheldCannotPass(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(t *testing.T, res *FixtureRunResult)
		passed int
		want   string
	}{
		{"all admitted", func(*testing.T, *FixtureRunResult) {}, 3, "passed"},
		{"restart withheld", func(t *testing.T, res *FixtureRunResult) { withhold(t, res, "r1") }, 2, "failed"},
		{"late attach withheld", func(t *testing.T, res *FixtureRunResult) { withhold(t, res, "r2") }, 2, "failed"},
		{"restart empty input", func(_ *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["r1"] = cudadiag.DiagnosticInput{}
		}, 2, "failed"},
		{"both withheld", func(t *testing.T, res *FixtureRunResult) {
			withhold(t, res, "r1")
			withhold(t, res, "r2")
		}, 1, "failed"},
		{"overflow admitted", func(_ *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["r0"] = res.DiagnosticInputs["h0"]
		}, 2, "failed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := resilienceRun(t, allModes...)
			tt.mutate(t, res)
			r, gates := evaluateDiagnostics(res, DefaultBudgets())
			g, ok := resilienceGate(gates)
			if !ok {
				t.Fatal("resilience gate not reported")
			}
			if r.ResilienceTotal != 3 || r.ResiliencePassed != tt.passed || g.Disposition != tt.want {
				t.Errorf("resilience %d/%d gate=%s (%s), want %d/3 %s", r.ResiliencePassed, r.ResilienceTotal, g.Disposition, g.Reason, tt.passed, tt.want)
			}
			if tt.want != "passed" && r.GateDisposition == "passed" {
				t.Errorf("receipt disposition passed with resilience gate %s", g.Disposition)
			}
		})
	}
}

// TestResiliencePartition checks that passed, withheld, missing and
// incorrect resilience cases are counted separately and sum to the total.
// resilienceRun(allModes) gives r0 overflow, r1 restart, r2 late attach.
func TestResiliencePartition(t *testing.T) {
	tests := []struct {
		name                                 string
		mutate                               func(t *testing.T, res *FixtureRunResult)
		passed, withheld, missing, incorrect int
		want                                 string
	}{
		{"all pass", func(*testing.T, *FixtureRunResult) {}, 3, 0, 0, 0, "passed"},
		{"restart withheld", func(t *testing.T, res *FixtureRunResult) { withhold(t, res, "r1") }, 2, 1, 0, 0, "failed"},
		{"late attach missing", func(_ *testing.T, res *FixtureRunResult) { delete(res.DiagnosticInputs, "r2") }, 2, 0, 1, 0, "withheld"},
		{"overflow admitted", func(_ *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["r0"] = res.DiagnosticInputs["n0"]
		}, 2, 0, 0, 1, "failed"},
		{"overflow missing", func(_ *testing.T, res *FixtureRunResult) { delete(res.DiagnosticInputs, "r0") }, 2, 0, 1, 0, "withheld"},
		{"one of each", func(t *testing.T, res *FixtureRunResult) {
			res.DiagnosticInputs["r0"] = res.DiagnosticInputs["h0"]
			withhold(t, res, "r1")
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
			incorrect := r.ResilienceTotal - r.ResiliencePassed - r.ResilienceWithheld - r.ResilienceMissing
			if r.ResilienceTotal != 3 || r.ResiliencePassed != tt.passed || r.ResilienceWithheld != tt.withheld || r.ResilienceMissing != tt.missing || incorrect != tt.incorrect {
				t.Errorf("resilience total=%d passed=%d withheld=%d missing=%d incorrect=%d, want 3 %d %d %d %d", r.ResilienceTotal, r.ResiliencePassed, r.ResilienceWithheld, r.ResilienceMissing, incorrect, tt.passed, tt.withheld, tt.missing, tt.incorrect)
			}
			g, ok := resilienceGate(gates)
			if !ok || g.Disposition != tt.want {
				t.Errorf("gate = %s (%s) reported %t, want %s", g.Disposition, g.Reason, ok, tt.want)
			}
		})
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
// which an overflow precedes restart and late attach cases. The resilience
// gate may pass only if neither later diagnosis is withheld.
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
	withheld := 0
	for i, id := range []string{"restart", "late"} {
		d := cudadiag.Diagnose(cudaworkload.WorkloadIdentity{Process: process, WorkloadName: "work"}, cudaworkload.OpID(101+i), res.DiagnosticInputs[id])
		if d.Mechanism == cudadiag.MechanismWithheld || !d.Admitted {
			withheld++
		}
	}
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	g, _ := resilienceGate(gates)
	if g.Disposition == "passed" && withheld > 0 {
		t.Fatalf("%d restart/late-attach diagnoses were withheld yet resilience passed %d/%d: %+v", withheld, r.ResiliencePassed, r.ResilienceTotal, g)
	}
	if withheld != 0 || r.ResiliencePassed != 3 || g.Disposition != "passed" {
		t.Errorf("withheld=%d resilience %d/%d gate=%s, want 0 withheld and 3/3 passed", withheld, r.ResiliencePassed, r.ResilienceTotal, g.Disposition)
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
