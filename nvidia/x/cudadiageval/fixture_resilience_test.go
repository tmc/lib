package cudadiageval

import (
	"context"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// resilienceRun runs two held-out (h0-h1), two no-fault (n0-n1) and two
// competing (d0-d1) cases, then one case per mode (r0, r1, ...).
func resilienceRun(t *testing.T, modes ...ResilienceMode) *FixtureRunResult {
	t.Helper()
	proc := cudaworkload.ProcessGenID{PID: 50000, Generation: 1}
	req := func(id string, corr int, fault FaultLabel, mode ResilienceMode) GroundTruthRequest {
		return GroundTruthRequest{ID: id, WorkloadName: "workload-inference", ProcessGen: proc, TID: 6000, StreamID: 1, CorrelationID: uint32(corr), Fault: fault, Resilience: mode, Data: []float64{1, 2, 3, 4}}
	}
	var reqs []GroundTruthRequest
	for i := 0; i < 2; i++ {
		reqs = append(reqs,
			req(fmt.Sprintf("h%d", i), 2000+i, FaultHostSubmissionDelay, ResilienceNormal),
			req(fmt.Sprintf("n%d", i), 3000+i, FaultNone, ResilienceNormal),
			req(fmt.Sprintf("d%d", i), 4000+i, FaultDeviceBottleneck, ResilienceNormal),
		)
	}
	for i, m := range modes {
		fault := FaultNone
		if m == ResilienceOverflow {
			fault = FaultHostSubmissionDelay
		}
		reqs = append(reqs, req(fmt.Sprintf("r%d", i), 5000+i, fault, m))
	}
	res, err := NewGroundTruthFixture().Execute(context.Background(), reqs)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func resilienceGate(gates []PropertyGate) (PropertyGate, bool) {
	for _, g := range gates {
		if g.GateName == "system_resilience_controls" {
			return g, true
		}
	}
	return PropertyGate{}, false
}

var allModes = []ResilienceMode{ResilienceOverflow, ResilienceRestart, ResilienceLateAttach}

func TestOverflowReachesDiagnosis(t *testing.T) {
	res := resilienceRun(t, ResilienceOverflow)
	in, ok := res.DiagnosticInputs["r0"]
	if !ok {
		t.Fatal("overflow case has no diagnostic input")
	}
	if len(in.Kernels) != 0 {
		t.Errorf("overflow input carries %d kernels; the device activity was dropped", len(in.Kernels))
	}
	if res.DroppedRecords != 1 || in.Receipt.LostRecords["activity"].Dropped != 1 {
		t.Errorf("dropped records = %d, receipt activity loss = %+v, want 1", res.DroppedRecords, in.Receipt.LostRecords["activity"])
	}
	for _, act := range res.DeviceActivities {
		if act.NativeCorrelationID == 5000 {
			t.Errorf("overflow device activity retained: %+v", act)
		}
	}
	gt := res.GroundTruth[len(res.GroundTruth)-1]
	d := cudadiag.Diagnose(cudaworkload.WorkloadIdentity{Process: gt.ProcessGen, WorkloadName: gt.WorkloadName}, cudaworkload.OpID(gt.CorrelationID), in)
	if d.Mechanism != cudadiag.MechanismWithheld || d.Admitted {
		t.Errorf("overflow diagnosis = %s admitted=%t, want withheld", d.Mechanism, d.Admitted)
	}
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	g, ok := resilienceGate(gates)
	if r.ResilienceTotal != 1 || r.ResiliencePassed != 1 || !ok || g.Disposition != "passed" {
		t.Errorf("resilience %d/%d gate=%+v (reported %t), want 1/1 passed", r.ResiliencePassed, r.ResilienceTotal, g, ok)
	}
}

func TestResilienceMissingInputs(t *testing.T) {
	tests := []struct {
		name    string
		modes   []ResilienceMode
		missing []string
		total   int
		maxPass int
		want    string
	}{
		{"all present", allModes, nil, 3, 3, "passed"},
		{"multiple overflow", []ResilienceMode{ResilienceOverflow, ResilienceOverflow, ResilienceRestart, ResilienceLateAttach}, nil, 4, 4, "passed"},
		{"overflow deleted", allModes, []string{"r0"}, 3, 2, "withheld"},
		{"restart deleted", allModes, []string{"r1"}, 3, 2, "withheld"},
		{"all deleted", allModes, []string{"r0", "r1", "r2"}, 3, 0, "withheld"},
		{"only case deleted", []ResilienceMode{ResilienceLateAttach}, []string{"r0"}, 1, 0, "withheld"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := resilienceRun(t, tt.modes...)
			for _, id := range tt.missing {
				delete(res.DiagnosticInputs, id)
			}
			r, gates := evaluateDiagnostics(res, DefaultBudgets())
			g, ok := resilienceGate(gates)
			if !ok {
				t.Fatalf("resilience requested by %d cases but gate not reported", len(tt.modes))
			}
			if r.ResilienceTotal != tt.total || r.ResiliencePassed > tt.maxPass {
				t.Errorf("resilience %d/%d, want total %d and at most %d passed", r.ResiliencePassed, r.ResilienceTotal, tt.total, tt.maxPass)
			}
			if r.ResilienceMissing != len(tt.missing) {
				t.Errorf("resilience missing = %d, want %d", r.ResilienceMissing, len(tt.missing))
			}
			if g.Disposition != tt.want {
				t.Errorf("gate = %s (%s), want %s", g.Disposition, g.Reason, tt.want)
			}
			// Every other diagnostic gate passes, so the receipt
			// follows the resilience gate.
			if r.GateDisposition != tt.want {
				t.Errorf("receipt disposition = %s, want %s", r.GateDisposition, tt.want)
			}
		})
	}
}

func TestNormalOnlyRunOmitsResilienceGate(t *testing.T) {
	r, gates := evaluateDiagnostics(resilienceRun(t), DefaultBudgets())
	if g, ok := resilienceGate(gates); ok || r.ResilienceTotal != 0 {
		t.Errorf("no resilience requested, got total %d gate %+v", r.ResilienceTotal, g)
	}
	if r.GateDisposition != "passed" {
		t.Errorf("receipt disposition = %s, want passed", r.GateDisposition)
	}
}

func TestAllDiagnosticInputsMissing(t *testing.T) {
	res := resilienceRun(t, allModes...)
	clear(res.DiagnosticInputs)
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	for _, g := range gates {
		if g.Disposition != "withheld" {
			t.Errorf("%s = %s (%s), want withheld", g.GateName, g.Disposition, g.Reason)
		}
	}
	if len(gates) != 4 {
		t.Errorf("reported %d gates, want 4", len(gates))
	}
	if r.HeldOutTotal != 2 || r.NoFaultTotal != 2 || r.CompetingTotal != 2 || r.ResilienceTotal != 3 {
		t.Errorf("totals held-out=%d no-fault=%d competing=%d resilience=%d, want 2 2 2 3", r.HeldOutTotal, r.NoFaultTotal, r.CompetingTotal, r.ResilienceTotal)
	}
	if r.HeldOutMissing != 2 || r.NoFaultMissing != 2 || r.CompetingMissing != 2 || r.ResilienceMissing != 3 {
		t.Errorf("missing held-out=%d no-fault=%d competing=%d resilience=%d, want 2 2 2 3", r.HeldOutMissing, r.NoFaultMissing, r.CompetingMissing, r.ResilienceMissing)
	}
	if r.HeldOutCorrect+r.NoFaultPassed+r.CompetingDiverted+r.ResiliencePassed != 0 {
		t.Errorf("cases without inputs passed: %+v", r)
	}
	if r.GateDisposition != "withheld" {
		t.Errorf("receipt disposition = %s, want withheld", r.GateDisposition)
	}
}

func TestRunEvaluationReportsOverflow(t *testing.T) {
	cfg := DefaultEvalConfig()
	cfg.ExerciseStress = true
	report, err := RunEvaluation(context.Background(), DefaultProtocol(PhasePilot), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if report.Diagnostic.ResilienceTotal != 3 || report.Diagnostic.ResiliencePassed != 3 {
		t.Errorf("resilience %d/%d, want 3/3 including overflow", report.Diagnostic.ResiliencePassed, report.Diagnostic.ResilienceTotal)
	}
}

// TestCaseScopedActivityLoss is the independent review control: an overflow
// earlier in a run must not withhold an unrelated later case.
func TestCaseScopedActivityLoss(t *testing.T) {
	process := cudaworkload.ProcessGenID{PID: 81, Generation: 1}
	requests := []GroundTruthRequest{
		{ID: "overflow", WorkloadName: "work", ProcessGen: process, TID: 8, StreamID: 1, CorrelationID: 100, Fault: FaultHostSubmissionDelay, Resilience: ResilienceOverflow, Data: []float64{1}},
		{ID: "after", WorkloadName: "work", ProcessGen: process, TID: 8, StreamID: 1, CorrelationID: 101, Fault: FaultNone, Resilience: ResilienceNormal, Data: []float64{1}},
	}
	res, err := NewGroundTruthFixture().Execute(context.Background(), requests)
	if err != nil {
		t.Fatal(err)
	}
	after := res.DiagnosticInputs["after"]
	if loss := after.Receipt.LostRecords["activity"]; loss.Dropped != 0 {
		t.Errorf("normal case inherited earlier overflow: loss=%+v", loss)
	}
	d := cudadiag.Diagnose(cudaworkload.WorkloadIdentity{Process: process, WorkloadName: "work"}, cudaworkload.OpID(101), after)
	if d.Mechanism != cudadiag.MechanismNoFault || !d.Admitted {
		t.Errorf("normal case contaminated by earlier overflow: diagnosis=%s admitted=%t reason=%s", d.Mechanism, d.Admitted, d.WithheldReason)
	}
	if res.DroppedRecords != 1 {
		t.Errorf("run dropped records = %d, want 1", res.DroppedRecords)
	}
}

func TestFixtureLossIsCaseLocal(t *testing.T) {
	tests := []struct {
		name  string
		modes []ResilienceMode
	}{
		{"overflow then normal", []ResilienceMode{ResilienceOverflow, ResilienceNormal}},
		{"normal then overflow", []ResilienceMode{ResilienceNormal, ResilienceOverflow}},
		{"two overflows then normal", []ResilienceMode{ResilienceOverflow, ResilienceOverflow, ResilienceNormal}},
		{"overflow then restart and late attach", []ResilienceMode{ResilienceOverflow, ResilienceRestart, ResilienceLateAttach}},
		{"interleaved", []ResilienceMode{ResilienceNormal, ResilienceOverflow, ResilienceNormal, ResilienceOverflow, ResilienceNormal}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proc := cudaworkload.ProcessGenID{PID: 90, Generation: 1}
			var reqs []GroundTruthRequest
			var wantRun uint64
			for i, m := range tt.modes {
				fault := FaultNone
				if m == ResilienceOverflow {
					fault = FaultHostSubmissionDelay
					wantRun++
				}
				reqs = append(reqs, GroundTruthRequest{ID: fmt.Sprintf("c%d", i), WorkloadName: "work", ProcessGen: proc, TID: 9, StreamID: 1, CorrelationID: uint32(700 + i), Fault: fault, Resilience: m, Data: []float64{1}})
			}
			res, err := NewGroundTruthFixture().Execute(context.Background(), reqs)
			if err != nil {
				t.Fatal(err)
			}
			if res.DroppedRecords != wantRun || res.TotalSubmissions != uint64(len(reqs)) {
				t.Errorf("run dropped/total = %d/%d, want %d/%d", res.DroppedRecords, res.TotalSubmissions, wantRun, len(reqs))
			}
			var sum uint64
			for i, m := range tt.modes {
				id := fmt.Sprintf("c%d", i)
				in := res.DiagnosticInputs[id]
				want := tracecontract.Loss{Dropped: 0, Total: 1}
				if m == ResilienceOverflow {
					want = tracecontract.Loss{Dropped: 1, Total: 1}
				}
				if got := in.Receipt.LostRecords["activity"]; got != want {
					t.Errorf("%s (%s) activity loss = %+v, want %+v", id, m, got, want)
				}
				if got := in.Receipt.LostRecords["sched"]; got != (tracecontract.Loss{Dropped: 0, Total: 1}) {
					t.Errorf("%s (%s) sched loss = %+v, want 0/1", id, m, got)
				}
				sum += in.Receipt.LostRecords["activity"].Dropped
				d := cudadiag.Diagnose(cudaworkload.WorkloadIdentity{Process: proc, WorkloadName: "work"}, cudaworkload.OpID(700+i), in)
				withheld := d.Mechanism == cudadiag.MechanismWithheld || !d.Admitted
				if withheld != (m == ResilienceOverflow) {
					t.Errorf("%s (%s) diagnosis = %s admitted=%t", id, m, d.Mechanism, d.Admitted)
				}
			}
			if sum != res.DroppedRecords {
				t.Errorf("per-case dropped sum %d != run dropped %d", sum, res.DroppedRecords)
			}
		})
	}
}
