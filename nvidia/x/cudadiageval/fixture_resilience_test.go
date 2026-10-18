package cudadiageval

import (
	"context"
	"fmt"
	"math"
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
				switch m {
				case ResilienceOverflow:
					want = tracecontract.Loss{Dropped: 1, Total: 1}
				case ResilienceLateAttach:
					want = tracecontract.Loss{Unknown: true}
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
				if withheld != (m != ResilienceNormal) {
					t.Errorf("%s (%s) diagnosis = %s admitted=%t", id, m, d.Mechanism, d.Admitted)
				}
			}
			if sum != res.DroppedRecords {
				t.Errorf("per-case dropped sum %d != run dropped %d", sum, res.DroppedRecords)
			}
		})
	}
}

// TestRestartFixtureEvidence checks that a restart case carries the new
// process generation where the diagnosis can observe it: in the host
// submission and in the scheduler evidence, while the device activity keeps
// the original generation.
func TestRestartFixtureEvidence(t *testing.T) {
	res := resilienceRun(t, ResilienceRestart)
	gt := res.GroundTruth[len(res.GroundTruth)-1]
	restarted := cudaworkload.ProcessGenID{PID: gt.ProcessGen.PID, Generation: gt.ProcessGen.Generation + 1}
	in := res.DiagnosticInputs["r0"]
	if in.Sched.Process != restarted {
		t.Errorf("scheduler evidence process = %v, want restarted %v", in.Sched.Process, restarted)
	}
	for _, s := range res.HostSubmissions {
		if s.NativeCorrelationID == gt.CorrelationID && s.Process != restarted {
			t.Errorf("host submission process = %v, want %v", s.Process, restarted)
		}
	}
	for _, a := range res.DeviceActivities {
		if a.NativeCorrelationID == gt.CorrelationID && a.Process != gt.ProcessGen {
			t.Errorf("device activity process = %v, want %v", a.Process, gt.ProcessGen)
		}
	}
	if got := in.Receipt.LostRecords["activity"]; got != (tracecontract.Loss{Dropped: 0, Total: 1}) {
		t.Errorf("restart activity loss = %+v, want 0/1", got)
	}
	d := cudadiag.Diagnose(cudaworkload.WorkloadIdentity{Process: gt.ProcessGen, WorkloadName: gt.WorkloadName}, cudaworkload.OpID(gt.CorrelationID), in)
	if d.Admitted || d.Mechanism != cudadiag.MechanismWithheld || d.WithheldReason != tracecontract.ReasonIdentityMismatch {
		t.Errorf("restart diagnosis = %s admitted=%t reason=%q, want withheld for %q", d.Mechanism, d.Admitted, d.WithheldReason, tracecontract.ReasonIdentityMismatch)
	}
}

// TestLateAttachFixtureEvidence checks that a late attach case leaves its
// device activity uncorrelated and that its receipt declares the attach gap
// instead of claiming full activity coverage.
func TestLateAttachFixtureEvidence(t *testing.T) {
	res := resilienceRun(t, ResilienceLateAttach)
	gt := res.GroundTruth[len(res.GroundTruth)-1]
	in := res.DiagnosticInputs["r0"]
	if loss := in.Receipt.LostRecords["activity"]; !loss.Unknown {
		t.Errorf("late attach activity loss = %+v, want unknown", loss)
	}
	if cov := in.Receipt.ProbeCoverage["cupti_activity"]; !cov.Present || cov.Ratio <= 0 || cov.Ratio >= 1 {
		t.Errorf("late attach activity coverage = %+v, want present with ratio in (0, 1)", cov)
	}
	if in.Sched.Process != gt.ProcessGen {
		t.Errorf("late attach scheduler process = %v, want %v", in.Sched.Process, gt.ProcessGen)
	}
	uncorrelated := 0
	for _, a := range res.DeviceActivities {
		if a.NativeCorrelationID == gt.CorrelationID {
			t.Errorf("late attach device activity keeps its correlation: %+v", a)
		}
		if a.NativeCorrelationID == 0 && a.ExternalCorrelationID == 0 {
			uncorrelated++
		}
	}
	if uncorrelated != 1 {
		t.Errorf("uncorrelated device activities = %d, want 1", uncorrelated)
	}
	d := cudadiag.Diagnose(cudaworkload.WorkloadIdentity{Process: gt.ProcessGen, WorkloadName: gt.WorkloadName}, cudaworkload.OpID(gt.CorrelationID), in)
	if d.Admitted || d.Mechanism != cudadiag.MechanismWithheld || d.WithheldReason != tracecontract.ReasonUnknownLoss {
		t.Errorf("late attach diagnosis = %s admitted=%t reason=%q, want withheld for %q", d.Mechanism, d.Admitted, d.WithheldReason, tracecontract.ReasonUnknownLoss)
	}
}

// TestFixtureRefusesInvalidResilienceIdentity checks that the fixture does
// not fabricate resilience evidence for a request without a complete
// explicit identity, or for a restart whose generation would wrap to zero.
func TestFixtureRefusesInvalidResilienceIdentity(t *testing.T) {
	proc := cudaworkload.ProcessGenID{PID: 70, Generation: 1}
	tests := []struct {
		name    string
		process cudaworkload.ProcessGenID
		corr    uint32
		mode    ResilienceMode
		wantErr bool
	}{
		{"restart", proc, 9, ResilienceRestart, false},
		{"restart generation wraps", cudaworkload.ProcessGenID{PID: 70, Generation: math.MaxUint64}, 9, ResilienceRestart, true},
		{"restart zero pid", cudaworkload.ProcessGenID{Generation: 1}, 9, ResilienceRestart, true},
		{"restart zero correlation", proc, 0, ResilienceRestart, true},
		{"late attach zero generation", cudaworkload.ProcessGenID{PID: 70}, 9, ResilienceLateAttach, true},
		{"overflow zero correlation", proc, 0, ResilienceOverflow, true},
		{"normal zero identity", cudaworkload.ProcessGenID{}, 0, ResilienceNormal, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := GroundTruthRequest{ID: "c", WorkloadName: "work", ProcessGen: tt.process, TID: 7, StreamID: 1, CorrelationID: tt.corr, Fault: FaultNone, Resilience: tt.mode, Data: []float64{1}}
			res, err := NewGroundTruthFixture().Execute(context.Background(), []GroundTruthRequest{req})
			if (err != nil) != tt.wantErr {
				t.Fatalf("Execute error = %v, want error %t", err, tt.wantErr)
			}
			if err != nil && res != nil {
				t.Errorf("refused run returned %+v", res)
			}
		})
	}
}

// TestResilienceJoinCapacityBoundary checks that a record expiring from
// the evaluator's bounded join queue never earns a restart or late attach
// pass. Each run starts with a restart case "evicted" whose host
// submission is restored to its device generation, so complete joins
// attribute it exactly; then an honest restart and a late attach case;
// then normal fillers up to hosts host submissions. The queue holds 4096
// pending submissions: at 4096 nothing expires and only the honest cases
// pass, and at 4097 "evicted" expires first and every restart and late
// attach case is withheld.
func TestResilienceJoinCapacityBoundary(t *testing.T) {
	tests := []struct {
		hosts            int
		expired          bool
		passed, withheld int
	}{
		{4096, false, 2, 1},
		{4097, true, 0, 3},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprint(tt.hosts), func(t *testing.T) {
			proc := cudaworkload.ProcessGenID{PID: 91, Generation: 1}
			req := func(id string, corr int, mode ResilienceMode) GroundTruthRequest {
				return GroundTruthRequest{ID: id, WorkloadName: "work", ProcessGen: proc, TID: 9, StreamID: 1, CorrelationID: uint32(corr), Fault: FaultNone, Resilience: mode, Data: []float64{1}}
			}
			reqs := []GroundTruthRequest{
				req("evicted", 1, ResilienceRestart),
				req("restart", 2, ResilienceRestart),
				req("late", 3, ResilienceLateAttach),
			}
			for i := len(reqs); i < tt.hosts; i++ {
				reqs = append(reqs, req(fmt.Sprintf("normal-%d", i), i+1, ResilienceNormal))
			}
			res, err := NewGroundTruthFixture().Execute(context.Background(), reqs)
			if err != nil {
				t.Fatal(err)
			}
			if len(res.HostSubmissions) != tt.hosts || res.HostSubmissions[0].NativeCorrelationID != 1 {
				t.Fatalf("setup: %d host submissions, first %+v", len(res.HostSubmissions), res.HostSubmissions[0])
			}
			res.HostSubmissions[0].Process = proc

			expired := false
			for _, j := range flushedJoins(res) {
				if j.Reason == cudaworkload.ReasonExpiredFIFO {
					expired = true
					if !(requestKey{proc, 1}).names(j) {
						t.Errorf("setup: expired record does not name the first request: %+v", j)
					}
				}
			}
			if expired != tt.expired {
				t.Fatalf("setup: join queue expired a record = %t, want %t", expired, tt.expired)
			}

			r, gates := evaluateDiagnostics(res, DefaultBudgets())
			if r.ResilienceTotal != 3 || r.ResiliencePassed != tt.passed || r.ResilienceWithheld != tt.withheld || r.ResilienceMissing != 0 {
				t.Errorf("resilience total=%d passed=%d withheld=%d missing=%d, want 3 %d %d 0", r.ResilienceTotal, r.ResiliencePassed, r.ResilienceWithheld, r.ResilienceMissing, tt.passed, tt.withheld)
			}
			if g, ok := resilienceGate(gates); !ok || g.Disposition != "failed" {
				t.Errorf("gate = %+v reported %t, want failed", g, ok)
			}
		})
	}
}
