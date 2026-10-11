package cudadiag

import (
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"testing"
)

func TestInterventionScope(t *testing.T) {
	f := BuildSyntheticFixture(InjectionHostDelay)
	in := tracecontract.InterventionReceipt{Verified: true, TestedRule: string(MechanismHostSubmissionDelay), WorkloadSHA256: workloadDigest(f.Workload), OperationID: f.OpID.String()}
	if !confirms(&in, MechanismHostSubmissionDelay, f.Workload, f.OpID) {
		t.Fatal("scoped receipt refused")
	}
	if confirms(&in, MechanismDeviceBottleneck, f.Workload, f.OpID) {
		t.Fatal("host intervention confirms device mechanism")
	}
	in.WorkloadSHA256 = ""
	if confirms(&in, MechanismHostSubmissionDelay, f.Workload, f.OpID) {
		t.Fatal("legacy unknown scope promoted")
	}
}

func TestInterventionRejectsUnadmittedPair(t *testing.T) {
	f := BuildSyntheticFixture(InjectionHostDelay)
	ctrl := Diagnose(f.Workload, f.OpID, f.Input)
	plan := InterventionPlan{HypothesisID: string(MechanismHostSubmissionDelay), TargetWorkload: f.Workload, ReferenceOutput: []float64{1}, CorrectnessOracle: "exact", InterventionType: "remove_delay", Ordering: "ABBA", StoppingRules: StoppingRules{MaxTrials: 4}}
	inv := ctrl
	inv.Admitted = false
	if _, err := EvaluateIntervention(plan, ctrl, []float64{1}, inv, []float64{1}); err == nil {
		t.Fatal("unadmitted evidence confirmed")
	}
	inv = ctrl
	if _, err := EvaluateIntervention(plan, ctrl, []float64{1}, inv, []float64{1.00000001}); err == nil {
		t.Fatal("zero tolerance silently relaxed")
	}
	if _, err := EvaluateIntervention(plan, ctrl, []float64{1}, inv, []float64{1}); err == nil {
		t.Fatal("no effect confirmed")
	}
}

func TestInterventionNegativeAccounting(t *testing.T) {
	f := BuildSyntheticFixture(InjectionHostDelay)
	ctrl := Diagnose(f.Workload, f.OpID, f.Input)
	plan := InterventionPlan{HypothesisID: string(MechanismHostSubmissionDelay), TargetWorkload: f.Workload, ReferenceOutput: []float64{1}, CorrectnessOracle: "exact", InterventionType: "remove_delay", Ordering: "declared AB", StoppingRules: StoppingRules{MaxTrials: 4}}
	for _, edit := range []func(*TimeAccounting){
		func(a *TimeAccounting) { a.HostAPIDuration = -1 },
		func(a *TimeAccounting) { a.BusyDuration = -1 },
		func(a *TimeAccounting) { a.BusyDuration = 1 << 62 },
		func(a *TimeAccounting) { a.WallInterval.End = a.WallInterval.Start - 1 },
	} {
		inv := ctrl
		edit(&inv.TimeAccounting)
		if receipt, err := EvaluateIntervention(plan, ctrl, []float64{1}, inv, []float64{1}); err == nil || receipt.Verified {
			t.Fatal("contradictory caller-admitted accounting confirmed")
		}
	}
}

func TestInterventionDistinctExecutionOperations(t *testing.T) {
	f := BuildSyntheticFixture(InjectionHostDelay)
	ctrl := Diagnose(f.Workload, f.OpID, f.Input)
	inv := ctrl
	inv.OpID++
	inv.TimeAccounting.HostAPIDuration /= 2
	plan := InterventionPlan{HypothesisID: string(MechanismHostSubmissionDelay), TargetWorkload: f.Workload, ReferenceOutput: []float64{1}, CorrectnessOracle: "exact", InterventionType: "remove_delay", Ordering: "declared AB", StoppingRules: StoppingRules{MaxTrials: 4}}
	r, err := EvaluateIntervention(plan, ctrl, []float64{1}, inv, []float64{1})
	if err != nil || !r.Verified || r.OperationID != ctrl.OpID.String() || r.InterventionOperationID != inv.OpID.String() {
		t.Fatalf("paired executions rejected: %v %+v", err, r)
	}
	if !confirms(&r, MechanismHostSubmissionDelay, f.Workload, ctrl.OpID) || confirms(&r, MechanismHostSubmissionDelay, f.Workload, inv.OpID) {
		t.Fatal("control operation scope lost")
	}
}

func TestInterventionRejectsEqualWrongOutputs(t *testing.T) {
	f := BuildSyntheticFixture(InjectionHostDelay)
	ctrl := Diagnose(f.Workload, f.OpID, f.Input)
	inv := ctrl
	inv.OpID++
	inv.TimeAccounting.HostAPIDuration /= 2
	plan := InterventionPlan{HypothesisID: string(MechanismHostSubmissionDelay), TargetWorkload: f.Workload, ReferenceOutput: []float64{1}, CorrectnessOracle: "independent oracle", InterventionType: "remove_delay", Ordering: "declared AB", StoppingRules: StoppingRules{MaxTrials: 4}}
	if receipt, err := EvaluateIntervention(plan, ctrl, []float64{2}, inv, []float64{2}); err == nil || receipt.Verified {
		t.Fatal("equal but independently incorrect outputs confirmed")
	}
}
