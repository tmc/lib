package cudadiag

import (
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"testing"
)

func TestMechanismRequiresComparableBaseline(t *testing.T) {
	for _, name := range []string{"workload", "shape", "stationarity", "correctness"} {
		t.Run(name, func(t *testing.T) {
			f := BuildSyntheticFixture(InjectionDeviceBottleneck)
			gate := f.Input.Receipt.Gates[name]
			gate.Disposition = "unchecked"
			f.Input.Receipt.Gates[name] = gate
			got := Diagnose(f.Workload, f.OpID, f.Input)
			if got.Admitted || got.PrimaryFinding.Class != tracecontract.FindingUnavailable {
				t.Fatal("unqualified baseline comparison admitted")
			}
			observations := Observe(f.Input)
			if !observations[0].Admission.Admitted || !observations[1].Admission.Admitted {
				t.Fatal("baseline qualification invalidated independent observed timing")
			}
		})
	}
}

func TestDiagnosticThresholdOverflow(t *testing.T) {
	f := BuildSyntheticFixture(InjectionNone)
	f.Input.Thresholds.HostDelayRatio = 1e300
	got := Diagnose(f.Workload, f.OpID, f.Input)
	if got.Admitted || got.Mechanism != MechanismWithheld {
		t.Fatal("overflowed finite threshold admitted")
	}
}
