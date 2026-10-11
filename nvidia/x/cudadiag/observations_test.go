package cudadiag

import (
	"encoding/json"
	"fmt"
	"testing"
)

func ExampleObserve() {
	f := BuildSyntheticFixture(InjectionMissingReturn)
	findings := Observe(f.Input)
	fmt.Println(findings[0].Admission.Admitted, findings[1].Admission.Admitted)
	// Output: false true
}

func TestConclusionSpecificAdmission(t *testing.T) {
	f := BuildSyntheticFixture(InjectionMissingReturn)
	got := Observe(f.Input)
	if got[0].Admission.Admitted || !got[1].Admission.Admitted {
		t.Fatalf("unrelated conclusion coupled: %v", got)
	}
	f.Input.Receipt.LostRecords["activity"] = f.Input.Receipt.LostRecords["sched"]
	loss := f.Input.Receipt.LostRecords["activity"]
	loss.Unknown = true
	f.Input.Receipt.LostRecords["activity"] = loss
	f.Input.HostCalls[0].HasReturn = true
	got = Observe(f.Input)
	if !got[0].Admission.Admitted || got[1].Admission.Admitted {
		t.Fatal("activity loss invalidated independent call or admitted device union")
	}
}

func TestObservationReplayOrder(t *testing.T) {
	f := BuildSyntheticFixture(InjectionNone)
	call := f.Input.HostCalls[0]
	call.Span.Start++
	call.Span.End++
	f.Input.HostCalls = append(f.Input.HostCalls, call)
	a, _ := json.Marshal(Observe(f.Input))
	f.Input.HostCalls[0], f.Input.HostCalls[1] = f.Input.HostCalls[1], f.Input.HostCalls[0]
	b, _ := json.Marshal(Observe(f.Input))
	if string(a) != string(b) {
		t.Fatal("input order changes findings")
	}
}

func TestMissingMechanismEvidenceWithholds(t *testing.T) {
	f := BuildSyntheticFixture(InjectionNone)
	f.Input.Kernels = nil
	got := Diagnose(f.Workload, f.OpID, f.Input)
	if got.Admitted || got.Mechanism != MechanismWithheld {
		t.Fatal("empty kernels admitted no-fault")
	}
	if !got.Findings[0].Admission.Admitted {
		t.Fatal("independent matched host call withheld")
	}
}
