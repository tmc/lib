package cudadiag

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func TestDiagnose_Mechanisms(t *testing.T) {
	cases := []struct {
		name            string
		injection       InjectionType
		wantMechanism   MechanismType
		wantPrimaryRule string
		wantClass       tracecontract.FindingClass
	}{
		{
			name:            "no_fault",
			injection:       InjectionNone,
			wantMechanism:   MechanismNoFault,
			wantPrimaryRule: "no_fault",
			wantClass:       tracecontract.FindingObserved,
		},
		{
			name:            "host_delay",
			injection:       InjectionHostDelay,
			wantMechanism:   MechanismHostSubmissionDelay,
			wantPrimaryRule: "host_submission_delay",
			wantClass:       tracecontract.FindingObserved,
		},
		{
			name:            "device_bottleneck",
			injection:       InjectionDeviceBottleneck,
			wantMechanism:   MechanismDeviceBottleneck,
			wantPrimaryRule: "competing_device_bottleneck",
			wantClass:       tracecontract.FindingObserved,
		},
		{
			name:            "ambiguous_simultaneous",
			injection:       InjectionAmbiguousSimultaneous,
			wantMechanism:   MechanismAmbiguousSimultaneous,
			wantPrimaryRule: "ambiguous_simultaneous_fault",
			wantClass:       tracecontract.FindingObserved,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fix := BuildSyntheticFixture(tc.injection)
			res := Diagnose(fix.Workload, fix.OpID, fix.Input)

			if !res.Admitted {
				t.Fatalf("expected admitted finding, got withheld with reason %q", res.WithheldReason)
			}
			if res.Mechanism != tc.wantMechanism {
				t.Fatalf("mechanism got %q, want %q", res.Mechanism, tc.wantMechanism)
			}
			if res.PrimaryFinding.Rule != tc.wantPrimaryRule {
				t.Fatalf("primary finding rule got %q, want %q", res.PrimaryFinding.Rule, tc.wantPrimaryRule)
			}
			if res.PrimaryFinding.Class != tc.wantClass {
				t.Fatalf("finding class got %q, want %q", res.PrimaryFinding.Class, tc.wantClass)
			}
			if err := res.PrimaryFinding.CheckClaim(tc.wantClass); err != nil {
				t.Fatalf("check claim failed: %v", err)
			}
		})
	}
}

func TestDiagnose_MissingReturn(t *testing.T) {
	fix := BuildSyntheticFixture(InjectionMissingReturn)
	res := Diagnose(fix.Workload, fix.OpID, fix.Input)

	if res.Admitted {
		t.Fatalf("expected diagnosis to be withheld when return event is missing")
	}
	if res.Mechanism != MechanismWithheld {
		t.Fatalf("mechanism got %q, want %q", res.Mechanism, MechanismWithheld)
	}
	if res.WithheldReason != tracecontract.ReasonMissingReturn {
		t.Fatalf("withheld reason got %q, want %q", res.WithheldReason, tracecontract.ReasonMissingReturn)
	}
	if res.PrimaryFinding.Class != tracecontract.FindingUnavailable {
		t.Fatalf("finding class got %q, want %q", res.PrimaryFinding.Class, tracecontract.FindingUnavailable)
	}
	if err := res.PrimaryFinding.CheckClaim(tracecontract.FindingObserved); err == nil {
		t.Fatalf("expected CheckClaim to reject observed claim on withheld finding")
	}
}

func TestDiagnose_MissingSchedulerEvidence(t *testing.T) {
	fix := BuildSyntheticFixture(InjectionMissingScheduler)
	res := Diagnose(fix.Workload, fix.OpID, fix.Input)

	if res.Admitted {
		t.Fatalf("expected diagnosis to be withheld when scheduler probe is inactive")
	}
	if res.WithheldReason != tracecontract.ReasonProbeMissing {
		t.Fatalf("withheld reason got %q, want %q", res.WithheldReason, tracecontract.ReasonProbeMissing)
	}
}

func TestDiagnose_SchedulerIdentityMismatch(t *testing.T) {
	fix := BuildSyntheticFixture(InjectionHostDelay)
	// Mutate target TID so it does not match scheduler evidence TID
	wrongTarget := fix.Workload
	wrongTarget.Process = cudaworkload.ProcessGenID{PID: 99999, Generation: 1}

	res := Diagnose(wrongTarget, fix.OpID, fix.Input)
	if res.Admitted {
		t.Fatalf("expected diagnosis to be withheld on scheduler identity mismatch")
	}
	if res.WithheldReason != tracecontract.ReasonIdentityMismatch {
		t.Fatalf("withheld reason got %q, want %q", res.WithheldReason, tracecontract.ReasonIdentityMismatch)
	}
}

func TestDiagnose_ClockUncertainty(t *testing.T) {
	fix := BuildSyntheticFixture(InjectionClockUncertainty)
	res := Diagnose(fix.Workload, fix.OpID, fix.Input)

	if res.Admitted {
		t.Fatalf("expected diagnosis to be withheld when clock uncertainty exceeds threshold")
	}
	if res.WithheldReason != tracecontract.ReasonClockUncertaintyExceeded {
		t.Fatalf("withheld reason got %q, want %q", res.WithheldReason, tracecontract.ReasonClockUncertaintyExceeded)
	}
}

func TestDiagnose_RecordLoss(t *testing.T) {
	fix := BuildSyntheticFixture(InjectionRecordLoss)
	res := Diagnose(fix.Workload, fix.OpID, fix.Input)

	if res.Admitted {
		t.Fatalf("expected diagnosis to be withheld when record loss exceeds threshold")
	}
	if res.WithheldReason != tracecontract.ReasonLossExceeded {
		t.Fatalf("withheld reason got %q, want %q", res.WithheldReason, tracecontract.ReasonLossExceeded)
	}
}

func TestDiagnose_PowerClockNotCausal(t *testing.T) {
	fix := BuildSyntheticFixture(InjectionHostDelay)
	// Add overlapping clock and power events
	fix.Input.PowerClock = PowerClockEvidence{
		ClockThrottled: true,
		PowerCapped:    true,
		Intervals:      []tracecontract.Interval{{Start: 2000, End: 3000, Clock: "monotonic"}},
	}

	res := Diagnose(fix.Workload, fix.OpID, fix.Input)

	if !res.Admitted {
		t.Fatalf("expected diagnosis to be admitted: %v", res.WithheldReason)
	}

	// Verify that power/clock overlap is recorded only as FindingAssociated (non-causal)
	foundPowerFinding := false
	for _, f := range res.Findings {
		if strings.Contains(f.Rule, "power_clock") {
			foundPowerFinding = true
			if f.Class != tracecontract.FindingAssociated {
				t.Fatalf("power/clock finding must be FindingAssociated, got %s", f.Class)
			}
			if err := f.CheckClaim(tracecontract.FindingObserved); err == nil {
				t.Fatalf("CheckClaim must reject causal FindingObserved claim for power/clock overlap")
			}
		}
	}
	if !foundPowerFinding {
		t.Fatalf("expected finding for power/clock association")
	}
}

func TestDiagnose_BlindedRules(t *testing.T) {
	fix1 := BuildSyntheticFixture(InjectionNone)
	fix2 := BuildSyntheticFixture(InjectionNone)
	// Mutate external label on fix2, but leave Input identical
	fix2.Injection = InjectionHostDelay

	res1 := Diagnose(fix1.Workload, fix1.OpID, fix1.Input)
	res2 := Diagnose(fix2.Workload, fix2.OpID, fix2.Input)

	if res1.Mechanism != res2.Mechanism {
		t.Fatalf("diagnostic rule was not blinded: changed label altered mechanism (%v vs %v)",
			res1.Mechanism, res2.Mechanism)
	}
}
