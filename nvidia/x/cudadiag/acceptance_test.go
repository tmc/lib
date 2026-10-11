package cudadiag

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

func TestAcceptance_Suite(t *testing.T) {
	cm := NewConfusionMatrix()

	injections := []struct {
		inj           InjectionType
		wantMechanism MechanismType
		wantReason    string
		isWithheld    bool
	}{
		{
			inj:           InjectionNone,
			wantMechanism: MechanismNoFault,
			isWithheld:    false,
		},
		{
			inj:           InjectionHostDelay,
			wantMechanism: MechanismHostSubmissionDelay,
			isWithheld:    false,
		},
		{
			inj:           InjectionDeviceBottleneck,
			wantMechanism: MechanismDeviceBottleneck,
			isWithheld:    false,
		},
		{
			inj:           InjectionAmbiguousSimultaneous,
			wantMechanism: MechanismAmbiguousSimultaneous,
			isWithheld:    false,
		},
		{
			inj:           InjectionMissingReturn,
			wantMechanism: MechanismWithheld,
			wantReason:    tracecontract.ReasonMissingReturn,
			isWithheld:    true,
		},
		{
			inj:           InjectionMissingScheduler,
			wantMechanism: MechanismWithheld,
			wantReason:    tracecontract.ReasonProbeMissing,
			isWithheld:    true,
		},
		{
			inj:           InjectionClockUncertainty,
			wantMechanism: MechanismWithheld,
			wantReason:    tracecontract.ReasonClockUncertaintyExceeded,
			isWithheld:    true,
		},
		{
			inj:           InjectionRecordLoss,
			wantMechanism: MechanismWithheld,
			wantReason:    tracecontract.ReasonLossExceeded,
			isWithheld:    true,
		},
	}

	for _, tc := range injections {
		fix := BuildSyntheticFixture(tc.inj)
		res := Diagnose(fix.Workload, fix.OpID, fix.Input)

		cm.Record(fix.ExpectedOutcome, res.Mechanism, res.WithheldReason)

		if tc.isWithheld {
			if res.Admitted {
				t.Fatalf("scenario %s: expected withheld, got admitted", tc.inj)
			}
			if res.WithheldReason != tc.wantReason {
				t.Fatalf("scenario %s: withheld reason got %q, want %q", tc.inj, res.WithheldReason, tc.wantReason)
			}
		} else {
			if !res.Admitted {
				t.Fatalf("scenario %s: expected admitted, got withheld with reason %q", tc.inj, res.WithheldReason)
			}
			if res.Mechanism != tc.wantMechanism {
				t.Fatalf("scenario %s: mechanism got %q, want %q", tc.inj, res.Mechanism, tc.wantMechanism)
			}
		}
	}

	// Verify ConfusionMatrix statistics
	if cm.TotalTrials != 8 {
		t.Fatalf("total trials got %d, want 8", cm.TotalTrials)
	}
	if cm.WithheldTrials != 4 {
		t.Fatalf("withheld trials got %d, want 4", cm.WithheldTrials)
	}

	wantWithheldFrac := 4.0 / 8.0
	if cm.WithheldFraction() != wantWithheldFrac {
		t.Fatalf("withheld fraction got %.4f, want %.4f", cm.WithheldFraction(), wantWithheldFrac)
	}

	// Evaluated non-withheld accuracy must be 100%
	if cm.Accuracy() != 1.0 {
		t.Fatalf("evaluated accuracy got %.4f, want 1.0", cm.Accuracy())
	}

	// Check withheld reason counts
	if cm.WithheldReasons[tracecontract.ReasonMissingReturn] != 1 {
		t.Fatalf("missing return count got %d, want 1", cm.WithheldReasons[tracecontract.ReasonMissingReturn])
	}
	if cm.WithheldReasons[tracecontract.ReasonProbeMissing] != 1 {
		t.Fatalf("probe missing count got %d, want 1", cm.WithheldReasons[tracecontract.ReasonProbeMissing])
	}
	if cm.WithheldReasons[tracecontract.ReasonClockUncertaintyExceeded] != 1 {
		t.Fatalf("clock uncertainty count got %d, want 1", cm.WithheldReasons[tracecontract.ReasonClockUncertaintyExceeded])
	}
	if cm.WithheldReasons[tracecontract.ReasonLossExceeded] != 1 {
		t.Fatalf("loss exceeded count got %d, want 1", cm.WithheldReasons[tracecontract.ReasonLossExceeded])
	}

	// Log formatted report
	report := cm.FormatReport()
	t.Logf("\n%s", report)
}
