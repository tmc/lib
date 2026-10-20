package cudadiag_test

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// One collected stream is diagnosed under different independently selected
// targets. Every reading is admitted in each row, so only the target contract
// decides the scheduler diagnosis; the stream's own scheduler TID never does.
func TestDiagnoseTelemetrySchedulerTarget(t *testing.T) {
	assertHostOnly(t)
	readings, opt := mixedProtocol()
	streamTID := opt.Thresholds.TargetTID
	raw := collectMixed(t, readings, opt)
	_, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0)
	if err != nil {
		t.Fatal(err)
	}
	var sched int
	for _, s := range samples {
		for _, r := range s.Readings {
			if o := r.Observation; o != nil && strings.HasPrefix(o.Kind, "sched_") {
				if o.TID != streamTID || o.Workload == nil || o.Workload.Process != opt.Workload.Process {
					t.Fatalf("stream scheduler identity: %+v", o)
				}
				sched++
			}
		}
	}
	if streamTID == 0 || sched == 0 {
		t.Fatal("stream has no scheduler readings for the caller thread")
	}
	for _, tc := range []struct {
		name     string
		target   uint64
		admitted bool
	}{
		{"explicit-independent", streamTID, true},
		{"mismatch", streamTID + 1, false},
		{"zero", 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opt := opt
			opt.Thresholds.TargetTID = tc.target
			got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Observations) != len(readings) {
				t.Fatal("observations lost")
			}
			for _, o := range got.Observations {
				if !o.Finding.Admission.Admitted {
					t.Fatalf("reading withheld before target contract: %+v", o)
				}
			}
			d := got.Diagnosis
			if tc.admitted {
				if !d.Admitted || d.Mechanism != cudadiag.MechanismNoFault || d.PrimaryFinding.Class != tracecontract.FindingAssociated {
					t.Fatalf("independent target not admitted: %+v", d)
				}
				return
			}
			if d.Admitted || d.Mechanism != cudadiag.MechanismWithheld || d.WithheldReason != tracecontract.ReasonIdentityMismatch || d.PrimaryFinding.Rule != "scheduler_identity_match" || d.PrimaryFinding.Admission.ReasonCode != tracecontract.ReasonIdentityMismatch {
				t.Fatalf("stream scheduler thread admitted without independent target: %+v", d)
			}
		})
	}
}
