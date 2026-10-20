package cudadiagdemo_test

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudadiagdemo"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

// An api_scheduler record is diagnosed for the independently logged request
// thread. Rows vary only the supplied target, scheduler and logged threads;
// the synthetic protocol evidence is otherwise admissible as a host delay.
func TestComposeNativeEvidenceSchedulerTarget(t *testing.T) {
	const caller, other = 12345, 4242
	for _, tc := range []struct {
		name                  string
		native                bool
		target, sched, logged uint64
		admitted              bool
		err                   string
	}{
		{name: "explicit-match", target: caller, sched: caller, logged: caller, admitted: true},
		{name: "unspecified-bound-to-log", target: 0, sched: caller, logged: caller, admitted: true},
		{name: "explicit-conflict", target: other, sched: caller, logged: caller, err: "scheduler target thread mismatch"},
		{name: "scheduler-other-thread", target: 0, sched: other, logged: caller, err: "scheduler target thread mismatch"},
		{name: "target-agrees-with-scheduler-only", target: other, sched: other, logged: caller, err: "scheduler target thread mismatch"},
		{name: "unknown-logged-thread", target: 0, sched: caller, logged: 0, err: "scheduler target thread mismatch"},
		{name: "native-unqualified", native: true, target: 0, sched: caller, logged: caller},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := cudadiag.BuildSyntheticFixture(cudadiag.InjectionHostDelay)
			in := f.Input
			in.Thresholds.TargetTID = tc.target
			in.Sched.TID = tc.sched
			log := cudadiagdemo.NativeRequestLog{Attempt: "attempt-0", StartedNS: in.WallInterval.Start, FinishedNS: in.WallInterval.End, Complete: true, GroundTruth: cudadiagdemo.GroundTruthLog{RequestID: "request-1", WorkloadName: f.Workload.WorkloadName, OpID: f.OpID, ProcessGen: f.Workload.Process, Workload: f.Workload, TID: tc.logged}}
			r := &cudadiagdemo.NativeResult{Synthetic: !tc.native, Requests: []cudadiagdemo.NativeRequestLog{log}}
			raw, err := json.Marshal(cudadiagdemo.NativeEvidence{Version: 1, Synthetic: !tc.native, Records: []cudadiagdemo.NativeEvidenceRecord{{RequestID: "request-1", Attempt: "attempt-0", OpID: f.OpID, Phase: "api_scheduler", Input: in}}})
			if err != nil {
				t.Fatal(err)
			}
			out, err := cudadiagdemo.ComposeNativeEvidence(context.Background(), r, cudadiagdemo.NativeCompositionOptions{Evidence: bytes.NewReader(raw)})
			if tc.err != "" {
				if err == nil || !strings.Contains(err.Error(), tc.err) {
					t.Fatalf("contradictory scheduler target not refused: %v %+v", err, out)
				}
				return
			}
			if err != nil || len(out.Requests) != 1 {
				t.Fatal(out, err)
			}
			row := out.Requests[0]
			if row.Observations.Thresholds.TargetTID != tc.target || row.Observations.Sched.TID != tc.sched || row.Observations.Receipt.ClockUncertaintyNS == nil {
				t.Fatalf("supplied observations not retained: %+v", row.Observations)
			}
			d := row.Diagnosis
			if !tc.admitted {
				if d.Admitted || d.Mechanism != cudadiag.MechanismWithheld || row.Refusal == "" {
					t.Fatalf("unqualified native evidence admitted: %+v", row)
				}
				return
			}
			if !d.Admitted || d.Mechanism != cudadiag.MechanismHostSubmissionDelay || d.PrimaryFinding.Class != tracecontract.FindingObserved || row.Refusal != "" {
				t.Fatalf("logged request thread did not bind scheduler evidence: %+v", row)
			}
		})
	}
}
