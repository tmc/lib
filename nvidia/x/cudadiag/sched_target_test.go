package cudadiag_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

// Scheduler evidence is admitted only for an independently selected thread.
// The host API call (10us) is below threshold, so a host delay can come only
// from the 55us runnable interval in the supplied scheduler evidence.
func TestDiagnoseSchedulerTarget(t *testing.T) {
	assertHostOnly(t)
	const caller, other = 12345, 4242
	for _, tc := range []struct {
		name          string
		target, sched uint64
		admitted      bool
	}{
		{"explicit-match", caller, caller, true},
		{"explicit-mismatch", caller, other, false},
		{"unset-target", 0, caller, false},
		{"unset-target-other-thread", 0, other, false},
		{"unknown-target", other, caller, false},
		{"zero-target-zero-evidence", 0, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := cudadiag.BuildSyntheticFixture(cudadiag.InjectionNone)
			in := f.Input
			in.Thresholds.TargetTID = tc.target
			in.Sched.TID = tc.sched
			in.Sched.Intervals = []cudadiag.SchedInterval{
				{State: cudadiag.SchedRunning, Span: tracecontract.Interval{Start: 2000, End: 11000, Clock: "monotonic"}},
				{State: cudadiag.SchedRunnable, Span: tracecontract.Interval{Start: 11000, End: 66000, Clock: "monotonic"}},
			}
			got := cudadiag.Diagnose(f.Workload, f.OpID, in)
			if tc.admitted {
				if !got.Admitted || got.Mechanism != cudadiag.MechanismHostSubmissionDelay || got.PrimaryFinding.Rule != "host_submission_delay" {
					t.Fatalf("matching target not admitted: %+v", got)
				}
				return
			}
			if got.Admitted || got.Mechanism != cudadiag.MechanismWithheld || got.WithheldReason != tracecontract.ReasonIdentityMismatch || got.PrimaryFinding.Rule != "scheduler_identity_match" || got.PrimaryFinding.Admission.Admitted {
				t.Fatalf("scheduler evidence admitted without independent target: %+v", got)
			}
		})
	}
}

// Pre-entry timing with an active scheduler probe binds the validated
// submission thread; a contradictory explicit target or scheduler thread is
// refused rather than chosen.
func TestDiagnosePreEntrySchedulerTarget(t *testing.T) {
	assertHostOnly(t)
	const submitter, other = 77, 78
	for _, tc := range []struct {
		name                  string
		target, submit, sched uint64
		admitted              bool
		reason, rule          string // for refused rows
	}{
		{"submission", 0, submitter, submitter, true, "", ""},
		{"explicit-match", submitter, submitter, submitter, true, "", ""},
		// validatePreEntry refuses a target that contradicts the submission.
		{"conflicting-target", other, submitter, submitter, false, tracecontract.ReasonEvidenceUnavailable, "mechanism_diagnosis"},
		{"scheduler-other-thread", 0, submitter, other, false, tracecontract.ReasonIdentityMismatch, "scheduler_identity_match"},
		{"target-agrees-with-scheduler-only", other, submitter, other, false, tracecontract.ReasonEvidenceUnavailable, "mechanism_diagnosis"},
		{"zero-submission", 0, 0, 0, false, tracecontract.ReasonIdentityMismatch, "scheduler_identity_match"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			workload, in := preEntryInput()
			in.Thresholds.TargetTID = tc.target
			in.PreEntries[0].Submission.TID = tc.submit
			in.Sched = cudadiag.SchedEvidence{Process: workload.Process, TID: tc.sched, ProbePresent: true, Intervals: []cudadiag.SchedInterval{
				{State: cudadiag.SchedRunning, Span: tracecontract.Interval{Start: 10, End: 410, Clock: "control"}},
			}}
			got := cudadiag.Diagnose(workload, 7, in)
			if tc.admitted {
				if !got.Admitted || got.Mechanism != cudadiag.MechanismHostSubmissionDelay || got.HostTimingBasis != "pre_entry" || got.PrimaryFinding.Rule != "host_pre_entry_delay" {
					t.Fatalf("validated submission thread not admitted: %+v", got)
				}
				return
			}
			if got.Admitted || got.Mechanism != cudadiag.MechanismWithheld || got.WithheldReason != tc.reason || got.PrimaryFinding.Rule != tc.rule || got.PrimaryFinding.Admission.ReasonCode != tc.reason || got.PrimaryFinding.Admission.Admitted {
				t.Fatalf("contradictory pre-entry target not refused by %s/%s: %+v", tc.reason, tc.rule, got)
			}
		})
	}
}
