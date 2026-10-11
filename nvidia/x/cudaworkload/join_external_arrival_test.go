package cudaworkload

import (
	"fmt"
	"testing"
)

func TestJoinExternalArrival(t *testing.T) {
	proc := ProcessGenID{PID: 10, Generation: 2}
	for _, deviceFirst := range []bool{false, true} {
		for _, ids := range [][2]uint32{{0, 0}, {8, 0}, {7, 0}, {8, 7}, {7, 7}} {
			t.Run(fmt.Sprintf("deviceFirst=%v/ids=%v", deviceFirst, ids), func(t *testing.T) {
				j := NewJoiner(JoinConfig{})
				host := HostSubmission{Process: proc, OpID: 23, NativeCorrelationID: 7}
				act := DeviceActivity{Process: proc, NativeCorrelationID: ids[0], RuntimeCorrelationID: ids[1], ExternalCorrelationID: 23}
				var got []JoinResult
				if deviceFirst {
					got = append(got, j.SubmitDevice(act)...)
					got = append(got, j.SubmitHost(host)...)
				} else {
					got = append(got, j.SubmitHost(host)...)
					got = append(got, j.SubmitDevice(act)...)
				}
				got = append(got, j.Flush()...)
				if len(got) != 1 || got[0].Outcome != OutcomeExact || got[0].Reason != ReasonExplicitExternalMatch {
					t.Fatalf("external correlation did not match: %+v", got)
				}
				if got[0].RawIDs.ExternalCorrelationID != 23 || got[0].RawIDs.NativeCorrelationID != ids[0] || got[0].RawIDs.RuntimeCorrelationID != ids[1] || got[0].DeviceActivity == nil || *got[0].DeviceActivity != act {
					t.Fatalf("raw activity changed: %+v", got[0])
				}
				before := j.Metrics()
				if before.TotalActivities != 1 || before.ExactJoins != 1 || before.MissingFraction() != 0 || before.AmbiguousFraction() != 0 || len(j.Flush()) != 0 || j.Metrics() != before {
					t.Fatalf("activity accounting changed: %+v", j.Metrics())
				}
			})
		}
	}
}

func TestJoinExternalCandidateConflict(t *testing.T) {
	proc := ProcessGenID{PID: 10, Generation: 2}
	for _, deviceFirst := range []bool{false, true} {
		t.Run(fmt.Sprintf("deviceFirst=%v", deviceFirst), func(t *testing.T) {
			j := NewJoiner(JoinConfig{})
			if deviceFirst {
				j.SubmitDevice(DeviceActivity{Process: proc, NativeCorrelationID: 7})
				j.SubmitDevice(DeviceActivity{Process: proc, ExternalCorrelationID: 23})
				got := j.SubmitHost(HostSubmission{Process: proc, OpID: 23, NativeCorrelationID: 7})
				if len(got) != 1 || got[0].Outcome != OutcomeAmbiguous || got[0].DeviceActivity != nil {
					t.Fatalf("different activity candidates admitted: %+v", got)
				}
				j.Flush()
				if j.Metrics().MissingActivities != 2 || j.Metrics().AmbiguousActivities != 0 {
					t.Fatalf("host-only ambiguity changed device counts: %+v", j.Metrics())
				}
			} else {
				// Both hosts assert the same external operation, but only one
				// matches the native ID. Explicit contradictory evidence remains
				// ambiguous even when the native index has a unique candidate.
				j.SubmitHost(HostSubmission{Process: proc, OpID: 23, NativeCorrelationID: 7})
				j.SubmitHost(HostSubmission{Process: proc, OpID: 23, NativeCorrelationID: 8})
				got := j.SubmitDevice(DeviceActivity{Process: proc, NativeCorrelationID: 7, ExternalCorrelationID: 23})
				if len(got) != 1 || got[0].Outcome != OutcomeAmbiguous || got[0].DeviceActivity == nil {
					t.Fatalf("duplicate external candidates admitted: %+v", got)
				}
				j.Flush()
				if j.Metrics().AmbiguousActivities != 1 || j.Metrics().AmbiguousFraction() != 1 || j.Metrics().MissingActivities != 0 {
					t.Fatalf("device ambiguity changed fractions: %+v", j.Metrics())
				}
			}
		})
	}
}

func TestJoinExternalIdentityConflict(t *testing.T) {
	for _, change := range []string{"process", "context", "replay", "zero operation", "zero external", "zero pid", "zero generation"} {
		for _, deviceFirst := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/deviceFirst=%v", change, deviceFirst), func(t *testing.T) {
				j := NewJoiner(JoinConfig{})
				proc := ProcessGenID{PID: 10, Generation: 2}
				host := HostSubmission{Process: proc, OpID: 23}
				act := DeviceActivity{Process: proc, ExternalCorrelationID: 23}
				switch change {
				case "process":
					host.Process.Generation++
				case "context":
					host.Context.Context++
				case "replay":
					host.GraphReplay.ReplayGeneration++
				case "zero operation":
					host.OpID = 0
				case "zero external":
					act.ExternalCorrelationID = 0
				case "zero pid":
					host.Process.PID, act.Process.PID = 0, 0
				case "zero generation":
					host.Process.Generation, act.Process.Generation = 0, 0
				}
				var got []JoinResult
				if deviceFirst {
					got = append(got, j.SubmitDevice(act)...)
					got = append(got, j.SubmitHost(host)...)
				} else {
					got = append(got, j.SubmitHost(host)...)
					got = append(got, j.SubmitDevice(act)...)
				}
				got = append(got, j.Flush()...)
				for _, result := range got {
					if result.Outcome == OutcomeExact {
						t.Fatalf("contradictory or absent identity admitted: %+v", result)
					}
				}
			})
		}
	}
}

func TestJoinExternalDuplicateActivities(t *testing.T) {
	j := NewJoiner(JoinConfig{})
	proc := ProcessGenID{PID: 10, Generation: 2}
	for i := 0; i < 2; i++ {
		j.SubmitDevice(DeviceActivity{Process: proc, ExternalCorrelationID: 23})
	}
	got := j.SubmitHost(HostSubmission{Process: proc, OpID: 23})
	if len(got) != 1 || got[0].Outcome != OutcomeAmbiguous || got[0].DeviceActivity != nil {
		t.Fatalf("duplicate activities were matched: %+v", got)
	}
	j.Flush()
	if j.Metrics().MissingActivities != 2 || j.Metrics().ExactJoins != 0 || j.Metrics().AmbiguousActivities != 0 {
		t.Fatalf("duplicate activity population changed: %+v", j.Metrics())
	}
}

func TestJoinExternalActivityReindex(t *testing.T) {
	proc := ProcessGenID{PID: 10, Generation: 2}
	j := NewJoiner(JoinConfig{MaxPendingRecords: 2})
	for id := uint64(1); id <= 3; id++ {
		j.SubmitDevice(DeviceActivity{Process: proc, ExternalCorrelationID: id})
	}
	for _, id := range []OpID{3, 2} {
		got := j.SubmitHost(HostSubmission{Process: proc, OpID: id})
		if len(got) != 1 || got[0].Outcome != OutcomeExact || got[0].OpID != id {
			t.Fatalf("external index not rebuilt for %d: %+v", id, got)
		}
	}
	before := j.Metrics()
	if before.MissingActivities != 1 || before.ExactJoins != 2 || len(j.Flush()) != 0 || j.Metrics() != before {
		t.Fatalf("eviction or consumption left stale external index: %+v", j.Metrics())
	}
	j.SubmitHost(HostSubmission{Process: proc, OpID: 3})
	got := j.Flush()
	if len(got) != 1 || got[0].DeviceActivity != nil || got[0].Outcome != OutcomeMissing {
		t.Fatalf("consumed activity reused after flush: %+v", got)
	}
}
