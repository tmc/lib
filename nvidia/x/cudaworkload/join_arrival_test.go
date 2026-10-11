package cudaworkload

import (
	"fmt"
	"testing"
)

func TestJoinRuntimeArrival(t *testing.T) {
	for _, ids := range [][2]uint32{{8, 7}, {7, 7}, {0, 7}} {
		for _, deviceFirst := range []bool{false, true} {
			t.Run(fmt.Sprintf("%v/deviceFirst=%v", ids, deviceFirst), func(t *testing.T) {
				j := NewJoiner(JoinConfig{})
				proc := ProcessGenID{PID: 1, Generation: 1}
				host := HostSubmission{Process: proc, OpID: 1, NativeCorrelationID: 7}
				act := DeviceActivity{Process: proc, NativeCorrelationID: ids[0], RuntimeCorrelationID: ids[1]}
				var got []JoinResult
				if deviceFirst {
					got = append(got, j.SubmitDevice(act)...)
					got = append(got, j.SubmitHost(host)...)
				} else {
					got = append(got, j.SubmitHost(host)...)
					got = append(got, j.SubmitDevice(act)...)
				}
				got = append(got, j.Flush()...)
				if len(got) != 1 || got[0].Outcome != OutcomeExact {
					t.Fatalf("results = %+v", got)
				}
				if got[0].RawIDs.RuntimeCorrelationID != ids[1] {
					t.Fatal("runtime ID lost")
				}
				before := j.Metrics()
				if len(j.Flush()) != 0 || j.Metrics() != before {
					t.Fatal("flush repeated a consumed activity")
				}
			})
		}
	}
}

func TestJoinActivityMetrics(t *testing.T) {
	j := NewJoiner(JoinConfig{MaxPendingRecords: 2})
	for id := uint32(1); id <= 3; id++ {
		j.SubmitHost(HostSubmission{NativeCorrelationID: id})
	}
	j.SubmitDevice(DeviceActivity{NativeCorrelationID: 99})
	j.Flush()
	m := j.Metrics()
	if m.MissingJoins != 4 || m.MissingActivities != 1 || m.MissingFraction() != 1 {
		t.Fatalf("metrics = %+v", m)
	}
	if len(j.Flush()) != 0 || j.Metrics() != m {
		t.Fatal("flush changed counters twice")
	}
}

func TestJoinZeroNativeActivityRetained(t *testing.T) {
	for _, act := range []DeviceActivity{{}, {RuntimeCorrelationID: 7}, {ExternalCorrelationID: 9}} {
		j := NewJoiner(JoinConfig{})
		j.SubmitDevice(act)
		got := j.Flush()
		if len(got) != 1 || got[0].DeviceActivity == nil || *got[0].DeviceActivity != act || got[0].Outcome != OutcomeMissing {
			t.Fatalf("activity %+v: %+v", act, got)
		}
		if j.Metrics().MissingActivities != 1 {
			t.Fatal("missing activity not counted")
		}
	}
}

func TestJoinRuntimeReindex(t *testing.T) {
	j := NewJoiner(JoinConfig{MaxPendingRecords: 2})
	proc := ProcessGenID{PID: 1, Generation: 1}
	for id := uint32(1); id <= 3; id++ {
		j.SubmitDevice(DeviceActivity{Process: proc, RuntimeCorrelationID: id})
	}
	for id := uint32(2); id <= 3; id++ {
		got := j.SubmitHost(HostSubmission{Process: proc, OpID: OpID(id), NativeCorrelationID: id})
		if len(got) != 1 || got[0].Outcome != OutcomeExact {
			t.Fatalf("ID %d: %+v", id, got)
		}
	}
	if len(j.Flush()) != 0 || j.Metrics().MissingActivities != 1 {
		t.Fatal("eviction or removal left stale index")
	}
}

func TestJoinRuntimeIdentityConflict(t *testing.T) {
	proc := ProcessGenID{PID: 1, Generation: 1}
	for _, change := range []string{"process", "context", "replay"} {
		t.Run(change, func(t *testing.T) {
			j := NewJoiner(JoinConfig{})
			act := DeviceActivity{Process: proc, RuntimeCorrelationID: 7}
			host := HostSubmission{Process: proc, OpID: 1, NativeCorrelationID: 7}
			switch change {
			case "process":
				host.Process.Generation = 2
			case "context":
				host.Context.Context = 2
			case "replay":
				host.GraphReplay.ReplayGeneration = 2
			}
			j.SubmitDevice(act)
			got := append(j.SubmitHost(host), j.Flush()...)
			for _, r := range got {
				if r.Outcome == OutcomeExact {
					t.Fatalf("conflict admitted: %+v", r)
				}
			}
		})
	}
}

func ExampleJoinMetrics_MissingFraction() {
	m := JoinMetrics{TotalActivities: 2, MissingActivities: 1, MissingJoins: 4}
	fmt.Println(m.MissingFraction())
	// Output: 0.5
}

func TestJoinRuntimeCandidateConflict(t *testing.T) {
	j := NewJoiner(JoinConfig{})
	proc := ProcessGenID{PID: 1, Generation: 1}
	j.SubmitHost(HostSubmission{Process: proc, OpID: 1, NativeCorrelationID: 7})
	j.SubmitHost(HostSubmission{Process: proc, OpID: 2, NativeCorrelationID: 8})
	got := j.SubmitDevice(DeviceActivity{Process: proc, NativeCorrelationID: 8, RuntimeCorrelationID: 7})
	if len(got) != 1 || got[0].Outcome != OutcomeAmbiguous {
		t.Fatalf("conflicting aliases: %+v", got)
	}
	j.Flush()
	m := j.Metrics()
	if m.AmbiguousActivities != 1 || m.AmbiguousFraction() != 1 || m.MissingActivities != 0 || m.MissingFraction() != 0 {
		t.Fatalf("metrics = %+v", m)
	}
}

func TestJoinHostOnlyAmbiguityMetric(t *testing.T) {
	j := NewJoiner(JoinConfig{})
	proc := ProcessGenID{PID: 1, Generation: 1}
	for i := 0; i < 2; i++ {
		j.SubmitDevice(DeviceActivity{Process: proc, RuntimeCorrelationID: 7})
	}
	got := j.SubmitHost(HostSubmission{Process: proc, OpID: 1, NativeCorrelationID: 7})
	if len(got) != 1 || got[0].Outcome != OutcomeAmbiguous {
		t.Fatalf("multiple pending activities: %+v", got)
	}
	m := j.Metrics()
	if m.AmbiguousJoins != 1 || m.AmbiguousActivities != 0 || m.AmbiguousFraction() != 0 {
		t.Fatalf("host-only ambiguity: %+v", m)
	}
	j.Flush()
	if j.Metrics().MissingActivities != 2 {
		t.Fatal("pending activity population lost")
	}
}
