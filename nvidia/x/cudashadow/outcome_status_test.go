package cudashadow

import (
	"bytes"
	"context"
	"reflect"
	"testing"
)

func TestRecordOutcomeStatus(t *testing.T) {
	for _, tt := range []struct {
		name   string
		status InputStatus
		change func(*Input)
	}{
		{"stale", StatusStale, func(in *Input) { in.Provenance.TimestampNano = 21 }},
		{"unsupported", StatusUnsupported, func(in *Input) { in.Provenance.Hook = "unknown" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			e, err := NewEvaluator(1)
			if err != nil {
				t.Fatal(err)
			}
			e.SetClock(func() int64 { return 20 })
			in := shadowInput()
			tt.change(&in)
			snap, err := e.Evaluate(context.Background(), &inspectionPolicy{e: e}, in)
			if snap.Status != tt.status || (err != nil) != (tt.status == StatusUnsupported) {
				t.Fatalf("Evaluate = status %q, %v; want %q", snap.Status, err, tt.status)
			}
			before := e.Snapshots()
			var history bytes.Buffer
			if err := WriteHistory(&history, before); err != nil {
				t.Fatalf("initial history: %v", err)
			}
			for _, outcome := range []ObservedOutcome{
				{TimestampNano: in.Provenance.TimestampNano, Action: "baseline"},
				{TimestampNano: in.Provenance.TimestampNano + 1, Action: "baseline"},
			} {
				if err := e.RecordOutcome(snap.ID, outcome); err == nil {
					t.Errorf("RecordOutcome admitted %s snapshot", snap.Status)
				}
				if got := e.Snapshots(); !reflect.DeepEqual(got, before) {
					t.Fatalf("rejected outcome changed snapshots: got %+v, want %+v", got, before)
				}
			}
			history.Reset()
			if err := WriteHistory(&history, e.Snapshots()); err != nil {
				t.Fatalf("history after rejection: %v", err)
			}
			got, err := ReadHistory(&history)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, before) {
				t.Fatalf("history changed: got %+v, want %+v", got, before)
			}
		})
	}
}

func TestRecordOutcomeErrorOrder(t *testing.T) {
	e, err := NewEvaluator(2)
	if err != nil {
		t.Fatal(err)
	}
	e.SetClock(func() int64 { return 20 })
	pol := &inspectionPolicy{e: e}
	valid, err := e.Evaluate(context.Background(), pol, shadowInput())
	if err != nil {
		t.Fatal(err)
	}
	in := shadowInput()
	in.Provenance.TimestampNano = 21
	stale, err := e.Evaluate(context.Background(), pol, in)
	if err != nil || stale.Status != StatusStale {
		t.Fatalf("Evaluate = %+v, %v; want stale", stale, err)
	}
	for _, tt := range []struct {
		name    string
		id      string
		outcome ObservedOutcome
		want    string
	}{
		{"invalid before missing", "missing", ObservedOutcome{}, "cudashadow: invalid observed outcome"},
		{"invalid before status", stale.ID, ObservedOutcome{}, "cudashadow: invalid observed outcome"},
		{"missing", "missing", ObservedOutcome{TimestampNano: 22, Action: "baseline"}, `cudashadow: snapshot "missing" not found`},
		{"timestamp before status", stale.ID, ObservedOutcome{TimestampNano: 20, Action: "baseline"}, "cudashadow: outcome precedes input"},
		{"timestamp", valid.ID, ObservedOutcome{TimestampNano: 9, Action: "baseline"}, "cudashadow: outcome precedes input"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			before := e.Snapshots()
			if err := e.RecordOutcome(tt.id, tt.outcome); err == nil || err.Error() != tt.want {
				t.Fatalf("RecordOutcome = %v, want %q", err, tt.want)
			}
			if !reflect.DeepEqual(e.Snapshots(), before) {
				t.Fatal("rejected outcome changed snapshots")
			}
		})
	}
	outcome := ObservedOutcome{TimestampNano: valid.Provenance.TimestampNano, Action: "baseline", Details: map[string]string{"source": "baseline"}}
	if err := e.RecordOutcome(valid.ID, outcome); err != nil {
		t.Fatal(err)
	}
	want := outcome
	want.Details = map[string]string{"source": "baseline"}
	outcome.Details["source"] = "changed"
	if got := e.Snapshots()[0].ObservedOutcome; !reflect.DeepEqual(got, &want) {
		t.Fatalf("recorded outcome = %+v, want %+v", got, &want)
	}
	before := e.Snapshots()
	for _, tt := range []struct {
		outcome ObservedOutcome
		want    string
	}{
		{ObservedOutcome{}, "cudashadow: invalid observed outcome"},
		{ObservedOutcome{TimestampNano: 9, Action: "baseline"}, "cudashadow: outcome already recorded"},
		{ObservedOutcome{TimestampNano: 22, Action: "baseline"}, "cudashadow: outcome already recorded"},
	} {
		if err := e.RecordOutcome(valid.ID, tt.outcome); err == nil || err.Error() != tt.want {
			t.Fatalf("duplicate RecordOutcome = %v, want %q", err, tt.want)
		}
		if !reflect.DeepEqual(e.Snapshots(), before) {
			t.Fatal("duplicate outcome changed snapshots")
		}
	}
	var history bytes.Buffer
	if err := WriteHistory(&history, e.Snapshots()); err != nil {
		t.Fatalf("history with valid outcome: %v", err)
	}
	got, err := ReadHistory(&history)
	if err != nil || !reflect.DeepEqual(got, before) {
		t.Fatalf("ReadHistory = %+v, %v; want %+v", got, err, before)
	}
}
