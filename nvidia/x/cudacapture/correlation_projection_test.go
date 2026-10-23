package cudacapture

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func projectionTracker(t *testing.T, rows []cudaprof.ExternalStackRecord) *CorrelationTracker {
	t.Helper()
	ct := NewCorrelationTracker()
	if rows == nil {
		rows = []cudaprof.ExternalStackRecord{}
	}
	data, err := json.Marshal(cudaprof.ExternalStackTable{Version: 1, Stacks: rows})
	if err != nil {
		t.Fatal(err)
	}
	if err := ct.ReadExternalStackTable(bytes.NewReader(data)); err != nil {
		t.Fatal(err)
	}
	return ct
}

func TestCorrelationProjectionSelectedKind(t *testing.T) {
	ct := projectionTracker(t, []cudaprof.ExternalStackRecord{{Process: 101, ExternalKind: 4, ExternalID: 7, Frames: []string{"Do", "selected"}}})
	if err := ct.RecordRuntime(RuntimeLaunch{CorrelationID: 77, HostThreadID: 9}); err != nil {
		t.Fatal(err)
	}
	for _, a := range []ActivityRecord{
		{Process: 101, Subtype: "external_correlation", Correlation: 77, ExternalKind: 3, ExternalID: 99},
		{Process: 101, Subtype: "external_correlation", Correlation: 77, ExternalKind: 4, ExternalID: 7},
		{Process: 101, Subtype: "kernel", Correlation: 77, Name: "kernel", Start: 100, End: 150, Valid: true},
	} {
		if err := ct.RecordActivity(a); err != nil {
			t.Fatal(err)
		}
	}
	joins, conflicts, _, _ := ct.JoinKind(4)
	if len(joins) != 1 || len(conflicts) != 0 || joins[0].ExternalID != 7 || joins[0].HostThreadID != 9 || !reflect.DeepEqual(joins[0].Callstack, []string{"Do", "selected"}) {
		t.Fatalf("join=%+v conflicts=%+v", joins, conflicts)
	}
	if !reflect.DeepEqual(ct.runtimes[77][0].Callstack, joins[0].Callstack) {
		t.Fatal("selected stack did not fill actual retained RuntimeLaunch")
	}
	joins[0].Callstack[0] = "mutated"
	id, frames, err := ct.ExternalCallstack(101, 77, 4)
	if id != 7 || err != nil || !reflect.DeepEqual(frames, []string{"Do", "selected"}) {
		t.Fatalf("sidecar lookup=%d %v %v", id, frames, err)
	}
	frames[0] = "mutated"
	_, again, _ := ct.ExternalCallstack(101, 77, 4)
	if again[0] != "Do" || ct.runtimes[77][0].Callstack[0] != "Do" {
		t.Fatal("projection returned aliased stack")
	}
	// Legacy Join still selects the lower namespace, even without its stack.
	legacy, _, _, _ := ct.Join()
	if len(legacy) != 1 || legacy[0].ExternalID != 99 || !legacy[0].CallstackMissing {
		t.Fatalf("legacy lowest-kind policy changed: %+v", legacy)
	}
}

func TestCorrelationProjectionMissingAmbiguous(t *testing.T) {
	for _, test := range []struct {
		name       string
		process    uint64
		links      []uint64
		rows       []cudaprof.ExternalStackRecord
		wantID     uint64
		wantFrames []string
		ambiguous  bool
	}{
		{name: "no link", process: 101, wantID: 0},
		{name: "no stack", process: 101, links: []uint64{7}, wantID: 7},
		{name: "foreign process", process: 202, links: []uint64{7}, rows: []cudaprof.ExternalStackRecord{{Process: 101, ExternalKind: 3, ExternalID: 7, Frames: []string{"foreign"}}}, wantID: 7},
		{name: "unknown process", process: 0, links: []uint64{7}, rows: []cudaprof.ExternalStackRecord{{Process: 101, ExternalKind: 3, ExternalID: 7, Frames: []string{"known"}}}, wantID: 7},
		{name: "conflicting links", process: 101, links: []uint64{7, 8}, ambiguous: true},
		{name: "conflicting stacks", process: 101, links: []uint64{7}, rows: []cudaprof.ExternalStackRecord{{Process: 101, ExternalKind: 3, ExternalID: 7, Frames: []string{"a"}}, {Process: 101, ExternalKind: 3, ExternalID: 7, Frames: []string{"b"}}}, ambiguous: true},
		{name: "idempotent duplicates", process: 101, links: []uint64{7, 7}, rows: []cudaprof.ExternalStackRecord{{Process: 101, ExternalKind: 3, ExternalID: 7, Frames: []string{"same"}}, {Process: 101, ExternalKind: 3, ExternalID: 7, Frames: []string{"same"}}}, wantID: 7, wantFrames: []string{"same"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			ct := projectionTracker(t, test.rows)
			for _, id := range test.links {
				if err := ct.RecordActivity(ActivityRecord{Process: test.process, Subtype: "external_correlation", Correlation: 77, ExternalKind: 3, ExternalID: id}); err != nil {
					t.Fatal(err)
				}
			}
			id, frames, err := ct.ExternalCallstack(test.process, 77, 3)
			if errors.Is(err, ErrAmbiguousCorrelation) != test.ambiguous || (!test.ambiguous && (err != nil || id != test.wantID || !reflect.DeepEqual(frames, test.wantFrames))) {
				t.Fatalf("id=%d frames=%v error=%v", id, frames, err)
			}
			if len(ct.runtimes) != 0 {
				t.Fatal("stack-only projection fabricated runtime records")
			}
		})
	}
}
