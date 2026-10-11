package cudacapture

import "testing"

var (
	_ func(*CorrelationTracker, uint64, uint64) error   = (*CorrelationTracker).RecordExternalCorrelation
	_ func(*CorrelationTracker, uint64, []string) error = (*CorrelationTracker).RecordExternalStack
)

func TestDefaultCorrelationNamespace(t *testing.T) {
	tracker := NewCorrelationTracker()
	if err := tracker.RecordExternalCorrelation(9, 1); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordExternalStack(9, []string{"default"}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordExternalStackKind(4, 9, []string{"other"}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordRuntime(RuntimeLaunch{CorrelationID: 1}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordActivity(ActivityRecord{Subtype: "kernel", Correlation: 1, Start: 1, End: 2}); err != nil {
		t.Fatal(err)
	}
	joined, ambiguous, _, _ := tracker.Join()
	if len(joined) != 1 || len(ambiguous) != 0 || len(joined[0].Callstack) != 1 || joined[0].Callstack[0] != "default" {
		t.Fatalf("joins = %v, ambiguous = %v", joined, ambiguous)
	}
}
