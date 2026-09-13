package egraph

import (
	"testing"
	"time"
)

func TestPGOTrackerMean(t *testing.T) {
	var tracker PGOTracker
	if _, ok := tracker.Score("missing"); ok {
		t.Fatal("Score reported an unmeasured candidate")
	}
	tracker.Record("candidate", 3*time.Nanosecond)
	tracker.Record("candidate", 5*time.Nanosecond)
	if got, ok := tracker.Score("candidate"); !ok || got != 4*time.Nanosecond {
		t.Fatalf("Score = %v, %v, want 4ns, true", got, ok)
	}
	tracker.Record("", time.Second)
	tracker.Record("candidate", 0)
	if got, _ := tracker.Score("candidate"); got != 4*time.Nanosecond {
		t.Fatalf("invalid observations changed score to %v", got)
	}
}
