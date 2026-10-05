package cudaschedule

import (
	"testing"
	"time"
)

func TestRecordedLimits(t *testing.T) {
	valid := recordedLimits{Queue: time.Second, Admission: 2 * time.Second, Runtime: 3 * time.Second, Cleanup: time.Second, Lease: 6 * time.Second}
	if total, err := valid.validate(); err != nil || total != 7*time.Second {
		t.Fatalf("queue/runtime budget merged: %v %v", total, err)
	}
	for _, tt := range []struct {
		name   string
		change func(*recordedLimits)
	}{
		{"no queue default", func(l *recordedLimits) { l.Queue = 0 }},
		{"negative runtime", func(l *recordedLimits) { l.Runtime = -1 }},
		{"oversized admission", func(l *recordedLimits) { l.Admission = 25 * time.Hour }},
		{"no cleanup", func(l *recordedLimits) { l.Cleanup = 0 }},
		{"short lease", func(l *recordedLimits) { l.Lease = 5 * time.Second }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			l := valid
			tt.change(&l)
			if _, err := l.validate(); err == nil {
				t.Fatal("accepted invalid explicit limits")
			}
		})
	}
}

func TestRecordedControllerOutsideLease(t *testing.T) {
	keys := []string{"GPU_LOCK_HELD", "GPU_LOCK_DISABLED", "CUDATEST_NO_LOCK"}
	for _, key := range keys {
		t.Setenv(key, "")
	}
	if err := recordedOutsideLease(); err != nil {
		t.Fatal(err)
	}
	for _, key := range keys {
		t.Run(key, func(t *testing.T) {
			t.Setenv(key, "123")
			if err := recordedOutsideLease(); err == nil {
				t.Fatal("accepted inherited or bypass controller")
			}
		})
	}
}
