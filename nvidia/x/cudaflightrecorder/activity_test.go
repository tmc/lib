package cudaflightrecorder

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestPushActivityRecord(t *testing.T) {
	tests := []struct {
		name            string
		activity        cuptiactivity.Record
		wantDuration    uint64
		wantCorrelation uint64
	}{
		{name: "complete", activity: cuptiactivity.Record{Name: "kernel", Start: 10, End: 42, Complete: true, CorrelationID: 7}, wantDuration: 32, wantCorrelation: 7},
		{name: "incomplete", activity: cuptiactivity.Record{Name: "pending", Start: 10, End: 42, Complete: false, CorrelationID: ^uint32(0)}, wantDuration: 0, wantCorrelation: uint64(^uint32(0))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := New(Config{Capacity: 1})
			if err != nil {
				t.Fatal(err)
			}
			if err := r.Start(); err != nil {
				t.Fatal(err)
			}
			r.PushActivityRecord(tt.activity)
			if r.count != 1 {
				t.Fatalf("record count = %d, want 1", r.count)
			}
			got := r.records[0]
			if got.DurationNs != tt.wantDuration {
				t.Errorf("duration = %d, want %d", got.DurationNs, tt.wantDuration)
			}
			if got.CorrelationID != tt.wantCorrelation {
				t.Errorf("correlation ID = %d, want %d", got.CorrelationID, tt.wantCorrelation)
			}
		})
	}
}

func TestMapCUPTITimestamp(t *testing.T) {
	if got, want := mapCUPTITimestamp(1_000, 500, 750), int64(1_250); got != want {
		t.Fatalf("mapped timestamp = %d, want %d", got, want)
	}
	if got, want := mapCUPTITimestamp(1_000, 1_500, 750), int64(250); got != want {
		t.Fatalf("mapped timestamp before anchor = %d, want %d", got, want)
	}
}
