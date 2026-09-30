package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGPUEdgeQueryReceipt(t *testing.T) {
	tests := []struct {
		name             string
		edge, start, end uint64
		unavailable      bool
	}{
		{"normal", 100, 120, 150, false},
		{"query before edge", 120, 100, 150, true},
		{"clock reversed", 100, 120, 110, true},
		{"lag overflow", 1, 1<<63 + 1, 1<<63 + 2, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newGPUEdgeSample(tt.edge, tt.start, tt.end, 5000000, fakeDevices)
			if (r.Unavailable != "") != tt.unavailable {
				t.Fatalf("receipt = %+v", r)
			}
			if tt.unavailable {
				return
			}
			if r.LagNanos != 20 || r.ReadNanos != 30 || r.ObservedPollGapNanos != 5000000 || r.RequestedPollNanos != 1000000 {
				t.Fatalf("receipt = %+v", r)
			}
			data, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{"CLOCK_MONOTONIC", "no sensor observation timestamp", "unmeasured"} {
				if !strings.Contains(string(data), want) {
					t.Errorf("receipt missing %q: %s", want, data)
				}
			}
		})
	}
}
