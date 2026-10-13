package main

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	languagestack "github.com/tmc/lib/nvidia/x/cudaprof/stack"
)

func TestLanguageSampleClock(t *testing.T) {
	for _, tt := range []struct {
		name       string
		session    string
		generation uint64
		unified    bool
		want       int
	}{
		{"same native clock", "session", 7, true, 2},
		{"other session", "other", 7, true, 1},
		{"other generation", "session", 8, true, 1},
		{"unknown native clock", "session", 7, false, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := languagestack.Sample{Version: 1, Type: "sample", Process: "producer", PID: 12, Session: "session", Generation: 7, Language: "python", TID: 13, Time: 100, Clock: "monotonic", Status: "clock_unavailable", Provenance: "wall_stack_snapshot", Frames: []languagestack.Frame{{Function: "waiting"}}}
			data, err := json.Marshal(s)
			if err != nil {
				t.Fatal(err)
			}
			data = append(data, '\n')
			cap := &cudacapture.Capture{Files: map[string][]byte{"stacks-python-wall-test.jsonl": data}}
			records := []traceRecord{{session: tt.session, process: 12, generation: tt.generation, clocksUnified: tt.unified}}
			got, err := appendLanguageSamples(context.Background(), cap, records, goTraceOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != tt.want {
				t.Fatalf("records = %d, want %d", len(got), tt.want)
			}
			if tt.want == 2 && (got[1].sample.Status != "native_clock_verified" || got[1].stack != nil) {
				t.Fatalf("invalid sample attribution: %+v", got[1])
			}
		})
	}
}
