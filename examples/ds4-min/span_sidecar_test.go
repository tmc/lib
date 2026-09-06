package main

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestHostEventSidecar checks the sidecar against the constraints
// gputrace's reader enforces (internal/hostevents): exact schema and
// clock names, kind "interval", one run ID, unique non-empty untrimmed
// IDs and names, and a non-zero duration. A file failing any of them is
// rejected whole, so the assertions are on every record.
func TestHostEventSidecar(t *testing.T) {
	r := newSpanRecorder(levelOp, "run-1")
	spans = r
	t.Cleanup(func() { spans = nil })

	outer := begin(levelPhase, "generation", nil)
	for i := range 3 {
		token := begin(levelToken, "token", nil)
		beginLayer(i).end()
		beginOp("batch.wait").end()
		token.end()
	}
	outer.end()
	// A level deeper than the recorder's is not recorded, and its end is
	// a no-op rather than a panic.
	quiet := newSpanRecorder(levelPhase, "run-1")
	spans = quiet
	beginOp("batch.wait").end()
	if got := quiet.Len(); got != 0 {
		t.Errorf("Len() = %d after a levelOp span on a levelPhase recorder, want 0", got)
	}
	spans = r

	path := filepath.Join(t.TempDir(), "spans.jsonl")
	if err := r.WriteHostEvents(path); err != nil {
		t.Fatalf("WriteHostEvents: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	ids := make(map[string]bool)
	names := make(map[string]int)
	for _, line := range strings.Split(strings.TrimSuffix(string(data), "\n"), "\n") {
		dec := json.NewDecoder(strings.NewReader(line))
		dec.DisallowUnknownFields()
		var rec hostEventRecord
		if err := dec.Decode(&rec); err != nil {
			t.Fatalf("decode %q: %v", line, err)
		}
		if err := dec.Decode(&struct{}{}); err != io.EOF {
			t.Fatalf("trailing data in %q", line)
		}
		switch {
		case rec.Schema != "gputrace.host-event/v1":
			t.Errorf("schema = %q", rec.Schema)
		case rec.Kind != "interval":
			t.Errorf("kind = %q", rec.Kind)
		case rec.ClockDomain != "cpu_uptime_ns":
			t.Errorf("clock_domain = %q", rec.ClockDomain)
		case rec.RunID != "run-1":
			t.Errorf("run_id = %q", rec.RunID)
		case rec.ID == "" || rec.ID != strings.TrimSpace(rec.ID):
			t.Errorf("id = %q", rec.ID)
		case rec.Name == "" || rec.Name != strings.TrimSpace(rec.Name):
			t.Errorf("name = %q", rec.Name)
		case rec.DurationNS == 0:
			t.Errorf("%s: duration_ns = 0", rec.Name)
		}
		if ids[rec.ID] {
			t.Errorf("duplicate id %q", rec.ID)
		}
		ids[rec.ID] = true
		names[rec.Name]++
	}
	want := map[string]int{"generation": 1, "token": 3, "layer": 3, "batch.wait": 3}
	for name, count := range want {
		if names[name] != count {
			t.Errorf("%q recorded %d times, want %d", name, names[name], count)
		}
	}
}

// TestGPUEventSidecar checks the spans appended to GPUTRACE_APP_EVENTS
// carry the clock declaration that lets a .gpucapture bundle translate
// them, and that the layer index survives as a label.
func TestGPUEventSidecar(t *testing.T) {
	r := newSpanRecorder(levelLayer, "run-2")
	spans = r
	t.Cleanup(func() { spans = nil })
	beginLayer(7).end()

	path := filepath.Join(t.TempDir(), "app_events.jsonl")
	if err := r.AppendGPUEvents(path); err != nil {
		t.Fatalf("AppendGPUEvents: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got gpuEventSpan
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode %q: %v", data, err)
	}
	// The decoder dispatches on kind; a record without it is dropped and
	// the capture reports a workload that emitted no spans.
	if got.Kind != "span" || got.Name != "layer" || got.Clock != "unix" || got.Labels["layer"] != "7" {
		t.Errorf("span = %+v, want kind span, name layer, clock unix, label layer=7", got)
	}
	if got.EndNS <= got.StartNS {
		t.Errorf("end_ns %d does not follow start_ns %d", got.EndNS, got.StartNS)
	}
}

func TestParseSpanLevel(t *testing.T) {
	for _, tt := range []struct {
		in   string
		want spanLevel
		bad  bool
	}{
		{in: "off", want: levelOff},
		{in: "phase", want: levelPhase},
		{in: "token", want: levelToken},
		{in: "layer", want: levelLayer},
		{in: "op", want: levelOp},
		{in: "Op", bad: true},
		{in: "", bad: true},
	} {
		got, err := parseSpanLevel(tt.in)
		if (err != nil) != tt.bad {
			t.Errorf("parseSpanLevel(%q) error = %v, want bad = %v", tt.in, err, tt.bad)
		}
		if err == nil && got != tt.want {
			t.Errorf("parseSpanLevel(%q) = %v, want %v", tt.in, got, tt.want)
		}
	}
}
