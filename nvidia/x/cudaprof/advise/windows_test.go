package advise

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestCaptureWindows(t *testing.T) {
	tests := []struct {
		name   string
		status string
		want   []captureWindow
	}{
		{"no status", ``, nil},
		{"no receipt", `{"status":{"dropped_records":0}}`, nil},
		{"unread", `{"status":{"phase_receipt":{"status":"UNREAD","reason":"record_loss"}}}`, nil},
		{"one range", `{"status":{"phase_receipt":{"status":"MEASURED","start_timestamp":10,"stop_timestamp":20}}}`, []captureWindow{{10, 20}}},
		{"unwrapped", `{"phase_receipt":{"status":"MEASURED","start_timestamp":10,"stop_timestamp":20}}`, []captureWindow{{10, 20}}},
		{
			"several ranges",
			`{"status":{"phase_receipt":{"status":"UNREAD","reason":"multiple_phases","phases":[
				{"status":"MEASURED","start_timestamp":10,"stop_timestamp":20},
				{"status":"MEASURED","start_timestamp":100,"stop_timestamp":120}]}}}`,
			[]captureWindow{{10, 20}, {100, 120}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := &cudacapture.Capture{Files: map[string][]byte{}}
			if tt.status != "" {
				cap.Files["capture-status.json"] = []byte(tt.status)
			}
			if got := captureWindows(cap); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("captureWindows = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestWindowedDutyCycle(t *testing.T) {
	ev := func(start, end uint64) timeline.Event {
		return timeline.Event{Subtype: "kernel", Start: start, End: end, DurationNs: end - start}
	}
	tests := []struct {
		name     string
		events   []timeline.Event
		windows  []captureWindow
		wantSpan uint64
		wantBusy uint64
		wantOK   bool
	}{
		{
			name:     "time between windows is not idle",
			events:   []timeline.Event{ev(0, 5), ev(5, 10), ev(1000, 1005), ev(1005, 1010)},
			windows:  []captureWindow{{0, 10}, {1000, 1010}},
			wantSpan: 20, wantBusy: 20, wantOK: true,
		},
		{
			name:     "idle inside a window is idle",
			events:   []timeline.Event{ev(0, 2), ev(8, 10), ev(1000, 1005), ev(1005, 1010)},
			windows:  []captureWindow{{0, 10}, {1000, 1010}},
			wantSpan: 20, wantBusy: 14, wantOK: true,
		},
		{
			name:     "activity crossing an edge is clipped",
			events:   []timeline.Event{ev(0, 5), ev(5, 15), ev(990, 1005), ev(1005, 1010)},
			windows:  []captureWindow{{0, 10}, {1000, 1010}},
			wantSpan: 20, wantBusy: 20, wantOK: true,
		},
		{
			name:     "activity outside every window is dropped",
			events:   []timeline.Event{ev(0, 5), ev(5, 10), ev(500, 600), ev(1000, 1005), ev(1005, 1010)},
			windows:  []captureWindow{{0, 10}, {1000, 1010}},
			wantSpan: 20, wantBusy: 20, wantOK: true,
		},
		{
			name:    "no activity in any window",
			events:  []timeline.Event{ev(500, 600), ev(700, 800)},
			windows: []captureWindow{{0, 10}, {1000, 1010}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			streams := map[uint32][]timeline.Event{7: tt.events}
			_, span, busy, _, ok := windowedDutyCycle(streams, tt.windows)
			if ok != tt.wantOK || span != tt.wantSpan || busy != tt.wantBusy {
				t.Errorf("windowedDutyCycle = span %d, busy %d, ok %v; want span %d, busy %d, ok %v", span, busy, ok, tt.wantSpan, tt.wantBusy, tt.wantOK)
			}
		})
	}
}

// TestWindowedCaptureIsNotIdle pins the scheduled-trainer report: a capture
// sampled in several short ranges read as a device idle for over 99% of its
// span, because the time between ranges was counted as idle.
func TestWindowedCaptureIsNotIdle(t *testing.T) {
	const ms = 1_000_000
	var samples []*profile.Sample
	for i, st := range []uint64{0, 5 * ms, 1000 * ms, 1005 * ms} {
		samples = append(samples, &profile.Sample{
			Location: []*profile.Location{{ID: uint64(i + 1), Line: []profile.Line{
				{Function: &profile.Function{ID: 1, Name: "work_kernel"}},
			}}},
			Value:    []int64{5 * ms},
			Label:    map[string][]string{"name": {"work_kernel"}},
			NumLabel: map[string][]int64{"stream": {7}, "start": {int64(st)}, "end": {int64(st + 5*ms)}},
		})
	}
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample:     samples,
		},
		Files: map[string][]byte{"capture-status.json": []byte(`{"status":{"phase_receipt":{"status":"UNREAD","reason":"multiple_phases","phases":[
			{"status":"MEASURED","start_timestamp":0,"stop_timestamp":10000000},
			{"status":"MEASURED","start_timestamp":1000000000,"stop_timestamp":1010000000}]}}}`)},
	}
	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "idle") {
			t.Errorf("a device busy throughout both capture windows was reported idle: %s", item.Finding)
		}
	}
	if report.CaptureWindows != 2 || report.SpanNanos != 20*ms {
		t.Errorf("report windows %d, span %d; want 2 windows, span %d", report.CaptureWindows, report.SpanNanos, 20*ms)
	}
	if md := report.MarkdownWithLimit(0); !strings.Contains(md, "2 capture windows") {
		t.Errorf("report does not say how many windows it measured:\n%s", md)
	}
}
