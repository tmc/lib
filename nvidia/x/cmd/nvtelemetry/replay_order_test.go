package main

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestReplayPhaseOutputIsSorted(t *testing.T) {
	phases := []nvtelemetry.PhaseMarker{
		{Name: "zeta", StartMonotonicNanos: 1, EndMonotonicNanos: 2},
		{Name: "alpha", StartMonotonicNanos: 2, EndMonotonicNanos: 3},
		{Name: "middle", StartMonotonicNanos: 3, EndMonotonicNanos: 4},
	}

	path := filepath.Join(t.TempDir(), "replay.jsonl")
	var stream bytes.Buffer
	writer := nvtelemetry.NewWriter(&stream, nvtelemetry.CollectorConfig{})
	manifest := &nvtelemetry.Manifest{
		RunID:            "replay-order",
		CollectorVersion: "test-collector",
		SchemaVersion:    nvtelemetry.SchemaVersion,
	}
	if err := writer.WriteManifest(manifest); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}
	for i := range phases {
		if err := writer.WritePhase(&phases[i]); err != nil {
			t.Fatalf("WritePhase(%q): %v", phases[i].Name, err)
		}
	}
	if err := os.WriteFile(path, stream.Bytes(), 0o600); err != nil {
		t.Fatalf("write replay fixture: %v", err)
	}

	var stdout, stderr bytes.Buffer
	if err := runReplay([]string{"-input=" + path}, &stdout, &stderr); err != nil {
		t.Fatalf("runReplay: %v; stderr: %s", err, stderr.String())
	}
	want := "Replayed run replay-order (collector: test-collector, devices: 0, complete: false)\n" +
		"  Phase: alpha (0 metrics, 0 energy summaries)\n" +
		"  Phase: middle (0 metrics, 0 energy summaries)\n" +
		"  Phase: zeta (0 metrics, 0 energy summaries)\n"
	if got := stdout.String(); got != want {
		t.Fatalf("replay output mismatch\ngot:\n%s\nwant:\n%s", got, want)
	}
}

func TestSortedPhaseNamesControls(t *testing.T) {
	tests := []struct {
		name string
		in   map[string][]nvtelemetry.WindowSummary
		want []string
	}{
		{name: "positive", in: map[string][]nvtelemetry.WindowSummary{"zeta": nil, "alpha": nil, "middle": nil}, want: []string{"alpha", "middle", "zeta"}},
		{name: "single", in: map[string][]nvtelemetry.WindowSummary{"only": nil}, want: []string{"only"}},
		{name: "empty", in: map[string][]nvtelemetry.WindowSummary{}, want: []string{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := sortedPhaseNames(tt.in); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("sortedPhaseNames() = %v, want %v", got, tt.want)
			}
		})
	}
}
