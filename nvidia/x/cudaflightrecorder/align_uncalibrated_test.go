package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

// checkClockUncertainty verifies a manifest's clock fields against each other
// and against the configured estimate, where configuredNs is the recorder's
// Config.ClockUncertaintyNs.
//
// It deliberately does not assert WHICH source won. Start calibrates against
// CUPTI whenever a device is present, and a measurement supersedes a configured
// estimate, so the source is a property of the host the test runs on. Asserting
// a literal makes the test pass only where the capability is absent. The
// precedence rule itself is covered by TestClockUncertaintyPrecedence and the
// reliability rule by TestAlignSpansUncertaintyInputs; what is left to check
// here is that Snapshot wires them together consistently.
func checkClockUncertainty(t *testing.T, m *cudaflightrecorder.Manifest, configuredNs int64) {
	t.Helper()
	switch m.ClockUncertaintySource {
	case "unavailable":
		if configuredNs > 0 {
			t.Errorf("source = unavailable with ClockUncertaintyNs = %d configured", configuredNs)
		}
		if m.ClockUncertaintyNs != -1 {
			t.Errorf("unavailable uncertainty = %d, want -1", m.ClockUncertaintyNs)
		}
		if m.OverlapReliable {
			t.Errorf("OverlapReliable = true with no clock uncertainty")
		}
	case "configured":
		if m.ClockUncertaintyNs != configuredNs {
			t.Errorf("configured uncertainty = %d, want %d", m.ClockUncertaintyNs, configuredNs)
		}
	case "measured":
		if m.ClockUncertaintyNs <= 0 {
			t.Errorf("measured uncertainty = %d, want > 0", m.ClockUncertaintyNs)
		}
	default:
		t.Errorf("source = %q, want measured, configured or unavailable", m.ClockUncertaintySource)
	}
}

func TestSnapshotClockUncertaintySource(t *testing.T) {
	for _, tt := range []struct {
		name        string
		uncertainty int64
	}{
		{"unconfigured", 0},
		{"configured", 1000},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := cudaflightrecorder.New(cudaflightrecorder.Config{GoTraceEnabled: true, ClockUncertaintyNs: tt.uncertainty})
			if err != nil {
				t.Fatal(err)
			}
			if err := r.Start(); err != nil {
				t.Fatal(err)
			}
			defer r.Stop()
			time.Sleep(5 * time.Millisecond)
			now := time.Now().UnixNano()
			r.Push(cudaflightrecorder.Record{TimestampNs: now - 2_000_000, DurationNs: 1_000_000})
			var buf bytes.Buffer
			manifest, err := r.Snapshot(context.Background(), &buf, tt.name)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.HasPrefix(manifest.ProviderStatus["go.trace"], "unvalidated:") || manifest.GoSpanNs != 0 || manifest.OverlapSpanNs != 0 || manifest.OverlapReliable {
				t.Fatalf("unvalidated trace admitted timing: %+v", manifest)
			}
			checkClockUncertainty(t, manifest, tt.uncertainty)
		})
	}
}

func TestAlignSpansAllUncalibrated(t *testing.T) {
	goSpan, gpuSpan, overlap, reliable := cudaflightrecorder.AlignSpans(
		200, 100, 100, true,
		[]cudaflightrecorder.Record{{TimestampNs: 0, DurationNs: 50, Labels: map[string]string{"activity.uncalibrated": "true"}}},
		0,
	)
	if goSpan != 100 || gpuSpan != 0 || overlap != 0 || reliable {
		t.Fatalf("AlignSpans = (%d, %d, %d, %t), want (100, 0, 0, false)", goSpan, gpuSpan, overlap, reliable)
	}
}

func TestAlignSpansUncertaintyInputs(t *testing.T) {
	tests := []struct {
		name         string
		uncertainty  int64
		wantReliable bool
	}{
		{"unavailable", -1, false},
		{"configured", 1, true},
		{"zero input", 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, reliable := cudaflightrecorder.AlignSpans(2_000_000, 1_000_000, 1_000_000, true,
				[]cudaflightrecorder.Record{{TimestampNs: 1_000_000, DurationNs: 1_000_000}}, tt.uncertainty)
			if reliable != tt.wantReliable {
				t.Fatalf("reliable = %t, want %t", reliable, tt.wantReliable)
			}
		})
	}
}
