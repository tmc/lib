package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func TestSnapshotClockUncertaintySource(t *testing.T) {
	for _, tt := range []struct {
		name         string
		uncertainty  int64
		wantSource   string
		wantReliable bool
	}{
		{"unavailable", 0, "unavailable", false},
		{"configured", 1000, "configured", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := cudaflightrecorder.New(cudaflightrecorder.Config{GoTraceEnabled: true, ClockUncertaintyNs: tt.uncertainty})
			if err != nil {
				t.Fatal(err)
			}
			if err := r.Start(); err != nil {
				t.Fatal(err)
			}
			r.Push(cudaflightrecorder.Record{TimestampNs: time.Now().UnixNano(), DurationNs: 1_000_000})
			var buf bytes.Buffer
			manifest, err := r.Snapshot(context.Background(), &buf, tt.name)
			if err != nil {
				t.Fatal(err)
			}
			if manifest.ClockUncertaintySource != tt.wantSource || manifest.ClockUncertaintyNs != map[int64]int64{0: -1, 1000: 1000}[tt.uncertainty] || manifest.OverlapReliable != tt.wantReliable {
				t.Fatalf("manifest source=%q uncertainty=%d reliable=%t", manifest.ClockUncertaintySource, manifest.ClockUncertaintyNs, manifest.OverlapReliable)
			}
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
