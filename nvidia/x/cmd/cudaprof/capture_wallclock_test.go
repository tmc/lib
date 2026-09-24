package main

import (
	"runtime"
	"testing"
	"time"
)

func TestCaptureWallClockAt(t *testing.T) {
	c := &captureWallClock{MonotonicNanos: 5e9, UnixNanos: time.Date(2026, 9, 24, 3, 0, 0, 0, time.UTC).UnixNano()}
	base := time.Unix(0, c.UnixNanos)
	tests := []struct {
		name string
		mono uint64
		want time.Time
	}{
		{"at the reading", 5e9, base},
		{"after", 5e9 + 1500e6, base.Add(1500 * time.Millisecond)},
		{"before", 5e9 - 2e9, base.Add(-2 * time.Second)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := c.at(tt.mono); !got.Equal(tt.want) {
				t.Errorf("at(%d) = %s, want %s", tt.mono, got, tt.want)
			}
		})
	}
}

func TestReadWallClock(t *testing.T) {
	c := readWallClock()
	if c == nil {
		if runtime.GOOS == "windows" {
			t.Skip("no CLOCK_MONOTONIC")
		}
		t.Fatal("no wall clock pairing on a unix host")
	}
	// A fresh monotonic reading placed through the pairing must land on the
	// wall clock read beside it.
	m, _ := monotonicNanos()
	now := time.Now()
	if d := c.at(m).Sub(now); d < -time.Millisecond || d > time.Millisecond {
		t.Errorf("pairing is off by %s", d)
	}
}
