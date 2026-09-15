package main

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

// TestBaseline samples the load without waiting, so the seams stand in
// for the clock and the kernel.
func TestBaseline(t *testing.T) {
	for _, tt := range []struct {
		name    string
		loads   []float64
		maxLoad float64
		want    []string
		wantErr error
	}{
		{
			name: "quiet", loads: []float64{1, 2, 1.5}, maxLoad: 4,
			want: []string{"min 1.00 max 2.00 mean 1.50, 0 over the limit", "limit is attainable"},
		},
		{
			// The Mac this gate runs on idles above its registered
			// 4.00 limit, so every measurement it takes is refused
			// before the model is loaded.
			name: "unstable", loads: []float64{3.45, 4.84, 4.09}, maxLoad: 4,
			want:    []string{"min 3.45 max 4.84 mean 4.13, 2 over the limit", "UNSTABLE"},
			wantErr: errDegraded,
		},
		{
			// A single reading would have passed at the first sample
			// and reported a quiet host.
			name: "one sample over", loads: []float64{1, 1, 9}, maxLoad: 4,
			want:    []string{"1 over the limit", "UNSTABLE"},
			wantErr: errDegraded,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			loads := tt.loads
			defer swap(&readLoad, func() (float64, error) {
				load := loads[0]
				loads = loads[1:]
				return load, nil
			})()
			var slept time.Duration
			defer swap(&sleep, func(d time.Duration) { slept += d })()

			var out bytes.Buffer
			cfg := config{maxLoad: tt.maxLoad, samples: len(tt.loads), interval: time.Second}
			err := baseline(cfg, &out)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("baseline: %v, want %v\n%s", err, tt.wantErr, out.String())
			}
			for _, want := range tt.want {
				if !strings.Contains(out.String(), want) {
					t.Errorf("output missing %q:\n%s", want, out.String())
				}
			}
			// One fewer wait than samples: nothing to wait for
			// before the first reading.
			if want := time.Duration(len(tt.loads)-1) * time.Second; slept != want {
				t.Errorf("slept %v, want %v", slept, want)
			}
			if len(loads) != 0 {
				t.Errorf("%d samples unread", len(loads))
			}
		})
	}
}

// TestBaselineReadError refuses to report a distribution it could not
// finish sampling.
func TestBaselineReadError(t *testing.T) {
	want := errors.New("boom")
	defer swap(&readLoad, func() (float64, error) { return 0, want })()
	defer swap(&sleep, func(time.Duration) {})()
	var out bytes.Buffer
	if err := baseline(config{maxLoad: 4, samples: 3}, &out); !errors.Is(err, want) {
		t.Fatalf("baseline: %v, want %v", err, want)
	}
	if out.Len() != 0 {
		t.Errorf("reported a distribution anyway:\n%s", out.String())
	}
}

// TestBaselineFlags rejects a baseline that would measure nothing or
// gate on a reference it does not read.
func TestBaselineFlags(t *testing.T) {
	for _, tt := range []struct {
		args    []string
		wantErr string
	}{
		{[]string{"-baseline", "-samples", "0"}, "samples must be positive"},
		{[]string{"-baseline", "-interval", "-1s"}, "interval must not be negative"},
		{[]string{"-baseline", "-reference", "r.csv", "-ctx", "2048"}, "baseline takes no reference"},
	} {
		_, err := parseConfig(tt.args, io.Discard)
		if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
			t.Errorf("parseConfig(%q) = %v, want %q", tt.args, err, tt.wantErr)
		}
	}
}

// swap replaces a seam and returns its restorer.
func swap[T any](p *T, v T) func() {
	old := *p
	*p = v
	return func() { *p = old }
}
