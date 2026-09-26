package main

import (
	"io"
	"strings"
	"testing"
)

// stretched is iterationRecs(n) with each iteration's gemm lengthened by
// extra ns, pushing later iterations back.
func stretched(n int, extra uint64) []rec {
	var recs []rec
	for i := range uint64(n) {
		t := 1000 + (1000+extra)*i
		recs = append(recs,
			rec{subtype: "kernel", name: "Embed", start: t, end: t + 100},
			rec{subtype: "kernel", name: "gemm_tf32", start: t + 100, end: t + 400 + extra},
		)
	}
	return recs
}

func TestCompareIterations(t *testing.T) {
	base := compareCapture(append([]rec{{subtype: "kernel", name: "load", start: 10, end: 900}}, stretched(5, 0)...), completeStatus())
	opts := func(args ...string) compareOptions {
		t.Helper()
		o, _, err := parseCompareArgs(append(args, "a", "b"), io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		return o
	}
	tests := []struct {
		name    string
		target  []rec
		opts    compareOptions
		window  [2]int64 // base, target
		note    string
		wantErr string
	}{
		{
			name:   "equal counts",
			target: stretched(5, 100),
			opts:   opts("-iteration-marker=^Embed$"),
			window: [2]int64{3000, 3300},
			note:   "window is 3 iterations per side; per iteration: window 0.001 -> 0.001 ms (+0.000)",
		},
		{
			name:    "counts differ",
			target:  stretched(7, 0),
			opts:    opts("-iteration-marker=^Embed$"),
			wantErr: "base holds 3 steady iterations, target 5; pass -iteration-count",
		},
		{
			name:   "explicit count",
			target: stretched(7, 0),
			opts:   opts("-iteration-marker=^Embed$", "-iteration-count=2", "-skip-iterations=2"),
			window: [2]int64{2000, 2000},
		},
		{
			name:    "count too large",
			target:  stretched(7, 0),
			opts:    opts("-iteration-marker=^Embed$", "-iteration-count=4"),
			wantErr: "base: -iteration-count 4: the window holds 3 steady iterations",
		},
		{
			name:    "no marker",
			target:  stretched(5, 0),
			opts:    opts("-iteration-marker=^Missing$"),
			wantErr: "fired 0 times",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := compareCaptures(base, compareCapture(tt.target, completeStatus()), nil, nil, tt.opts)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got := [2]int64{c.Base.WindowNanos, c.Target.WindowNanos}; got != tt.window {
				t.Errorf("windows = %v, want %v", got, tt.window)
			}
			if tt.note != "" && !strings.Contains(strings.Join(c.Notes, "\n"), tt.note) {
				t.Errorf("notes = %q, want one containing %q", c.Notes, tt.note)
			}
		})
	}
}

func TestParseCompareIterations(t *testing.T) {
	tests := []struct {
		args []string
		want string
	}{
		{[]string{"-skip-iterations=2"}, "need -iteration-marker"},
		{[]string{"-iteration-marker=x", "-warmup-ms=5"}, "exclusive"},
		{[]string{"-iteration-marker=x", "-iteration-count=-1"}, "invalid -iteration-count -1"},
		{[]string{"-iteration-marker=("}, "-iteration-marker:"},
	}
	for _, tt := range tests {
		_, _, err := parseCompareArgs(append(tt.args, "a", "b"), io.Discard)
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("parseCompareArgs(%q) err = %v, want %q", tt.args, err, tt.want)
		}
	}
}
