package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// timedCapture builds a capture whose profile spans spanNanos and measures
// busyNanos of device time across samples, plus samples PC samples.
func timedCapture(spanNanos, busyNanos, samples int64) *cudacapture.Capture {
	p := &profile.Profile{
		DurationNanos: spanNanos,
		SampleType: []*profile.ValueType{
			{Type: "pc_samples", Unit: "count"},
			{Type: "kernel_time", Unit: "nanoseconds"},
		},
	}
	if busyNanos > 0 || samples > 0 {
		p.Sample = []*profile.Sample{{Value: []int64{samples, busyNanos}}}
	}
	return &cudacapture.Capture{Profile: p}
}

// TestCoverageFallsWhenTheSpanGrows is the positive control the request names:
// a workload with a deliberate host-side stall between launches has the same
// device busy time over a longer span, and the reported coverage must fall.
// A coverage figure that does not move under that change is not measuring what
// it claims, and would read as confidently on a profile that saw 40% of the
// wall clock as on one that saw all of it.
func TestCoverageFallsWhenTheSpanGrows(t *testing.T) {
	tight := captureCoverage(timedCapture(1e9, 8e8, 1000))
	stalled := captureCoverage(timedCapture(4e9, 8e8, 1000))
	if !tight.Determined() || !stalled.Determined() {
		t.Fatalf("device time present but coverage undetermined: %v / %v", tight, stalled)
	}
	if !(stalled.Fraction() < tight.Fraction()) {
		t.Fatalf("coverage did not fall under a 3s stall: %.3f -> %.3f", tight.Fraction(), stalled.Fraction())
	}
	if got := tight.Fraction(); got < 0.79 || got > 0.81 {
		t.Errorf("tight coverage = %.3f, want 0.8", got)
	}
	// The denominator must travel with the share, in the same string. Here the
	// profile is ranked by device time, so that is what the line names.
	if !strings.Contains(tight.String(), "device time") {
		t.Errorf("coverage line does not name the share's denominator: %q", tight)
	}
}

// TestSampleOnlyProfileLeavesCoverageUnknown pins the honest half: a count of
// samples has no duration, so no arithmetic over it yields a fraction of the
// span. Reporting 100% here is the error the whole item exists to prevent.
func TestSampleOnlyProfileLeavesCoverageUnknown(t *testing.T) {
	c := captureCoverage(timedCapture(1e9, 0, 1000))
	if c.Determined() {
		t.Fatalf("coverage determined from sample counts alone: %v", c)
	}
	if !strings.Contains(c.String(), "unknown") {
		t.Errorf("undetermined coverage does not say so: %q", c)
	}
	// With no device time the samples are the denominator, so the line must
	// say so -- and say what they are not.
	if !strings.Contains(c.String(), "1000 observed samples") {
		t.Errorf("sample denominator not named: %q", c)
	}
	if !strings.Contains(c.String(), "not of elapsed time") {
		t.Errorf("the share is not distinguished from a time share: %q", c)
	}
}

func TestRankingIsRefusedWithoutADenominator(t *testing.T) {
	sampleOnly := timedCapture(1e9, 0, 1000)
	timed := timedCapture(1e9, 8e8, 1000)

	tests := []struct {
		name      string
		cap       *cudacapture.Capture
		args      []string
		allow     bool
		wantError bool
	}{
		{"ranking without coverage", sampleOnly, []string{"-top"}, false, true},
		{"ranking with coverage", timed, []string{"-top"}, false, false},
		{"ranking acknowledged", sampleOnly, []string{"-top"}, true, false},
		// Device time states its own denominator -- "75.95% of 868.79us" names
		// both halves -- so a missing span weakens the ranking without
		// invalidating it. Refusing here would take the tool down on every
		// capture that does not also trace activity.
		{"device time, unknown span", timedCapture(0, 8e8, 0), []string{"-top"}, false, false},
		// A reader opening the interactive view is looking at the profile, not
		// reading an ordered list off it; gating that would make the tool
		// unusable for the inspection that establishes what the profile is.
		{"interactive view", sampleOnly, []string{"-http=:0"}, false, false},
		{"tree is a ranking", sampleOnly, []string{"-tree"}, false, true},
		{"flag with a value", sampleOnly, []string{"-top=10"}, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := preflightCoverage(&buf, tt.cap, tt.args, tt.allow)
			if (err != nil) != tt.wantError {
				t.Fatalf("preflightCoverage error = %v, want error = %v", err, tt.wantError)
			}
			// The statement is printed whatever the verdict: a reader who is
			// allowed through must still see what the share is a share of.
			if !strings.Contains(buf.String(), "coverage") {
				t.Errorf("no coverage statement printed: %q", buf.String())
			}
		})
	}
}

func TestStripCoverageFlag(t *testing.T) {
	args, found := stripCoverageFlag([]string{"-top", allowUnknownCoverageFlag, "gpu.pprof"})
	if !found {
		t.Error("flag not detected")
	}
	if strings.Join(args, " ") != "-top gpu.pprof" {
		t.Errorf("args = %v, want the flag removed; pprof rejects what it does not know", args)
	}
	if _, found := stripCoverageFlag([]string{"-top", "gpu.pprof"}); found {
		t.Error("flag reported present when it was absent")
	}
}
