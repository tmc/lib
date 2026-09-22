package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// TestDiffCaptures compares a capture with itself and then with a capture
// whose samples moved, which is what the measured columns are for: the two
// builds are the same, so every static column matches and only the
// measurements differ.
func TestDiffCaptures(t *testing.T) {
	base, _ := testCaptureWithLines(t)
	target, _ := testCaptureWithLines(t)
	target.Profile.Sample[0].Value[0] = 60

	var buf bytes.Buffer
	if err := diffCaptures(&buf, base, target, "fused_gate", "", "base.pprof", "target.pprof"); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, "BASE.PPROF PC_SAMPLES") || !strings.Contains(out, "TARGET.PPROF PC_SAMPLES") {
		t.Fatalf("no measured columns:\n%s", out)
	}
	var row string
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "kernels.go:14") {
			row = line
		}
	}
	if row == "" {
		t.Fatalf("no row for the sampled line:\n%s", out)
	}
	if !strings.Contains(row, "9") || !strings.Contains(row, "63") {
		t.Errorf("row does not carry both measurements: %q", row)
	}
	// The same build on both sides adds and removes no opcode.
	if strings.Count(row, "-") < 2 {
		t.Errorf("one build differs from itself: %q", row)
	}
}

// A capture of several builds is none of them until the kernel is named.
func TestDiffCapturesNeedsAKernel(t *testing.T) {
	base, _ := testCaptureWithLines(t)
	base.Modules = []cudacapture.Module{{ID: "m0", DriverID: 22}, {ID: "m1", DriverID: 23}}
	target, _ := testCaptureWithLines(t)
	if err := diffCaptures(&bytes.Buffer{}, base, target, "", "", "base", "target"); err == nil {
		t.Error("no error diffing a capture of several builds with no kernel named")
	}
}

// A sample type one capture has and the other does not cannot be compared,
// and says so rather than comparing the type against zero.
func TestDiffCapturesUnknownSampleType(t *testing.T) {
	base, _ := testCaptureWithLines(t)
	target, _ := testCaptureWithLines(t)
	if err := diffCaptures(&bytes.Buffer{}, base, target, "fused_gate", "executed", "base", "target"); err == nil {
		t.Error("no error for a sample type neither capture carries")
	}
}
