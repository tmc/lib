package cudacapture

import (
	"strings"
	"testing"
)

// TestSampleLinesBoundsTheDiagnostic is the control on the error size: a real
// PTX/SASS mismatch named 67741 line numbers, which is roughly 400 KB of error
// text and unreadable wherever it lands. The rendering must stay bounded as the
// set grows while still naming enough lines to identify the region.
func TestSampleLinesBoundsTheDiagnostic(t *testing.T) {
	small := []int{206, 214, 215}
	if got := sampleLines(small, maxReportedLines); got != "[206 214 215]" {
		t.Errorf("a set that fits must render whole, got %q", got)
	}
	if strings.Contains(sampleLines(small, maxReportedLines), "more") {
		t.Error("a complete set must not claim it withheld anything")
	}

	huge := make([]int, 67741)
	for i := range huge {
		huge[i] = i + 200
	}
	got := sampleLines(huge, maxReportedLines)
	if len(got) > 200 {
		t.Errorf("diagnostic for %d lines rendered %d bytes, want a bounded sample", len(huge), len(got))
	}
	if !strings.Contains(got, "67725 more") {
		t.Errorf("the withheld count must be stated, got %q", got)
	}
	if !strings.Contains(got, "200") {
		t.Errorf("the sample must name real lines, got %q", got)
	}
}
