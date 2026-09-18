package cuptibench

import (
	"strings"
	"testing"
)

// TestGpuNsLimitsNameTheFormAndTheContents pins two distinct claims about the
// same metric: what is inside the span, and what the span is. They were one
// line, and a reader who had the first still guessed the second wrong -- taking
// gpu-ns/op for a per-launch figure and comparing it against a timer that spans
// each launch separately, a 2x disagreement that was entirely the two forms.
func TestGpuNsLimitsNameTheFormAndTheContents(t *testing.T) {
	r := &Result{Operations: 32, Metadata: map[string]string{"mode": "whole"},
		Timing: &Timing{Operations: 32, Total: 1, Method: "cuda-events"}}

	var contents, form string
	for _, s := range benchLimits(r) {
		if !strings.Contains(s, "gpu-ns/op") {
			continue
		}
		switch {
		case strings.Contains(s, "inter-kernel gaps"):
			contents = s
		case strings.Contains(s, "not comparable"):
			form = s
		}
	}
	if contents == "" || form == "" {
		t.Fatalf("both gpu-ns/op caveats must be stated: contents=%q form=%q", contents, form)
	}
	if !strings.Contains(form, "32 operation(s)") {
		t.Errorf("the form caveat must name the operation count it divides by: %q", form)
	}

	// A timing-less result reports no gpu-ns/op, so neither caveat is anything
	// but noise there.
	for _, s := range benchLimits(&Result{Operations: 32, Metadata: map[string]string{"mode": "whole"}}) {
		if strings.Contains(s, "gpu-ns/op") {
			t.Errorf("no gpu-ns/op is reported, so the caveat is noise: %q", s)
		}
	}
}
