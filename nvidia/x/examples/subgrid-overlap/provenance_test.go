package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestReportStatesItIsModeled covers every rendering of a SuiteReport. Nothing
// in this package opens a device, so a microsecond column here is a model
// output; printed without that word it is indistinguishable from a profiler's.
func TestReportStatesItIsModeled(t *testing.T) {
	report, err := DefaultBenchmarkSuite().Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if report.Provenance != modeledProvenance {
		t.Fatalf("report provenance %q, want %q", report.Provenance, modeledProvenance)
	}

	data, err := report.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	renderings := map[string]string{"text": report.Table(), "markdown": report.Markdown(), "json": string(data)}
	for name, out := range renderings {
		if !strings.Contains(out, modeledProvenance) {
			t.Errorf("%s rendering carries no provenance statement:\n%s", name, out)
		}
		// The words this example used to print. Each one asserts a measurement
		// that never happened, and each is the reason the caveat above is not
		// enough on its own: a reader who believes the column header does not
		// read the banner.
		for _, claim := range []string{"Empirical", "Target Hardware", "Total Time (us)", "Total Latency"} {
			if strings.Contains(out, claim) {
				t.Errorf("%s rendering claims a measurement: %q", name, claim)
			}
		}
	}

	// The near-miss half: the banner is the report's own field, not a constant
	// the renderers spell for themselves. A renderer that hardcoded its text
	// would pass every assertion above while reporting the wrong provenance for
	// any report that carried a different one.
	report.Provenance = "measured on a device"
	for name, out := range map[string]string{"text": report.Table(), "markdown": report.Markdown()} {
		if !strings.Contains(out, "measured on a device") {
			t.Errorf("%s rendering ignores the report's provenance field:\n%s", name, out)
		}
	}
	var round map[string]any
	data, err = report.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	if err := json.Unmarshal(data, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round["provenance"] != "measured on a device" {
		t.Errorf("json provenance = %v", round["provenance"])
	}
}
