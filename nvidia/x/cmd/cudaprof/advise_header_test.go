package main

import (
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// TestSampleOnlyReportStatesTheInstrumentLimit is the §7 control. PC sampling
// cannot observe device idle between launches. That is a property of the
// instrument, not a defect, but a reader who does not know it reads
// "17.66% of observed samples" as "17.66% of elapsed time" -- the same
// category error B1 used to make in the tool's own voice. The report must say
// so, in the header, whenever it has no device timing to offer.
func TestSampleOnlyReportStatesTheInstrumentLimit(t *testing.T) {
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "stall_scoreboard", Unit: "count"},
				{Type: "stall_mem", Unit: "count"},
			},
			Location: []*profile.Location{{ID: 1, Address: 0x1000}},
			Sample: []*profile.Sample{
				{
					Location: []*profile.Location{{ID: 1, Address: 0x1000}},
					Value:    []int64{900, 100},
					Label:    map[string][]string{"kernel": {"sampled_kernel"}},
				},
			},
		},
	}
	if hasDeviceTiming(cap) {
		t.Fatal("fixture is not sample-only; control cannot fire")
	}

	rep, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture: %v", err)
	}
	if len(rep.Items) == 0 {
		t.Fatal("no findings produced; control cannot fire")
	}
	md := rep.Markdown()

	for _, want := range []string{"device idle", "not a share of elapsed time"} {
		if !strings.Contains(md, want) {
			t.Errorf("sample-only report header missing %q:\n%s", want, md)
		}
	}
	// The caveat belongs above the findings, where it is read before the
	// percentages it qualifies.
	if ci, fi := strings.Index(md, "device idle"), strings.Index(md, "## Findings"); ci > fi {
		t.Errorf("instrument caveat appears after the findings it qualifies (at %d, findings at %d)", ci, fi)
	}
}

// TestTimedReportOmitsSampleOnlyCaveat is the other half: a capture that does
// carry device timing must not claim its numbers are sample shares. Without
// this half the fix could pass by printing the caveat unconditionally.
func TestTimedReportOmitsSampleOnlyCaveat(t *testing.T) {
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
			Location:   []*profile.Location{{ID: 1, Address: 0x1000}},
			Sample: []*profile.Sample{
				{
					Location: []*profile.Location{{ID: 1, Address: 0x1000}},
					Value:    []int64{5_000_000},
					Label:    map[string][]string{"kernel": {"timed_kernel"}},
				},
			},
		},
	}
	if !hasDeviceTiming(cap) {
		t.Fatal("fixture carries no device timing; control cannot fire")
	}
	rep, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture: %v", err)
	}
	if strings.Contains(rep.Markdown(), "not a share of elapsed time") {
		t.Errorf("timed report carries the sample-only caveat:\n%s", rep.Markdown())
	}
}
