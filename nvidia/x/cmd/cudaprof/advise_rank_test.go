package main

import (
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestAdvice_PCSamplingRankingAndDurationFallback(t *testing.T) {
	cap := &cudacapture.Capture{
		Metadata: []byte(`{"duration_ms": 100}`),
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "pc_samples", Unit: "count"},
				{Type: "stall_mem", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{900, 800},
					Label: map[string][]string{"kernel": {"heavy_kernel"}},
				},
				{
					Value: []int64{100, 90},
					Label: map[string][]string{"kernel": {"light_kernel"}},
				},
			},
		},
	}

	// The capture's 100 ms is wall clock. Spreading it across the samples
	// would yield a per-kernel nanosecond figure nothing measured, so the
	// counts stay counts and the ranking below works off their share.
	times, totalTime := computeProfileTimes(cap)
	if totalTime != 1000 {
		t.Errorf("totalTime = %d, want 1000 samples", totalTime)
	}
	if times["heavy_kernel"] != 900 {
		t.Errorf("heavy_kernel = %d, want its 900 samples", times["heavy_kernel"])
	}
	if times["light_kernel"] != 100 {
		t.Errorf("light_kernel = %d, want its 100 samples", times["light_kernel"])
	}

	report := &AdviceReport{
		Items: []AdviceItem{
			{
				Severity: SeverityCritical,
				Category: CategoryMemory,
				Location: Location{Kernel: "light_kernel"},
				Finding:  "light finding",
			},
			{
				Severity: SeverityCritical,
				Category: CategoryMemory,
				Location: Location{Kernel: "heavy_kernel"},
				Finding:  "heavy finding",
			},
		},
	}
	finalizeAndRankFindings(report, times, totalTime)
	if len(report.Items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(report.Items))
	}
	if report.Items[0].Location.Kernel != "heavy_kernel" {
		t.Errorf("item 0 kernel = %s, want heavy_kernel", report.Items[0].Location.Kernel)
	}
	if report.Items[0].ShareOfProfile != 90.0 {
		t.Errorf("item 0 share = %.2f, want 90.0", report.Items[0].ShareOfProfile)
	}
}

func TestAdvice_PurePCSampleProfileShareFormatting(t *testing.T) {
	// Pure PC-sample capture: no duration metadata, sample type is pc_samples.
	purePCCap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "pc_samples", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{800},
					Label: map[string][]string{
						"kernel": {"heavy_kernel"},
					},
				},
				{
					Value: []int64{200},
					Label: map[string][]string{
						"kernel": {"light_kernel"},
					},
				},
			},
		},
	}

	times, totalTime := computeProfileTimes(purePCCap)
	if hasDeviceTiming(purePCCap) {
		t.Fatal("hasDeviceTiming returned true for pure PC-sample capture")
	}

	report := &AdviceReport{
		Items: []AdviceItem{
			{
				Severity: SeverityWarning,
				Category: CategoryMemory,
				Location: Location{Kernel: "heavy_kernel"},
				Finding:  "Heavy load finding",
			},
			{
				Severity: SeverityInfo,
				Category: CategoryMemory,
				Location: Location{Kernel: "light_kernel"},
				Finding:  "Light load finding",
			},
		},
	}

	finalizeAndRankFindingsWithTiming(report, times, totalTime, false)

	for _, it := range report.Items {
		if it.DeviceTimeNanos != 0 {
			t.Errorf("kernel %q assigned non-zero DeviceTimeNanos %d on pure PC-sample profile", it.Location.Kernel, it.DeviceTimeNanos)
		}
		if it.Location.Kernel == "heavy_kernel" && it.ShareOfProfile != 80.0 {
			t.Errorf("heavy_kernel share = %.2f, want 80.0", it.ShareOfProfile)
		}
		if it.Location.Kernel == "light_kernel" && it.ShareOfProfile != 20.0 {
			t.Errorf("light_kernel share = %.2f, want 20.0", it.ShareOfProfile)
		}
	}

	md := report.Markdown()
	if !strings.Contains(md, "(80.00% of observed samples)") {
		t.Errorf("Markdown output missing '(80.00%% of observed samples)':\n%s", md)
	}
	if strings.Contains(md, "GPU time") {
		t.Errorf("Markdown output improperly claims 'GPU time' on pure PC-sample profile:\n%s", md)
	}
	if strings.Contains(md, "0.00 ms") {
		t.Errorf("Markdown output improperly claims '0.00 ms' on pure PC-sample profile:\n%s", md)
	}

	// A wall-clock duration on a sample-only profile is not device time. This
	// half of the test used to assert the opposite -- that 800 of 1000 samples
	// over a 100 ms capture should render as "80.00 ms of GPU time" -- which is
	// the fabrication itself, written down as intent.
	durCap := &cudacapture.Capture{
		Profile: &profile.Profile{
			DurationNanos: 100_000_000, // 100 ms
			SampleType: []*profile.ValueType{
				{Type: "pc_samples", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{800},
					Label: map[string][]string{
						"kernel": {"heavy_kernel"},
					},
				},
				{
					Value: []int64{200},
					Label: map[string][]string{
						"kernel": {"light_kernel"},
					},
				},
			},
		},
	}

	if hasDeviceTiming(durCap) {
		t.Fatal("hasDeviceTiming returned true for a sample-only capture with a wall duration")
	}

	durTimes, durTotalTime := computeProfileTimes(durCap)
	if got := durTimes["heavy_kernel"]; got != 800 {
		t.Errorf("heavy_kernel = %d, want its 800 samples unscaled by the 100 ms duration", got)
	}
	durReport := &AdviceReport{
		Items: []AdviceItem{
			{
				Severity: SeverityWarning,
				Category: CategoryMemory,
				Location: Location{Kernel: "heavy_kernel"},
				Finding:  "Heavy load finding",
			},
		},
	}
	finalizeAndRankFindingsWithTiming(durReport, durTimes, durTotalTime, hasDeviceTiming(durCap))

	if got := durReport.Items[0].DeviceTimeNanos; got != 0 {
		t.Errorf("DeviceTimeNanos = %d on a sample-only capture, want 0", got)
	}
	durMd := durReport.Markdown()
	if strings.Contains(durMd, "GPU time") || strings.Contains(durMd, " ms)") {
		t.Errorf("Markdown claims device time from a wall-clock duration:\n%s", durMd)
	}
	if !strings.Contains(durMd, "80.00% of observed samples") {
		t.Errorf("Markdown missing sample-share formatting:\n%s", durMd)
	}
}

// TestTopTierIsEarnedByKindNotByCost pins the rule that replaced the share
// floor, and the two things it must not disturb.
//
// Severity is the reader's ordering, not a defect level. Cost is already
// stated in the header -- share of device time, milliseconds, share of the
// captured span -- so letting cost also buy the top tier said the same thing
// twice and said it as alarm: a kernel family holding most of the device time
// is what a well-tuned program looks like, not a finding to read first. So no
// amount of device time promotes an item; only severityByKind does, for
// findings that are what they are regardless of what they cost -- the module
// targets the wrong architecture, or the report's own ordering is misleading.
//
// The one thing cost still decides is demotion out of Cost: a finding worth
// less than negligibleDeviceNanos is context, not an expense.
func TestTopTierIsEarnedByKindNotByCost(t *testing.T) {
	const total = 1_000_000_000
	report := &AdviceReport{Items: []AdviceItem{
		{Severity: SeverityReadFirst, Category: CategoryArchitectureCompute, Finding: "tiny by cost", DeviceTimeNanos: 6_000},
		{Severity: SeverityReadFirst, Category: CategoryArchitecture, Finding: "big by cost", DeviceTimeNanos: 536_000_000},
		{Severity: SeverityReadFirst, Category: CategoryArchitecture, Finding: "wrong architecture", DeviceTimeNanos: 6_000, severityByKind: true},
		{Severity: SeverityCost, Category: CategoryMemory, Finding: "a negligible cost becomes context", DeviceTimeNanos: 6_000},
		{Severity: SeverityCost, Category: CategoryMemory, Finding: "a real cost stays one", DeviceTimeNanos: 536_000_000},
	}}
	finalizeAndRankFindingsWithTiming(report, nil, total, true)

	got := make(map[string]Severity, len(report.Items))
	for _, it := range report.Items {
		got[it.Finding] = it.Severity
	}
	for _, tc := range []struct {
		finding string
		want    Severity
	}{
		{"tiny by cost", SeverityContext},
		{"big by cost", SeverityCost},
		{"wrong architecture", SeverityReadFirst},
		{"a negligible cost becomes context", SeverityContext},
		{"a real cost stays one", SeverityCost},
	} {
		if got[tc.finding] != tc.want {
			t.Errorf("%q = %s, want %s", tc.finding, got[tc.finding], tc.want)
		}
	}

	// The negligible boundary must not be reachable by rounding: a finding
	// exactly at it stays a Cost, so the boundary is stated rather than
	// discovered.
	at := &AdviceReport{Items: []AdviceItem{
		{Severity: SeverityCost, Category: CategoryMemory, Finding: "exactly at the boundary", DeviceTimeNanos: negligibleDeviceNanos},
	}}
	finalizeAndRankFindingsWithTiming(at, nil, total, true)
	if at.Items[0].Severity != SeverityCost {
		t.Errorf("a finding exactly at the negligible boundary was demoted to %s", at.Items[0].Severity)
	}

	// A finding with no device time of its own is not negligible, it is
	// unmeasured: it must keep whatever tier its kind gave it.
	none := &AdviceReport{Items: []AdviceItem{
		{Severity: SeverityCost, Category: CategoryDispatch, Finding: "no device time", DeviceTimeNanos: 0},
	}}
	finalizeAndRankFindingsWithTiming(none, nil, total, true)
	if none.Items[0].Severity != SeverityCost {
		t.Errorf("a finding with no device time was demoted to %s", none.Items[0].Severity)
	}
}
