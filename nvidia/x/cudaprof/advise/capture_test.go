package advise

import (
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/internal/journaltest"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestAdvice_DetermineSMCountPrecedence(t *testing.T) {
	// 1. Explicit SM count from capture-provenance.json takes highest precedence.
	capExplicit := &cudacapture.Capture{
		Files: map[string][]byte{
			"capture-provenance.json": []byte(`{"device":"NVIDIA GB10","arch":"sm_121a","sm_count":48}`),
			"kernels.ptx":             []byte(".version 8.0\n.target sm_90\n"),
		},
	}
	count, ok := determineSMCount(capExplicit)
	if !ok || count != 48 {
		t.Fatalf("determineSMCount(capExplicit) = (%d, %v), want (48, true)", count, ok)
	}

	// 2. An architecture whose products differ implies no SM count.
	capArchOnly := &cudacapture.Capture{
		Files: map[string][]byte{
			"kernels.ptx": []byte(".version 8.0\n.target sm_90\n"),
		},
	}
	if count, ok := determineSMCount(capArchOnly); ok {
		t.Fatalf("determineSMCount(capArchOnly) = (%d, true), want unknown", count)
	}

	// 3. A capture with no device identity is unknown, on every host.
	if count, ok := determineSMCount(&cudacapture.Capture{}); ok {
		t.Fatalf("determineSMCount(no identity) = (%d, true), want unknown", count)
	}
	if count, ok := determineSMCount(nil); ok {
		t.Fatalf("determineSMCount(nil) = (%d, true), want unknown", count)
	}
	if w, b, ok := determineDeviceLimits(&cudacapture.Capture{}); ok {
		t.Fatalf("determineDeviceLimits(no identity) = (%d, %d, true), want unknown", w, b)
	}
}

// TestCaptureSMCount checks where each SM count comes from, for every
// architecture advise may see.
func TestCaptureSMCount(t *testing.T) {
	tests := []struct {
		name      string
		metadata  string
		want      int
		datasheet bool
	}{
		{"recorded wins", `{"device":"NVIDIA B200","arch":"sm_100a","sm_count":144}`, 144, false},
		{"GB10 probed", `{"device":"NVIDIA GB10"}`, 48, false},
		{"sm_121a alone is GB10", `{"arch":"sm_121a"}`, 48, false},
		{"H100 SXM", `{"device":"NVIDIA H100 80GB HBM3"}`, 132, true},
		{"H100 PCIe refused", `{"device":"NVIDIA H100 PCIe"}`, 0, false},
		{"H200", `{"device":"NVIDIA H200"}`, 132, true},
		{"GB200 not B200", `{"device":"NVIDIA GB200"}`, 148, true},
		{"B200", `{"device":"NVIDIA B200"}`, 148, true},
		{"GB300", `{"device":"NVIDIA GB300"}`, 160, true},
		{"RTX 5090", `{"device":"NVIDIA GeForce RTX 5090"}`, 170, true},
		{"RTX PRO 6000", `{"device":"NVIDIA RTX PRO 6000 Blackwell Server Edition"}`, 188, true},
		{"sm_70 alone", `{"arch":"sm_70"}`, 0, false},
		{"sm_75 alone", `{"arch":"sm_75"}`, 0, false},
		{"sm_80 alone", `{"arch":"sm_80"}`, 0, false},
		{"sm_86 alone", `{"arch":"sm_86"}`, 0, false},
		{"sm_87 alone", `{"arch":"sm_87"}`, 0, false},
		{"sm_89 alone", `{"arch":"sm_89"}`, 0, false},
		{"sm_90a alone", `{"arch":"sm_90a"}`, 0, false},
		{"sm_100a alone", `{"arch":"sm_100a"}`, 0, false},
		{"sm_103a alone", `{"arch":"sm_103a"}`, 0, false},
		{"sm_110a alone", `{"arch":"sm_110a"}`, 0, false},
		{"sm_120a alone", `{"arch":"sm_120a"}`, 0, false},
		{"nothing", `{}`, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := captureSMCount(&cudacapture.Capture{Metadata: []byte(tt.metadata)})
			if ok != (tt.want > 0) || got.n != tt.want || got.datasheet != tt.datasheet {
				t.Errorf("captureSMCount = %+v, %v; want %d SMs, datasheet %v", got, ok, tt.want, tt.datasheet)
			}
		})
	}
}

// TestCapturedGraphReplays measures the correlation-multiplicity signature that
// stands in for CUPTI's dropped graphId. The two cases are the two arms of the
// GB10 measurement the detector is derived from: the same kernel executed 20
// times, once as 20 direct launches and once as a four-node graph replayed five
// times.
func TestCapturedGraphReplays(t *testing.T) {
	tests := []struct {
		name        string
		correlation func(i int) int
		graphID     func(i int) int // 0 means the capture carries no graph ID
		want        int
	}{
		{
			name:        "direct launches give every kernel its own correlation",
			correlation: func(i int) int { return 100 + i },
			want:        0,
		},
		{
			name:        "graph replay shares one correlation per cudaGraphLaunch",
			correlation: func(i int) int { return 100 + i/4 },
			want:        5,
		},
		{
			name:        "captures without correlation IDs claim nothing",
			correlation: func(int) int { return 0 },
			want:        0,
		},
		{
			// The agent now forwards graphId, so the same four-node graph is
			// read directly rather than inferred from shared correlations.
			name:        "graph ID counts replays directly",
			correlation: func(i int) int { return 100 + i/4 },
			graphID:     func(int) int { return 42 },
			want:        5,
		},
		{
			// The case correlation multiplicity cannot see: every replay of a
			// one-node graph has its own correlation and exactly one kernel
			// record, which is indistinguishable from an ordinary launch.
			name:        "single-node graph replay is invisible without a graph ID",
			correlation: func(i int) int { return 100 + i },
			want:        0,
		},
		{
			name:        "single-node graph replay is counted with a graph ID",
			correlation: func(i int) int { return 100 + i },
			graphID:     func(int) int { return 42 },
			want:        20,
		},
		{
			// A graph ID on some kernels and not others: only the graph nodes
			// witness replays, and the plain launches must not inflate them.
			name:        "mixed graph and non-graph work counts only the graph launches",
			correlation: func(i int) int { return 100 + i },
			graphID: func(i int) int {
				if i < 8 {
					return 42
				}
				return 0
			},
			want: 8,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var samples []*profile.Sample
			for i := range 20 {
				start := uint64(1000 + i*10000)
				sample := microKernelSample("bump", 7, start, start+6800, tt.correlation(i))
				if tt.graphID != nil {
					if id := tt.graphID(i); id != 0 {
						sample.NumLabel["graph_id"] = []int64{int64(id)}
					}
				}
				samples = append(samples, sample)
			}
			cap := &cudacapture.Capture{Profile: &profile.Profile{Sample: samples}}
			if got := capturedGraphReplays(cap); got != tt.want {
				t.Errorf("capturedGraphReplays = %d, want %d", got, tt.want)
			}
		})
	}

	if got := capturedGraphReplays(nil); got != 0 {
		t.Errorf("capturedGraphReplays(nil) = %d, want 0", got)
	}
}

// TestAdvice_WallDurationIsNotDeviceTime pins the case that made the advisor
// print a fabricated millisecond figure: a profile whose only sample type is
// pc_samples, carrying a wall-clock duration. Scaling the samples by it gave
// this single kernel a "device time" equal to the entire capture.
func TestAdvice_WallDurationIsNotDeviceTime(t *testing.T) {
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			DurationNanos: 250_000_000,
			SampleType: []*profile.ValueType{
				{Type: "pc_samples", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{100},
					Label: map[string][]string{"kernel": {"k1"}},
				},
			},
		},
	}
	times, totalTime := computeProfileTimes(cap)
	if totalTime != 100 {
		t.Errorf("totalTime = %d, want 100 samples", totalTime)
	}
	if times["k1"] != 100 {
		t.Errorf("times[k1] = %d, want its 100 samples, not the 250 ms wall duration", times["k1"])
	}
	if hasDeviceTiming(cap) {
		t.Error("a wall-clock duration was accepted as device timing")
	}
}

// TestAdviseReadsTheJournalItWasHanded covers the reader that announced the
// absence of data the same file carries.
//
// advise read the pprof projection, which is an aggregate and has no activity
// timestamps by construction. On a capture whose journal was attached it
// therefore reported that the capture carries no timestamps, and recommended a
// re-capture -- a run under GPU lock to reproduce a file already on disk.
func TestAdviseReadsTheJournalItWasHanded(t *testing.T) {
	b := journaltest.New()
	b.KernelOn(0, 1, 1_000, 3_000, 1)
	b.KernelOn(0, 1, 6_000, 7_000, 2)
	cap := testCapture(map[string][]byte{"capture.journal": b.Finalize()}, journaltest.CompleteStatus())

	// The projection is present and useless: samples with names and no
	// timestamps, which is what a real capture of this shape holds. A guard on
	// "did the projection produce anything" never fires here.
	cap.Profile = &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
		Sample: []*profile.Sample{
			{Value: []int64{2_000}, Label: map[string][]string{"subtype": {"kernel"}, "name": {"gemm"}}},
			{Value: []int64{1_000}, Label: map[string][]string{"subtype": {"kernel"}, "name": {"softmax"}}},
		},
	}

	streams, _ := timeline.Events(cap)
	var got []timeline.Event
	for _, evs := range streams {
		got = append(got, evs...)
	}
	if len(got) != 2 {
		t.Fatalf("stream events = %d, want the journal's 2: the untimed projection was kept, or the two sources were merged", len(got))
	}
	for _, ev := range got {
		if ev.Start == 0 || ev.End == 0 {
			t.Errorf("event %+v carries no timestamps: the journal has them", ev)
		}
	}
	pct, span, busy, ok := profileDutyCycle(streams)
	if !ok || span != 6_000 || busy != 3_000 {
		t.Fatalf("duty = %.2f span/busy = %d/%d (ok %v), want 50.00 6000/3000", pct, span, busy, ok)
	}
}
