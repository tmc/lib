package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestPhaseOf(t *testing.T) {
	r := func(names ...string) []timeline.Range {
		var out []timeline.Range
		for _, n := range names {
			out = append(out, timeline.Range{Name: n})
		}
		return out
	}
	tests := []struct {
		stack []timeline.Range
		want  string
	}{
		{nil, groupNoRange},
		{r("encoder:fwd", "encoder.layers.3:fwd"), groupForward},
		{r("encoder.layers.3:bwd"), groupBackward},
		{r("MulBackward0"), groupBackward},
		{r("optimizer"), groupOptimizer},
		{r("step", "encoder:fwd"), groupForward},
		{r("cub::DeviceRadixSort::SortPairs"), groupOther},
	}
	for _, tt := range tests {
		if got := phaseOf(tt.stack); got != tt.want {
			t.Errorf("phaseOf(%v) = %q, want %q", tt.stack, got, tt.want)
		}
	}
}

// phasedCapture is a capture whose host thread 1 runs a forward range, a
// backward range and an optimizer range, launching kernels in each, plus a
// kernel launched from thread 2, which has no ranges, and one with no
// launch record.
func phasedCapture(bwd uint64) []byte {
	b := newJournalBuilder()
	marker := func(name string, id, flags uint32, tid uint32, at uint64) {
		b.append(wire.KindActivity, map[string]any{
			"kind": "activity", "subtype": "marker", "valid": true, "name": name,
			"marker_id": id, "marker_flags": flags, "process_id": 1, "thread_id": tid, "start": at, "end": at,
		})
	}
	launch := func(name string, corr uint32, tid uint32, at, start, end uint64) {
		b.append(wire.KindAPI, map[string]any{"kind": "api", "start": at, "end": at + 1, "correlation": corr, "thread_id": tid})
		b.namedKernel(name, start, end, corr)
	}
	marker("model:fwd", 1, 2, 1, 1000)
	launch("gemm", 1, 1, 1010, 1100, 1200)
	marker("", 1, 4, 1, 1020)
	marker("model:bwd", 2, 2, 1, 1030)
	launch("gemm", 2, 1, 1040, 1200, 1200+bwd)
	marker("", 2, 4, 1, 1050)
	marker("optimizer", 3, 2, 1, 1060)
	launch("adam", 3, 1, 1070, 1500, 1520)
	marker("", 3, 4, 1, 1080)
	launch("copy", 4, 2, 1090, 1600, 1610)
	b.namedKernel("orphan", 1700, 1705, 99)
	return b.finalize()
}

func TestCompareByPhase(t *testing.T) {
	capture := func(bwd uint64) *cudacapture.Capture {
		return testCapture(map[string][]byte{"capture.journal": phasedCapture(bwd)}, phaseStatus(1000, 2000))
	}
	c, err := compareCaptures(capture(200), capture(250), nil, nil, compareOptions{phase: -1, by: "phase"})
	if err != nil {
		t.Fatal(err)
	}
	got := make(map[string][2]int64)
	var groups []string
	for _, g := range c.Groups {
		groups = append(groups, g.Group)
		got[g.Group] = [2]int64{g.BaseExclusiveNanos, g.TargetExclusiveNanos}
	}
	want := []string{groupForward, groupBackward, groupOptimizer, groupNoRange, groupNoLaunch}
	if strings.Join(groups, ",") != strings.Join(want, ",") {
		t.Errorf("groups = %v, want %v", groups, want)
	}
	if got[groupBackward] != [2]int64{200, 250} || got[groupForward] != [2]int64{100, 100} {
		t.Errorf("forward, backward = %v, %v; want [100 100], [200 250]", got[groupForward], got[groupBackward])
	}
	if a := c.Base.Groups; a.OwnThread != 3 || a.OtherThread != 1 || a.NoLaunchRecord != 1 || a.Ranges != 3 {
		t.Errorf("attribution = %+v, want 3 own thread, 1 other, 1 without a record, 3 ranges", *a)
	}
	var out bytes.Buffer
	if err := c.write(&out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(squash(out.String()), "1 1 200 250 +50 +0 +50 backward gemm") {
		t.Errorf("output lacks the backward gemm cell:\n%s", out.String())
	}

	c, err = compareCaptures(capture(200), capture(250), nil, nil, compareOptions{phase: -1, by: "range"})
	if err != nil {
		t.Fatal(err)
	}
	if c.Groups[0].Group != "model:bwd" {
		t.Errorf("-by range first group = %q, want model:bwd", c.Groups[0].Group)
	}
}
