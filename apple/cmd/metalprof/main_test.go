package main

import (
	"bytes"
	"testing"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/apple/x/metal/metalcapture"
)

func TestFetcherProfileSeparation(t *testing.T) {
	fn1 := &profile.Function{ID: 1, Name: "vector_add", Filename: "kernels.metal"}
	loc1 := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn1}}}
	liveProf := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		Sample: []*profile.Sample{
			{
				Location: []*profile.Location{loc1},
				Value:    []int64{1000},
			},
		},
		Function: []*profile.Function{fn1},
		Location: []*profile.Location{loc1},
	}

	fn2 := &profile.Function{ID: 2, Name: "compute", Filename: "kernels.metal"}
	loc2 := &profile.Location{ID: 2, Line: []profile.Line{{Function: fn2}}}
	replayProf := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "replay_gpu_time", Unit: "nanoseconds"}},
		Sample: []*profile.Sample{
			{
				Location: []*profile.Location{loc2},
				Value:    []int64{5000},
			},
		},
		Function: []*profile.Function{fn2},
		Location: []*profile.Location{loc2},
	}

	var replayBuf bytes.Buffer
	if err := replayProf.Write(&replayBuf); err != nil {
		t.Fatalf("write replay profile: %v", err)
	}

	cap := &metalcapture.Capture{
		Profile: liveProf,
		Files: map[string][]byte{
			"kernels.metal": []byte("kernel void vector_add() {}\nkernel void compute() {}"),
			"replay.pprof":  replayBuf.Bytes(),
		},
		Code: metalcapture.CodeFiles{
			Source: "kernels.metal",
		},
		Observations: []metalcapture.Observation{
			{
				ID:          "occupancy",
				Origin:      metalcapture.OriginReplay,
				NonAdditive: true,
				Value:       metalcapture.NewFloatObservationValue(0.75),
			},
		},
	}

	dir := t.TempDir()
	fetchLive := &fetcher{path: "/test/capture.pprof", dir: dir, cap: cap, selected: "live"}
	pLive, _, err := fetchLive.Fetch("/test/capture.pprof", 0, time.Second)
	if err != nil {
		t.Fatalf("Fetch live: %v", err)
	}
	if pLive.SampleType[0].Type != "gpu_time" {
		t.Errorf("live SampleType = %q, want gpu_time", pLive.SampleType[0].Type)
	}
	if pLive.Sample[0].Value[0] != 1000 {
		t.Errorf("live Sample value = %d, want 1000", pLive.Sample[0].Value[0])
	}

	fetchReplay := &fetcher{path: "/test/capture.pprof", dir: dir, cap: cap, selected: "replay"}
	pReplay, _, err := fetchReplay.Fetch("/test/capture.pprof", 0, time.Second)
	if err != nil {
		t.Fatalf("Fetch replay: %v", err)
	}
	if pReplay.SampleType[0].Type != "replay_gpu_time" {
		t.Errorf("replay SampleType = %q, want replay_gpu_time", pReplay.SampleType[0].Type)
	}
	if pReplay.Sample[0].Value[0] != 5000 {
		t.Errorf("replay Sample value = %d, want 5000", pReplay.Sample[0].Value[0])
	}

	if pLive.Sample[0].Value[0] == pReplay.Sample[0].Value[0] {
		t.Errorf("expected live and replay profiles to remain strictly separate")
	}

	capNoReplay := &metalcapture.Capture{
		Profile: liveProf,
		Files: map[string][]byte{
			"kernels.metal": []byte("kernel void vector_add() {}"),
		},
		Code: metalcapture.CodeFiles{Source: "kernels.metal"},
	}
	fetchNoReplay := &fetcher{path: "/test/capture.pprof", dir: dir, cap: capNoReplay, selected: "replay"}
	if _, _, err := fetchNoReplay.Fetch("/test/capture.pprof", 0, time.Second); err == nil {
		t.Errorf("expected error when fetching missing replay profile")
	}
}
