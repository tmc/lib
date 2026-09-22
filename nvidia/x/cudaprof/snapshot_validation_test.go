package cudaprof_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func validProfileSnapshot() cudaprof.ProfileSnapshot {
	return cudaprof.ProfileSnapshot{Version: 1, CodeDigest: strings.Repeat("a", 64), Region: "kernel", Workload: "work", Device: "device", CollectorMode: "pc_sampling", Coverage: 1, MetricUnits: map[string]string{"stall_mem": "samples"}, Metrics: map[string]int64{"stall_mem": 5}, PCStalls: map[uint64]map[string]int64{16: {"stall_mem": 5}}}
}

func TestProfileSnapshotValidationAtConsumption(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*cudaprof.ProfileSnapshot)
	}{
		{"nan", func(s *cudaprof.ProfileSnapshot) { s.Coverage = math.NaN() }},
		{"infinity", func(s *cudaprof.ProfileSnapshot) { s.Coverage = math.Inf(1) }},
		{"negative_count", func(s *cudaprof.ProfileSnapshot) { s.Metrics["stall_mem"] = -1 }},
		{"unnamed_count", func(s *cudaprof.ProfileSnapshot) { s.Metrics[""] = 1 }},
		{"negative_pc_count", func(s *cudaprof.ProfileSnapshot) { s.PCStalls[16]["stall_mem"] = -1 }},
		{"unnamed_pc_count", func(s *cudaprof.ProfileSnapshot) { s.PCStalls[16][""] = 1 }},
		{"digest", func(s *cudaprof.ProfileSnapshot) { s.CodeDigest = "invalid" }},
		{"version", func(s *cudaprof.ProfileSnapshot) { s.Version = 99 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := validProfileSnapshot()
			tc.change(&raw)
			if _, err := cudaprof.NewProfileSnapshot(raw); err == nil {
				t.Fatal("constructor accepted malformed evidence")
			}
			snapshot, err := cudaprof.NewProfileSnapshot(validProfileSnapshot())
			if err != nil {
				t.Fatal(err)
			}
			tc.change(snapshot)
			if guidance, err := snapshot.Guidance(profileTarget(snapshot, "work", "", "device")); err == nil || guidance != nil {
				t.Fatalf("consumed mutated evidence: %+v %v", guidance, err)
			}
		})
	}
	var nilSnapshot *cudaprof.ProfileSnapshot
	if _, err := nilSnapshot.Guidance(cudaprof.ProfileTarget{}); err == nil {
		t.Fatal("accepted nil snapshot")
	}
}

func TestProfileSnapshotCopiesNestedCounters(t *testing.T) {
	raw := validProfileSnapshot()
	snapshot, err := cudaprof.NewProfileSnapshot(raw)
	if err != nil {
		t.Fatal(err)
	}
	raw.Metrics["stall_mem"] = 99
	raw.PCStalls[16]["stall_mem"] = 99
	if snapshot.Metrics["stall_mem"] != 5 || snapshot.PCStalls[16]["stall_mem"] != 5 {
		t.Fatal("source maps alias snapshot")
	}
	guidance, err := snapshot.Guidance(profileTarget(snapshot, "work", "", "device"))
	if err != nil {
		t.Fatal(err)
	}
	guidance.StallCounts["stall_mem"] = 100
	if snapshot.Metrics["stall_mem"] != 5 {
		t.Fatal("guidance aliases snapshot")
	}
}
