package cudaprof_test

import (
	"math"
	"reflect"
	"testing"
	"time"
)

func TestGuidanceRanksMetricsIndependently(t *testing.T) {
	s := validProfileSnapshot()
	s.Timestamp = time.Unix(123, 0)
	s.Losses = 7
	s.PCStalls = map[uint64]map[string]int64{
		16: {"cycles": math.MaxInt64, "warps": 1},
		32: {"cycles": 1, "warps": math.MaxInt64},
		48: {"cycles": math.MaxInt64, "warps": 0},
	}
	s.Metrics = map[string]int64{"cycles": math.MaxInt64, "warps": math.MaxInt64}
	s.MetricUnits = map[string]string{"cycles": "cycles", "warps": "warps"}
	g, err := s.Guidance(profileTarget(&s, "work", "", "device"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(g.HotPCs["cycles"], []uint64{16, 48, 32}) || !reflect.DeepEqual(g.HotPCs["warps"], []uint64{32, 16}) {
		t.Fatalf("cross-metric ranking: %v", g.HotPCs)
	}
	if g.Losses != 7 || !g.Timestamp.Equal(s.Timestamp) || g.PCStalls[16]["cycles"] != math.MaxInt64 {
		t.Fatalf("lost raw attribution: %+v", g)
	}
	g.PCStalls[16]["cycles"] = 0
	if s.PCStalls[16]["cycles"] != math.MaxInt64 {
		t.Fatal("guidance aliases snapshot")
	}
}

func TestGuidanceBoundsEachMetric(t *testing.T) {
	s := validProfileSnapshot()
	s.MetricUnits["samples"] = "samples"
	s.PCStalls = map[uint64]map[string]int64{}
	for pc := uint64(0); pc < 30; pc++ {
		s.PCStalls[pc] = map[string]int64{"samples": int64(pc)}
	}
	g, err := s.Guidance(profileTarget(&s, "work", "", "device"))
	if err != nil {
		t.Fatal(err)
	}
	if got := g.HotPCs["samples"]; len(got) != 10 || got[0] != 29 || got[9] != 20 {
		t.Fatalf("bounded PCs=%v", got)
	}
}

func TestGuidanceRequiresTokenIdentity(t *testing.T) {
	for _, hash := range []string{"", "tokens"} {
		s := validProfileSnapshot()
		s.TokensHash = hash
		target := "tokens"
		if hash != "" {
			target = ""
		}
		if _, err := s.Guidance(profileTarget(&s, "work", target, "device")); err == nil {
			t.Fatal("missing identity treated as wildcard")
		}
	}
}
