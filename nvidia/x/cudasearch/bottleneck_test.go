package cudasearch

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func metricsFixture(memory, compute, occupancy float64) ProfileMetrics {
	return ProfileMetrics{MemoryThroughput: &memory, ComputeThroughput: &compute, WarpOccupancy: &occupancy}
}

func TestBottleneckClassifier_MemoryBandwidthLimited(t *testing.T) {
	s := ClassifyBottleneck(metricsFixture(90, 20, 70))
	if s.StateName != cudatrial.MemoryBandwidthLimited || !s.PendingHardware || s.PrimaryBottleneck != "memory_bound" {
		t.Fatalf("state: %+v", s)
	}
}

func TestBottleneckClassifier_ComputeThroughputLimited(t *testing.T) {
	s := ClassifyBottleneck(metricsFixture(20, 90, 70))
	if s.StateName != cudatrial.ComputeThroughputLimited || !s.PendingHardware {
		t.Fatalf("state: %+v", s)
	}
}

func TestBottleneckClassifierBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name string
		m    ProfileMetrics
		want cudatrial.BottleneckFamily
	}{
		{"missing", ProfileMetrics{}, ""},
		{"nan", metricsFixture(math.NaN(), 20, 70), ""},
		{"negative", metricsFixture(-1, 20, 70), ""},
		{"over100", metricsFixture(101, 20, 70), ""},
		{"hybrid", metricsFixture(70, 70, 70), cudatrial.HybridBound},
		{"bandwidth-edge", metricsFixture(70, 69, 70), cudatrial.MemoryBandwidthLimited},
		{"low-occupancy", metricsFixture(20, 20, 34), cudatrial.LatencyOccupancyLimited},
		{"known-zero", metricsFixture(0, 0, 35), cudatrial.MemoryLatencyBound},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := ClassifyBottleneck(tc.m)
			if s.StateName != tc.want || !s.PendingHardware {
				t.Fatalf("state: %+v", s)
			}
		})
	}
	api := 50.0
	conflicts := uint64(1)
	m := metricsFixture(90, 90, 10)
	m.APIOverhead = &api
	m.BankConflicts = &conflicts
	if ClassifyBottleneck(m).StateName != cudatrial.APIOverheadDominant {
		t.Fatal("API precedence")
	}
	m.APIOverhead = nil
	if ClassifyBottleneck(m).StateName != cudatrial.MemoryBankConflicts {
		t.Fatal("bank precedence")
	}
	m = metricsFixture(20, 20, 70)
	hit := 0.0
	m.L1HitRate = &hit
	m.L2HitRate = &hit
	if ClassifyBottleneck(m).StateName != cudatrial.CacheInefficient {
		t.Fatal("known-zero cache")
	}
	m.L1HitRate = nil
	if ClassifyBottleneck(m).StateName != cudatrial.MemoryLatencyBound {
		t.Fatal("unknown cache invented")
	}
}
