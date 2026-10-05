package cudatrial

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// This is a synthetic integrity fixture, not a hardware qualification receipt.
func trajectoryFixture(rule string, latency int64, correct bool) TrajectoryReceipt {
	d := strings.Repeat("a", 64)
	r := TrajectoryReceipt{Version: 1, WorkloadKey: d, Domain: d, TargetArch: "sm_121", ArtifactDigest: d, CompilerKey: d, BaselineArtifactDigest: d, BaselineCompilerKey: d, InputDigest: d, PolicyID: d, AppliedRules: []string{rule}, BottleneckState: MemoryBandwidthLimited, MeasuredLatencyNs: latency, BaselineLatencyNs: 100, SpeedupRatio: 100 / float64(latency), VerifiedCorrect: correct, DeviceUUID: "synthetic-integrity-fixture", LockPID: 123}
	r.SHA256 = r.checksum()
	return r
}

func TestTrajectoryMemory_ReplayRewardRanking(t *testing.T) {
	m, err := NewTrajectoryMemory(8)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range []TrajectoryReceipt{trajectoryFixture("fast", 50, true), trajectoryFixture("regression", 200, true), trajectoryFixture("divergence", 10, false)} {
		if err := m.ImportTrusted(r); err != nil {
			t.Fatal(err)
		}
	}
	rank := m.Rank(strings.Repeat("a", 64), MemoryBandwidthLimited, "sm_121")
	if len(rank) != 3 || rank[0].Rule != "fast" || rank[0].Reward != 1 || rank[1].Rule != "regression" || rank[1].Reward != -1 || rank[2].Reward != -8 {
		t.Fatalf("rank: %+v", rank)
	}
	if got := m.Rank(strings.Repeat("b", 64), MemoryBandwidthLimited, "sm_121"); len(got) != 0 {
		t.Fatalf("domain leak: %+v", got)
	}
	if got := m.Rank(strings.Repeat("a", 64), MemoryBandwidthLimited, "sm_120"); len(got) != 0 {
		t.Fatalf("target leak: %+v", got)
	}
}

func TestTrialReceipt_PendingHardwareWithoutDevice(t *testing.T) {
	r := trajectoryFixture("fast", 50, true)
	fake := Receipt{Complete: true, Timings: map[TimingKind]time.Duration{TimingSummedKernel: time.Nanosecond}, Provenance: map[string]string{"timer": "cuda_event"}}
	got, err := ObserveTrajectory(nil, r, fake, fake)
	if !errors.Is(err, ErrUnqualifiedOutcome) || !errors.Is(err, cudaerrors.ErrHardwarePending) || !got.PendingHardware || got.SpeedupRatio != 0 || got.MeasuredLatencyNs != 0 {
		t.Fatalf("fake timing admitted: %+v %v", got, err)
	}
	m, _ := NewTrajectoryMemory(1)
	if err := m.Record(got); !errors.Is(err, ErrUnqualifiedOutcome) {
		t.Fatalf("record pending: %v", err)
	}
}

func TestMutant_FakeTimingRejected(t *testing.T) {
	TestTrialReceipt_PendingHardwareWithoutDevice(t)
	m, _ := NewTrajectoryMemory(1)
	r := trajectoryFixture("forged", 1, true)
	if r.HardwareObserved() {
		t.Fatal("producer declaration is not live evidence")
	}
	if err := m.Record(r); !errors.Is(err, ErrUnqualifiedOutcome) {
		t.Fatalf("forged declaration admitted: %v", err)
	}
}

func TestTrajectoryIntegrityCapacityAndCopies(t *testing.T) {
	m, _ := NewTrajectoryMemory(2)
	r := trajectoryFixture("fast", 50, true)
	if err := m.ImportTrusted(r); err != nil {
		t.Fatal(err)
	}
	r.AppliedRules[0] = "caller-mutated"
	if got := m.Rank(r.Domain, r.BottleneckState, r.TargetArch); got[0].Rule != "fast" {
		t.Fatal(got)
	}
	bad := trajectoryFixture("bad", 50, true)
	bad.MeasuredLatencyNs++
	if err := m.ImportTrusted(bad); err == nil {
		t.Fatal("tampered timing admitted")
	}
	for _, name := range []string{"second", "third"} {
		if err := m.ImportTrusted(trajectoryFixture(name, 100, true)); err != nil {
			t.Fatal(err)
		}
	}
	got := m.Rank(r.Domain, r.BottleneckState, r.TargetArch)
	if len(got) != 2 || got[0].Rule != "second" || got[1].Rule != "third" {
		t.Fatal(got)
	}
}
