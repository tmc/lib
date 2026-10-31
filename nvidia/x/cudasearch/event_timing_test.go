package cudasearch

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestEventProducerTimingChannels(t *testing.T) {
	ctx := context.Background()
	p := &prefillExecutable{checkLease: func() bool { return true }}
	pr, err := p.runWith(ctx, true, prefillRunOps{poison: func() error { return nil }, record: func(int) error { return nil }, launch: func() error { return nil }, wait: func(context.Context) error { return nil }, elapsed: func() (float32, error) { return 0.5, nil }})
	if err != nil {
		t.Fatal(err)
	}
	s := &sequenceExecutable{checkLease: func() bool { return true }, stages: make([]sequenceStage, 2)}
	sr, err := s.runWith(ctx, true, sequenceRunOps{poison: func() error { return nil }, record: func(int) error { return nil }, launch: func(int) error { return nil }, wait: func(context.Context) error { return nil }, elapsed: func() (float32, error) { return 0.5, nil }})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range []cudatrial.Receipt{pr, sr} {
		if r.TimingKind != cudatrial.TimingCUDAEvent {
			t.Fatalf("kind = %s", r.TimingKind)
		}
		if d, err := r.DurationFor(cudatrial.TimingCUDAEvent); err != nil || d != 500*time.Microsecond {
			t.Fatalf("event = %v, %v", d, err)
		}
		for _, kind := range []cudatrial.TimingKind{cudatrial.TimingSummedKernel, cudatrial.TimingGPUBusy} {
			if r.HasTiming(kind) {
				t.Fatalf("event mislabeled as %s", kind)
			}
		}
	}
}

func TestExecutableEventTrajectoryBoundary(t *testing.T) {
	e, base, compiled, _ := executableFixture(t)
	e.TimingKind = cudatrial.TimingCUDAEvent
	base.kind, compiled.kind = cudatrial.TimingCUDAEvent, cudatrial.TimingCUDAEvent
	digest := strings.Repeat("a", 64)
	e.Run = func(ctx context.Context, arm string, x Executable) (cudatrial.Receipt, error) {
		r, err := x.Run(ctx)
		r.Provenance = map[string]string{"timer": "cuda_event", "verified_correct": "true", "input_digest": digest, "artifact_digest": digest, "compiler_key": digest, "policy_id": digest, "device_uuid": "synthetic-host-fixture"}
		return r, err
	}
	s := executableState()
	s.Derivation = []egraph.Action{{RuleName: "vectorize"}}
	candidate, err := e.Evaluate(context.Background(), s, s.Roots["root"])
	if err != nil {
		t.Fatal(err)
	}
	if candidate.Cost != float64((5*time.Microsecond).Nanoseconds()) || len(candidate.Trials) != 4 {
		t.Fatalf("candidate = %+v", candidate)
	}
	if err := admitSearchCandidate(candidate); err != nil {
		t.Fatalf("beam rejected event: %v", err)
	}
	other := *candidate
	otherReceipt := *candidate.Receipt
	otherReceipt.TimingKind = cudatrial.TimingGPUBusy
	other.Receipt = &otherReceipt
	if err := admitComparableCandidate(candidate, &other); !errors.Is(err, ErrTimingKindMismatch) {
		t.Fatalf("mixed beam candidate: %v", err)
	}
	score, err := (ScoringModel{BaseLatencyWeight: 1, TimingKind: cudatrial.TimingCUDAEvent}).Score(candidate.Receipt, nil)
	if err != nil || score != 5000 {
		t.Fatalf("event score = %v, %v", score, err)
	}
	if _, err := (ScoringModel{BaseLatencyWeight: 1, TimingKind: cudatrial.TimingGPUBusy}).Score(candidate.Receipt, nil); !errors.Is(err, ErrTimingKindMismatch) {
		t.Fatalf("mixed score: %v", err)
	}
	baseReceipt, err := trajectoryArmMedian(candidate.Trials, "baseline")
	if err != nil || baseReceipt.TimingKind != cudatrial.TimingCUDAEvent {
		t.Fatalf("baseline = %+v, %v", baseReceipt, err)
	}
	got, err := ObserveCandidateTrajectory(nil, cudatrial.TrajectoryReceipt{}, candidate)
	if !errors.Is(err, cudatrial.ErrUnqualifiedOutcome) || !got.PendingHardware || got.InputDigest != digest {
		t.Fatalf("trajectory = %+v, %v", got, err)
	}
}
