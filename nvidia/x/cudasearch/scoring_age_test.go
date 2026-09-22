package cudasearch

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestScoringEvidenceAge(t *testing.T) {
	now := time.Date(2026, 9, 20, 12, 0, 0, 0, time.UTC)
	model := DefaultScoringModel("epoch")
	model.EvidenceTime, model.EvidenceMaxAge = now, time.Minute
	r := &cudatrial.Receipt{Complete: true, SummedKernel: time.Microsecond}
	base, err := model.Score(r, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name      string
		collected time.Time
		admitted  bool
	}{
		{"fresh", now.Add(-time.Second), true},
		{"boundary", now.Add(-time.Minute), true},
		{"expired", now.Add(-time.Minute - time.Nanosecond), false},
		{"future", now.Add(time.Nanosecond), false},
		{"missing", time.Time{}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			evidence := &ProfileEvidence{Epoch: "epoch", CollectedAt: tc.collected, Occupancy: 1, StallMemoryThrottle: 0.5}
			got, err := model.Score(r, evidence)
			if err != nil || got != base || model.acceptsEvidence(evidence) != tc.admitted {
				t.Fatalf("score=%v base=%v err=%v", got, base, err)
			}
		})
	}
	model.EvidenceMaxAge = -1
	if _, err := model.Score(r, nil); err == nil {
		t.Fatal("accepted negative age limit")
	}
}

func TestSearchEvidenceClockFrozen(t *testing.T) {
	eval := &MockEvaluator{Scoring: DefaultScoringModel("epoch")}
	eval.Scoring.EvidenceMaxAge = time.Minute
	before := time.Now()
	ctx, work, err := searchWorkBudget(context.Background(), DefaultBudget(), eval)
	if err != nil {
		t.Fatal(err)
	}
	after := time.Now()
	frozen := work.scoring.model.EvidenceTime
	if frozen.Before(before) || frozen.After(after) {
		t.Fatalf("reference time=%v", frozen)
	}
	eval.Scoring.EvidenceTime = frozen.Add(time.Hour)
	if got := mockScoring(ctx, eval).model.EvidenceTime; !got.Equal(frozen) {
		t.Fatalf("reference time changed to %v", got)
	}
	if got := snapshotScoring(eval).model.EvidenceTime; !got.Equal(eval.Scoring.EvidenceTime) {
		t.Fatalf("new search reference=%v", got)
	}
}
