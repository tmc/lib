package cudasearch

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestMockEvaluatorDoesNotChangeObjectiveOnScoreError(t *testing.T) {
	s := executableState()
	eval := &MockEvaluator{Scoring: DefaultScoringModel("epoch")}
	eval.Scoring.BaseLatencyWeight = math.NaN()
	if candidate, err := eval.Evaluate(context.Background(), s, s.Roots["root"]); err == nil || candidate != nil {
		t.Fatalf("invalid score fell back to AST cost: %v %v", candidate, err)
	}
}

func TestScoreRequiresSelectedDeviceTiming(t *testing.T) {
	for _, tc := range []struct {
		name    string
		receipt cudatrial.Receipt
		want    float64
	}{
		{"summed", cudatrial.Receipt{Complete: true, SummedKernel: 100 * time.Nanosecond, WallNs: 1}, 100},
		{"wall", cudatrial.Receipt{Complete: true, WallElapsed: time.Second, WallNs: 1}, 0},
		{"generic_device", cudatrial.Receipt{Complete: true, DeviceNs: 1, DeviceDuration: time.Nanosecond}, 0},
		{"absent", cudatrial.Receipt{Complete: true}, 0},
		{"mixed", cudatrial.Receipt{Complete: true, TimingKind: cudatrial.TimingGPUBusy, SummedKernel: time.Nanosecond, GPUBusy: time.Nanosecond}, 0},
		{"negative", cudatrial.Receipt{Complete: true, SummedKernel: -time.Nanosecond}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DefaultScoringModel("epoch").Score(&tc.receipt, nil)
			if tc.want == 0 {
				if err == nil {
					t.Fatalf("accepted score %v", got)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("got=%v err=%v", got, err)
			}
		})
	}
}

func TestScoreRejectsInvalidNumbers(t *testing.T) {
	r := &cudatrial.Receipt{Complete: true, SummedKernel: time.Second}
	for _, value := range []float64{-1, math.NaN(), math.Inf(1), math.MaxFloat64} {
		model := DefaultScoringModel("epoch")
		model.BaseLatencyWeight = value
		if _, err := model.Score(r, nil); err == nil {
			t.Fatalf("accepted weight %v", value)
		}
	}
	model := DefaultScoringModel("epoch")
	for _, value := range []float64{-1, math.NaN(), math.Inf(1), 1.1} {
		evidence := &ProfileEvidence{Epoch: "epoch", Occupancy: value}
		if _, err := model.Score(r, evidence); err == nil {
			t.Fatalf("accepted fraction %v", value)
		}
		evidence.Epoch = "old"
		if got, err := model.Score(r, evidence); err != nil || got != float64(time.Second) {
			t.Fatalf("stale fallback=%v %v", got, err)
		}
	}
}

func TestProfileObservationsAreNotLatency(t *testing.T) {
	model := DefaultScoringModel("epoch")
	r := &cudatrial.Receipt{Complete: true, SummedKernel: 17 * time.Nanosecond}
	for _, evidence := range []*ProfileEvidence{nil, {Epoch: "epoch", StallMemoryThrottle: 1, StallScoreboard: 1, StallLGThrottle: 1, TargetOccupancy: 1, RegisterSpillBytes: ^uint64(0)}} {
		got, err := model.Score(r, evidence)
		if err != nil || got != 17 {
			t.Fatalf("profile converted to latency: %v %v", got, err)
		}
	}
}
