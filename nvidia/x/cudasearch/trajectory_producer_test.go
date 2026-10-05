package cudasearch

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func trajectoryProducerFixture() *CandidatePlan {
	digest := strings.Repeat("a", 64)
	c := &CandidatePlan{Validated: true, CostSource: CostMeasured, Cost: 30, WorkloadKey: digest, Derivation: []egraph.Action{{RuleName: "vectorize"}}}
	for i, arm := range []string{"baseline", "candidate", "candidate", "baseline"} {
		r := cudatrial.Receipt{Name: arm, Complete: true, TimingKind: cudatrial.TimingSummedKernel, Timings: map[cudatrial.TimingKind]time.Duration{cudatrial.TimingSummedKernel: time.Duration([]int{40, 20, 30, 50}[i])}, Provenance: map[string]string{"timer": "cuda_event", "verified_correct": "true", "input_digest": digest, "artifact_digest": digest, "compiler_key": digest, "policy_id": digest, "device_uuid": "synthetic-host-fixture"}}
		c.Trials = append(c.Trials, r)
	}
	return c
}

func TestTrajectoryProducerMedianAndRefusal(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*CandidatePlan)
	}{
		{"positive", func(*CandidatePlan) {}},
		{"wall", func(c *CandidatePlan) { c.Trials[1].TimingKind = cudatrial.TimingWallElapsed }},
		{"not-correct", func(c *CandidatePlan) { c.Trials[1].Provenance["verified_correct"] = "false" }},
		{"mixed-input", func(c *CandidatePlan) { c.Trials[2].Provenance["input_digest"] = strings.Repeat("b", 64) }},
		{"reordered", func(c *CandidatePlan) { c.Trials[0], c.Trials[1] = c.Trials[1], c.Trials[0] }},
		{"zero", func(c *CandidatePlan) { c.Trials[2].Timings[cudatrial.TimingSummedKernel] = 0 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := trajectoryProducerFixture()
			tt.mutate(c)
			got, err := trajectoryArmMedian(c.Trials, "candidate")
			if tt.name == "positive" {
				if err != nil || got.Timings[cudatrial.TimingSummedKernel] != 30 {
					t.Fatalf("median=%+v err=%v", got, err)
				}
				base, err := trajectoryArmMedian(c.Trials, "baseline")
				if err != nil || base.Timings[cudatrial.TimingSummedKernel] != 50 {
					t.Fatal("baseline median")
				}
			} else if !errors.Is(err, cudatrial.ErrUnqualifiedOutcome) {
				t.Fatalf("accepted trials: %v", err)
			}
			if _, err := ObserveCandidateTrajectory(nil, cudatrial.TrajectoryReceipt{}, c); !errors.Is(err, cudatrial.ErrUnqualifiedOutcome) {
				t.Fatal("host trials admitted as hardware")
			}
		})
	}
}

func TestTrajectoryProducerRejectsUnmeasuredCandidates(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*CandidatePlan)
	}{
		{"unvalidated", func(c *CandidatePlan) { c.Validated = false }},
		{"estimated", func(c *CandidatePlan) { c.CostSource = CostEstimated }},
		{"no-derivation", func(c *CandidatePlan) { c.Derivation = nil }},
		{"bad-cost", func(c *CandidatePlan) { c.Cost = 1 }},
		{"unbalanced", func(c *CandidatePlan) { c.Trials = c.Trials[:3] }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := trajectoryProducerFixture()
			tt.mutate(c)
			if _, err := ObserveCandidateTrajectory(nil, cudatrial.TrajectoryReceipt{}, c); !errors.Is(err, cudatrial.ErrUnqualifiedOutcome) {
				t.Fatal("accepted candidate")
			}
		})
	}
}
