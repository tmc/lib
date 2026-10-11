package cudaregress_test

import (
	"context"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"testing"
	"time"
)

func TestReplayRejectsUnverifiedReceipts(t *testing.T) {
	for _, test := range []struct {
		name     string
		complete bool
		output   float64
		warmup   int
	}{
		{"incomplete", false, 1, 0}, {"baseline-oracle", true, 1.0000001, 0}, {"warmup-oracle", true, 1.0000001, 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			inc := validAdmittedIncident()
			plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{WarmupRuns: test.warmup, Rounds: 3, Oracle: cudaregress.CorrectnessOracle{Method: "numerical-tolerance"}})
			if err != nil {
				t.Fatal(err)
			}
			runner := cudaregress.ReplayFunc(func(context.Context, string, int) (cudatrial.Receipt, []float64, error) {
				return cudatrial.Receipt{Complete: test.complete, WallElapsed: time.Second}, []float64{test.output}, nil
			})
			execution, comparison, err := cudaregress.Replay(context.Background(), plan, runner, cudaregress.ReplayOptions{CurrentEnvironment: inc.ObservedEnvironment, ReferenceValues: []float64{1}})
			if err == nil && comparison != nil && comparison.Verdict == "pass" {
				t.Fatal("unverified run passed")
			}
			if execution != nil && execution.OutputsVerified {
				t.Fatal("unverified outputs promoted")
			}
		})
	}
}
