package cudaregress_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestReplayCUDAEvent(t *testing.T) {
	inc := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Oracle:         cudaregress.CorrectnessOracle{Method: "numerical-tolerance"},
		Rounds:         3,
		ExecutionScope: cudaregress.TimingScope{Channel: cudatrial.TimingCUDAEvent},
		StoppingRules:  cudaregress.StoppingRules{RelativeThreshold: 0.1, MinSamples: 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	runner := cudaregress.ReplayFunc(func(ctx context.Context, name string, round int) (cudatrial.Receipt, []float64, error) {
		d := 100 * time.Nanosecond
		if name == "candidate" {
			d = 130 * time.Nanosecond
		}
		r := cudatrial.Receipt{Name: name, Round: round, Complete: true}
		if err := r.RecordTiming(cudatrial.TimingCUDAEvent, d); err != nil {
			return r, nil, err
		}
		return r, []float64{1}, nil
	})
	opts := cudaregress.ReplayOptions{CurrentEnvironment: inc.ObservedEnvironment, ReferenceValues: []float64{1}}
	exec, comp, err := cudaregress.Replay(context.Background(), plan, runner, opts)
	if err != nil {
		t.Fatal(err)
	}
	if comp.TimingChannel != cudatrial.TimingCUDAEvent || comp.BaselineDuration != 100*time.Nanosecond || comp.CandidateDuration != 130*time.Nanosecond || !comp.RegressionDetected {
		t.Fatalf("event comparison = %+v", comp)
	}
	for _, r := range exec.ArmReceipts {
		for _, kind := range []cudatrial.TimingKind{cudatrial.TimingWallElapsed, cudatrial.TimingSummedKernel, cudatrial.TimingGPUBusy, cudatrial.TimingAPICall} {
			if r.HasTiming(kind) {
				t.Errorf("%s unexpectedly measured %s", r.Name, kind)
			}
		}
	}
	second, err := cudaregress.CompareExecution(plan, exec, opts)
	if err != nil || second.TimingChannel != cudatrial.TimingCUDAEvent || second.CandidateDuration != comp.CandidateDuration {
		t.Fatalf("compare = %+v, %v", second, err)
	}
	missing := *exec
	missing.ArmReceipts = append([]cudatrial.Receipt(nil), exec.ArmReceipts...)
	missing.ArmReceipts[0].Timings = nil
	if _, err := cudaregress.CompareExecution(plan, &missing, opts); err == nil || !strings.Contains(err.Error(), "missing requested timing") {
		t.Fatalf("missing event: %v", err)
	}

	unknown, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{Oracle: cudaregress.CorrectnessOracle{Method: "numerical-tolerance"}, Rounds: 3, ExecutionScope: cudaregress.TimingScope{Channel: "unknown"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := cudaregress.Replay(context.Background(), unknown, runner, opts); !errors.Is(err, cudaregress.ErrUnsupportedTimingChannel) {
		t.Fatalf("unknown channel: %v", err)
	}
}
