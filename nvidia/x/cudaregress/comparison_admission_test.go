package cudaregress_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func comparisonAdmissionPlan(t *testing.T, order string) (*cudaregress.ExperimentPlan, cudaregress.ReplayOptions) {
	t.Helper()
	incident := validAdmittedIncident()
	plan, _, err := cudaregress.NewPlanFromIncident(incident, cudaregress.PlanOptions{Rounds: 3, Order: order, Oracle: cudaregress.CorrectnessOracle{Method: "digest-exact", ExpectedDigest: "control-output"}})
	if err != nil {
		t.Fatal(err)
	}
	return plan, cudaregress.ReplayOptions{CurrentEnvironment: incident.ObservedEnvironment}
}
func comparisonAdmissionRunner(context.Context, string, int) (cudatrial.Receipt, []float64, error) {
	return cudatrial.Receipt{Complete: true, Digest: "control-output", WallElapsed: time.Millisecond}, nil, nil
}
func comparisonAdmissionReplay(t *testing.T, order string) (*cudaregress.ExperimentPlan, *cudaregress.ExecutionReceipt, cudaregress.ReplayOptions) {
	t.Helper()
	plan, opts := comparisonAdmissionPlan(t, order)
	execution, comparison, err := cudaregress.Replay(context.Background(), plan, cudaregress.ReplayFunc(comparisonAdmissionRunner), opts)
	if err != nil {
		t.Fatal(err)
	}
	if comparison == nil || comparison.Verdict != cudaprofilehistory.Pass {
		t.Fatalf("valid replay: %+v", comparison)
	}
	return plan, execution, opts
}

func TestComparisonAdmissionDistinctRounds(t *testing.T) {
	for _, order := range []string{"ABBA", "interleaved", "sequential"} {
		t.Run(order, func(t *testing.T) {
			plan, execution, opts := comparisonAdmissionReplay(t, order)
			if execution.RoundsCompleted != 3 || len(execution.ArmReceipts) != 6 {
				t.Fatal("producer population")
			}
			comparison, err := cudaregress.CompareExecution(plan, execution, opts)
			if err != nil || comparison.Verdict != cudaprofilehistory.Pass {
				t.Fatalf("equal durations across real rounds: %+v %v", comparison, err)
			}
		})
	}
}

func TestComparisonAdmissionIdentity(t *testing.T) {
	for _, change := range []string{"execution", "plan id", "empty plan id", "digest", "empty digest", "foreign valid plan", "unsealed plan"} {
		t.Run(change, func(t *testing.T) {
			plan, execution, opts := comparisonAdmissionReplay(t, "ABBA")
			switch change {
			case "execution":
				execution.ExecutionID = ""
			case "plan id":
				execution.PlanID = "foreign"
			case "empty plan id":
				execution.PlanID = ""
			case "digest":
				execution.PlanDigest = "foreign"
			case "empty digest":
				execution.PlanDigest = ""
			case "foreign valid plan":
				other := *plan
				other.PlanID += "-other"
				var err error
				other.PlanDigest, err = cudaregress.ComputePlanDigest(&other)
				if err != nil {
					t.Fatal(err)
				}
				plan = &other
			case "unsealed plan":
				plan.StoppingRules.RelativeThreshold += 0.1
			}
			if comparison, err := cudaregress.CompareExecution(plan, execution, opts); err == nil {
				t.Fatalf("accepted %s: %+v", change, comparison)
			}
		})
	}
}

func TestComparisonAdmissionPopulation(t *testing.T) {
	for _, change := range []string{"duplicate round", "extra pair", "missing arm", "out of range", "zero round", "unknown arm", "wrong order", "completed count", "incomplete not partial"} {
		t.Run(change, func(t *testing.T) {
			plan, execution, opts := comparisonAdmissionReplay(t, "ABBA")
			switch change {
			case "duplicate round":
				a, b := execution.ArmReceipts[0], execution.ArmReceipts[1]
				execution.ArmReceipts = []cudatrial.Receipt{a, b, a, b, a, b}
			case "extra pair":
				execution.ArmReceipts = append(execution.ArmReceipts, execution.ArmReceipts[:2]...)
			case "missing arm":
				execution.ArmReceipts = execution.ArmReceipts[:5]
			case "out of range":
				execution.ArmReceipts[5].Round = 4
			case "zero round":
				execution.ArmReceipts[0].Round = 0
			case "unknown arm":
				execution.ArmReceipts[0].Name = "null"
			case "wrong order":
				execution.ArmReceipts[2], execution.ArmReceipts[3] = execution.ArmReceipts[3], execution.ArmReceipts[2]
			case "completed count":
				execution.RoundsCompleted = 1
			case "incomplete not partial":
				execution.ArmReceipts[0].Complete = false
			}
			if comparison, err := cudaregress.CompareExecution(plan, execution, opts); err == nil {
				t.Fatalf("accepted %s: %+v", change, comparison)
			}
		})
	}
}

func TestComparisonAdmissionActualPartial(t *testing.T) {
	for _, mode := range []string{"stop on failed candidate", "continue failed candidate", "cancel after baseline"} {
		t.Run(mode, func(t *testing.T) {
			plan, opts := comparisonAdmissionPlan(t, "ABBA")
			if mode == "stop on failed candidate" {
				plan.StoppingRules.MaxConsecutiveFailures = 1
			}
			var err error
			plan.PlanDigest, err = cudaregress.ComputePlanDigest(plan)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			failure := errors.New("controlled arm failure")
			runner := cudaregress.ReplayFunc(func(ctx context.Context, name string, round int) (cudatrial.Receipt, []float64, error) {
				r, values, err := comparisonAdmissionRunner(ctx, name, round)
				if name == "candidate" && round == 1 && mode != "cancel after baseline" {
					return r, values, failure
				}
				if name == "baseline" && round == 2 && mode == "cancel after baseline" {
					cancel()
				}
				return r, values, err
			})
			execution, _, _ := cudaregress.Replay(ctx, plan, runner, opts)
			if execution == nil || !execution.Partial {
				t.Fatalf("missing actual partial receipt: %+v", execution)
			}
			before := len(execution.ArmReceipts)
			comparison, err := cudaregress.CompareExecution(plan, execution, opts)
			if err != nil || comparison.Verdict != cudaprofilehistory.Unknown {
				t.Fatalf("partial diagnostic: %+v %v", comparison, err)
			}
			if len(execution.ArmReceipts) != before {
				t.Fatal("changed retained arm population")
			}
		})
	}
}
