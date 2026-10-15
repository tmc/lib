package cudasearch

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type objectiveCostModel struct{ calls int }

func (m *objectiveCostModel) NodeCost(egraph.ENode, []float64) float64 {
	m.calls++
	return 1
}

func TestExecutableObjectiveAdmission(t *testing.T) {
	for _, test := range []struct {
		name      string
		objective Objective
		valid     bool
	}{
		{"default", "", true},
		{"latency", ObjectiveMinLatency, true},
		{"throughput", ObjectiveMaxThroughput, false},
		{"profile", ObjectiveProfileCost, false},
		{"unknown", Objective("unknown"), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			// Both executables and their timings are synthetic CPU fixtures.
			e, base, compiled, order := executableFixture(t)
			e.Workload.Objective = test.objective
			if err := e.Workload.Validate(); err != nil {
				t.Fatalf("generic workload validation rejected objective: %v", err)
			}
			model := &objectiveCostModel{}
			e.CostModel = model
			leaseChecks := 0
			e.HardwareLockChecker = func() bool { leaseChecks++; return true }
			s := executableState()
			candidate, err := e.Evaluate(context.Background(), s, s.Roots["root"])
			if !test.valid {
				if candidate != nil || !errors.Is(err, ErrMissingBinding) || !strings.Contains(err.Error(), "unsupported executable objective") {
					t.Fatalf("candidate=%v error=%v", candidate, err)
				}
				if model.calls != 0 || leaseChecks != 0 || len(*order) != 0 || base.closed != 0 || compiled.closed != 0 {
					t.Fatalf("unsupported objective reached work: extraction=%d lease=%d order=%v closed=%d/%d", model.calls, leaseChecks, *order, base.closed, compiled.closed)
				}
				return
			}
			if err != nil || candidate == nil || !candidate.Validated || candidate.CostSource != CostMeasured || candidate.Cost != float64((5*time.Microsecond).Nanoseconds()) || len(candidate.Trials) != 4 {
				t.Fatalf("candidate=%v error=%v", candidate, err)
			}
			if model.calls == 0 || leaseChecks == 0 || len(*order) == 0 || base.closed != 0 || compiled.closed != 1 {
				t.Fatalf("latency objective missed work: extraction=%d lease=%d order=%v closed=%d/%d", model.calls, leaseChecks, *order, base.closed, compiled.closed)
			}
		})
	}
}

func TestExecutableObjectiveAdmissionOrder(t *testing.T) {
	for _, test := range []struct {
		name   string
		change func(*ExecutableEvaluator)
		want   string
	}{
		{"missing_binding", func(e *ExecutableEvaluator) { e.Workload.InputIdentities = nil }, "no identity for required input"},
		{"rounds", func(e *ExecutableEvaluator) { e.Rounds = 1 }, "ABBA rounds"},
		{"timing", func(e *ExecutableEvaluator) { e.TimingKind = cudatrial.TimingWallElapsed }, ErrTimingKindMismatch.Error()},
		{"tolerance", func(e *ExecutableEvaluator) { e.Policy.Absolute = math.NaN() }, "invalid output tolerance"},
		{"exact", func(e *ExecutableEvaluator) { e.Workload.NumericalPolicy = NumericalPolicyExact; e.Policy.Absolute = 1 }, "nonzero tolerance conflicts"},
		{"compile_timeout", func(e *ExecutableEvaluator) { e.CompileTimeout = -1 }, "negative compile timeout"},
		{"generation", func(e *ExecutableEvaluator) { e.Generation = -1 }, ErrStaleGeneration.Error()},
	} {
		t.Run(test.name, func(t *testing.T) {
			e, base, compiled, order := executableFixture(t)
			e.Workload.Objective = Objective("unknown")
			test.change(e)
			leaseChecks := 0
			e.HardwareLockChecker = func() bool { leaseChecks++; return true }
			s := executableState()
			candidate, err := e.Evaluate(context.Background(), s, s.Roots["root"])
			if candidate != nil || err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("candidate=%v error=%v, want %q", candidate, err, test.want)
			}
			if leaseChecks != 0 || len(*order) != 0 || base.closed != 0 || compiled.closed != 0 {
				t.Fatalf("configuration refusal reached work: lease=%d order=%v closed=%d/%d", leaseChecks, *order, base.closed, compiled.closed)
			}
		})
	}
}
