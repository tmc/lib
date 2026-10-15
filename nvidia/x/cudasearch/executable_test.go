package cudasearch

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type testExecutable struct {
	id       string
	value    float32
	kind     cudatrial.TimingKind
	duration time.Duration
	order    *[]string
	closed   int
	closeErr error
}

func executableState() *egraph.SearchState {
	g := egraph.New()
	x := g.Add(egraph.ENode{Op: "fixture"})
	return egraph.NewSearchState(g, map[string]egraph.Id{"root": x})
}

func (x *testExecutable) Identity() string { return x.id }
func (x *testExecutable) Outputs(context.Context) (map[string][]float32, error) {
	*x.order = append(*x.order, x.id+" check")
	return map[string][]float32{"out": {x.value}}, nil
}
func (x *testExecutable) Run(context.Context) (cudatrial.Receipt, error) {
	*x.order = append(*x.order, x.id+" run")
	r := cudatrial.Receipt{Complete: true, TimingKind: x.kind}
	if x.duration > 0 {
		_ = r.RecordTiming(x.kind, x.duration)
	}
	return r, nil
}
func (x *testExecutable) Close() error { x.closed++; return x.closeErr }

func executableFixture(t *testing.T) (*ExecutableEvaluator, *testExecutable, *testExecutable, *[]string) {
	t.Helper()
	order := []string{}
	base := &testExecutable{id: "base", value: 1, kind: cudatrial.TimingSummedKernel, duration: 10 * time.Microsecond, order: &order}
	compiled := &testExecutable{id: "image-and-launch", value: 1, kind: cudatrial.TimingSummedKernel, duration: 5 * time.Microsecond, order: &order}
	e := &ExecutableEvaluator{
		Workload: WorkloadBindings{Name: "fixture", InputNames: []string{"x"}, InputIdentities: map[string]string{"x": "synthetic-fixture-x"}, NumericalPolicy: NumericalPolicyFP32, ResetState: func() error { order = append(order, "reset"); return nil }},
		Compile: func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) {
			order = append(order, "compile")
			return compiled, nil
		},
		Baseline: base, References: map[string][]float32{"out": {1}}, Rounds: 2, TimingKind: cudatrial.TimingSummedKernel,
		// Synthetic CPU adapters: this checker does not authorize GPU work.
		HardwareLockChecker: func() bool { return true },
	}
	return e, base, compiled, &order
}

func TestExecutableEvaluatorABBA(t *testing.T) {
	e, base, compiled, order := executableFixture(t)
	e.Policy.Absolute = 0.01
	compiled.value = 1.001
	s := executableState()
	c, err := e.Evaluate(context.Background(), s, s.Roots["root"])
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"compile", "reset", "base check", "reset", "image-and-launch check", "reset", "base run", "reset", "image-and-launch run", "reset", "image-and-launch run", "reset", "base run"}
	if !reflect.DeepEqual(*order, want) {
		t.Fatalf("order=%v", *order)
	}
	if !c.Validated || c.Cost != 5000 || c.ID != compiled.id || c.Comparison["out"].MaxAbsolute <= 0 || len(e.Receipts) != 4 || compiled.closed != 1 || base.closed != 0 {
		t.Fatalf("candidate=%+v receipts=%v", c, e.Receipts)
	}
	if c.Receipt.Provenance["executable_identity"] != compiled.id {
		t.Fatal("lost execution identity")
	}
}

func TestExecutableEvaluatorFailureGates(t *testing.T) {
	for _, stage := range []string{"wrong_output", "secondary_output", "mixed_timing", "unmeasured", "no_lease", "cleanup"} {
		t.Run(stage, func(t *testing.T) {
			e, _, compiled, order := executableFixture(t)
			failure := errors.New("cleanup failed")
			switch stage {
			case "wrong_output":
				compiled.value = 2
			case "secondary_output":
				e.References["secondary"] = []float32{4}
			case "mixed_timing":
				compiled.kind = cudatrial.TimingGPUBusy
			case "unmeasured":
				compiled.duration = 0
			case "no_lease":
				e.HardwareLockChecker = func() bool { return false }
			case "cleanup":
				compiled.closeErr = failure
			}
			s := executableState()
			c, err := e.Evaluate(context.Background(), s, s.Roots["root"])
			if err == nil || c != nil {
				t.Fatalf("candidate=%v error=%v", c, err)
			}
			if stage == "wrong_output" || stage == "secondary_output" {
				if len(e.Receipts) != 0 {
					t.Fatal("timed incorrect output")
				}
			}
			if stage == "no_lease" && len(*order) != 0 {
				t.Fatal("used executable without lease")
			}
			if stage == "cleanup" {
				if !errors.Is(err, failure) {
					t.Fatal(err)
				}
				compiled.closeErr = nil
				if _, err := e.Evaluate(context.Background(), s, s.Roots["root"]); err == nil {
					t.Fatal("ignored pending owner")
				}
				if err := e.Close(); err != nil {
					t.Fatal(err)
				}
				if compiled.closed != 2 {
					t.Fatalf("close calls=%d", compiled.closed)
				}
			}
		})
	}
}

func TestExecutableEvaluatorCompilerErrorOwner(t *testing.T) {
	e, _, compiled, _ := executableFixture(t)
	failure := errors.New("compile failure")
	e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { return compiled, failure }
	s := executableState()
	c, err := e.Evaluate(context.Background(), s, s.Roots["root"])
	if c != nil || !errors.Is(err, failure) || compiled.closed != 1 {
		t.Fatalf("candidate=%v error=%v closes=%d", c, err, compiled.closed)
	}
}

func TestExecutableCompileDeadline(t *testing.T) {
	for _, kind := range []string{"timeout", "late_success", "parent_cancel", "cleanup_failure"} {
		t.Run(kind, func(t *testing.T) {
			e, base, compiled, order := executableFixture(t)
			e.CompileTimeout = time.Millisecond
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cleanupErr := errors.New("close failed")
			if kind == "cleanup_failure" {
				compiled.closeErr = cleanupErr
			}
			e.Compile = func(child context.Context, _ *egraph.Expr, _ WorkloadBindings) (Executable, error) {
				if _, ok := child.Deadline(); !ok {
					t.Fatal("compiler has no deadline")
				}
				if kind == "parent_cancel" {
					cancel()
				}
				<-child.Done()
				if kind == "timeout" {
					return compiled, child.Err()
				}
				return compiled, nil // Simulate a non-cooperative late return.
			}
			s := executableState()
			candidate, err := e.Evaluate(ctx, s, s.Roots["root"])
			want := context.DeadlineExceeded
			if kind == "parent_cancel" {
				want = context.Canceled
			}
			if candidate != nil || !errors.Is(err, want) || compiled.closed != 1 || base.closed != 0 || len(*order) != 0 {
				t.Fatalf("candidate=%v err=%v closed=%d order=%v", candidate, err, compiled.closed, *order)
			}
			if kind == "cleanup_failure" {
				if !errors.Is(err, cleanupErr) || len(e.pending) != 1 {
					t.Fatalf("lost cleanup owner: %v", err)
				}
				compiled.closeErr = nil
				if err := e.Close(); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func TestExecutableBoundsExtractionBeforeCompile(t *testing.T) {
	e, _, _, _ := executableFixture(t)
	e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) {
		t.Fatal("compiled oversized extracted tree")
		return nil, nil
	}
	g := egraph.New()
	root := g.Add(egraph.ENode{Op: "leaf"})
	for i := 0; i < 24; i++ {
		root = g.Add(egraph.ENode{Op: "pair", Children: []egraph.Id{root, root}})
	}
	s := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	if candidate, err := e.Evaluate(context.Background(), s, root); candidate != nil || !errors.Is(err, egraph.ErrExtractionLimit) {
		t.Fatalf("candidate=%v err=%v", candidate, err)
	}
}

func TestExecutableEvaluatorAdmission(t *testing.T) {
	for _, change := range []func(*ExecutableEvaluator){
		func(e *ExecutableEvaluator) { e.Rounds = math.MaxInt - 1 },
		func(e *ExecutableEvaluator) { e.Policy.Absolute = math.NaN() },
		func(e *ExecutableEvaluator) { e.Workload.NumericalPolicy = NumericalPolicyExact; e.Policy.Absolute = 1 },
	} {
		e, _, _, order := executableFixture(t)
		change(e)
		s := executableState()
		if _, err := e.Evaluate(context.Background(), s, s.Roots["root"]); err == nil {
			t.Fatal("accepted invalid configuration")
		}
		if len(*order) != 0 {
			t.Fatal("executed before configuration admission")
		}
	}
}

func TestOutputPolicyIEEE(t *testing.T) {
	for _, tc := range []struct {
		name      string
		got, want float32
		policy    OutputPolicy
		ok        bool
	}{
		{"absolute", 1.01, 1, OutputPolicy{Absolute: 0.02}, true},
		{"outside", 1.1, 1, OutputPolicy{Absolute: 0.02}, false},
		{"relative", 101, 100, OutputPolicy{Relative: 0.02}, true},
		{"nan", float32(math.NaN()), float32(math.NaN()), OutputPolicy{}, false},
		{"nan_equal", float32(math.NaN()), float32(math.NaN()), OutputPolicy{NaNsEqual: true}, true},
		{"infinity", float32(math.Inf(1)), float32(math.Inf(1)), OutputPolicy{}, true},
		{"opposite_infinity", float32(math.Inf(1)), float32(math.Inf(-1)), OutputPolicy{}, false},
		{"signed_zero", math.Float32frombits(0x80000000), 0, OutputPolicy{}, false},
		{"ignore_zero", math.Float32frombits(0x80000000), 0, OutputPolicy{IgnoreZeroSign: true}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.policy.compare(map[string][]float32{"o": {tc.got}}, map[string][]float32{"o": {tc.want}})
			if (err == nil) != tc.ok {
				t.Fatal(err)
			}
		})
	}
}

func TestExecutableEvaluatorBeamConsumer(t *testing.T) {
	e, _, _, _ := executableFixture(t)
	s := executableState()
	beam := NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: SearchBudget{MaxStates: 1, MaxDepth: 1, MaxEvaluations: 1}})
	r, err := beam.Search(context.Background(), s, nil, e)
	if err != nil {
		t.Fatal(err)
	}
	if r.Winner == nil || !r.Winner.Validated || r.Winner.Cost != 5000 {
		t.Fatalf("result=%+v", r)
	}
}
