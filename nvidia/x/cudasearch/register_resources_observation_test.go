package cudasearch

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// registerObserver refuses the start or outcome of selected attempts. Attempt
// numbers start at 1, as in EvaluationStart.
type registerObserver struct {
	log                 *[]string
	starts              []EvaluationStart
	finishes            []EvaluationFinish
	refuseStart         uint64
	refuseFinish        uint64
	startErr, finishErr error
	atFinish            func(EvaluationFinish)
}

func (o *registerObserver) StartEvaluation(_ context.Context, s EvaluationStart) error {
	*o.log = append(*o.log, "start")
	if s.Attempt == o.refuseStart {
		return o.startErr
	}
	o.starts = append(o.starts, s)
	return nil
}

func (o *registerObserver) FinishEvaluation(_ context.Context, f EvaluationFinish) error {
	*o.log = append(*o.log, "finish")
	if o.atFinish != nil {
		o.atFinish(f)
	}
	o.finishes = append(o.finishes, f)
	if f.Attempt == o.refuseFinish {
		return o.finishErr
	}
	return nil
}

// registerCompiler returns a compiler for registerOptions whose second option
// is fastest, and the owners it returned.
func registerCompiler(base *testExecutable, mutate func(i int, x *testExecutable) error) (func(context.Context, *egraph.Expr, WorkloadBindings, RegisterOption) (Executable, error), *[]*testExecutable) {
	var owners []*testExecutable
	return func(_ context.Context, _ *egraph.Expr, _ WorkloadBindings, option RegisterOption) (Executable, error) {
		durations := []time.Duration{9, 3, 6}
		x := &testExecutable{id: option.Identity, value: 1, kind: base.kind, duration: durations[len(owners)] * time.Microsecond, order: base.order}
		owners = append(owners, x)
		*base.order = append(*base.order, "compile")
		var err error
		if mutate != nil {
			err = mutate(len(owners)-1, x)
		}
		return x, err
	}, &owners
}

// The resource search admits each option through the evaluator's observed
// attempt owner: one start and one outcome per option, none for its preflight.
func TestRegisterSearchObservesEachOption(t *testing.T) {
	e, base, _, order := executableFixture(t)
	defer e.Close()
	options := registerOptions(t)
	o := &registerObserver{log: order}
	closedAtFinish := []int{}
	compile, owners := registerCompiler(base, nil)
	o.atFinish = func(f EvaluationFinish) {
		closedAtFinish = append(closedAtFinish, (*owners)[f.Attempt-1].closed)
	}
	e.Observer = o
	previous := reflect.ValueOf(e.Compile).Pointer()
	r, err := e.SearchRegisters(t.Context(), &egraph.Expr{Op: "fixture"}, options, registerLimits(), compile)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if r.Evaluations != 3 || len(o.starts) != 3 || len(o.finishes) != 3 || r.Winner.ID != options[1].Identity {
		t.Fatalf("evaluations=%d starts=%d finishes=%d winner=%v", r.Evaluations, len(o.starts), len(o.finishes), r.Winner)
	}
	// Each attempt is bracketed: start, then compile, then its outcome.
	var bracket []string
	for _, step := range *order {
		if step == "start" || step == "compile" || step == "finish" {
			bracket = append(bracket, step)
		}
	}
	if want := slices.Repeat([]string{"start", "compile", "finish"}, 3); !slices.Equal(bracket, want) {
		t.Fatalf("order=%v", bracket)
	}
	for i := range 3 {
		s, f := o.starts[i], o.finishes[i]
		if s.Attempt != uint64(i+1) || f.Attempt != s.Attempt || s.Kind != EvaluationRegister || f.Kind != EvaluationRegister {
			t.Fatalf("attempt %d: start=%+v finish=%+v", i, s, f)
		}
		if s.Derivation != nil || s.ConfirmationOf != "" || s.WorkloadKey != e.Workload.Key() || s.BaselineID != "base" {
			t.Fatalf("register start invented search state: %+v", s)
		}
		if f.Err != nil || f.Candidate == nil || f.Candidate.ID != options[i].Identity || f.Evidence.ExecutableID != options[i].Identity || f.MeasurementReused {
			t.Fatalf("attempt %d outcome=%+v", i, f)
		}
		if closedAtFinish[i] != 1 {
			t.Fatalf("attempt %d recorded before its owner closed: closes=%d", i, closedAtFinish[i])
		}
	}
	if reflect.ValueOf(e.Compile).Pointer() != previous {
		t.Fatal("observed search changed the compiler")
	}
}

func TestRegisterSearchObserverRefusals(t *testing.T) {
	stored := errors.New("archive refused")
	cleanup := errors.New("close failed")
	for _, tt := range []struct {
		name                      string
		refuseStart, refuseFinish uint64
		mutate                    func(int, *testExecutable) error
		compiles, finishes        int
		winner                    int // option index, or -1 for none
		want                      []error
	}{
		// Refusing the first start prevents every compile and admits nothing.
		{name: "first start", refuseStart: 1, compiles: 0, finishes: 0, winner: -1, want: []error{stored}},
		// Refusing a later start keeps the admitted best and compiles no further.
		{name: "second start", refuseStart: 2, compiles: 1, finishes: 1, winner: 0, want: []error{stored}},
		// The fastest option is measured but its outcome is not recorded: it
		// is withheld and the earlier admitted best remains the winner.
		{name: "winning outcome", refuseFinish: 2, compiles: 2, finishes: 2, winner: 0, want: []error{stored}},
		// A failed attempt whose outcome is refused keeps both errors.
		{name: "failed outcome", refuseFinish: 2, compiles: 2, finishes: 2, winner: 0, want: []error{stored, ErrVerificationFailed},
			mutate: func(i int, x *testExecutable) error {
				if i == 1 {
					x.value = 2
				}
				return nil
			}},
		// A failed cleanup whose outcome is refused stays owned by e.
		{name: "cleanup outcome", refuseFinish: 2, compiles: 2, finishes: 2, winner: 0, want: []error{stored, cleanup},
			mutate: func(i int, x *testExecutable) error {
				if i == 1 {
					x.closeErr = cleanup
				}
				return nil
			}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			e, base, _, order := executableFixture(t)
			options := registerOptions(t)
			o := &registerObserver{log: order, refuseStart: tt.refuseStart, refuseFinish: tt.refuseFinish, startErr: stored, finishErr: stored}
			e.Observer = o
			compile, owners := registerCompiler(base, tt.mutate)
			previous := reflect.ValueOf(e.Compile).Pointer()
			r, err := e.SearchRegisters(t.Context(), &egraph.Expr{Op: "fixture"}, options, registerLimits(), compile)
			for _, want := range tt.want {
				if !errors.Is(err, want) {
					t.Fatalf("err=%v, want %v", err, want)
				}
			}
			if r == nil {
				t.Fatal("no result")
			}
			defer r.Close()
			if len(*owners) != tt.compiles || len(o.finishes) != tt.finishes {
				t.Fatalf("compiles=%d finishes=%d", len(*owners), len(o.finishes))
			}
			if tt.winner < 0 && r.Winner != nil || tt.winner >= 0 && (r.Winner == nil || r.Winner.ID != options[tt.winner].Identity) {
				t.Fatalf("winner=%v", r.Winner)
			}
			if tt.refuseFinish != 0 {
				f := o.finishes[tt.refuseFinish-1]
				// The refused record carries the attempt's own outcome, not the
				// recording failure.
				if errors.Is(f.Err, stored) || (len(tt.want) == 1) != (f.Err == nil) || (f.Err == nil) != (f.Candidate != nil) {
					t.Fatalf("refused outcome=%+v", f)
				}
			}
			for i, x := range *owners {
				if x.closed != 1 {
					t.Fatalf("owner %d closes=%d", i, x.closed)
				}
			}
			if base.closed != 0 || reflect.ValueOf(e.Compile).Pointer() != previous {
				t.Fatal("closed borrowed baseline or changed compiler")
			}
			if tt.name == "cleanup outcome" {
				if err := e.Close(); !errors.Is(err, cleanup) {
					t.Fatalf("lost pending owner: %v", err)
				}
				(*owners)[1].closeErr = nil
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// Observation copies stay charged with the attempt and the winner's metadata
// with the result; closing both returns the ledger to its setup charge.
func TestRegisterSearchObserverAccounting(t *testing.T) {
	for _, refuse := range []uint64{0, 2} {
		e, base, _, order := executableFixture(t)
		o := &registerObserver{log: order, refuseFinish: refuse, finishErr: errors.New("archive refused")}
		e.Observer = o
		ctx, work, err := searchWorkBudget(t.Context(), SearchBudget{MaxMemoryBytes: 1 << 22, CallbackMemoryBytes: 32 << 10}, e)
		if err != nil {
			t.Fatal(err)
		}
		setup := work.memory.live
		compile, _ := registerCompiler(base, nil)
		r, err := e.SearchRegisters(ctx, &egraph.Expr{Op: "fixture"}, registerOptions(t), registerLimits(), compile)
		if (refuse == 0) != (err == nil) || r == nil || r.Winner == nil {
			t.Fatalf("refuse=%d result=%v err=%v", refuse, r, err)
		}
		if work.memory.live <= setup || e.attemptMemory.Bytes() == 0 {
			t.Fatal("winner or observation copies uncharged")
		}
		r.Close()
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
		if work.memory.live != setup {
			t.Fatalf("refuse=%d: storage leaked: live=%d setup=%d", refuse, work.memory.live, setup)
		}
		work.closeSearch()
	}
}
