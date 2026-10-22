package cudasearch

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// recordingObserver is a host test double for a durable attempt store.
type recordingObserver struct {
	log                 *[]string
	starts              []EvaluationStart
	finishes            []EvaluationFinish
	startErr, finishErr error
	atFinish            func(EvaluationFinish)
}

func (o *recordingObserver) StartEvaluation(_ context.Context, s EvaluationStart) error {
	*o.log = append(*o.log, "start")
	if o.startErr != nil {
		return o.startErr
	}
	o.starts = append(o.starts, s)
	return nil
}

func (o *recordingObserver) FinishEvaluation(ctx context.Context, f EvaluationFinish) error {
	*o.log = append(*o.log, "finish")
	if ctx.Err() != nil {
		return errors.New("finish context canceled")
	}
	if o.atFinish != nil {
		o.atFinish(f)
	}
	o.finishes = append(o.finishes, f)
	return o.finishErr
}

type countingCost struct{ calls *int }

func (c countingCost) NodeCost(egraph.ENode, []float64) float64 { *c.calls++; return 1 }

// observedFixture returns an evaluator configured for Evaluate or Confirm,
// and the subject each entry point consumes.
func observedFixture(t *testing.T, kind EvaluationKind) (*ExecutableEvaluator, *testExecutable, *[]string, *egraph.SearchState, *CandidatePlan) {
	t.Helper()
	var screened *CandidatePlan
	if kind == EvaluationConfirmation {
		screened = screenedFixture(t)
		t.Cleanup(screened.Close)
	}
	e, _, compiled, order := executableFixture(t)
	if kind == EvaluationConfirmation {
		e.Workload.InputIdentities = map[string]string{"x": "held-input"}
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Error(err)
		}
	})
	return e, compiled, order, executableState(), screened
}

func observedRun(ctx context.Context, e *ExecutableEvaluator, s *egraph.SearchState, screened *CandidatePlan) (*CandidatePlan, error) {
	if screened != nil {
		return e.Confirm(ctx, screened)
	}
	return e.Evaluate(ctx, s, s.Roots["root"])
}

func TestObserverStartFailurePreventsWork(t *testing.T) {
	stored := errors.New("archive refused start")
	for _, kind := range []EvaluationKind{EvaluationSearch, EvaluationConfirmation} {
		for _, refuse := range []bool{false, true} {
			name := string(kind) + "/accepted"
			if refuse {
				name = string(kind) + "/refused"
			}
			t.Run(name, func(t *testing.T) {
				e, compiled, order, s, screened := observedFixture(t, kind)
				extractions := 0
				e.CostModel = countingCost{&extractions}
				if screened != nil {
					// Copying this expression fails, so a copy before the
					// start record would replace the refusal error.
					screened.Expr.Children = []*egraph.Expr{nil}
				}
				o := &recordingObserver{log: order}
				if refuse {
					o.startErr = stored
				}
				e.Observer = o
				c, err := observedRun(t.Context(), e, s, screened)
				if refuse {
					// Refusal: no extraction, copy, compile or executable
					// call, and no terminal record.
					if c != nil || !errors.Is(err, stored) || extractions != 0 || compiled.closed != 0 || !slices.Equal(*order, []string{"start"}) || len(o.finishes) != 0 {
						t.Fatalf("refused start did work: c=%v err=%v extractions=%d order=%v finishes=%d", c, err, extractions, *order, len(o.finishes))
					}
					return
				}
				// Control: with the start accepted, the same fixture reaches the
				// work the refusal must prevent, and the outcome is recorded.
				c.Close()
				if len(o.finishes) != 1 || !errors.Is(o.finishes[0].Err, err) {
					t.Fatalf("control: err=%v finishes=%v", err, o.finishes)
				}
				if screened == nil && (err != nil || extractions == 0 || compiled.closed != 1) {
					t.Fatalf("control did not extract and compile: err=%v extractions=%d closed=%d", err, extractions, compiled.closed)
				}
				if screened != nil && (err == nil || errors.Is(err, stored) || !slices.Equal(*order, []string{"start", "finish"})) {
					t.Fatalf("control did not copy the expression: err=%v order=%v", err, *order)
				}
			})
		}
	}
}

func TestObserverRecordsAfterCleanupAndConfirmation(t *testing.T) {
	for _, kind := range []EvaluationKind{EvaluationSearch, EvaluationConfirmation} {
		t.Run(string(kind), func(t *testing.T) {
			e, compiled, order, s, screened := observedFixture(t, kind)
			o := &recordingObserver{log: order}
			o.atFinish = func(f EvaluationFinish) {
				// The fixture compiler returns one owner, closed once per attempt.
				if compiled.closed != int(f.Attempt) {
					t.Error("outcome recorded before candidate cleanup")
				}
			}
			e.Observer = o
			c, err := observedRun(t.Context(), e, s, screened)
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			if (*order)[0] != "start" || (*order)[1] != "compile" || (*order)[len(*order)-1] != "finish" {
				t.Fatalf("order=%v", *order)
			}
			if len(o.starts) != 1 || len(o.finishes) != 1 {
				t.Fatal(o.starts, o.finishes)
			}
			start, finish := o.starts[0], o.finishes[0]
			if start.Attempt != 1 || finish.Attempt != 1 || start.Kind != kind || finish.Kind != kind || start.WorkloadKey != e.Workload.Key() || start.InputKey != inputKey(e.Workload) || start.BaselineID != "base" || start.Rounds != 2 || start.TimingKind != cudatrial.TimingSummedKernel {
				t.Fatalf("start=%+v finish=%+v", start, finish)
			}
			got := finish.Candidate
			if got == nil || finish.Err != nil || got.ID != c.ID || got.Cost != c.Cost || !got.Validated || got.CostSource != CostMeasured || len(finish.Evidence.Trials) != 4 || finish.Evidence.ExecutableID != compiled.id || finish.Evidence.ComparisonArm != "candidate" {
				t.Fatalf("finish=%+v candidate=%+v", finish, got)
			}
			if screened != nil && (start.ConfirmationOf != screened.ID || got.ConfirmationOf != screened.ID || c.ConfirmationOf != screened.ID) {
				t.Fatalf("confirmation association: start=%q finish=%q", start.ConfirmationOf, got.ConfirmationOf)
			}
			if screened == nil && (start.ConfirmationOf != "" || got.ConfirmationOf != "") {
				t.Fatal("search attempt recorded as confirmation")
			}
			// A second admitted attempt has its own sequence number and pair.
			c2, err := observedRun(t.Context(), e, s, screened)
			if err != nil {
				t.Fatal(err)
			}
			c2.Close()
			if len(o.finishes) != 2 || o.starts[1].Attempt != 2 || o.finishes[1].Attempt != 2 {
				t.Fatal(o.starts, o.finishes)
			}
		})
	}
}

func TestObserverFinishFailureRejectsCandidate(t *testing.T) {
	stored := errors.New("archive refused outcome")
	for _, stage := range []string{"admitted", "wrong_output", "cleanup", "canceled"} {
		t.Run(stage, func(t *testing.T) {
			e, compiled, order, s, _ := observedFixture(t, EvaluationSearch)
			o := &recordingObserver{log: order, finishErr: stored}
			e.Observer = o
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			cleanup := errors.New("close failed")
			var want error
			switch stage {
			case "wrong_output":
				compiled.value = 2
				want = ErrVerificationFailed
			case "cleanup":
				compiled.closeErr = cleanup
				want = cleanup
			case "canceled":
				// Cancel after the start record: the outcome still records.
				e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) {
					cancel()
					return compiled, nil
				}
				want = context.Canceled
			}
			c, err := e.Evaluate(ctx, s, s.Roots["root"])
			if c != nil || !errors.Is(err, stored) {
				t.Fatal(c, err)
			}
			if len(o.finishes) != 1 {
				t.Fatal("missing terminal record", o.finishes)
			}
			f := o.finishes[0]
			if want == nil {
				// The candidate was admitted before recording and is withdrawn.
				if f.Err != nil || f.Candidate == nil || f.Candidate.ID != compiled.id {
					t.Fatalf("finish=%+v", f)
				}
				return
			}
			if !errors.Is(err, want) || !errors.Is(f.Err, want) || errors.Is(f.Err, stored) || f.Candidate != nil {
				t.Fatalf("err=%v finish=%+v", err, f)
			}
			if stage == "cleanup" {
				if len(e.pending) != 1 {
					t.Fatal("failed cleanup owner not retained")
				}
				compiled.closeErr = nil
			}
		})
	}
}

func TestObserverFinishFailureLeavesNoWinner(t *testing.T) {
	for _, fail := range []bool{false, true} {
		e, _, order, s, _ := observedFixture(t, EvaluationSearch)
		o := &recordingObserver{log: order}
		if fail {
			o.finishErr = errors.New("archive refused outcome")
		}
		e.Observer = o
		beam := NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: SearchBudget{MaxStates: 1, MaxDepth: 1, MaxEvaluations: 1, MaxCompilations: 1}})
		r, err := beam.Search(t.Context(), s, nil, e)
		if !fail {
			// Control: the observed concrete evaluator keeps its work cap.
			if err != nil || r.Winner == nil || !r.WorkAccounted || r.Compilations != 1 {
				t.Fatalf("control: %+v %v", r, err)
			}
			r.Close()
			continue
		}
		if r != nil && r.Winner != nil {
			t.Fatal("candidate admitted after failed outcome record", r.Winner)
		}
		if r != nil && (len(r.Rejected) != 1 || !errors.Is(r.Rejected[0].Err, o.finishErr)) {
			t.Fatalf("rejection=%+v err=%v", r.Rejected, err)
		}
		r.Close()
	}
}

func TestObserverRecordsAbandonedAttempt(t *testing.T) {
	e, _, order, s, _ := observedFixture(t, EvaluationSearch)
	o := &recordingObserver{log: order}
	e.Observer = o
	e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { panic("compiler bug") }
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("panic was swallowed")
			}
		}()
		e.Evaluate(t.Context(), s, s.Roots["root"])
	}()
	if len(o.finishes) != 1 || !errors.Is(o.finishes[0].Err, errEvaluationAbandoned) {
		t.Fatal(o.finishes)
	}
}

func TestObserverSnapshotsAreDetached(t *testing.T) {
	e, _, order, s, _ := observedFixture(t, EvaluationSearch)
	s.Derivation = []egraph.Action{{RuleName: "r", Subst: egraph.Subst{"x": 1}}}
	o := &recordingObserver{log: order}
	e.Observer = o
	c, err := e.Evaluate(t.Context(), s, s.Roots["root"])
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	s.Derivation[0].Subst["x"] = 9
	e.Receipts[0].Name = "mutated"
	e.Comparison["out"] = OutputComparison{Elements: 99}
	start, finish := o.starts[0], o.finishes[0]
	if start.Derivation[0].Subst["x"] != 1 || finish.Evidence.Trials[0].Name == "mutated" || finish.Evidence.Comparison["out"].Elements != 1 {
		t.Fatalf("observation aliases evaluator storage: %+v %+v", start.Derivation, finish.Evidence)
	}
	c.Derivation[0].Subst["x"] = 7
	if start.Derivation[0].Subst["x"] != 1 {
		t.Fatal("start derivation aliases candidate")
	}
}

// observedLedger runs one capped attempt and reports the attempt-scope bytes
// that remain charged, or the attempt error.
func observedLedger(t *testing.T, limit int64, observe bool) (attempt int64, o *recordingObserver, err error) {
	t.Helper()
	e, compiled, order, s, _ := observedFixture(t, EvaluationSearch)
	for i := range 64 {
		s.Derivation = append(s.Derivation, egraph.Action{RuleName: "r", StepIndex: i, Subst: egraph.Subst{"x": egraph.Id(i), "y": egraph.Id(i)}})
	}
	if observe {
		o = &recordingObserver{log: order}
		e.Observer = o
	}
	ctx, work, err := searchWorkBudget(t.Context(), SearchBudget{MaxMemoryBytes: limit, CallbackMemoryBytes: 32 << 10}, e)
	if err != nil {
		return 0, o, err
	}
	defer work.closeSearch()
	setup := work.memory.live
	c, err := evaluateSearch(ctx, e, s, s.Roots["root"])
	if err != nil && c != nil {
		t.Fatal("candidate returned with error")
	}
	attempt = e.attemptMemory.Bytes()
	c.Close()
	if cerr := e.Close(); cerr != nil {
		t.Fatal(cerr)
	}
	if work.memory.live != setup {
		t.Fatalf("observation storage leaked: live=%d setup=%d", work.memory.live, setup)
	}
	if err != nil && compiled.closed != 0 && observe && o.finishes == nil {
		t.Fatal("compiled without a terminal record")
	}
	return attempt, o, err
}

func TestObserverSnapshotsChargedToAttempt(t *testing.T) {
	plain, _, err := observedLedger(t, 1<<22, false)
	if err != nil {
		t.Fatal(err)
	}
	observed, o, err := observedLedger(t, 1<<22, true)
	if err != nil {
		t.Fatal(err)
	}
	// 64 copied actions and their maps are at least 64 action slots.
	if min := plain + 64*int64(reflect.TypeFor[egraph.Action]().Size()); len(o.starts) != 1 || observed < min {
		t.Fatalf("observation copies uncharged: observed=%d plain=%d", observed, plain)
	}
	// The smallest limit admitting an unobserved attempt cannot also hold the
	// observation copies. The observed attempt is refused, not over-admitted.
	lo, hi := int64(1), int64(1<<22)
	for lo < hi {
		mid := lo + (hi-lo)/2
		if _, _, err := observedLedger(t, mid, false); err == nil {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	if _, _, err := observedLedger(t, lo, false); err != nil {
		t.Fatal("no admitting limit", err)
	}
	_, o, err = observedLedger(t, lo, true)
	if !errors.Is(err, ErrMemoryBudgetExceeded) {
		t.Fatalf("observed attempt at unobserved minimum %d: %v", lo, err)
	}
	if len(o.starts) == 1 && (len(o.finishes) != 1 || o.finishes[0].Candidate != nil) {
		t.Fatal("refused attempt lacks its terminal record", o.finishes)
	}
}
