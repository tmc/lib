package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type measurementExecutable struct {
	*testExecutable
	outputs, runs   int
	runError        error
	runReceiptError error
	runIncomplete   bool
	onClose         func()
	preparationWall int64
}

func (x *measurementExecutable) Preparation() []PreparationPhase {
	return []PreparationPhase{{Name: "current-compile", WallNs: x.preparationWall, Complete: true}}
}
func (x *measurementExecutable) Outputs(ctx context.Context) (map[string][]float32, error) {
	x.outputs++
	return x.testExecutable.Outputs(ctx)
}
func (x *measurementExecutable) Run(ctx context.Context) (cudatrial.Receipt, error) {
	x.runs++
	r, err := x.testExecutable.Run(ctx)
	r.Err = x.runReceiptError
	if x.runIncomplete {
		r.Complete = false
	}
	r.Provenance = map[string]string{"raw-source": "host-fixture"}
	return r, errors.Join(err, x.runError)
}
func (x *measurementExecutable) Close() error {
	if x.onClose != nil {
		x.onClose()
	}
	return x.testExecutable.Close()
}

type measurementCost struct{ calls int }

func (m *measurementCost) NodeCost(n egraph.ENode, c []float64) float64 {
	m.calls++
	return (egraph.AstSizeCost{}).NodeCost(n, c)
}
func (m *measurementCost) NodeCostWithMemory(n egraph.ENode, c []float64, _ *egraphcore.MemoryScope) (float64, error) {
	return m.NodeCost(n, c), nil
}

type measurementFixture struct {
	e              *ExecutableEvaluator
	base           *measurementExecutable
	compiled       []*measurementExecutable
	model          *measurementCost
	scoreAtCompile []int
	ledger         *memoryBudget
	beforeCompile  func(int)
	configure      func(*measurementExecutable, int)
}

func newMeasurementFixture(t *testing.T) *measurementFixture {
	t.Helper()
	e, b, c, order := executableFixture(t)
	f := &measurementFixture{e: e, base: &measurementExecutable{testExecutable: b}, model: &measurementCost{}}
	e.Baseline = f.base
	e.CostModel = f.model
	e.Generation = 7
	e.MeasurementContext = MeasurementContext{TargetIdentity: "host-fixture/device0/driver-none/toolchain-fake/context1", ResetIdentity: "reset-state-v1"}
	e.Compile = func(ctx context.Context, _ *egraph.Expr, _ WorkloadBindings) (Executable, error) {
		f.ledger = searchMemoryBudget(ctx)
		f.scoreAtCompile = append(f.scoreAtCompile, f.model.calls)
		if f.beforeCompile != nil {
			f.beforeCompile(len(f.compiled))
		}
		v := *c
		v.order = order
		v.closed = 0
		x := &measurementExecutable{testExecutable: &v, preparationWall: int64(len(f.compiled) + 1)}
		if f.configure != nil {
			f.configure(x, len(f.compiled))
		}
		f.compiled = append(f.compiled, x)
		return x, nil
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Errorf("evaluator cleanup: %v", err)
		}
	})
	return f
}
func measurementBudget(memory int64) SearchBudget {
	return SearchBudget{MaxStates: 3, MaxDepth: 1, MaxActions: 1, MaxTotalActions: 1, MaxEvaluations: 2, MaxMemoryBytes: memory, CallbackMemoryBytes: 32 << 10}
}
func runMeasurementSearch(t *testing.T, kind string, eval Evaluator, memory int64) *SearchResult {
	t.Helper()
	var r *SearchResult
	var err error
	initial := executableState()
	rules := []*egraph.Rewrite{egraph.MustRewrite("expand", "fixture", "(expanded fixture)")}
	if kind == "beam" {
		r, err = NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: measurementBudget(memory)}).Search(context.Background(), initial, rules, eval)
	} else {
		r, err = NewMCTSSearch(MCTSConfig{Iterations: 1, RolloutDepth: 1, Budget: measurementBudget(memory)}).Search(context.Background(), initial, rules, eval)
	}
	if r != nil {
		t.Cleanup(r.Close)
	}
	if err != nil {
		t.Fatalf("%s search: %v result=%+v", kind, err, r)
	}
	if len(r.AllCandidates) != 2 {
		t.Fatalf("%s candidates=%d evaluations=%d rejected=%+v stop=%v", kind, len(r.AllCandidates), r.Evaluations, r.Rejected, r.StopReason)
	}
	return r
}
func checkMeasurementRuns(t *testing.T, f *measurementFixture, want int) {
	t.Helper()
	runs, outputs := f.base.runs, f.base.outputs
	for _, x := range f.compiled {
		runs += x.runs
		outputs += x.outputs
		if x.closed != 1 {
			t.Fatalf("current compiler owner closes=%d", x.closed)
		}
	}
	if runs != want*4 || outputs != want*2 || f.base.closed != 0 {
		t.Fatalf("runs=%d outputs=%d baseline closes=%d, want %d measurements", runs, outputs, f.base.closed, want)
	}
}
func TestMeasurementReuseSearch(t *testing.T) {
	for _, kind := range []string{"beam", "mcts"} {
		for _, memory := range []int64{0, 1 << 20} {
			t.Run(fmt.Sprintf("%s/memory%d", kind, memory), func(t *testing.T) {
				f := newMeasurementFixture(t)
				r := runMeasurementSearch(t, kind, f.e, memory)
				checkMeasurementRuns(t, f, 1)
				if r.Evaluations != 2 || r.Compilations != 2 || r.Extractions != 2 || !r.WorkAccounted || len(f.scoreAtCompile) != 2 || f.scoreAtCompile[1] <= f.scoreAtCompile[0] {
					t.Fatalf("work erased: result=%+v scores=%v", r, f.scoreAtCompile)
				}
				a, b := r.AllCandidates[0], r.AllCandidates[1]
				if len(a.Derivation) != 0 || len(b.Derivation) != 1 || a.Expr.Op != "fixture" || b.Expr.Op != "fixture" {
					t.Fatalf("distinct states/derivations lost: %v %v", a.Derivation, b.Derivation)
				}
				for i, c := range []*CandidatePlan{a, b} {
					if !c.Validated || c.Cost != 5000 || c.CostSource != CostMeasured || len(c.Trials) != 4 || c.Comparison["out"].Elements != 1 {
						t.Fatalf("candidate%d=%+v", i, c)
					}
					want := []string{"false", "true"}[i]
					if c.Receipt.Provenance["measurement_reused"] != want || c.Receipt.Provenance["measurement_target_identity"] != f.e.MeasurementContext.TargetIdentity {
						t.Fatalf("candidate%d provenance=%v", i, c.Receipt.Provenance)
					}
				}
				if len(a.Preparation) != 1 || len(b.Preparation) != 1 || a.Preparation[0].WallNs != 1 || b.Preparation[0].WallNs != 2 {
					t.Fatal("reused facts replaced current compilation preparation")
				}
				if a.Receipt.Provenance["measurement_key"] != b.Receipt.Provenance["measurement_key"] || len(a.Receipt.Provenance["measurement_key"]) != 64 {
					t.Fatal("reuse not linked to complete tuple")
				}
				for i := range a.Trials {
					if !a.Trials[i].Started.Equal(b.Trials[i].Started) || !a.Trials[i].Finished.Equal(b.Trials[i].Finished) {
						t.Fatal("reuse manufactured new timestamps")
					}
				}
				// Memo, attempts and candidates have independent metadata owners.
				f.e.Receipts[0].Provenance["raw-source"] = "mutated-last-attempt"
				for _, record := range f.e.measurements.entries {
					if record.trials[0].Provenance["raw-source"] != "host-fixture" {
						t.Fatal("memo aliases attempt")
					}
				}
				a.Trials[0].Provenance["raw-source"] = "mutated-result"
				if b.Trials[0].Provenance["raw-source"] != "host-fixture" {
					t.Fatal("candidate aliases another result")
				}
				a.Close()
				memo := f.e.measurements
				ledger := memo.ledger
				memoBytes := memo.memory.Bytes()
				for _, record := range memo.entries {
					memoBytes += record.memory.Bytes()
				}
				beforeClose := ledger.live
				if ledger.live == 0 {
					t.Fatal("retained memo/result storage uncharged")
				}
				if err := f.e.Close(); err != nil {
					t.Fatal(err)
				}
				if b.Trials[0].Provenance["raw-source"] != "host-fixture" || b.Receipt.Provenance["measurement_reused"] != "true" {
					t.Fatal("evaluator close invalidated result evidence")
				}
				if memory == 0 && beforeClose-ledger.live != memoBytes {
					t.Fatalf("unlimited memo release=%d want %d", beforeClose-ledger.live, memoBytes)
				}
				if memo.memory.Bytes() != 0 || len(memo.entries) != 0 {
					t.Fatal("memo owner still live after close")
				}
				r.Close()
				// Existing unlimited Search post-hoc accounting has no result book and
				// discards its reservation token. This feature tests its own exact release;
				// capped Search owners must still release the entire shared ledger.
				if memory != 0 && ledger.live != 0 {
					t.Fatalf("closed owners retain %d bytes", ledger.live)
				}
			})
		}
	}
}

func TestMeasurementReuseIdentity(t *testing.T) {
	changes := []struct {
		name   string
		change func(*measurementFixture)
	}{
		{"target", func(f *measurementFixture) { f.e.MeasurementContext.TargetIdentity += "/device1" }},
		{"reset", func(f *measurementFixture) { f.e.MeasurementContext.ResetIdentity += "/v2" }},
		{"generation", func(f *measurementFixture) { f.e.Generation++ }},
		{"inputs", func(f *measurementFixture) { f.e.Workload.InputIdentities["x"] = "changed-input" }},
		{"weights", func(f *measurementFixture) {
			f.e.Workload.WeightIdentities = map[string]string{"weight": "changed-weight"}
		}},
		{"baseline", func(f *measurementFixture) { f.base.id = "base-v2" }},
		{"reference", func(f *measurementFixture) { f.e.References["out"][0] = 1.001 }},
		{"policy", func(f *measurementFixture) { f.e.Policy.NaNsEqual = true }},
		{"rounds", func(f *measurementFixture) { f.e.Rounds = 4 }},
	}
	for _, row := range changes {
		t.Run(row.name, func(t *testing.T) {
			f := newMeasurementFixture(t)
			if row.name == "reference" {
				f.e.Policy.Absolute = .01
			}
			f.beforeCompile = func(i int) {
				if i == 1 {
					row.change(f)
				}
			}
			r := runMeasurementSearch(t, "beam", f.e, 1<<20)
			wantRuns := 8
			if row.name == "rounds" {
				wantRuns = 12
			}
			total := f.base.runs
			for _, x := range f.compiled {
				total += x.runs
				if x.closed != 1 {
					t.Fatal("current owner not retired")
				}
			}
			if total != wantRuns || f.base.outputs != 2 || f.compiled[1].outputs != 1 {
				t.Fatalf("identity change reused facts: runs=%d outputs=%d/%d", total, f.base.outputs, f.compiled[1].outputs)
			}
			for _, c := range r.AllCandidates {
				if c.Receipt.Provenance["measurement_reused"] != "false" {
					t.Fatal("changed identity reported reuse")
				}
			}
			if r.AllCandidates[0].Receipt.Provenance["measurement_key"] == r.AllCandidates[1].Receipt.Provenance["measurement_key"] {
				t.Fatal("identity omitted from key")
			}
		})
	}
}
func TestMeasurementReuseEpochAndAdmission(t *testing.T) {
	t.Run("new_search_epoch", func(t *testing.T) {
		f := newMeasurementFixture(t)
		a := runMeasurementSearch(t, "beam", f.e, 1<<20)
		old := f.e.measurements.ledger
		b := runMeasurementSearch(t, "beam", f.e, 1<<20)
		if f.e.measurements.ledger == old {
			t.Fatal("memo crossed Search epoch")
		}
		checkMeasurementRuns(t, f, 2)
		if b.AllCandidates[0].Receipt.Provenance["measurement_reused"] != "false" {
			t.Fatal("new search skipped measurement")
		}
		a.Close()
		b.Close()
		if old.live != 0 {
			t.Fatalf("old epoch retains %d bytes", old.live)
		}
	})
	for _, name := range []string{"no_target", "no_reset", "zero_generation", "missing_input_identity"} {
		t.Run(name, func(t *testing.T) {
			f := newMeasurementFixture(t)
			switch name {
			case "no_target":
				f.e.MeasurementContext.TargetIdentity = ""
			case "no_reset":
				f.e.MeasurementContext.ResetIdentity = ""
			case "zero_generation":
				f.e.Generation = 0
			case "missing_input_identity":
				f.e.Workload.InputIdentities = nil
				f.e.Workload.InputNames = nil
				f.e.Workload.Shape = mustMeasurementShape()
			}
			runMeasurementSearch(t, "beam", f.e, 1<<20)
			checkMeasurementRuns(t, f, 2)
			if f.e.measurements != nil {
				t.Fatal("incomplete identity populated cache")
			}
		})
	}
	for _, kind := range []string{"beam", "mcts"} {
		// Compile is a public callback and may retain the Search context.
		// Once Search returns, that context must not revive the epoch.
		t.Run("retired_search_context/"+kind, func(t *testing.T) {
			f := newMeasurementFixture(t)
			compile := f.e.Compile
			var retained context.Context
			f.e.Compile = func(ctx context.Context, expr *egraph.Expr, w WorkloadBindings) (Executable, error) {
				retained = ctx
				return compile(ctx, expr, w)
			}
			result := runMeasurementSearch(t, kind, f.e, 1<<20)
			if result.AllCandidates[1].Receipt.Provenance["measurement_reused"] != "true" {
				t.Fatal("live Search epoch did not reuse")
			}
			result.Close()
			for i := 0; i < 2; i++ {
				c, err := measurementEvaluate(t, f, retained)
				if err != nil {
					t.Fatal(err)
				}
				last := f.compiled[len(f.compiled)-1]
				reused := c.Receipt.Provenance["measurement_reused"]
				c.Close()
				if reused != "" || last.runs != 2 || last.outputs != 1 || last.closed != 1 {
					t.Fatalf("evaluate %d after Search: reused=%q runs=%d outputs=%d closes=%d", i, reused, last.runs, last.outputs, last.closed)
				}
				if f.e.measurements != nil {
					t.Fatal("retired Search context retained or seeded a memo")
				}
			}
			checkMeasurementRuns(t, f, 3)
		})
	}
	// Compile may hand the live Search context to another evaluator. That
	// evaluator does not own the epoch and must not memoize on its ledger.
	t.Run("foreign_evaluator", func(t *testing.T) {
		x, y := newMeasurementFixture(t), newMeasurementFixture(t)
		compile := x.e.Compile
		var reused []string
		x.e.Compile = func(ctx context.Context, expr *egraph.Expr, w WorkloadBindings) (Executable, error) {
			for len(reused) < 2 {
				c, err := measurementEvaluate(t, y, ctx)
				if err != nil {
					t.Fatal(err)
				}
				reused = append(reused, c.Receipt.Provenance["measurement_reused"])
				c.Close()
				if y.e.measurements != nil {
					t.Fatal("foreign evaluator joined the Search epoch")
				}
			}
			return compile(ctx, expr, w)
		}
		r := runMeasurementSearch(t, "beam", x.e, 1<<20)
		if r.AllCandidates[1].Receipt.Provenance["measurement_reused"] != "true" {
			t.Fatal("owning evaluator did not reuse")
		}
		if reused[0] != "" || reused[1] != "" {
			t.Fatalf("foreign evaluator reuse=%q", reused)
		}
		checkMeasurementRuns(t, y, 2)
		r.Close()
		if err := x.e.Close(); err != nil {
			t.Fatal(err)
		}
		// Only y's last-attempt evidence remains, as without reuse.
		if x.ledger.live != y.e.attemptMemory.Bytes() {
			t.Fatalf("foreign evaluator holds %d bytes on the Search ledger, last attempt %d", x.ledger.live, y.e.attemptMemory.Bytes())
		}
		if err := y.e.Close(); err != nil {
			t.Fatal(err)
		}
		if x.ledger.live != 0 {
			t.Fatalf("closed owners retain %d bytes", x.ledger.live)
		}
	})
	t.Run("retired_epoch_memo", func(t *testing.T) {
		f := newMeasurementFixture(t)
		ctx, w, err := searchWorkBudget(context.Background(), measurementBudget(1<<20), f.e)
		if err != nil {
			t.Fatal(err)
		}
		prime, err := measurementEvaluate(t, f, ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer prime.Close()
		memo := f.e.measurements
		if memo == nil || len(memo.entries) != 1 {
			t.Fatal("live epoch did not seed memo")
		}
		w.closeSearch()
		// A memo obtained while live must not be extended after retirement.
		if err := memo.remember([32]byte{1}, prime.Trials, prime.Comparison); err != nil || len(memo.entries) != 1 {
			t.Fatalf("retired memo seeded: err=%v entries=%d", err, len(memo.entries))
		}
		c, err := measurementEvaluate(t, f, ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		if c.Receipt.Provenance["measurement_reused"] != "" || f.compiled[1].runs != 2 || f.compiled[1].outputs != 1 {
			t.Fatalf("retired epoch reused facts: %v", c.Receipt.Provenance)
		}
		if f.e.measurements != nil || memo.memory.Bytes() != 0 || len(memo.entries) != 0 {
			t.Fatal("retired memo not released")
		}
	})
	t.Run("direct_and_confirmation", func(t *testing.T) {
		f := newMeasurementFixture(t)
		result := runMeasurementSearch(t, "beam", f.e, 1<<20)
		screened := result.AllCandidates[0]
		s := executableState()
		for i := 0; i < 2; i++ {
			c, err := f.e.Evaluate(context.Background(), s, s.Roots["root"])
			if err != nil {
				t.Fatal(err)
			}
			c.Close()
		}
		if f.e.measurements != nil {
			t.Fatal("direct measurement retained an old search epoch")
		}
		ctx, _ := measurementTestEpoch(t, f)
		f.e.Workload.InputIdentities["x"] = "distinct-held-out-inputs"
		prime, err := measurementEvaluate(t, f, ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer prime.Close()
		for i := 0; i < 2; i++ {
			c, err := f.e.Confirm(ctx, screened)
			if err != nil {
				t.Fatal(err)
			}
			if c.ConfirmationOf != screened.ID || c.Receipt.Provenance["measurement_reused"] != "" {
				t.Fatal("held-out confirmation reused screening facts")
			}
			c.Close()
		}
		checkMeasurementRuns(t, f, 6)
		if f.e.measurements != nil {
			t.Fatal("confirmation retained epoch memo")
		}
	})
}

func mustMeasurementShape() cudashape.Shape {
	s, err := cudashape.New(1)
	if err != nil {
		panic(err)
	}
	return s
}

// The main feature tests above use public Search calls. These focused ownership
// controls borrow the same private per-Search capability without a policy loop.
func measurementTestEpoch(t *testing.T, f *measurementFixture) (context.Context, *memoryBudget) {
	t.Helper()
	ctx, w, err := searchWorkBudget(context.Background(), measurementBudget(1<<20), f.e)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(w.closeSearch)
	return ctx, w.memory
}
func measurementEvaluate(t *testing.T, f *measurementFixture, ctx context.Context) (*CandidatePlan, error) {
	t.Helper()
	s := executableState()
	return f.e.Evaluate(ctx, s, s.Roots["root"])
}
func TestMeasurementReuseFailures(t *testing.T) {
	for _, name := range []string{"run_error", "receipt_error", "incomplete", "wrong_output", "cleanup"} {
		t.Run(name, func(t *testing.T) {
			f := newMeasurementFixture(t)
			failure := errors.New("failed empirical attempt")
			f.configure = func(x *measurementExecutable, _ int) {
				switch name {
				case "run_error":
					x.runError = failure
				case "receipt_error":
					x.runReceiptError = failure
				case "incomplete":
					x.runIncomplete = true
				case "wrong_output":
					x.value = 2
				case "cleanup":
					x.closeErr = failure
				}
			}
			ctx, _ := measurementTestEpoch(t, f)
			c, err := measurementEvaluate(t, f, ctx)
			if c != nil || err == nil {
				t.Fatalf("failed attempt admitted: c=%v err=%v", c, err)
			}
			if name == "run_error" || name == "receipt_error" || name == "cleanup" {
				if !errors.Is(err, failure) {
					t.Fatalf("lost primary identity: %v", err)
				}
			}
			if f.e.measurements != nil && len(f.e.measurements.entries) != 0 {
				t.Fatal("failed attempt seeded reusable facts")
			}
			if name == "cleanup" {
				before := len(f.compiled)
				last := f.e.Receipts
				if _, err := measurementEvaluate(t, f, ctx); err == nil {
					t.Fatal("pending owner bypassed")
				}
				if len(f.compiled) != before || len(f.e.Receipts) != len(last) {
					t.Fatal("pending refusal replaced last evidence")
				}
				f.compiled[0].closeErr = nil
				if err := f.e.Close(); err != nil {
					t.Fatal(err)
				}
				if f.compiled[0].closed != 2 {
					t.Fatal("failed closer not retried once")
				}
			}
			f.configure = nil
			c, err = measurementEvaluate(t, f, ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			if c.Receipt.Provenance["measurement_reused"] != "false" || f.compiled[len(f.compiled)-1].outputs != 1 {
				t.Fatal("failed attempt reused instead of remeasured")
			}
		})
	}
	t.Run("current_hit_cleanup_failure", func(t *testing.T) {
		f := newMeasurementFixture(t)
		ctx, _ := measurementTestEpoch(t, f)
		c, err := measurementEvaluate(t, f, ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		failure := errors.New("new current owner did not retire")
		f.configure = func(x *measurementExecutable, _ int) { x.closeErr = failure }
		c, err = measurementEvaluate(t, f, ctx)
		if c != nil || !errors.Is(err, failure) || f.compiled[1].runs != 0 || f.compiled[1].outputs != 0 {
			t.Fatalf("cached facts bypassed current owner: %v %v", c, err)
		}
		if len(f.e.pending) != 1 {
			t.Fatal("current compiler owner lost")
		}
		f.compiled[1].closeErr = nil
		if err := f.e.Close(); err != nil {
			t.Fatal(err)
		}
		if f.compiled[1].closed != 2 {
			t.Fatal("current owner retry lost")
		}
	})
	t.Run("lease_lost_during_compile", func(t *testing.T) {
		f := newMeasurementFixture(t)
		ctx, _ := measurementTestEpoch(t, f)
		c, err := measurementEvaluate(t, f, ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		f.beforeCompile = func(_ int) { f.e.HardwareLockChecker = func() bool { return false } }
		c, err = measurementEvaluate(t, f, ctx)
		if c != nil || !errors.Is(err, cudatrial.ErrHardwareLockRequired) || f.compiled[1].closed != 1 {
			t.Fatalf("reused after lease loss: %v %v", c, err)
		}
	})
	t.Run("cancellation_before_hit", func(t *testing.T) {
		f := newMeasurementFixture(t)
		parent, _ := measurementTestEpoch(t, f)
		ctx, cancel := context.WithCancel(parent)
		defer cancel()
		c, err := measurementEvaluate(t, f, ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		f.beforeCompile = func(_ int) { cancel() }
		c, err = measurementEvaluate(t, f, ctx)
		if c != nil || !errors.Is(err, context.Canceled) || f.compiled[1].closed != 1 {
			t.Fatalf("reused after cancellation: %v %v", c, err)
		}
	})
}
func TestMeasurementReusePreflight(t *testing.T) {
	t.Run("identity_before_outputs", func(t *testing.T) {
		f := newMeasurementFixture(t)
		ctx, _ := measurementTestEpoch(t, f)
		f.e.MeasurementContext.TargetIdentity = strings.Repeat("t", 2<<20)
		c, err := measurementEvaluate(t, f, ctx)
		if c != nil || !errors.Is(err, ErrMemoryBudgetExceeded) {
			t.Fatalf("unadmitted key copied: %v %v", c, err)
		}
		if len(f.compiled) != 1 || f.compiled[0].closed != 1 || f.compiled[0].outputs != 0 || f.base.outputs != 0 || f.e.measurements != nil {
			t.Fatal("identity refusal came after empirical work or populated memo")
		}
	})
	t.Run("failed_entry_unwinds", func(t *testing.T) {
		ledger := &memoryBudget{limit: 128}
		scope, err := egraphcore.NewMemoryScope(ledger)
		if err != nil {
			t.Fatal(err)
		}
		memo := &measurementMemo{epoch: &searchEpoch{}, ledger: ledger, memory: scope, entries: make(map[[32]byte]*measurementRecord)}
		before := ledger.live
		err = memo.remember([32]byte{}, []cudatrial.Receipt{{Complete: true, Provenance: map[string]string{"key": strings.Repeat("v", 1024)}}}, map[string]OutputComparison{"out": {Elements: 1}})
		if !errors.Is(err, ErrMemoryBudgetExceeded) || len(memo.entries) != 0 || ledger.live != before {
			t.Fatalf("failed entry owner: err=%v entries=%d live=%d before=%d", err, len(memo.entries), ledger.live, before)
		}
		memo.close()
		if ledger.live != 0 {
			t.Fatalf("memo close retains %d", ledger.live)
		}
	})
}

func TestMeasurementReuseCurrentGates(t *testing.T) {
	t.Run("quarantine_before_hit", func(t *testing.T) {
		f := newMeasurementFixture(t)
		ctx, _ := measurementTestEpoch(t, f)
		q, err := NewCandidateQuarantine(1, 16384)
		if err != nil {
			t.Fatal(err)
		}
		f.e.FailureEvidence = &FailureEvidence{Quarantine: q}
		c, err := measurementEvaluate(t, f, ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		association, owner, err := f.e.candidateAssociation(ctx, f.compiled[0].Identity())
		if err != nil {
			t.Fatal(err)
		}
		q.entries = append(q.entries, *association)
		owner.release()
		c, err = measurementEvaluate(t, f, ctx)
		if c != nil || !errors.Is(err, ErrCandidateQuarantined) || len(f.compiled) != 2 || f.compiled[1].closed != 1 || f.compiled[1].runs != 0 || f.compiled[1].outputs != 0 {
			t.Fatalf("quarantine bypass: %v %v", c, err)
		}
	})
	t.Run("different_compiled_artifact", func(t *testing.T) {
		f := newMeasurementFixture(t)
		f.configure = func(x *measurementExecutable, i int) { x.id = fmt.Sprintf("artifact-%d/launch", i) }
		r := runMeasurementSearch(t, "beam", f.e, 1<<20)
		checkMeasurementRuns(t, f, 2)
		if r.AllCandidates[0].Receipt.Provenance["measurement_key"] == r.AllCandidates[1].Receipt.Provenance["measurement_key"] {
			t.Fatal("compiled identity missing from key")
		}
	})
	t.Run("key_structure", func(t *testing.T) {
		f := newMeasurementFixture(t)
		ctx, _ := measurementTestEpoch(t, f)
		f.e.References = map[string][]float32{"a": {1, 2}, "b": {3}}
		_, a, err := f.e.measurementAttempt(ctx, false, "artifact")
		if err != nil {
			t.Fatal(err)
		}
		f.e.References = map[string][]float32{"b": {3}, "a": {1, 2}}
		_, b, err := f.e.measurementAttempt(ctx, false, "artifact")
		if err != nil {
			t.Fatal(err)
		}
		if a != b {
			t.Fatal("map insertion order changed key")
		}
		f.e.References = map[string][]float32{"a": {1}, "b": {2, 3}}
		_, c, err := f.e.measurementAttempt(ctx, false, "artifact")
		if err != nil {
			t.Fatal(err)
		}
		if a == c {
			t.Fatal("output extents omitted from key")
		}
		f.e.References = map[string][]float32{"c": {1, 2}, "b": {3}}
		_, d, err := f.e.measurementAttempt(ctx, false, "artifact")
		if err != nil {
			t.Fatal(err)
		}
		if a == d {
			t.Fatal("reference names omitted from key")
		}
	})
}

// wantMemoBytes is the declared storage of an empty memo, including the
// Search epoch and ledger objects it keeps reachable after Search returns.
func wantMemoBytes() int64 {
	return int64(reflect.TypeFor[egraphcore.MemoryScope]().Size() +
		reflect.TypeFor[measurementMemo]().Size() +
		reflect.TypeFor[map[[32]byte]*measurementRecord]().Size() +
		reflect.TypeFor[searchEpoch]().Size() +
		reflect.TypeFor[memoryBudget]().Size())
}

func TestMeasurementReuseRetainedStorage(t *testing.T) {
	for _, kind := range []string{"beam", "mcts"} {
		t.Run(kind, func(t *testing.T) {
			f := newMeasurementFixture(t)
			r := runMeasurementSearch(t, kind, f.e, 1<<20)
			r.Close()
			memo := f.e.measurements
			if memo == nil || len(memo.entries) != 1 {
				t.Fatal("Search did not leave a memo")
			}
			slot := int64(reflect.TypeFor[struct {
				key   [32]byte
				value *measurementRecord
			}]().Size())
			if want := wantMemoBytes() + slot; memo.memory.Bytes() != want {
				t.Fatalf("memo with one entry charges %d, want %d", memo.memory.Bytes(), want)
			}
			held := memo.memory.Bytes() + f.e.attemptMemory.Bytes()
			for _, record := range memo.entries {
				held += record.memory.Bytes()
			}
			// After Search and its result close, the ledger holds exactly the
			// retained memo and the evaluator's last attempt.
			if f.ledger.live != held {
				t.Fatalf("ledger holds %d bytes, memo and last attempt %d", f.ledger.live, held)
			}
			if err := f.e.Close(); err != nil {
				t.Fatal(err)
			}
			if f.ledger.live != 0 {
				t.Fatalf("closed memo retains %d bytes", f.ledger.live)
			}
		})
	}
	t.Run("search_setup", func(t *testing.T) {
		f := newMeasurementFixture(t)
		_, w, err := searchWorkBudget(context.Background(), measurementBudget(1<<20), f.e)
		if err != nil {
			t.Fatal(err)
		}
		want := int64(reflect.TypeFor[workBudget]().Size() + reflect.TypeFor[memoryBudget]().Size() + reflect.TypeFor[egraphcore.MemoryScope]().Size() + reflect.TypeFor[searchEpoch]().Size())
		if w.epoch == nil || w.memory.live != want {
			t.Fatalf("Search setup charges %d, want %d with its epoch", w.memory.live, want)
		}
		w.closeSearch()
		if w.memory.live != 0 || !w.epoch.retired.Load() {
			t.Fatalf("closeSearch left %d bytes, retired=%v", w.memory.live, w.epoch.retired.Load())
		}
	})
	for _, row := range []struct {
		name  string
		limit int64
		ok    bool
	}{
		{"exact", wantMemoBytes(), true},
		{"short", wantMemoBytes() - 1, false},
	} {
		t.Run("cap/"+row.name, func(t *testing.T) {
			ledger := &memoryBudget{limit: row.limit}
			memo, err := newMeasurementMemo(&searchEpoch{}, ledger)
			if !row.ok {
				if memo != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || ledger.live != 0 || ledger.peak > row.limit {
					t.Fatalf("memo=%v err=%v live=%d peak=%d", memo, err, ledger.live, ledger.peak)
				}
				return
			}
			if err != nil || ledger.live != row.limit {
				t.Fatalf("err=%v live=%d want %d", err, ledger.live, row.limit)
			}
			memo.close()
			if ledger.live != 0 {
				t.Fatalf("memo close retains %d", ledger.live)
			}
		})
	}
}

func TestMeasurementReuseDetachedSource(t *testing.T) {
	f := newMeasurementFixture(t)
	ctx, _ := measurementTestEpoch(t, f)
	first, err := measurementEvaluate(t, f, ctx)
	if err != nil {
		t.Fatal(err)
	}
	// Last-attempt data is caller-visible; the memo must own independent copies.
	f.e.Receipts[0].Provenance["raw-source"] = "mutated-source"
	f.e.Comparison["out"] = OutputComparison{Elements: 999}
	first.Close()
	hit, err := measurementEvaluate(t, f, ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer hit.Close()
	if hit.Trials[0].Provenance["raw-source"] != "host-fixture" || hit.Comparison["out"].Elements != 1 {
		t.Fatal("memo retained last-attempt aliases")
	}
	if f.compiled[1].runs != 0 || f.compiled[1].outputs != 0 {
		t.Fatal("detachment control did not exercise a hit")
	}
}
func TestMeasurementReuseStateless(t *testing.T) {
	f := newMeasurementFixture(t)
	f.e.Workload.ResetState = nil
	f.e.MeasurementContext.ResetIdentity = ""
	runMeasurementSearch(t, "beam", f.e, 1<<20)
	checkMeasurementRuns(t, f, 1)
}
