package cudasearch

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func executableMemoryContext(limit int64) (context.Context, *memoryBudget) {
	b := &memoryBudget{limit: limit}
	return context.WithValue(context.Background(), memoryBudgetKey{}, b), b
}

func TestExecutableAttemptResultOwners(t *testing.T) {
	e, _, _, _ := executableFixture(t)
	ctx, b := executableMemoryContext(1 << 20)
	s := executableState()
	c, err := e.Evaluate(ctx, s, s.Roots["root"])
	if err != nil {
		t.Fatal(err)
	}
	if e.attemptMemory == nil || c.metadata == nil || b.live == 0 {
		t.Fatal("missing attempt/result owner", b.live)
	}
	first := c.Trials[1].Timings[cudatrial.TimingSummedKernel]
	alias := *c
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if b.live == 0 || c.Trials[1].Timings[cudatrial.TimingSummedKernel] != first {
		t.Fatal("evaluator retired result", b.live)
	}
	c.Close()
	if b.live != 0 {
		t.Fatal("result leaked", b.live)
	}
	alias.Close()
	if b.live != 0 {
		t.Fatal("alias released twice", b.live)
	}
}

func TestExecutableAttemptPendingPreservesEvidence(t *testing.T) {
	e, _, x, _ := executableFixture(t)
	ctx, b := executableMemoryContext(1 << 20)
	s := executableState()
	failure := errors.New("close failed")
	x.closeErr = failure
	if c, err := e.Evaluate(ctx, s, s.Roots["root"]); c != nil || !errors.Is(err, failure) {
		t.Fatal(c, err)
	}
	receipts := len(e.Receipts)
	identity := e.attemptIdentity
	live := b.live
	if receipts != 4 || identity == "" {
		t.Fatal("failed attempt lost", receipts, identity)
	}
	if c, err := e.Evaluate(ctx, s, s.Roots["root"]); c != nil || err == nil {
		t.Fatal(c, err)
	}
	if len(e.Receipts) != receipts || e.attemptIdentity != identity || b.live != live {
		t.Fatal("pending refusal erased attempt", b.live, live)
	}
	x.closeErr = nil
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if b.live != 0 {
		t.Fatal("pending cleanup leak", b.live)
	}
}

func TestExecutableMetadataPreflight(t *testing.T) {
	e, _, _, order := executableFixture(t)
	ctx, b := executableMemoryContext(1024)
	s := executableState()
	if c, err := e.Evaluate(ctx, s, s.Roots["root"]); c != nil || !errors.Is(err, ErrMemoryBudgetExceeded) {
		t.Fatal(c, err)
	}
	if len(*order) != 0 || b.live != 0 {
		t.Fatal("work before metadata admission", *order, b.live)
	}
}

type ownedOutputFixture struct {
	*testExecutable
	owner   *outputCloserFixture
	failure error
}
type outputCloserFixture struct {
	calls   int
	failure error
	release func()
}

func (x *outputCloserFixture) Close() error {
	x.calls++
	if x.failure != nil {
		return x.failure
	}
	if x.release != nil {
		x.release()
		x.release = nil
	}
	return nil
}
func (x *ownedOutputFixture) outputsOwned(ctx context.Context) (map[string][]float32, io.Closer, error) {
	release, err := ReserveSearchMemory(ctx, 1024)
	if err != nil {
		return nil, nil, err
	}
	x.owner.release = release
	out, err := x.testExecutable.Outputs(ctx)
	return out, x.owner, errors.Join(err, x.failure)
}
func TestExecutableOwnedOutputsRetry(t *testing.T) {
	for _, readFailure := range []bool{false, true} {
		t.Run(map[bool]string{false: "comparison", true: "read-error"}[readFailure], func(t *testing.T) {
			e, base, _, _ := executableFixture(t)
			ctx, b := executableMemoryContext(1 << 20)
			s := executableState()
			failure := errors.New("output close failed")
			primary := errors.New("read failed")
			owner := &outputCloserFixture{failure: failure}
			x := &ownedOutputFixture{testExecutable: base, owner: owner}
			if readFailure {
				x.failure = primary
			}
			e.Baseline = x
			c, err := e.Evaluate(ctx, s, s.Roots["root"])
			if c != nil || !errors.Is(err, failure) || (readFailure && !errors.Is(err, primary)) {
				t.Fatal(c, err)
			}
			if owner.calls != 1 || base.closed != 0 || e.pendingOutput != owner || b.live == 0 {
				t.Fatal("output owner discarded", owner.calls, base.closed, b.live)
			}
			owner.failure = nil
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if b.live != 0 || base.closed != 0 || owner.calls != 2 {
				t.Fatal("output cleanup", b.live, base.closed, owner.calls)
			}
		})
	}
}

type mutableReceiptFixture struct {
	*testExecutable
	receipt cudatrial.Receipt
	failure error
}

func (x *mutableReceiptFixture) Run(context.Context) (cudatrial.Receipt, error) {
	return x.receipt, x.failure
}
func TestExecutableReceiptDeepDetachment(t *testing.T) {
	for _, failed := range []bool{false, true} {
		t.Run(map[bool]string{false: "accepted", true: "failed"}[failed], func(t *testing.T) {
			e, _, compiled, _ := executableFixture(t)
			s := executableState()
			ctx, b := executableMemoryContext(1 << 20)
			m, _ := cudatrial.NewThreadEdgeMap(16, 1)
			m.Record(0, 3)
			primary := errors.New("raw failed run")
			x := &mutableReceiptFixture{testExecutable: compiled, receipt: cudatrial.Receipt{Complete: true, TimingKind: cudatrial.TimingSummedKernel, Timings: map[cudatrial.TimingKind]time.Duration{cudatrial.TimingSummedKernel: time.Microsecond}, Before: cudatrial.Conditions{Values: map[cudatrial.ConditionKind]int64{cudatrial.ConditionHostAvailableBytes: 0}}, After: cudatrial.Conditions{Values: map[cudatrial.ConditionKind]int64{cudatrial.ConditionHostTotalBytes: 20}}, Coverage: m.Snapshot()}}
			if failed {
				x.failure = primary
			}
			e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { return x, nil }
			c, err := e.Evaluate(ctx, s, s.Roots["root"])
			if failed {
				if c != nil || !errors.Is(err, primary) {
					t.Fatal(c, err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			x.receipt.Before.Values[cudatrial.ConditionHostAvailableBytes] = 10
			x.receipt.After.Values[cudatrial.ConditionHostTotalBytes] = 99
			x.receipt.Coverage.Reset()
			got := e.Receipts[1]
			if got.Before.Values[cudatrial.ConditionHostAvailableBytes] != 0 || got.After.Values[cudatrial.ConditionHostTotalBytes] != 20 || got.Coverage.Bytes()[11] != 1 {
				t.Fatal("raw producer aliases attempt", got)
			}
			if failed && !errors.Is(got.Err, primary) {
				t.Fatal("raw failure lost", got.Err)
			}
			if c != nil {
				e.Receipts[1].Coverage.Reset()
				e.Receipts[1].Before.Values[cudatrial.ConditionHostAvailableBytes] = 55
				if c.Trials[1].Coverage.Bytes()[11] != 1 || c.Trials[1].Before.Values[cudatrial.ConditionHostAvailableBytes] != 0 {
					t.Fatal("attempt aliases result")
				}
				c.Close()
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if b.live != 0 {
				t.Fatal("receipt owners leaked", b.live)
			}
		})
	}
}

func TestExecutableRefusedReplacementPreservesAttempt(t *testing.T) {
	e, _, _, _ := executableFixture(t)
	ctx, b := executableMemoryContext(1 << 20)
	s := executableState()
	c, err := e.Evaluate(ctx, s, s.Roots["root"])
	if err != nil {
		t.Fatal(err)
	}
	c.Close()
	identity := e.attemptIdentity
	count := len(e.Receipts)
	live := b.live
	e.Workload.InputIdentities["large"] = strings.Repeat("x", 1<<20)
	if c, err := e.Evaluate(ctx, s, s.Roots["root"]); c != nil || !errors.Is(err, ErrMemoryBudgetExceeded) {
		t.Fatal(c, err)
	}
	if e.attemptIdentity != identity || len(e.Receipts) != count || b.live != live {
		t.Fatal("unadmitted replacement cleared evidence", b.live, live)
	}
	if err := e.Close(); err != nil || b.live != 0 {
		t.Fatal("attempt leak", err, b.live)
	}
}

func TestExecutableRejectedEvidenceOwned(t *testing.T) {
	e, _, compiled, _ := executableFixture(t)
	ctx, b := executableMemoryContext(1 << 20)
	s := executableState()
	primary := errors.New("run failed")
	m, _ := cudatrial.NewThreadEdgeMap(16, 1)
	m.Record(0, 3)
	x := &mutableReceiptFixture{testExecutable: compiled, failure: primary, receipt: cudatrial.Receipt{Before: cudatrial.Conditions{Values: map[cudatrial.ConditionKind]int64{cudatrial.ConditionHostAvailableBytes: 0}}, Coverage: m.Snapshot()}}
	e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { return x, nil }
	_, failure := e.Evaluate(ctx, s, s.Roots["root"])
	if !errors.Is(failure, primary) {
		t.Fatal(failure)
	}
	scope, err := mockMemoryScope(ctx)
	if err != nil {
		t.Fatal(err)
	}
	rejected, err := rejectionWithMemory(scope, s, e, failure)
	if err != nil {
		t.Fatal(err)
	}
	e.Receipts[1].Before.Values[cudatrial.ConditionHostAvailableBytes] = 99
	e.Receipts[1].Coverage.Reset()
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	raw := rejected.Evidence.Trials[1]
	if raw.Before.Values[cudatrial.ConditionHostAvailableBytes] != 0 || raw.Coverage.Bytes()[11] != 1 || !errors.Is(raw.Err, primary) || b.live == 0 {
		t.Fatal("rejection borrowed attempt evidence", raw, b.live)
	}
	scope.Close()
	if b.live != 0 {
		t.Fatal("rejection metadata leaked", b.live)
	}
}

func TestExecutablePendingDoesNotRetainUnusedEnvelope(t *testing.T) {
	e, _, x, _ := executableFixture(t)
	s := executableState()
	failure := errors.New("close failed")
	x.closeErr = failure
	ctx, work, err := searchWorkBudget(t.Context(), SearchBudget{MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: 32 << 10}, e)
	if err != nil {
		t.Fatal(err)
	}
	defer work.closeSearch()
	setup := work.memory.live
	if c, err := evaluateSearch(ctx, e, s, s.Roots["root"]); c != nil || !errors.Is(err, failure) {
		t.Fatal(c, err)
	}
	live := work.memory.live
	receipts := len(e.Receipts)
	if len(e.memoryReservations) != 1 {
		t.Fatal("missing original callback envelope", len(e.memoryReservations))
	}
	for i := 0; i < 3; i++ {
		if c, err := evaluateSearch(ctx, e, s, s.Roots["root"]); c != nil || err == nil {
			t.Fatal(c, err)
		}
	}
	if work.memory.live != live || len(e.memoryReservations) != 1 || len(e.Receipts) != receipts {
		t.Fatal("refusal retained unused envelope", work.memory.live, live, len(e.memoryReservations))
	}
	x.closeErr = nil
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if work.memory.live != setup {
		t.Fatal("retired attempt leaked", work.memory.live, setup)
	}
}

func TestExecutableMetadataPopulationsPreflight(t *testing.T) {
	for _, kind := range []string{"small", "reference-name", "input-name", "derivation"} {
		t.Run(kind, func(t *testing.T) {
			e, _, _, order := executableFixture(t)
			s := executableState()
			ctx, b := executableMemoryContext(64 << 10)
			switch kind {
			case "reference-name":
				e.References = map[string][]float32{strings.Repeat("n", 1<<20): {1}}
			case "input-name":
				e.Workload.InputNames = []string{strings.Repeat("n", 1<<20)}
			case "derivation":
				s.Derivation = make([]egraph.Action, 4096)
			}
			c, err := e.Evaluate(ctx, s, s.Roots["root"])
			if kind == "small" {
				if err != nil || c == nil {
					t.Fatal(c, err)
				}
				c.Close()
			} else if c != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || len(*order) != 0 {
				t.Fatal("unadmitted metadata reached compile", c, err, *order)
			}
			if err := e.Close(); err != nil || b.live != 0 {
				t.Fatal("metadata owner leaked", err, b.live)
			}
		})
	}
}
