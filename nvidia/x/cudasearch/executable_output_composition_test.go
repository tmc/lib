package cudasearch

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

var _ ownedExecutableOutputs = (*prefillExecutable)(nil)
var _ ownedExecutableOutputs = (*sequenceExecutable)(nil)

// These controls use actual built-in readback/owner methods with E3's narrow
// per-call host device-operation seam. The thin executable only supplies
// synthetic timing and resource operations; it is not a CUDA execution claim.
type jointOutputExecutable struct {
	fixture                                                         *outputFixture
	id                                                              string
	t                                                               *testing.T
	ledger                                                          *memoryBudget
	evaluator                                                       *ExecutableEvaluator
	outputCalls, legacyCalls, runCalls, resourceCalls, outputCloses int
	changed                                                         bool
	runFailure, resourceFailure, outputCloseFailure                 error
	owners                                                          []*jointOutputOwner
}

func (x *jointOutputExecutable) Identity() string { return x.id }
func (x *jointOutputExecutable) Outputs(context.Context) (map[string][]float32, error) {
	x.legacyCalls++
	return nil, errors.New("legacy Outputs must not be selected")
}
func (x *jointOutputExecutable) outputsOwned(ctx context.Context) (map[string][]float32, io.Closer, error) {
	x.outputCalls++
	values, owner, err := x.fixture.owned(ctx)
	if x.changed {
		for _, data := range values {
			if len(data) > 0 {
				data[0] += 5
				break
			}
		}
	}
	if owner == nil {
		return values, nil, err
	}
	bytes := owner.(*outputMemoryOwner).reservation.bytes
	wrapped := &jointOutputOwner{Closer: owner, executable: x, bytes: bytes}
	x.owners = append(x.owners, wrapped)
	return values, wrapped, err
}
func (x *jointOutputExecutable) Run(context.Context) (cudatrial.Receipt, error) {
	x.runCalls++
	if x.runFailure != nil {
		return cudatrial.Receipt{Digest: "failed-raw"}, x.runFailure
	}
	r := cudatrial.Receipt{Complete: true, TimingKind: cudatrial.TimingSummedKernel}
	r.RecordTiming(cudatrial.TimingSummedKernel, time.Microsecond)
	return r, nil
}
func (x *jointOutputExecutable) Close() error {
	return x.fixture.close(func() error { x.resourceCalls++; return x.resourceFailure })
}

type jointOutputOwner struct {
	io.Closer
	executable *jointOutputExecutable
	bytes      int64
	released   bool
}

func (o *jointOutputOwner) Close() error {
	x := o.executable
	x.outputCloses++
	// On these successful readbacks comparison must have consumed the returned
	// output before the evaluator retires this real E3 reservation.
	if len(x.evaluator.Comparison) == 0 {
		x.t.Error("output retired before comparison")
	}
	if x.outputCloseFailure != nil {
		return x.outputCloseFailure
	}
	before := x.ledger.live
	if err := o.Closer.Close(); err != nil {
		return err
	}
	expected := o.bytes
	if o.released {
		expected = 0
	}
	if before-x.ledger.live != expected {
		x.t.Errorf("output retirement delta=%d want %d", before-x.ledger.live, expected)
	}
	o.released = true
	return nil
}
func jointOutputWant(kind string) map[string][]float32 {
	switch kind {
	case "prefill":
		return map[string][]float32{"out": {1, 2, 3, 4}}
	case "bytes":
		return map[string][]float32{"float": {7, 8, 9}}
	case "multiple":
		return map[string][]float32{"float": {1.25, -2.5}, "bytes": {7, 8, 9}}
	default:
		return map[string][]float32{"float": {1.25, -2.5}}
	}
}
func jointOutputOpen(x *jointOutputExecutable) bool {
	if x.fixture.prefill != nil {
		return !x.fixture.prefill.closed && x.fixture.prefill.outputMemory == nil
	}
	return !x.fixture.sequence.closed && x.fixture.sequence.outputMemory == nil
}

func TestJointExecutableOutputOwnership(t *testing.T) {
	checkOutputLibraries(t)
	for _, kind := range []string{"prefill", "float", "bytes", "multiple"} {
		for _, mode := range []string{"success", "comparison", "run-error", "cancel-read", "output-close", "comparison-close", "resource-close"} {
			t.Run(kind+"/"+mode, func(t *testing.T) {
				e, _, _, _ := executableFixture(t)
				ctx, b := executableMemoryContext(1 << 20)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()
				baseline := &jointOutputExecutable{fixture: newOutputFixture(kind), id: "baseline", t: t, ledger: b, evaluator: e}
				e.Baseline = baseline
				e.References = jointOutputWant(kind)
				failure := errors.New("joint injected failure")
				var candidate *jointOutputExecutable
				e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) {
					candidate = &jointOutputExecutable{fixture: newOutputFixture(kind), id: "candidate", t: t, ledger: b, evaluator: e}
					switch mode {
					case "comparison":
						candidate.changed = true
					case "run-error":
						candidate.runFailure = failure
					case "cancel-read":
						candidate.fixture.cancel = cancel
					case "output-close":
						candidate.outputCloseFailure = failure
					case "comparison-close":
						candidate.changed = true
						candidate.outputCloseFailure = failure
					case "resource-close":
						candidate.resourceFailure = failure
					}
					return candidate, nil
				}
				s := executableState()
				plan, err := e.Evaluate(ctx, s, s.Roots["root"])
				switch mode {
				case "success":
					if err != nil || plan == nil {
						t.Fatal(plan, err)
					}
				case "comparison":
					if !errors.Is(err, ErrVerificationFailed) || plan != nil {
						t.Fatal(plan, err)
					}
				case "cancel-read":
					if !errors.Is(err, context.Canceled) || plan != nil {
						t.Fatal(plan, err)
					}
				default:
					if !errors.Is(err, failure) || plan != nil {
						t.Fatal(plan, err)
					}
				}
				if mode == "comparison-close" && !errors.Is(err, ErrVerificationFailed) {
					t.Fatal("lost primary comparison failure", err)
				}
				if mode == "run-error" && (len(e.Receipts) != 2 || e.Receipts[1].Digest != "failed-raw" || !errors.Is(e.Receipts[1].Err, failure)) {
					t.Fatal("lost raw failed run", e.Receipts)
				}
				if baseline.outputCalls != 1 || candidate.outputCalls != 1 || baseline.legacyCalls != 0 || candidate.legacyCalls != 0 || !jointOutputOpen(baseline) || baseline.resourceCalls != 0 {
					t.Fatal("borrowed baseline or owned route", baseline, candidate)
				}
				for _, x := range []*jointOutputExecutable{baseline, candidate} {
					for _, o := range x.owners {
						if mode != "output-close" && mode != "comparison-close" || x == baseline {
							if !o.released {
								t.Fatal("output still retained before evaluator Close")
							}
						}
					}
				}
				if mode == "output-close" || mode == "comparison-close" {
					if e.pendingOutput == nil {
						t.Fatal("failed output owner lost")
					}
					candidate.outputCloseFailure = nil
				}
				candidate.resourceFailure = nil
				if err := e.Close(); err != nil {
					t.Fatal(err)
				}
				if !jointOutputOpen(baseline) || baseline.resourceCalls != 0 {
					t.Fatal("evaluator closed borrowed builtin baseline")
				}
				if mode == "success" {
					if b.live == 0 || len(plan.Trials) != 4 {
						t.Fatal("attempt Close retired candidate metadata", b.live)
					}
					plan.Close()
				}
				if b.live != 0 {
					t.Fatal("joint owner leaked", b.live)
				}
				if mode == "resource-close" && candidate.resourceCalls != 2 {
					t.Fatal("failed resource close not retried", candidate.resourceCalls)
				}
				for _, x := range []*jointOutputExecutable{baseline, candidate} {
					for _, o := range x.owners {
						if !o.released {
							t.Fatal("output owner not retired")
						}
					}
				}
				if err := e.Close(); err != nil || b.live != 0 {
					t.Fatal("joint close not idempotent", err, b.live)
				}
			})
		}
	}
}

func TestJointBorrowedBuiltinRepeatedReadback(t *testing.T) {
	checkOutputLibraries(t)
	for _, kind := range []string{"prefill", "multiple"} {
		t.Run(kind, func(t *testing.T) {
			e, _, _, _ := executableFixture(t)
			ctx, b := executableMemoryContext(1 << 20)
			base := &jointOutputExecutable{fixture: newOutputFixture(kind), id: "borrowed", t: t, ledger: b, evaluator: e}
			e.Baseline = base
			e.References = jointOutputWant(kind)
			e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) {
				return &jointOutputExecutable{fixture: newOutputFixture(kind), id: "candidate", t: t, ledger: b, evaluator: e}, nil
			}
			s := executableState()
			var last int64
			for i := 0; i < 3; i++ {
				c, err := e.Evaluate(ctx, s, s.Roots["root"])
				if err != nil {
					t.Fatal(err)
				}
				c.Close()
				if b.live != e.attemptMemory.Bytes() {
					t.Fatal("per-call output or result retained", b.live, e.attemptMemory.Bytes())
				}
				if i > 0 && b.live != last {
					t.Fatal("repeated baseline output accumulated", b.live, last)
				}
				last = b.live
			}
			if base.outputCloses != 3 || base.resourceCalls != 0 || !jointOutputOpen(base) {
				t.Fatal("borrowed baseline lifetime", base.outputCloses, base.resourceCalls)
			}
			if err := e.Close(); err != nil || b.live != 0 {
				t.Fatal(err, b.live)
			}
		})
	}
}

func TestJointActualBuiltinEvaluatorRefusal(t *testing.T) {
	checkOutputLibraries(t)
	for _, kind := range []string{"prefill", "sequence"} {
		t.Run(kind, func(t *testing.T) {
			e, _, candidate, _ := executableFixture(t)
			ctx, b := executableMemoryContext(1 << 20)
			var baseline Executable
			if kind == "prefill" {
				baseline = &prefillExecutable{identity: "actual-prefill", checkLease: func() bool { return false }}
			} else {
				baseline = &sequenceExecutable{identity: "actual-sequence", checkLease: func() bool { return false }, outputs: []sequenceOutput{{name: "out", buffer: "out"}}, buffers: map[string]cudamem.Slice[byte]{"out": {}}}
			}
			e.Baseline = baseline
			s := executableState()
			c, err := e.Evaluate(ctx, s, s.Roots["root"])
			if c != nil || !errors.Is(err, cudatrial.ErrHardwareLockRequired) || candidate.closed != 1 {
				t.Fatal(c, err, candidate.closed)
			}
			switch x := baseline.(type) {
			case *prefillExecutable:
				if x.closed || x.outputMemory != nil {
					t.Fatal("actual baseline retired")
				}
			case *sequenceExecutable:
				if x.closed || x.outputMemory != nil {
					t.Fatal("actual baseline retired")
				}
			}
			if err := e.Close(); err != nil || b.live != 0 {
				t.Fatal(err, b.live)
			}
		})
	}
}
