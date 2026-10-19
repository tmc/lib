package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestMemoryReservation(t *testing.T) {
	b := &memoryBudget{limit: 100}
	a, err := b.reserve(60)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.reserve(41); !errors.Is(err, ErrMemoryBudgetExceeded) {
		t.Fatal(err)
	}
	if b.live != 60 || b.accountedBytes() != 60 {
		t.Fatal(b.live, b.accountedBytes())
	}
	a.release()
	a.release()
	if b.live != 0 {
		t.Fatal(b.live)
	}
	c, err := b.reserve(100)
	if err != nil {
		t.Fatal(err)
	}
	c.release()
	if b.accountedBytes() != 100 || b.live != 0 {
		t.Fatal(b.accountedBytes(), b.live)
	}
	overflow := &memoryBudget{live: math.MaxInt64}
	if _, err := overflow.reserve(1); !errors.Is(err, ErrMemoryBudgetExceeded) {
		t.Fatal(err)
	}
	if _, err := b.reserve(-1); err == nil {
		t.Fatal("accepted negative reservation")
	}
}

func TestMemoryPayloadStorage(t *testing.T) {
	type payload struct {
		next *payload
		data []byte
	}
	p := &payload{data: make([]byte, 0, 1<<20)}
	p.next = p
	bytes, err := storageBytes(p)
	if err != nil || bytes < 1<<20 {
		t.Fatalf("bytes=%d err=%v", bytes, err)
	}
	for _, value := range []any{make(chan int), func() {}} {
		if _, err := storageBytes(value); !errors.Is(err, ErrMemoryAccountingUnavailable) {
			t.Fatal(err)
		}
	}
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			g := egraph.New()
			root := g.Add(egraph.ENode{Op: "payload", Data: make([]byte, 64<<10)})
			state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
			budget := SearchBudget{MaxMemoryBytes: 8 << 10, CallbackMemoryBytes: 4096}
			calls := 0
			eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
				calls++
				return &CandidatePlan{Cost: 1, Validated: true}, nil
			})
			_, err := memoryStrategy(strategy, budget, state, nil, eval)
			if !errors.Is(err, ErrMemoryBudgetExceeded) || calls != 0 {
				t.Fatalf("calls=%d err=%v", calls, err)
			}
		})
	}
}

func memoryStrategy(strategy string, budget SearchBudget, state *egraph.SearchState, rules []*egraph.Rewrite, eval Evaluator) (*SearchResult, error) {
	if strategy == "beam" {
		return NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), state, rules, eval)
	}
	return NewMCTSSearch(MCTSConfig{Budget: budget}).Search(context.Background(), state, rules, eval)
}

func TestMemoryOpaqueCallbackContract(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		for _, kind := range []string{"undeclared", "declared", "unlimited", "exhausted"} {
			t.Run(strategy+"/"+kind, func(t *testing.T) {
				state, _ := beamContractState()
				budget := SearchBudget{MaxMemoryBytes: 64 << 10, MaxStates: 1}
				if kind == "declared" {
					budget.CallbackMemoryBytes = 4096
				}
				if kind == "unlimited" {
					budget.MaxMemoryBytes = 0
				}
				if kind == "exhausted" {
					budget.CallbackMemoryBytes = 128 << 10
				}
				calls := 0
				eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
					calls++
					return &CandidatePlan{Cost: 1, Validated: true}, nil
				})
				result, err := memoryStrategy(strategy, budget, state, nil, eval)
				switch kind {
				case "undeclared":
					if calls != 0 || !errors.Is(err, ErrMemoryAccountingUnavailable) {
						t.Fatalf("calls=%d err=%v", calls, err)
					}
				case "exhausted":
					if calls != 0 || !errors.Is(err, ErrMemoryBudgetExceeded) {
						t.Fatalf("calls=%d err=%v", calls, err)
					}
				default:
					if err != nil || calls != 1 || result.Winner == nil || result.AccountedBytes <= 0 {
						t.Fatalf("calls=%d result=%+v err=%v", calls, result, err)
					}
					result.Close()
				}
			})
		}
	}
}

func TestMemoryCloneAndNoOpReservation(t *testing.T) {
	state, _ := beamContractState()
	b := &memoryBudget{limit: 1}
	ctx := context.WithValue(context.Background(), memoryBudgetKey{}, b)
	if cloned, err := cloneSearchState(ctx, state); cloned != nil || !errors.Is(err, ErrMemoryBudgetExceeded) {
		t.Fatalf("clone=%v err=%v", cloned, err)
	}
	actions := egraph.EnumerateActionsBounded(state.Graph, []*egraph.Rewrite{egraph.MustRewrite("same", "base", "base")}, 1)
	if len(actions) != 1 {
		t.Fatal(actions)
	}
	b.limit = 64 << 10
	var peak int64
	for i := 0; i < 3; i++ {
		result, owner, err := applySearchAction(ctx, state, actions[0])
		if err != nil || !result.NoOp || b.live <= 0 || owner == nil {
			t.Fatalf("result=%+v live=%d peak=%d err=%v", result, b.live, b.accountedBytes(), err)
		}
		owner.close()
		owner.close()
		if i == 0 {
			peak = b.accountedBytes()
		}
		if b.live != 0 || b.accountedBytes() != peak {
			t.Fatalf("live=%d peak=%d want=%d", b.live, b.accountedBytes(), peak)
		}
	}
}

func TestMemorySequenceBeforeDeviceAllocation(t *testing.T) {
	layout, err := cudalayout.NewTile(4, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	b := &memoryBudget{limit: 63}
	// The injected allocator is a host-only trap, so even an omitted quota
	// check cannot call CUDA in this test.
	x := &sequenceExecutable{memory: b}
	calls := 0
	allocate := func(int) (cudamem.Slice[byte], error) {
		calls++
		return cudamem.Slice[byte]{}, errors.New("allocator reached")
	}
	if err := x.allocateBufferWith("output", layout, allocate); !errors.Is(err, ErrMemoryBudgetExceeded) {
		t.Fatal(err)
	}
	if calls != 0 || b.live != 0 || len(x.buffers) != 0 || len(x.memoryReservations) != 0 {
		t.Fatal("allocation refusal changed owner")
	}
}

type memoryExecutable struct {
	Executable
	data    []byte
	release func()
	failure error
}

func (x *memoryExecutable) Close() error {
	if x.failure != nil {
		return x.failure
	}
	if err := x.Executable.Close(); err != nil {
		return err
	}
	x.data = nil
	if x.release != nil {
		x.release()
		x.release = nil
	}
	return nil
}

func TestMemoryExecutableReservationCleanup(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(fmt.Sprint(fail), func(t *testing.T) {
			eval, _, prototype, _ := executableFixture(t)
			state := executableState()
			ctx, work, err := searchWorkBudget(context.Background(), SearchBudget{MaxMemoryBytes: 128 << 10, CallbackMemoryBytes: 32 << 10}, eval)
			if err != nil {
				t.Fatal(err)
			}
			defer work.closeSearch()
			setupBytes := work.memory.live
			var owner *memoryExecutable
			eval.Compile = func(ctx context.Context, _ *egraph.Expr, _ WorkloadBindings) (Executable, error) {
				release, err := ReserveSearchMemory(ctx, 1024)
				if err != nil {
					return nil, err
				}
				owner = &memoryExecutable{Executable: prototype, data: make([]byte, 1024), release: release}
				if fail {
					owner.failure = errors.New("free failed")
				}
				return owner, nil
			}
			candidate, err := evaluateSearch(ctx, eval, state, state.Roots["root"])
			if fail {
				if err == nil || candidate != nil || work.memory.live < 1024+(32<<10) {
					t.Fatalf("live=%d err=%v", work.memory.live, err)
				}
				owner.failure = nil
				if err := eval.Close(); err != nil {
					t.Fatal(err)
				}
				if work.memory.live != setupBytes {
					t.Fatal("pending reservation leaked", work.memory.live-setupBytes)
				}
				work.closeSearch()
				if work.memory.live != 0 {
					t.Fatal("search setup reservation leaked", work.memory.live)
				}
			} else if err != nil || candidate == nil || owner.data != nil || work.memory.live == 0 || work.memory.live >= work.memory.accountedBytes() {
				t.Fatalf("live=%d peak=%d candidate=%v err=%v", work.memory.live, work.memory.accountedBytes(), candidate, err)
			}
		})
	}
}

// These host callbacks allocate a candidate and, at most, an append-grown
// derivation slice. Rule strings, maps, receipt fixtures and mutated capture
// storage belong to their caller. This is a conservative positive declaration,
// not an assertion that a callback allocates this many bytes on every call.
func fixtureCallbackMemoryBytes(depth int) int64 {
	if depth == 0 {
		depth = DefaultBudget().MaxDepth
	}
	return int64(reflect.TypeOf(CandidatePlan{}).Size()) + 2*int64(depth)*int64(reflect.TypeOf(egraph.Action{}).Size())
}

func TestMemoryProductOverflow(t *testing.T) {
	if _, err := memoryProduct(math.MaxInt, 5); !errors.Is(err, ErrMemoryBudgetExceeded) {
		t.Fatal(err)
	}
	if n, err := memoryProduct(16, 4); err != nil || n != 64 {
		t.Fatal(n, err)
	}
}

func TestMemoryDeviceReservationFreeFailure(t *testing.T) {
	b := &memoryBudget{limit: 64}
	r, err := b.reserve(64)
	if err != nil {
		t.Fatal(err)
	}
	x := &sequenceExecutable{buffers: map[string]cudamem.Slice[byte]{"output": {}}, memoryReservations: map[string]*memoryReservation{"output": r}}
	failed := errors.New("free failed")
	if err := x.freeBuffer("output", func() error { return failed }); !errors.Is(err, failed) {
		t.Fatal(err)
	}
	if b.live != 64 || len(x.buffers) != 1 || x.memoryReservations["output"] != r {
		t.Fatal("failed free relinquished ownership")
	}
	if err := x.freeBuffer("output", func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if b.live != 0 || len(x.buffers) != 0 || len(x.memoryReservations) != 0 {
		t.Fatal("successful free retained reservation")
	}
}

func TestMemoryBorrowedBaselineBeforeCompile(t *testing.T) {
	for _, kind := range []string{"sequence", "prefill"} {
		for _, quota := range []bool{false, true} {
			t.Run(fmt.Sprint(kind, "/", quota), func(t *testing.T) {
				eval, _, _, _ := executableFixture(t)
				var baseline Executable
				if kind == "sequence" {
					baseline = &sequenceExecutable{identity: "borrowed", buffers: map[string]cudamem.Slice[byte]{"output": {}}, memoryReservations: map[string]*memoryReservation{"output": {bytes: 64}}}
				} else {
					baseline = &prefillExecutable{identity: "borrowed", deviceBytes: 64}
				}
				eval.Baseline = baseline
				budget := SearchBudget{MaxMemoryBytes: 8192, CallbackMemoryBytes: 4096}
				if quota {
					budget.MaxMemoryBytes = budget.CallbackMemoryBytes + 63 + int64(reflect.TypeFor[workBudget]().Size()+reflect.TypeFor[memoryBudget]().Size()+reflect.TypeFor[egraphcore.MemoryScope]().Size())
				}
				ctx, work, err := searchWorkBudget(context.Background(), budget, eval)
				if err != nil {
					t.Fatal(err)
				}
				defer work.closeSearch()
				setupBytes := work.memory.live
				calls := 0
				stopped := errors.New("compile stopped")
				eval.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) {
					calls++
					if work.memory.live < budget.CallbackMemoryBytes+64 {
						t.Fatal("borrowed device storage uncharged")
					}
					return nil, stopped
				}
				state := executableState()
				_, err = evaluateSearch(ctx, eval, state, state.Roots["root"])
				if quota {
					if calls != 0 || !errors.Is(err, ErrMemoryBudgetExceeded) {
						t.Fatalf("calls=%d err=%v", calls, err)
					}
				} else if calls != 1 || !errors.Is(err, stopped) {
					t.Fatalf("calls=%d err=%v", calls, err)
				}
				if work.memory.live != setupBytes {
					t.Fatal("borrowed accounting leaked", work.memory.live-setupBytes)
				}
				work.closeSearch()
				if work.memory.live != 0 {
					t.Fatal("search setup reservation leaked", work.memory.live)
				}
				switch x := baseline.(type) {
				case *sequenceExecutable:
					if x.closed || len(x.buffers) != 1 {
						t.Fatal("borrowed baseline closed")
					}
				case *prefillExecutable:
					if x.closed || x.deviceBytes != 64 {
						t.Fatal("borrowed baseline closed")
					}
				}
			})
		}
	}
}

// Concrete built-in device buffers reserve separately. This host envelope
// covers readback/decoding/comparison copies and the finite ABBA receipt copies;
// reference and fixture strings are borrowed. It is not a measured peak.
func fixtureEvaluatorMemoryBytes(eval Evaluator) int64 {
	var e *ExecutableEvaluator
	switch x := eval.(type) {
	case *ExecutableEvaluator:
		e = x
	case recordingEvaluator:
		e = x.inner
	default:
		return fixtureCallbackMemoryBytes(1)
	}
	var outputs int64
	for _, values := range e.References {
		outputs += int64(len(values)) * 4
	}
	receipt, _ := storageBytes(cudatrial.Receipt{})
	return 6*outputs + 4*int64(2*e.Rounds+2)*(receipt+fixtureCallbackMemoryBytes(1))
}

func TestMemoryInitialRejectedEvidence(t *testing.T) {
	eval, _, prototype, _ := executableFixture(t)
	prototype.value = 2
	state := executableState()
	result, err := NewBeamSearch(BeamConfig{Budget: SearchBudget{MaxMemoryBytes: 128 << 10, CallbackMemoryBytes: 32 << 10}}).Search(context.Background(), state, nil, eval)
	if !errors.Is(err, ErrVerificationFailed) || len(result.Rejected) != 1 || result.Rejected[0].Evidence == nil {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	rejectionBytes, sizeErr := storageBytes(result.Rejected[0])
	if sizeErr != nil || result.AccountedBytes < estimateStateMemory(state)+rejectionBytes {
		t.Fatalf("bytes=%d rejection=%d err=%v", result.AccountedBytes, rejectionBytes, sizeErr)
	}
}
