package cudaspec

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabatch"
)

type draftCloserRecorder struct {
	name   string
	order  *[]string
	closed int
	err    error
}

func (c *draftCloserRecorder) Error() string {
	return "draft producer error: " + c.name
}

func (c *draftCloserRecorder) Close() error {
	c.closed++
	if c.order != nil {
		*c.order = append(*c.order, c.name)
	}
	return c.err
}

func TestBatchDraftProducerInvocationAndProposalFlow(t *testing.T) {
	// 1. Reference mode (DraftProducer == nil) remains unchanged and produces [2, 3].
	aRef, reqRef := batchFixture(t)
	engRef := &batchTokenEngine{tokens: []int32{2, 3, 1}}
	resRef, err := ExecuteBatchCycle(context.Background(), aRef, reqRef, verifier(engRef))
	if err != nil {
		t.Fatalf("reference ExecuteBatchCycle: %v", err)
	}
	if !resRef.Outcome.Committed || !reflect.DeepEqual(resRef.Outcome.DraftTokens, []int32{2, 3}) || !reflect.DeepEqual(resRef.Outcome.AcceptedTokens, []int32{2, 3}) || resRef.Outcome.BonusToken != 1 {
		t.Fatalf("unexpected reference outcome: %+v", resRef.Outcome)
	}
	if err := resRef.KV.Close(); err != nil {
		t.Fatal(err)
	}

	// 2. Configured DraftProducer is invoked with admitted inputs and its produced
	// logits directly determine proposed DraftTokens ([1, 0] instead of [2, 3]).
	aProd, reqProd := batchFixture(t)
	var called, waited int
	var gotStep BatchDraftStep
	reqProd.DraftProducer = &BatchDraftProducer{
		Engine: DraftLogitsFunc(func(ctx context.Context, step *BatchDraftStep) ([]float32, error) {
			called++
			gotStep = *step
			// Mutate step slices to verify ExecuteBatchCycle isolated request slices.
			step.Hidden[0] = 999
			// Head 0 selects token 1; Head 1 selects token 0.
			return []float32{
				0.1, 4.5, 0.2, -1.0,
				3.8, -0.5, 0.0, 0.2,
			}, nil
		}),
		Wait: func(step *BatchDraftStep) error {
			if step.StepID != 1 {
				t.Fatalf("Wait step ID = %d, want 1", step.StepID)
			}
			waited++
			return nil
		},
		Done: func(step *BatchDraftStep) bool {
			return step.StepID == 1 && waited == 1
		},
	}

	engAcceptProduced := &batchTokenEngine{tokens: []int32{1, 0, 2}}
	resProd, err := ExecuteBatchCycle(context.Background(), aProd, reqProd, verifier(engAcceptProduced))
	if err != nil {
		t.Fatalf("produced ExecuteBatchCycle: %v", err)
	}
	if called != 1 || waited != 1 {
		t.Fatalf("producer called=%d waited=%d, want 1, 1", called, waited)
	}
	if gotStep.DraftDigest != reqProd.DraftDigest || gotStep.Key != resProd.Key {
		t.Fatalf("step identity mismatch: got (%x, %+v), want (%x, %+v)", gotStep.DraftDigest, gotStep.Key, reqProd.DraftDigest, resProd.Key)
	}
	if reqProd.Hidden[0] != 1 {
		t.Fatalf("producer mutation leaked into request Hidden: %v", reqProd.Hidden)
	}
	if !resProd.Outcome.Committed || !reflect.DeepEqual(resProd.Outcome.DraftTokens, []int32{1, 0}) || !reflect.DeepEqual(resProd.Outcome.AcceptedTokens, []int32{1, 0}) || resProd.Outcome.BonusToken != 2 {
		t.Fatalf("produced outcome = %+v, want Committed=true DraftTokens=[1 0] AcceptedTokens=[1 0] BonusToken=2", resProd.Outcome)
	}
	if err := resProd.KV.Close(); err != nil {
		t.Fatal(err)
	}

	// 3. Causal omitted-producer control: with the exact same verifier expecting
	// produced drafts [1, 0, 2], omitting DraftProducer falls back to CPU reference
	// drafts [2, 3], causing immediate rejection at step 1 and branch rollback.
	aOmit, reqOmit := batchFixture(t)
	engOmit := &batchTokenEngine{tokens: []int32{1, 0, 2}}
	resOmit, err := ExecuteBatchCycle(context.Background(), aOmit, reqOmit, verifier(engOmit))
	if err != nil {
		t.Fatalf("omitted-producer control ExecuteBatchCycle: %v", err)
	}
	if resOmit.Outcome.Committed || !reflect.DeepEqual(resOmit.Outcome.DraftTokens, []int32{2, 3}) || len(resOmit.Outcome.AcceptedTokens) != 0 || resOmit.Outcome.BonusToken != 1 {
		t.Fatalf("omitted-producer control outcome = %+v, want Committed=false DraftTokens=[2 3] AcceptedTokens=[] BonusToken=1", resOmit.Outcome)
	}
}

func TestBatchDraftProducerRefusals(t *testing.T) {
	tests := []struct {
		name     string
		mutate   func(*BatchCycleRequest)
		producer *BatchDraftProducer
	}{
		{
			name:     "nil-engine",
			producer: &BatchDraftProducer{Wait: func(*BatchDraftStep) error { return nil }},
		},
		{
			name: "nil-wait",
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					return []float32{0, 1, 0, 0, 0, 0, 1, 0}, nil
				}),
			},
		},
		{
			name: "typed-nil-func",
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(nil),
				Wait:   func(*BatchDraftStep) error { return nil },
			},
		},
		{
			name: "hidden-length-mismatch",
			mutate: func(r *BatchCycleRequest) {
				r.Hidden = []float32{1, 2}
			},
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					t.Fatal("producer must not run on invalid hidden length")
					return nil, nil
				}),
				Wait: func(*BatchDraftStep) error { return nil },
			},
		},
		{
			name: "producer-error",
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					return nil, errors.New("kernel launch failed")
				}),
				Wait: func(*BatchDraftStep) error { return nil },
			},
		},
		{
			name: "nil-logits",
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					return nil, nil
				}),
				Wait: func(*BatchDraftStep) error { return nil },
			},
		},
		{
			name: "short-logits",
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					return []float32{1, 2, 3, 4, 5, 6, 7}, nil
				}),
				Wait: func(*BatchDraftStep) error { return nil },
			},
		},
		{
			name: "overlong-logits",
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					return []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, nil
				}),
				Wait: func(*BatchDraftStep) error { return nil },
			},
		},
		{
			name: "nan-first-element",
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					return []float32{float32(math.NaN()), 1, 2, 3, 0, 1, 2, 3}, nil
				}),
				Wait: func(*BatchDraftStep) error { return nil },
			},
		},
		{
			name: "nan-later-element",
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					return []float32{0, 1, 2, 3, 0, 1, float32(math.NaN()), 3}, nil
				}),
				Wait: func(*BatchDraftStep) error { return nil },
			},
		},
		{
			name: "pos-inf-element",
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					return []float32{0, 1, float32(math.Inf(1)), 3, 0, 1, 2, 3}, nil
				}),
				Wait: func(*BatchDraftStep) error { return nil },
			},
		},
		{
			name: "neg-inf-element",
			producer: &BatchDraftProducer{
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					return []float32{0, 1, 2, 3, float32(math.Inf(-1)), 1, 2, 3}, nil
				}),
				Wait: func(*BatchDraftStep) error { return nil },
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, r := batchFixture(t)
			if tt.mutate != nil {
				tt.mutate(&r)
			}
			r.DraftProducer = tt.producer
			e := &batchTokenEngine{tokens: []int32{2, 3, 1}}
			res, err := ExecuteBatchCycle(context.Background(), a, r, verifier(e))
			if err == nil {
				t.Fatalf("expected refusal for %s, got result=%+v", tt.name, res)
			}
			if res != nil && (res.Outcome.Committed || res.KV != nil) {
				t.Fatalf("refusal committed or leaked branch KV: %+v", res)
			}
			if e.calls != 0 || a.NumFreeBlocks() != 7 {
				t.Fatalf("refusal had verifier/KV side effects: calls=%d freeBlocks=%d", e.calls, a.NumFreeBlocks())
			}
		})
	}
}

func TestBatchDraftProducerQuiescenceAndOwnerLifecycle(t *testing.T) {
	t.Run("quiescent-error-retains-failed-prerequisite", func(t *testing.T) {
		a, r := batchFixture(t)
		var order []string
		stepOwner := &draftCloserRecorder{name: "step-owner", order: &order}
		errCloseSentinel := errors.New("err-closer-cleanup-failed")
		errOwner := &draftCloserRecorder{name: "err-owner", order: &order, err: errCloseSentinel}
		r.DraftProducer = &BatchDraftProducer{
			Owner: stepOwner,
			Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
				return nil, errOwner
			}),
			Wait: func(*BatchDraftStep) error { return nil },
			Done: func(*BatchDraftStep) bool { return true },
		}
		e := &batchTokenEngine{tokens: []int32{2, 3, 1}}
		res, err := ExecuteBatchCycle(context.Background(), a, r, verifier(e))
		if !errors.Is(err, errOwner) || !errors.Is(err, errCloseSentinel) {
			t.Fatalf("expected joined run+close errors, got %v", err)
		}
		if res == nil || res.DraftOwner == nil || res.KV != nil {
			t.Fatalf("unexpected result state after quiescent close: %+v", res)
		}
		if !reflect.DeepEqual(order, []string{"err-owner"}) {
			t.Fatalf("close order = %v, want [err-owner]", order)
		}
		errOwner.err = nil
		if err := res.DraftOwner.Close(); err != nil {
			t.Fatal(err)
		}
		if err := res.DraftOwner.Close(); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(order, []string{"err-owner", "err-owner", "step-owner"}) {
			t.Fatalf("retry close order = %v", order)
		}
		if e.calls != 0 || a.NumFreeBlocks() != 7 {
			t.Fatalf("unexpected verifier/KV activity: calls=%d free=%d", e.calls, a.NumFreeBlocks())
		}
	})

	t.Run("inflight-drain-error-retains-owners-until-caller-closes", func(t *testing.T) {
		a, r := batchFixture(t)
		var order []string
		stepOwner := &draftCloserRecorder{name: "step-owner", order: &order}
		errOwner := &draftCloserRecorder{name: "err-owner", order: &order}
		r.DraftProducer = &BatchDraftProducer{
			Owner: stepOwner,
			Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
				return nil, errOwner
			}),
			Wait: func(*BatchDraftStep) error { return errors.New("stream still active") },
			Done: func(*BatchDraftStep) bool { return false },
		}
		e := &batchTokenEngine{tokens: []int32{2, 3, 1}}
		res, err := ExecuteBatchCycle(context.Background(), a, r, verifier(e))
		if err == nil || res == nil || res.DraftOwner == nil {
			t.Fatalf("expected retained DraftOwner on inflight drain failure, got res=%+v err=%v", res, err)
		}
		if len(order) != 0 {
			t.Fatalf("owners closed before quiescence: %v", order)
		}
		if err := res.DraftOwner.Close(); err != nil {
			t.Fatalf("DraftOwner.Close: %v", err)
		}
		if !reflect.DeepEqual(order, []string{"err-owner", "step-owner"}) {
			t.Fatalf("caller recovery close order = %v, want [err-owner step-owner]", order)
		}
	})

	t.Run("completed-failed-drain-with-done-closes-owners", func(t *testing.T) {
		a, r := batchFixture(t)
		var order []string
		stepOwner := &draftCloserRecorder{name: "step-owner", order: &order}
		r.DraftProducer = &BatchDraftProducer{
			Owner: stepOwner,
			Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
				return []float32{0, 0, 2, 0, 0, 0, 0, 3}, nil
			}),
			Wait: func(*BatchDraftStep) error { return errors.New("terminal stream error after completion") },
			Done: func(*BatchDraftStep) bool { return true },
		}
		e := &batchTokenEngine{tokens: []int32{2, 3, 1}}
		res, err := ExecuteBatchCycle(context.Background(), a, r, verifier(e))
		if err == nil || res == nil || res.DraftOwner != nil {
			t.Fatalf("expected completed-failed drain to close owner, got res=%+v err=%v", res, err)
		}
		if !reflect.DeepEqual(order, []string{"step-owner"}) {
			t.Fatalf("close order = %v, want [step-owner]", order)
		}
	})

	t.Run("false-success-done-false-retains-owner", func(t *testing.T) {
		a, r := batchFixture(t)
		var order []string
		stepOwner := &draftCloserRecorder{name: "step-owner", order: &order}
		r.DraftProducer = &BatchDraftProducer{
			Owner: stepOwner,
			Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
				return []float32{0, 0, 2, 0, 0, 0, 0, 3}, nil
			}),
			Wait: func(*BatchDraftStep) error { return nil },
			Done: func(*BatchDraftStep) bool { return false },
		}
		e := &batchTokenEngine{tokens: []int32{2, 3, 1}}
		res, err := ExecuteBatchCycle(context.Background(), a, r, verifier(e))
		if err == nil || res == nil || res.DraftOwner == nil {
			t.Fatalf("expected retained DraftOwner when Done=false, got res=%+v err=%v", res, err)
		}
		if len(order) != 0 {
			t.Fatalf("owner closed while Done=false: %v", order)
		}
		if err := res.DraftOwner.Close(); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(order, []string{"step-owner"}) {
			t.Fatalf("close order = %v, want [step-owner]", order)
		}
	})

	t.Run("cancellation-during-produce-still-drains-and-closes-owner", func(t *testing.T) {
		a, r := batchFixture(t)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var order []string
		stepOwner := &draftCloserRecorder{name: "step-owner", order: &order}
		waited := false
		r.DraftProducer = &BatchDraftProducer{
			Owner: stepOwner,
			Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
				cancel()
				return []float32{0, 0, 2, 0, 0, 0, 0, 3}, nil
			}),
			Wait: func(*BatchDraftStep) error {
				waited = true
				return nil
			},
		}
		e := &batchTokenEngine{tokens: []int32{2, 3, 1}}
		res, err := ExecuteBatchCycle(ctx, a, r, verifier(e))
		if !errors.Is(err, context.Canceled) || !waited || res == nil || res.DraftOwner != nil {
			t.Fatalf("cancel during produce: res=%+v err=%v waited=%v", res, err, waited)
		}
		if !reflect.DeepEqual(order, []string{"step-owner"}) || e.calls != 0 || a.NumFreeBlocks() != 7 {
			t.Fatalf("cleanup/side-effect check failed: order=%v calls=%d free=%d", order, e.calls, a.NumFreeBlocks())
		}
	})

	t.Run("verifier-cancellation-after-produced-drafts", func(t *testing.T) {
		a, r := batchFixture(t)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var order []string
		stepOwner := &draftCloserRecorder{name: "step-owner", order: &order}
		r.DraftProducer = &BatchDraftProducer{
			Owner: stepOwner,
			Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
				return []float32{0, 3, 0, 0, 4, 0, 0, 0}, nil
			}),
			Wait: func(*BatchDraftStep) error { return nil },
		}
		e := &batchTokenEngine{
			tokens:  []int32{1, 0, 2},
			inspect: func(*cudabatch.ExecutionStep) { cancel() },
		}
		v := verifier(e)
		verifyWaited := false
		v.Wait = func(*cudabatch.ExecutionStep) error {
			verifyWaited = true
			return nil
		}
		res, err := ExecuteBatchCycle(ctx, a, r, v)
		if !errors.Is(err, context.Canceled) || !verifyWaited || res == nil || a.NumFreeBlocks() != 7 {
			t.Fatalf("verifier cancel after produced drafts: res=%+v err=%v verifyWaited=%v free=%d", res, err, verifyWaited, a.NumFreeBlocks())
		}
		if !reflect.DeepEqual(order, []string{"step-owner"}) {
			t.Fatalf("step-owner not closed after quiescent draft stage: %v", order)
		}
	})
}
