package cudaspec

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestBatchDraftOwnerRetryAfterDependentFailure(t *testing.T) {
	a, req := batchFixture(t)
	var order []string
	closeErr := errors.New("step close")
	stepOwner := &draftCloserRecorder{name: "step", order: &order, err: closeErr}
	errOwner := &draftCloserRecorder{name: "error", order: &order}
	req.DraftProducer = &BatchDraftProducer{
		Owner: stepOwner,
		Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
			return nil, errOwner
		}),
		Wait: func(*BatchDraftStep) error { return nil },
	}
	e := &batchTokenEngine{tokens: []int32{2, 3, 1}}
	res, err := ExecuteBatchCycle(context.Background(), a, req, verifier(e))
	if !errors.Is(err, errOwner) || !errors.Is(err, closeErr) || res == nil || res.DraftOwner == nil {
		t.Fatalf("result=%+v error=%v", res, err)
	}
	if !reflect.DeepEqual(order, []string{"error", "step"}) {
		t.Fatalf("initial closes=%v", order)
	}
	stepOwner.err = nil
	if err := res.DraftOwner.Close(); err != nil {
		t.Fatal(err)
	}
	if err := res.DraftOwner.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []string{"error", "step", "step"}) {
		t.Fatalf("retry closes=%v", order)
	}
	if e.calls != 0 || res.KV != nil || a.NumFreeBlocks() != 7 {
		t.Fatalf("unexpected verifier/KV activity: calls=%d result=%+v free=%d", e.calls, res, a.NumFreeBlocks())
	}
}

func TestBatchDraftFailureIdentityDuringDrain(t *testing.T) {
	for _, tc := range []struct {
		name    string
		waitErr bool
		done    bool
	}{
		{"inflight-drain-error", true, false},
		{"completed-drain-error", true, true},
		{"completion-not-established", false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, req := batchFixture(t)
			runErr := errors.New("producer failure")
			drainErr := errors.New("drain failure")
			owner := &draftCloserRecorder{name: "step"}
			req.DraftProducer = &BatchDraftProducer{
				Owner: owner,
				Engine: DraftLogitsFunc(func(context.Context, *BatchDraftStep) ([]float32, error) {
					return nil, runErr
				}),
				Wait: func(*BatchDraftStep) error {
					if tc.waitErr {
						return drainErr
					}
					return nil
				},
				Done: func(*BatchDraftStep) bool { return tc.done },
			}
			e := &batchTokenEngine{tokens: []int32{2, 3, 1}}
			res, err := ExecuteBatchCycle(context.Background(), a, req, verifier(e))
			if !errors.Is(err, runErr) || tc.waitErr && !errors.Is(err, drainErr) {
				t.Fatalf("lost producer/drain identity: %v", err)
			}
			if res == nil || e.calls != 0 || res.KV != nil || a.NumFreeBlocks() != 7 {
				t.Fatalf("unexpected activity: result=%+v calls=%d free=%d", res, e.calls, a.NumFreeBlocks())
			}
			if tc.done {
				if owner.closed != 1 || res.DraftOwner != nil {
					t.Fatalf("quiescent owner state: %+v", res)
				}
			} else {
				if owner.closed != 0 || res.DraftOwner == nil {
					t.Fatalf("inflight owner state: %+v", res)
				}
				// Caller recovery follows independently established quiescence.
				if err := res.DraftOwner.Close(); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
