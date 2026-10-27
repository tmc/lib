package cudaspec_test

import (
	"context"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamtp"
	"github.com/tmc/lib/nvidia/x/cudaspec"
)

type identityDraftOwner struct {
	name  string
	calls *int
	order *[]string
}

func (o identityDraftOwner) Error() string { return o.name }
func (o identityDraftOwner) Close() error {
	(*o.calls)++
	*o.order = append(*o.order, o.name)
	return nil
}

type sliceDraftOwner struct {
	identityDraftOwner
	data []int
}

type interfaceDraftOwner struct {
	identityDraftOwner
	data any
}

type identityVerifier struct{ calls int }

func (v *identityVerifier) ExecuteForward(context.Context, *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
	v.calls++
	return nil, nil, errors.New("unexpected verifier call")
}

func draftIdentityFixture(t *testing.T) (*cudakv.BlockAllocator, cudaspec.BatchCycleRequest, cudaspec.BatchVerifier, *identityVerifier) {
	t.Helper()
	a := cudakv.NewBlockAllocator(2, 8)
	if err := a.AllocSequence(10, nil, 2); err != nil {
		t.Fatal(err)
	}
	r := cudaspec.BatchCycleRequest{
		Prefix: []int32{0, 1}, PrefixSequence: 10, Sequence: 11,
		Hidden: []float32{1}, DraftWeights: []float32{1, 1},
		HeadWeights: []float32{0, 0, 2, 0, 0, 0, 0, 3},
		MTP:         cudamtp.MTPConfig{NumDraftHeads: 2, Dim: 1, VocabSize: 4},
	}
	var model [32]byte
	model[0] = 1
	r.Key = cudaspec.ComputeSpeculationKey(model[:], r.Prefix, 1, 2, 2, "fp32", 1)
	r.ExpectedKey = r.Key
	var err error
	r.DraftDigest, err = cudaspec.MTPDraftDigest(r.MTP, r.DraftWeights, r.HeadWeights)
	if err != nil {
		t.Fatal(err)
	}
	e := &identityVerifier{}
	return a, r, cudaspec.BatchVerifier{ReadOnlyPrefix: true, Engine: e, Wait: func(*cudabatch.ExecutionStep) error { return nil }}, e
}

func checkDraftIdentityKV(t *testing.T, a *cudakv.BlockAllocator, prefix []int) {
	t.Helper()
	if a.NumFreeBlocks() != 7 || !reflect.DeepEqual(a.Table(10), prefix) || len(a.Table(11)) != 0 {
		t.Fatal("draft admission changed KV allocations")
	}
	for _, block := range prefix {
		if holders, err := a.Holders(block); err != nil || holders != 1 {
			t.Fatalf("prefix holders = %d, %v; want 1", holders, err)
		}
	}
}

func TestBatchDraftOwnerRejectsNoncomparable(t *testing.T) {
	for _, name := range []string{"slice", "interface field"} {
		t.Run(name, func(t *testing.T) {
			a, req, verifier, engine := draftIdentityFixture(t)
			prefix := a.Table(10)
			calls, produced, waited := 0, 0, 0
			var order []string
			base := identityDraftOwner{name: "owner", calls: &calls, order: &order}
			var owner interface {
				error
				io.Closer
			} = sliceDraftOwner{identityDraftOwner: base, data: []int{1}}
			if name == "interface field" {
				owner = interfaceDraftOwner{identityDraftOwner: base, data: []int{1}}
				if !reflect.TypeOf(owner).Comparable() {
					t.Fatal("fixture type must be comparable despite its noncomparable value")
				}
			}
			if reflect.ValueOf(owner).Comparable() {
				t.Fatal("fixture value is comparable")
			}
			req.DraftProducer = &cudaspec.BatchDraftProducer{
				Owner: owner,
				Engine: cudaspec.DraftLogitsFunc(func(context.Context, *cudaspec.BatchDraftStep) ([]float32, error) {
					produced++
					return nil, owner
				}),
				Wait: func(*cudaspec.BatchDraftStep) error { waited++; return nil },
			}
			res, err := cudaspec.ExecuteBatchCycle(context.Background(), a, req, verifier)
			if err == nil || !strings.Contains(err.Error(), "configured draft owner has noncomparable identity") || res != nil {
				t.Fatalf("admission = %+v, %v; want explicit refusal", res, err)
			}
			if produced != 0 || waited != 0 || calls != 0 || engine.calls != 0 {
				t.Fatalf("refusal side effects: produce=%d wait=%d close=%d verify=%d", produced, waited, calls, engine.calls)
			}
			checkDraftIdentityKV(t, a, prefix)
			if err := owner.Close(); err != nil || calls != 1 {
				t.Fatalf("caller lost cleanup ownership: calls=%d err=%v", calls, err)
			}
		})
	}
}

func TestBatchDraftComparableSharedOwner(t *testing.T) {
	for _, pointer := range []bool{false, true} {
		name := "value"
		if pointer {
			name = "pointer"
		}
		t.Run(name, func(t *testing.T) {
			a, req, verifier, engine := draftIdentityFixture(t)
			prefix := a.Table(10)
			calls := 0
			var order []string
			base := identityDraftOwner{name: "shared", calls: &calls, order: &order}
			var owner interface {
				error
				io.Closer
			} = base
			if pointer {
				owner = &base
			}
			req.DraftProducer = &cudaspec.BatchDraftProducer{
				Owner: owner,
				Engine: cudaspec.DraftLogitsFunc(func(context.Context, *cudaspec.BatchDraftStep) ([]float32, error) {
					order = append(order, "produce")
					return nil, owner
				}),
				Wait: func(*cudaspec.BatchDraftStep) error { order = append(order, "wait"); return nil },
			}
			res, err := cudaspec.ExecuteBatchCycle(context.Background(), a, req, verifier)
			if !errors.Is(err, owner) || res == nil || res.DraftOwner != nil || res.KV != nil || engine.calls != 0 || calls != 1 || !reflect.DeepEqual(order, []string{"produce", "wait", "shared"}) {
				t.Fatalf("shared cleanup: result=%+v err=%v calls=%d order=%v", res, err, calls, order)
			}
			checkDraftIdentityKV(t, a, prefix)
		})
	}
}

func TestBatchDraftDistinctOwnerIdentity(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		configured, sliceError bool
	}{
		{"error only", false, true},
		{"distinct comparable", true, false},
		{"distinct noncomparable error", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, req, verifier, engine := draftIdentityFixture(t)
			prefix := a.Table(10)
			stepCalls, errCalls := 0, 0
			var order []string
			stepOwner := &identityDraftOwner{name: "step", calls: &stepCalls, order: &order}
			errBase := identityDraftOwner{name: "error", calls: &errCalls, order: &order}
			var errOwner interface {
				error
				io.Closer
			} = &errBase
			if tc.sliceError {
				errOwner = sliceDraftOwner{identityDraftOwner: errBase, data: []int{1}}
			}
			req.DraftProducer = &cudaspec.BatchDraftProducer{
				Engine: cudaspec.DraftLogitsFunc(func(context.Context, *cudaspec.BatchDraftStep) ([]float32, error) {
					order = append(order, "produce")
					return nil, errOwner
				}),
				Wait: func(*cudaspec.BatchDraftStep) error { order = append(order, "wait"); return nil },
			}
			wantOrder := []string{"produce", "wait", "error"}
			wantStepCalls := 0
			if tc.configured {
				req.DraftProducer.Owner = stepOwner
				wantStepCalls = 1
				wantOrder = append(wantOrder, "step")
			}
			res, err := cudaspec.ExecuteBatchCycle(context.Background(), a, req, verifier)
			var preserved interface {
				error
				io.Closer
			}
			if !errors.As(err, &preserved) || preserved.Error() != "error" || res == nil || res.DraftOwner != nil || res.KV != nil || engine.calls != 0 || errCalls != 1 || stepCalls != wantStepCalls || !reflect.DeepEqual(order, wantOrder) {
				t.Fatalf("distinct cleanup: result=%+v err=%v step=%d error=%d order=%v", res, err, stepCalls, errCalls, order)
			}
			checkDraftIdentityKV(t, a, prefix)
		})
	}
}
