package cudaspec

import (
	"context"
	"errors"
	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamtp"
	"math"
	"reflect"
	"testing"
)

type batchTokenEngine struct {
	tokens  []int32
	calls   int
	inspect func(*cudabatch.ExecutionStep)
	err     error
	foreign bool
}

func (e *batchTokenEngine) ExecuteForward(ctx context.Context, s *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
	if e.inspect != nil {
		e.inspect(s)
	}
	e.calls++
	if e.err != nil {
		return nil, nil, e.err
	}
	id := s.Metadata.RowBindings[0].SeqID
	if e.foreign {
		id = "wrong-branch"
	}
	return map[string]int32{id: e.tokens[(e.calls-1)%len(e.tokens)]}, nil, nil
}
func batchFixture(t *testing.T) (*cudakv.BlockAllocator, BatchCycleRequest) {
	t.Helper()
	a := cudakv.NewBlockAllocator(2, 8)
	if err := a.AllocSequence(10, nil, 2); err != nil {
		t.Fatal(err)
	}
	r := BatchCycleRequest{Prefix: []int32{0, 1}, PrefixSequence: 10, Sequence: 11, Hidden: []float32{1}, DraftWeights: []float32{1, 1}, HeadWeights: []float32{0, 0, 2, 0, 0, 0, 0, 3}, MTP: cudamtp.MTPConfig{NumDraftHeads: 2, Dim: 1, VocabSize: 4}}
	var model [32]byte
	model[0] = 1
	r.Key = ComputeSpeculationKey(model[:], r.Prefix, 1, 2, 2, "fp32", 1)
	r.ExpectedKey = r.Key
	var err error
	r.DraftDigest, err = MTPDraftDigest(r.MTP, r.DraftWeights, r.HeadWeights)
	if err != nil {
		t.Fatal(err)
	}
	return a, r
}
func verifier(e *batchTokenEngine) BatchVerifier {
	return BatchVerifier{Engine: e, ReadOnlyPrefix: true, Wait: func(*cudabatch.ExecutionStep) error { return nil }}
}
func TestMTPBatchCycleCommitAndRollback(t *testing.T) {
	for _, tt := range []struct {
		name     string
		target   []int32
		commit   bool
		accepted int
		bonus    int32
	}{{"accept", []int32{2, 3, 1}, true, 2, 1}, {"partial", []int32{2, 1}, false, 1, 1}, {"reject", []int32{1}, false, 0, 1}} {
		t.Run(tt.name, func(t *testing.T) {
			a, r := batchFixture(t)
			prefix := a.Table(10)
			e := &batchTokenEngine{tokens: tt.target, inspect: func(s *cudabatch.ExecutionStep) {
				row := s.Metadata.RowBindings[0]
				if row.StartPos != row.SequenceLength-1 || len(row.Tokens) != 1 {
					t.Fatal("invalid binding")
				}
				n, _ := a.Holders(prefix[0])
				if n != 2 {
					t.Fatal("prefix not borrowed")
				}
			}}
			result, err := ExecuteBatchCycle(context.Background(), a, r, verifier(e))
			if err != nil {
				t.Fatal(err)
			}
			o := result.Outcome
			if o.Committed != tt.commit || len(o.AcceptedTokens) != tt.accepted || o.BonusToken != tt.bonus || !reflect.DeepEqual(o.DraftTokens, []int32{2, 3}) {
				t.Fatalf("outcome %+v", o)
			}
			if err := result.KV.Close(); err != nil {
				t.Fatal(err)
			}
			if a.NumFreeBlocks() != 7 || !reflect.DeepEqual(a.Table(10), prefix) {
				t.Fatal("borrowed prefix lost")
			}
			if err := result.KV.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
func TestBatchCycleIdentityAndBoundsRefusal(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*BatchCycleRequest)
	}{{"weights", func(r *BatchCycleRequest) { r.HeadWeights[2]++ }},
		{"model", func(r *BatchCycleRequest) { r.ExpectedKey.ModelDigest[0]++ }}, {"prefix", func(r *BatchCycleRequest) { r.Prefix[0] = 2 }}, {"generation", func(r *BatchCycleRequest) { r.ExpectedKey.Generation++ }}, {"branch", func(r *BatchCycleRequest) { r.Key.BranchID = 2 }}, {"numeric", func(r *BatchCycleRequest) { r.Hidden[0] = float32(math.NaN()) }}, {"bound", func(r *BatchCycleRequest) { r.MTP.NumDraftHeads = 17 }}} {
		t.Run(tt.name, func(t *testing.T) {
			a, r := batchFixture(t)
			tt.mutate(&r)
			e := &batchTokenEngine{tokens: []int32{2}}
			if _, err := ExecuteBatchCycle(context.Background(), a, r, verifier(e)); err == nil {
				t.Fatal("invalid admission accepted")
			}
			if e.calls != 0 || a.NumFreeBlocks() != 7 {
				t.Fatal("side effect before admission")
			}
		})
	}
}
func TestBatchCycleDrainOwnsKVLifetime(t *testing.T) {
	a, r := batchFixture(t)
	e := &batchTokenEngine{err: context.Canceled}
	v := verifier(e)
	v.Wait = func(*cudabatch.ExecutionStep) error {
		n, _ := a.Holders(a.Table(10)[0])
		if n != 2 {
			t.Fatal("KV released before drain")
		}
		return errors.New("device not drained")
	}
	result, err := ExecuteBatchCycle(context.Background(), a, r, v)
	if err == nil || result == nil {
		t.Fatal("missing drain receipt")
	}
	if a.NumFreeBlocks() != 6 {
		t.Fatal("undrained KV freed")
	}
	if err := result.KV.Close(); err != nil {
		t.Fatal(err)
	}
	if a.NumFreeBlocks() != 7 {
		t.Fatal("cleanup failed")
	}
}
func TestBatchCycleForeignRowAndMutatingEngineRefused(t *testing.T) {
	a, r := batchFixture(t)
	e := &batchTokenEngine{tokens: []int32{2}, foreign: true}
	if _, err := ExecuteBatchCycle(context.Background(), a, r, verifier(e)); err == nil {
		t.Fatal("foreign row accepted")
	}
	if a.NumFreeBlocks() != 7 {
		t.Fatal("failure leaked KV")
	}
	v := verifier(e)
	v.ReadOnlyPrefix = false
	e.calls = 0
	if _, err := ExecuteBatchCycle(context.Background(), a, r, v); err == nil || e.calls != 0 {
		t.Fatal("mutating verifier admitted")
	}
}
func TestBranchKVBorrowedPrefixAndDuplicateRelease(t *testing.T) {
	a, r := batchFixture(t)
	kv, err := NewBranchKV(a, r.PrefixSequence, r.Sequence, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := kv.Release(a.Table(10)); err == nil {
		t.Fatal("borrowed prefix released")
	}
	if err := kv.Release(kv.private); err != nil {
		t.Fatal(err)
	}
	if err := kv.Release(kv.private); err != nil {
		t.Fatal(err)
	}
	n, _ := a.Holders(a.Table(10)[0])
	if n != 1 {
		t.Fatal("original prefix lost")
	}
	a.FreeSequence(10)
	if err := a.AllocSequence(10, nil, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := NewBranchKV(a, 10, 12, 1, 2); err == nil {
		t.Fatal("partial prefix sharing accepted")
	}
}

func TestBranchKVRejectsDuplicatedPrefix(t *testing.T) {
	a, _ := batchFixture(t)
	root := a.Table(10)
	before := a.NumFreeBlocks()
	holders, err := a.Holders(root[0])
	if err != nil {
		t.Fatal(err)
	}
	// The public allocator refuses this prefix before BranchKV can observe
	// it. This does not independently cover BranchKV's private duplicate guard.
	if err := a.AllocSequence(12, []int{root[0], root[0]}, 4); err == nil {
		t.Fatal("duplicate shared prefix admitted by allocator")
	}
	after, err := a.Holders(root[0])
	if err != nil || after != holders || a.NumFreeBlocks() != before || len(a.Table(12)) != 0 {
		t.Fatal("duplicate-prefix refusal changed allocations or holders")
	}
}

func TestBatchCycleSeparatesCompletionAndTerminalError(t *testing.T) {
	for _, tt := range []struct {
		name    string
		waitErr error
		done    bool
		free    int
	}{{"completed-failed", errors.New("failed completed step"), true, 7}, {"inflight-failed", errors.New("still pending"), false, 6}, {"false-success", nil, false, 6}} {
		t.Run(tt.name, func(t *testing.T) {
			a, r := batchFixture(t)
			e := &batchTokenEngine{tokens: []int32{2}}
			v := verifier(e)
			v.Wait = func(*cudabatch.ExecutionStep) error { return tt.waitErr }
			v.Done = func(s *cudabatch.ExecutionStep) bool {
				if s.StepID != 1 {
					t.Fatal("wrong step completion")
				}
				return tt.done
			}
			result, err := ExecuteBatchCycle(context.Background(), a, r, v)
			if err == nil || result == nil {
				t.Fatal("missing failed cycle")
			}
			if a.NumFreeBlocks() != tt.free {
				t.Fatalf("free blocks %d want %d", a.NumFreeBlocks(), tt.free)
			}
			if err := result.KV.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestBatchCycleCancellationAfterCompletedForward(t *testing.T) {
	a, r := batchFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e := &batchTokenEngine{tokens: []int32{2}, inspect: func(*cudabatch.ExecutionStep) { cancel() }}
	v := verifier(e)
	waited := false
	v.Wait = func(*cudabatch.ExecutionStep) error { waited = true; return nil }
	result, err := ExecuteBatchCycle(ctx, a, r, v)
	if !errors.Is(err, context.Canceled) || !waited || result == nil || a.NumFreeBlocks() != 7 {
		t.Fatalf("cancel/drain result=%v err=%v waited=%v free=%d", result, err, waited, a.NumFreeBlocks())
	}
}
