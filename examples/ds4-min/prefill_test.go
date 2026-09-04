package main

import (
	"errors"
	"reflect"
	"testing"
)

// batchBackend is a referenceBackend that also evaluates whole runs. Its
// batched path must leave exactly the state the single-token path leaves.
type batchBackend struct {
	referenceBackend
	runs [][]int
	fail bool
}

func (b *batchBackend) EvalBatch(tokens []int, position int) error {
	if b.fail {
		return errors.New("batch refused")
	}
	b.runs = append(b.runs, append([]int(nil), tokens...))
	for i, token := range tokens {
		if err := b.Eval(token, position+i); err != nil {
			return err
		}
	}
	return nil
}

func TestSessionUsesBatchedPrefill(t *testing.T) {
	b := new(batchBackend)
	s, err := newSession(b, 32, stopSet{1: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{10, 11, 12}); err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{10, 11, 12, 13, 14}); err != nil {
		t.Fatal(err)
	}
	want := [][]int{{10, 11, 12}, {13, 14}}
	if !reflect.DeepEqual(b.runs, want) {
		t.Fatalf("runs = %v, want %v", b.runs, want)
	}
}

// TestSessionBatchedPrefillMatchesLoop pins the seam's contract: a backend that
// batches must reach the same timeline and logits as one that does not.
func TestSessionBatchedPrefillMatchesLoop(t *testing.T) {
	prompts := [][]int{{10, 11, 12}, {10, 11, 12, 13}, {10, 99}, {10, 99, 7, 8, 9}}
	batched, err := newSession(new(batchBackend), 32, stopSet{1: true})
	if err != nil {
		t.Fatal(err)
	}
	looped, err := newSession(new(referenceBackend), 32, stopSet{1: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, prompt := range prompts {
		if err := batched.Sync(prompt); err != nil {
			t.Fatal(err)
		}
		if err := looped.Sync(prompt); err != nil {
			t.Fatal(err)
		}
		if _, err := batched.Accept(5); err != nil {
			t.Fatal(err)
		}
		if _, err := looped.Accept(5); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(batched.tokens, looped.tokens) || !reflect.DeepEqual(batched.logits, looped.logits) {
			t.Fatalf("prompt %v: batched %v/%v, looped %v/%v", prompt, batched.tokens, batched.logits, looped.tokens, looped.logits)
		}
	}
}

func TestSessionBatchedPrefillErrorNamesTheRun(t *testing.T) {
	s, err := newSession(&batchBackend{fail: true}, 32, stopSet{1: true})
	if err != nil {
		t.Fatal(err)
	}
	err = s.Sync([]int{10, 11})
	if err == nil || err.Error() != "evaluate tokens 0 through 1: batch refused" {
		t.Fatalf("error = %v", err)
	}
}
