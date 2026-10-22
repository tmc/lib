package dsv4bench

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type evaluationBackend struct {
	states []*evaluationState
	fail   string
}
type evaluationState struct {
	benchState
	owner      *evaluationBackend
	id         int
	closeCalls int
}

func (s *evaluationState) Close() error {
	s.closeCalls++
	s.closed = true
	if s.owner.fail == "close" {
		return errors.New("close failed")
	}
	return nil
}
func (b *evaluationBackend) NewState(int) (dsv4.State, error) {
	s := &evaluationState{owner: b, id: len(b.states)}
	b.states = append(b.states, s)
	if b.fail == "partial" {
		return s, errors.New("partial state")
	}
	return s, nil
}
func (b *evaluationBackend) Eval(v dsv4.State, token, pos int) error {
	s := v.(*evaluationState)
	if len(s.tokens) != pos {
		return errors.New("wrong position")
	}
	s.tokens = append(s.tokens, token)
	return nil
}
func (b *evaluationBackend) Logits(v dsv4.State) []float32 {
	s := v.(*evaluationState)
	x := float32(len(s.tokens))
	if b.fail == "fresh-mutation" && s.id > 0 {
		x += 1
	}
	if b.fail == "nan" {
		x = float32(math.NaN())
	}
	return []float32{0, x, 2}
}
func (b *evaluationBackend) Close() error { return nil }
func evaluationFixture() EvaluationConfig {
	return EvaluationConfig{Prompt: []int{1, 2}, Continuation: []int{2, 1}, Provenance: Provenance{Revision: "revision", BinarySHA256: strings.Repeat("1", 64), ModelSHA256: strings.Repeat("2", 64), Host: "fixture", Backend: "fixture", Context: 8}}
}

func TestEvaluateReceipts(t *testing.T) {
	b := new(evaluationBackend)
	cfg := evaluationFixture()
	r, err := Evaluate(context.Background(), b, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Complete || r.Expected != 2 || r.Ran != 2 || r.Skipped != 0 || r.Perplexity.Scored != 2 || r.Consistency.Positions != 2 || r.Consistency.MaxAbs != 0 || len(b.states) != 3 {
		t.Fatalf("incomplete receipt: %+v", r)
	}
	wantNLL := math.Log(2+math.Exp(-2)) + math.Log(1+math.Exp(-3)+math.Exp(-1))
	if math.Abs(r.Perplexity.NLL-wantNLL) > 1e-12 {
		t.Fatalf("perplexity NLL=%g want independent %g", r.Perplexity.NLL, wantNLL)
	}
	for i, row := range r.Steps {
		history := append(append([]int(nil), cfg.Prompt...), cfg.Continuation[:i]...)
		if row.Position != len(history) || row.HistorySHA256 != historyHash(history) || row.ResultHistorySHA256 != historyHash(append(history, row.Token)) || len(row.Logits) != 3 {
			t.Fatal("history/row mismatch")
		}
	}
	for _, s := range b.states {
		if !s.closed {
			t.Fatal("state leaked")
		}
	}
	r.Steps[0].Logits[0] = 88
	if b.Logits(b.states[0])[0] != 0 {
		t.Fatal("row aliases backend")
	}
}

func TestEvaluateFailsClosed(t *testing.T) {
	for _, failure := range []string{"close", "partial", "nan"} {
		t.Run(failure, func(t *testing.T) {
			b := &evaluationBackend{fail: failure}
			r, err := Evaluate(context.Background(), b, evaluationFixture())
			if err == nil || r.Complete || len(r.Steps) != 0 {
				t.Fatal("failed producer returned receipt", r, err)
			}
			for _, s := range b.states {
				if !s.closed {
					t.Fatal("failed producer leaked state")
				}
			}
		})
	}
}

func TestEvaluateDetectsFreshMutation(t *testing.T) {
	b := &evaluationBackend{fail: "fresh-mutation"}
	r, err := Evaluate(context.Background(), b, evaluationFixture())
	if err != nil {
		t.Fatal(err)
	}
	if r.Consistency.MaxAbs != 1 || r.Consistency.Positions != 2 {
		t.Fatal("independent fresh mutation hidden")
	}
}

func TestEvaluateRetainsStateCleanup(t *testing.T) {
	b := &evaluationBackend{fail: "close"}
	r, err := Evaluate(context.Background(), b, evaluationFixture())
	if err == nil || r.Complete || len(r.Steps) != 0 {
		t.Fatal("cleanup failure returned measurement", err)
	}
	var owners []io.Closer
	var visit func(error)
	visit = func(err error) {
		if owner, ok := err.(io.Closer); ok {
			owners = append(owners, owner)
		} else if joined, ok := err.(interface{ Unwrap() []error }); ok {
			for _, e := range joined.Unwrap() {
				visit(e)
			}
		} else if wrapped, ok := err.(interface{ Unwrap() error }); ok {
			visit(wrapped.Unwrap())
		}
	}
	visit(err)
	if len(owners) != 1 {
		t.Fatalf("retained owners=%d want ordered aggregate", len(owners))
	}
	b.fail = ""
	for _, owner := range owners {
		if err := owner.Close(); err != nil {
			t.Fatal(err)
		}
		if err := owner.Close(); err != nil {
			t.Fatal(err)
		}
	}
	for _, state := range b.states {
		if state.closeCalls != 2 {
			t.Fatalf("state%d closecalls=%d want failed attempt plus retry", state.id, state.closeCalls)
		}
	}
}

func TestEvaluateProvenance(t *testing.T) {
	for _, mutate := range []func(*EvaluationConfig){func(c *EvaluationConfig) { c.Provenance.ModelSHA256 = "" }, func(c *EvaluationConfig) { c.Prompt = nil }, func(c *EvaluationConfig) { c.Continuation = nil }, func(c *EvaluationConfig) { c.Continuation[0] = -1 }, func(c *EvaluationConfig) { c.Continuation[0] = 3 }, func(c *EvaluationConfig) { c.Provenance.Context = 4 }} {
		c := evaluationFixture()
		mutate(&c)
		r, err := Evaluate(context.Background(), new(evaluationBackend), c)
		if err == nil || r.Complete {
			t.Fatal("bad provenance/input admitted")
		}
	}
}

func ExampleEvaluate() {
	r, err := Evaluate(context.Background(), new(evaluationBackend), evaluationFixture())
	fmt.Println(r.Complete, r.Ran, r.Perplexity.Scored, err)
	// Output: true 2 2 <nil>
}
func ExampleEvaluationConfig() {
	cfg := EvaluationConfig{Prompt: []int{1}, Continuation: []int{2}}
	fmt.Println(len(cfg.Prompt), len(cfg.Continuation))
	// Output: 1 1
}
func ExampleEvaluationStep() {
	s := EvaluationStep{Position: 4, Token: 2}
	fmt.Println(s.Position, s.Token)
	// Output: 4 2
}
func ExampleEvaluationReceipt() {
	r := EvaluationReceipt{Expected: 2}
	fmt.Println(r.Expected, r.Complete)
	// Output: 2 false
}
