package main

import (
	"errors"
	"fmt"
	"unicode/utf8"
)

var (
	errEmptyPrompt     = errors.New("empty prompt")
	errContextOverflow = errors.New("prompt exceeds context")
)

type sessionBackend interface {
	Reset() error
	Eval(token, position int) error
	Logits() []float32
}

// A batchEvaluator evaluates a whole run of tokens starting at position. A
// backend implements it when the run costs less than the sum of its Eval
// calls. The result must be indistinguishable from calling Eval for each token
// in order: the same timeline advance, and the same Logits afterwards as after
// evaluating the final token of the run. Logits for the earlier positions of a
// run are never observed, so a backend may skip work only those logits need.
type batchEvaluator interface {
	EvalBatch(tokens []int, position int) error
}

type session struct {
	backend sessionBackend
	context int
	stop    stopSet
	tokens  []int
	logits  []float32
}

func newSession(backend sessionBackend, context int, stop stopSet) (*session, error) {
	if backend == nil {
		return nil, errors.New("missing session backend")
	}
	if context <= 1 {
		return nil, errors.New("context must exceed one token")
	}
	if len(stop) == 0 {
		return nil, errors.New("empty stop-token set")
	}
	return &session{backend: backend, context: context, stop: stop}, nil
}

func (s *session) Sync(prompt []int) error {
	if len(prompt) == 0 {
		return errEmptyPrompt
	}
	if len(prompt) >= s.context {
		return fmt.Errorf("%w: length %d, context %d", errContextOverflow, len(prompt), s.context)
	}
	common := commonPrefix(s.tokens, prompt)
	if common != len(s.tokens) {
		if err := s.backend.Reset(); err != nil {
			return fmt.Errorf("reset session: %w", err)
		}
		s.tokens = s.tokens[:0]
		s.logits = nil
		common = 0
	}
	if run := prompt[common:]; len(run) != 0 {
		if err := s.evalRun(run, common); err != nil {
			return err
		}
		s.tokens = append(s.tokens, run...)
	}
	if common != len(prompt) || s.logits == nil {
		s.logits = append(s.logits[:0], s.backend.Logits()...)
	}
	return nil
}

func (s *session) Accept(token int) (stop bool, err error) {
	if s.stop.Contains(token) {
		return true, nil
	}
	if len(s.tokens) >= s.context {
		return false, fmt.Errorf("%w: generation has no free token", errContextOverflow)
	}
	if err := s.backend.Eval(token, len(s.tokens)); err != nil {
		return false, fmt.Errorf("evaluate generated token: %w", err)
	}
	s.tokens = append(s.tokens, token)
	s.logits = append(s.logits[:0], s.backend.Logits()...)
	return false, nil
}

// evalRun evaluates tokens starting at position, preferring the backend's
// batched prefill when it has one.
func (s *session) evalRun(tokens []int, position int) error {
	if b, ok := s.backend.(batchEvaluator); ok {
		if err := b.EvalBatch(tokens, position); err != nil {
			return fmt.Errorf("evaluate tokens %d through %d: %w", position, position+len(tokens)-1, err)
		}
		return nil
	}
	for i, token := range tokens {
		if err := s.backend.Eval(token, position+i); err != nil {
			return fmt.Errorf("evaluate token %d: %w", position+i, err)
		}
	}
	return nil
}

func commonPrefix(a, b []int) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

type textAssembler struct {
	pending []byte
}

func (a *textAssembler) Add(fragment []byte) (string, error) {
	a.pending = append(a.pending, fragment...)
	if !utf8.FullRune(a.pending) {
		return "", nil
	}
	if !utf8.Valid(a.pending) {
		a.pending = a.pending[:0]
		return "", errors.New("invalid UTF-8 output")
	}
	out := string(a.pending)
	a.pending = a.pending[:0]
	return out, nil
}

func (a *textAssembler) Finish() error {
	if len(a.pending) != 0 {
		return errors.New("incomplete UTF-8 output")
	}
	return nil
}
