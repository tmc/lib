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

type session struct {
	backend sessionBackend
	context int
	eos     int
	tokens  []int
	logits  []float32
}

func newSession(backend sessionBackend, context, eos int) (*session, error) {
	if backend == nil {
		return nil, errors.New("missing session backend")
	}
	if context <= 1 {
		return nil, errors.New("context must exceed one token")
	}
	return &session{backend: backend, context: context, eos: eos}, nil
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
	for i := common; i < len(prompt); i++ {
		if err := s.backend.Eval(prompt[i], i); err != nil {
			return fmt.Errorf("evaluate token %d: %w", i, err)
		}
		s.tokens = append(s.tokens, prompt[i])
	}
	if common != len(prompt) || s.logits == nil {
		s.logits = append(s.logits[:0], s.backend.Logits()...)
	}
	return nil
}

func (s *session) Accept(token int) (stop bool, err error) {
	if token == s.eos {
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
