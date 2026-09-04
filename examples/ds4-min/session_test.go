package main

import (
	"errors"
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

type fakeSessionBackend struct {
	events []string
	logits []float32
}

func (b *fakeSessionBackend) Reset() error {
	b.events = append(b.events, "reset")
	return nil
}

func (b *fakeSessionBackend) Eval(token, position int) error {
	b.events = append(b.events, fmt.Sprintf("eval %d@%d", token, position))
	b.logits = []float32{float32(token), float32(position)}
	return nil
}

func (b *fakeSessionBackend) Logits() []float32 {
	b.events = append(b.events, "logits")
	return b.logits
}

func TestSessionSyncSuffixReuse(t *testing.T) {
	b := new(fakeSessionBackend)
	s, err := newSession(b, 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{10, 11}); err != nil {
		t.Fatal(err)
	}
	b.events = nil
	if err := s.Sync([]int{10, 11, 12}); err != nil {
		t.Fatal(err)
	}
	want := []string{"eval 12@2", "logits"}
	if !reflect.DeepEqual(b.events, want) {
		t.Fatalf("events = %v, want %v", b.events, want)
	}
	if want := []float32{12, 2}; !reflect.DeepEqual(s.logits, want) {
		t.Fatalf("logits = %v, want %v", s.logits, want)
	}
}

func TestSessionSyncDivergentPrefixResets(t *testing.T) {
	b := new(fakeSessionBackend)
	s, _ := newSession(b, 8, 1)
	if err := s.Sync([]int{10, 11}); err != nil {
		t.Fatal(err)
	}
	b.events = nil
	if err := s.Sync([]int{10, 99}); err != nil {
		t.Fatal(err)
	}
	want := []string{"reset", "eval 10@0", "eval 99@1", "logits"}
	if !reflect.DeepEqual(b.events, want) {
		t.Fatalf("events = %v, want %v", b.events, want)
	}
}

func TestSessionEOSBeforeEval(t *testing.T) {
	b := new(fakeSessionBackend)
	s, _ := newSession(b, 4, 1)
	if err := s.Sync([]int{10}); err != nil {
		t.Fatal(err)
	}
	b.events = nil
	stop, err := s.Accept(1)
	if err != nil || !stop {
		t.Fatalf("Accept = %t, %v", stop, err)
	}
	if len(b.events) != 0 {
		t.Fatalf("events after EOS = %v", b.events)
	}
}

func TestSessionBounds(t *testing.T) {
	b := new(fakeSessionBackend)
	s, _ := newSession(b, 3, 1)
	if err := s.Sync(nil); !errors.Is(err, errEmptyPrompt) {
		t.Fatalf("empty error = %v", err)
	}
	if err := s.Sync([]int{1, 2, 3}); !errors.Is(err, errContextOverflow) {
		t.Fatalf("overflow error = %v", err)
	}
}

func TestTextAssemblerUTF8Fragments(t *testing.T) {
	var a textAssembler
	if got, err := a.Add([]byte{0xe4, 0xb8}); err != nil || got != "" {
		t.Fatalf("partial = %q", got)
	}
	if got, err := a.Add([]byte{0x96}); err != nil || got != "世" {
		t.Fatalf("complete = %q", got)
	}
	if err := a.Finish(); err != nil {
		t.Fatal(err)
	}
	if got, err := a.Add([]byte{0xe4}); err != nil || got != "" {
		t.Fatalf("partial = %q", got)
	}
	if err := a.Finish(); err == nil {
		t.Fatal("incomplete UTF-8 succeeded")
	}
}

func TestTextAssemblerRejectsInvalidUTF8(t *testing.T) {
	var a textAssembler
	if _, err := a.Add([]byte{0x80}); err == nil {
		t.Fatal("invalid UTF-8 succeeded")
	}
	if got, err := a.Add([]byte("A")); err != nil || got != "A" {
		t.Fatalf("recovery = %q, %v", got, err)
	}
}

type referenceBackend struct {
	tokens []int
}

func (b *referenceBackend) Reset() error {
	b.tokens = b.tokens[:0]
	return nil
}

func (b *referenceBackend) Eval(token, position int) error {
	if position != len(b.tokens) {
		return fmt.Errorf("position %d, want %d", position, len(b.tokens))
	}
	b.tokens = append(b.tokens, token)
	return nil
}

func (b *referenceBackend) Logits() []float32 {
	var sum int
	for i, token := range b.tokens {
		sum += (i + 1) * token
	}
	return []float32{float32(len(b.tokens)), float32(sum)}
}

func TestSessionDifferential(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	liveBackend := new(referenceBackend)
	live, _ := newSession(liveBackend, 32, 1)
	for attempt := 0; attempt < 100; attempt++ {
		n := 1 + rng.Intn(20)
		prompt := make([]int, n)
		for i := range prompt {
			prompt[i] = 2 + rng.Intn(100)
		}
		if err := live.Sync(prompt); err != nil {
			t.Fatal(err)
		}
		freshBackend := new(referenceBackend)
		fresh, _ := newSession(freshBackend, 32, 1)
		if err := fresh.Sync(prompt); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(live.tokens, fresh.tokens) || !reflect.DeepEqual(live.logits, fresh.logits) {
			t.Fatalf("attempt %d: live tokens/logits %v/%v, fresh %v/%v", attempt, live.tokens, live.logits, fresh.tokens, fresh.logits)
		}
	}
}
