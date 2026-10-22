package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type fakeSessionBackend struct {
	events []string
	logits []float32
}

func (b *fakeSessionBackend) Save(w io.Writer) error {
	var size [4]byte
	binary.LittleEndian.PutUint32(size[:], uint32(len(b.logits)))
	if _, err := w.Write(size[:]); err != nil {
		return err
	}
	for _, value := range b.logits {
		var word [4]byte
		binary.LittleEndian.PutUint32(word[:], math.Float32bits(value))
		if _, err := w.Write(word[:]); err != nil {
			return err
		}
	}
	return nil
}

func (b *fakeSessionBackend) Load(r io.Reader) error {
	var size [4]byte
	if _, err := io.ReadFull(r, size[:]); err != nil {
		return err
	}
	n := binary.LittleEndian.Uint32(size[:])
	if n > 16 {
		return errors.New("fake state too large")
	}
	logits := make([]float32, n)
	for i := range logits {
		var word [4]byte
		if _, err := io.ReadFull(r, word[:]); err != nil {
			return err
		}
		logits[i] = math.Float32frombits(binary.LittleEndian.Uint32(word[:]))
	}
	b.logits = logits
	return nil
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
	s, err := newSession(b, 8, stopSet{1: true})
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

func TestConfigurePrefillChunkRequiresBackendSupport(t *testing.T) {
	backend := new(fakeSessionBackend)
	if err := configurePrefillChunk(backend, 256); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("configure unsupported backend = %v, want ErrUnimplemented", err)
	}
	if err := configurePrefillChunk(backend, 2048); err != nil {
		t.Fatalf("configure default chunk = %v", err)
	}
}

func TestSessionSyncDivergentPrefixResets(t *testing.T) {
	b := new(fakeSessionBackend)
	s, _ := newSession(b, 8, stopSet{1: true})
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
	s, _ := newSession(b, 4, stopSet{1: true})
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
	s, _ := newSession(b, 3, stopSet{1: true})
	if err := s.Sync(nil); !errors.Is(err, errEmptyPrompt) {
		t.Fatalf("empty error = %v", err)
	}
	if err := s.Sync([]int{1, 2, 3}); !errors.Is(err, errContextOverflow) {
		t.Fatalf("overflow error = %v", err)
	}
}

func TestSessionSaveLoadRoundTrip(t *testing.T) {
	backend := new(fakeSessionBackend)
	s, err := newSession(backend, 16, stopSet{1: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{10, 11}); err != nil {
		t.Fatal(err)
	}
	var saved bytes.Buffer
	if err := s.Save(&saved); err != nil {
		t.Fatal(err)
	}
	restoredBackend := new(fakeSessionBackend)
	restored, _ := newSession(restoredBackend, 16, stopSet{1: true})
	if err := restored.Load(bytes.NewReader(saved.Bytes())); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored.tokens, s.tokens) || !reflect.DeepEqual(restored.logits, s.logits) {
		t.Fatalf("restored session = %v/%v, want %v/%v", restored.tokens, restored.logits, s.tokens, s.logits)
	}
	if err := restored.Sync([]int{10, 11, 12}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored.tokens, []int{10, 11, 12}) {
		t.Fatalf("continued tokens = %v", restored.tokens)
	}
}

func TestSessionLoadRejectsMutationBeforeBackend(t *testing.T) {
	backend := new(fakeSessionBackend)
	s, _ := newSession(backend, 16, stopSet{1: true})
	if err := s.Sync([]int{10, 11}); err != nil {
		t.Fatal(err)
	}
	beforeTokens := append([]int(nil), s.tokens...)
	beforeLogits := append([]float32(nil), s.logits...)
	if err := s.Load(bytes.NewReader([]byte("bad"))); err == nil {
		t.Fatal("malformed session accepted")
	}
	if !reflect.DeepEqual(s.tokens, beforeTokens) || !reflect.DeepEqual(s.logits, beforeLogits) {
		t.Fatal("malformed load changed session")
	}
}

func TestSessionLoadRejectsTrailingAndContextMismatch(t *testing.T) {
	backend := new(fakeSessionBackend)
	s, err := newSession(backend, 16, stopSet{1: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{10, 11}); err != nil {
		t.Fatal(err)
	}
	var saved bytes.Buffer
	if err := s.Save(&saved); err != nil {
		t.Fatal(err)
	}
	before := append([]int(nil), s.tokens...)
	trailing := append(append([]byte(nil), saved.Bytes()...), 0x7f)
	if err := s.Load(bytes.NewReader(trailing)); err == nil {
		t.Fatal("trailing session data accepted")
	}
	if !reflect.DeepEqual(s.tokens, before) {
		t.Fatal("trailing-data rejection changed session")
	}
	short, err := newSession(new(fakeSessionBackend), 8, stopSet{1: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := short.Load(bytes.NewReader(saved.Bytes())); err == nil {
		t.Fatal("context-mismatched session accepted")
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
	live, _ := newSession(liveBackend, 32, stopSet{1: true})
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
		fresh, _ := newSession(freshBackend, 32, stopSet{1: true})
		if err := fresh.Sync(prompt); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(live.tokens, fresh.tokens) || !reflect.DeepEqual(live.logits, fresh.logits) {
			t.Fatalf("attempt %d: live tokens/logits %v/%v, fresh %v/%v", attempt, live.tokens, live.logits, fresh.tokens, fresh.logits)
		}
	}
}

func TestSessionSplitPrefixMatchesDirectEvaluation(t *testing.T) {
	prompt := []int{7, 11, 13, 17, 19, 23}
	directBackend := new(referenceBackend)
	direct, err := newSession(directBackend, 32, stopSet{1: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := direct.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	splitBackend := new(referenceBackend)
	split, err := newSession(splitBackend, 32, stopSet{1: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := split.Sync(prompt[:3]); err != nil {
		t.Fatal(err)
	}
	if err := split.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(split.tokens, direct.tokens) || !reflect.DeepEqual(split.logits, direct.logits) {
		t.Fatalf("split state = %v/%v, direct = %v/%v", split.tokens, split.logits, direct.tokens, direct.logits)
	}
}
