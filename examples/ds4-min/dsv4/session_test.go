package dsv4

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"reflect"
	"testing"
)

type sessionBackend struct{ failToken int }

type sessionState struct {
	values    []int
	closed    bool
	failToken int
}

type closeRetrySessionState struct {
	*sessionState
	calls int
}

func (s *closeRetrySessionState) Close() error {
	s.calls++
	if s.calls == 1 {
		return io.ErrUnexpectedEOF
	}
	return s.sessionState.Close()
}

func TestSessionCloseRetry(t *testing.T) {
	state := &closeRetrySessionState{sessionState: &sessionState{}}
	s := &Session{backend: sessionBackend{}, state: state, context: 32, stop: NewStopSet(0)}
	if err := s.Close(); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("first Close = %v", err)
	}
	if s.state != state {
		t.Fatal("failed Close lost state ownership")
	}
	checks := []struct {
		name string
		call func() error
	}{
		{"Sync", func() error { return s.Sync([]int{1}) }},
		{"Reset", s.Reset},
		{"Accept", func() error { _, err := s.Accept(1); return err }},
		{"Save", func() error { return s.Save(io.Discard) }},
		{"Load", func() error { return s.Load(bytes.NewReader(nil)) }},
	}
	for _, check := range checks {
		if err := check.call(); !errors.Is(err, ErrClosed) {
			t.Errorf("%s after failed Close = %v", check.name, err)
		}
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil || state.calls != 2 || s.state != nil {
		t.Fatalf("final Close = %v, calls=%d state=%v", err, state.calls, s.state)
	}
}

func (b sessionBackend) NewState(int) (State, error) {
	return &sessionState{failToken: b.failToken}, nil
}
func (b sessionBackend) Eval(state State, token, position int) error {
	s := state.(*sessionState)
	if token == s.failToken {
		return errors.New("injected evaluation failure")
	}
	if position != len(s.values) {
		return errors.New("unexpected position")
	}
	s.values = append(s.values, token)
	return nil
}
func (sessionBackend) Logits(state State) []float32 {
	s := state.(*sessionState)
	if len(s.values) == 0 {
		return nil
	}
	return []float32{float32(s.values[len(s.values)-1])}
}
func (sessionBackend) Close() error { return nil }

func (s *sessionState) Reset() error {
	if s.closed {
		return ErrClosed
	}
	s.values = nil
	return nil
}
func (s *sessionState) Snapshot() (Snapshot, error) {
	return &sessionSnapshot{values: append([]int(nil), s.values...)}, nil
}
func (s *sessionState) Restore(snapshot Snapshot) error {
	copyOf, ok := snapshot.(*sessionSnapshot)
	if !ok {
		return errors.New("foreign snapshot")
	}
	s.values = append(s.values[:0], copyOf.values...)
	return nil
}
func (s *sessionState) Save(w io.Writer) error {
	var word [4]byte
	binary.LittleEndian.PutUint32(word[:], uint32(len(s.values)))
	if _, err := w.Write(word[:]); err != nil {
		return err
	}
	for _, value := range s.values {
		binary.LittleEndian.PutUint32(word[:], uint32(value))
		if _, err := w.Write(word[:]); err != nil {
			return err
		}
	}
	return nil
}
func (s *sessionState) Load(r io.Reader) error {
	b, err := io.ReadAll(r)
	if err != nil || len(b) < 4 {
		return errors.New("invalid state")
	}
	n := int(binary.LittleEndian.Uint32(b[:4]))
	if n < 0 || len(b) != 4+4*n {
		return errors.New("invalid state")
	}
	values := make([]int, n)
	for i := range values {
		values[i] = int(binary.LittleEndian.Uint32(b[4+4*i:]))
	}
	s.values = values
	return nil
}
func (s *sessionState) Close() error { s.closed = true; return nil }

type sessionSnapshot struct{ values []int }

func (s *sessionSnapshot) Close() error { return nil }

func TestSessionUsesIndependentState(t *testing.T) {
	b := sessionBackend{}
	a, err := NewSession(b, 8, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	c, err := NewSession(b, 8, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Sync([]int{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := c.Sync([]int{7}); err != nil {
		t.Fatal(err)
	}
	if got := a.Tokens(); len(got) != 2 || got[0] != 1 {
		t.Fatalf("session A tokens = %v", got)
	}
	if got := c.Tokens(); len(got) != 1 || got[0] != 7 {
		t.Fatalf("session C tokens = %v", got)
	}
}

func TestSessionSyncFailureClosesOverDesynchronizedState(t *testing.T) {
	s, err := NewSession(sessionBackend{failToken: 9}, 16, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{7, 9}); !errors.Is(err, ErrDesynchronized) {
		t.Fatalf("divergent Sync error = %v, want ErrDesynchronized", err)
	}
	if got := s.Tokens(); len(got) != 0 {
		t.Fatalf("tokens after failed Sync = %v, want empty", got)
	}
	if got := s.Logits(); len(got) != 0 {
		t.Fatalf("logits after failed Sync = %v, want empty", got)
	}
	if err := s.Sync([]int{3}); !errors.Is(err, ErrDesynchronized) {
		t.Fatalf("reuse after failed Sync = %v, want ErrDesynchronized", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionCloseReleasesState(t *testing.T) {
	b := sessionBackend{}
	s, err := NewSession(b, 8, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	state := s.state.(*sessionState)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if !state.closed {
		t.Fatal("Close did not release the request state")
	}
	if err := s.Reset(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Reset after Close = %v, want ErrClosed", err)
	}
	if stop, err := s.Accept(1); stop || !errors.Is(err, ErrClosed) {
		t.Fatalf("Accept after Close = (%v, %v), want (false, ErrClosed)", stop, err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestSessionSaveLoadRoundTrip(t *testing.T) {
	b := sessionBackend{}
	s, err := NewSession(b, 8, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{1, 2}); err != nil {
		t.Fatal(err)
	}
	if stop, err := s.Accept(3); stop || err != nil {
		t.Fatalf("Accept = (%v, %v)", stop, err)
	}
	var saved bytes.Buffer
	if err := s.Save(&saved); err != nil {
		t.Fatal(err)
	}
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{7}); err != nil {
		t.Fatal(err)
	}
	if err := s.Load(bytes.NewReader(saved.Bytes())); err != nil {
		t.Fatal(err)
	}
	if got := s.Tokens(); !reflect.DeepEqual(got, []int{1, 2, 3}) {
		t.Fatalf("tokens after load = %v", got)
	}
	if got := s.Logits(); !reflect.DeepEqual(got, []float32{3}) {
		t.Fatalf("logits after load = %v", got)
	}
	if stop, err := s.Accept(4); stop || err != nil {
		t.Fatalf("Accept after load = (%v, %v)", stop, err)
	}
}

func TestSessionSelfOracleRoundTripContinuation(t *testing.T) {
	b := sessionBackend{}
	prompt := make([]int, 64)
	for i := range prompt {
		prompt[i] = i + 1
	}
	original, err := NewSession(b, 256, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := original.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	var saved bytes.Buffer
	if err := original.Save(&saved); err != nil {
		t.Fatal(err)
	}
	restored, err := NewSession(b, 256, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Load(bytes.NewReader(saved.Bytes())); err != nil {
		t.Fatal(err)
	}
	for _, token := range []int{1001, 1002, 1003, 1004} {
		stop, err := original.Accept(token)
		if stop || err != nil {
			t.Fatalf("original Accept(%d) = (%v, %v)", token, stop, err)
		}
		stop, err = restored.Accept(token)
		if stop || err != nil {
			t.Fatalf("restored Accept(%d) = (%v, %v)", token, stop, err)
		}
	}
	if !reflect.DeepEqual(original.Tokens(), restored.Tokens()) ||
		!reflect.DeepEqual(original.Logits(), restored.Logits()) {
		t.Fatalf("restored continuation diverged: tokens=%v/%v logits=%v/%v",
			original.Tokens(), restored.Tokens(), original.Logits(), restored.Logits())
	}
}

func TestSessionFormatSelfConsistency(t *testing.T) {
	b := sessionBackend{}
	s, err := NewSession(b, 128, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	var first bytes.Buffer
	if err := s.Save(&first); err != nil {
		t.Fatal(err)
	}
	loaded, err := NewSession(b, 128, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := loaded.Load(bytes.NewReader(first.Bytes())); err != nil {
		t.Fatal(err)
	}
	var second bytes.Buffer
	if err := loaded.Save(&second); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatal("session save-load-save changed the bytes")
	}
}

func TestSessionFormatMutationFailsContinuation(t *testing.T) {
	b := sessionBackend{}
	s, err := NewSession(b, 128, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	var saved bytes.Buffer
	if err := s.Save(&saved); err != nil {
		t.Fatal(err)
	}
	data := append([]byte(nil), saved.Bytes()...)
	// The logits follow the fixed session header and token timeline. A changed
	// logit is accepted structurally but must fail the bit-exact self-oracle.
	logitOffset := sessionHeaderBytes + 3*4
	data[logitOffset] ^= 1
	mutated, err := NewSession(b, 128, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := mutated.Load(bytes.NewReader(data)); err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(mutated.Logits(), s.Logits()) {
		t.Fatal("mutated session unexpectedly passed the bit-exact gate")
	}
}

func TestSessionSplitInvariance(t *testing.T) {
	b := sessionBackend{}
	prompt := make([]int, 2048)
	for i := range prompt {
		prompt[i] = i + 10
	}
	full, err := NewSession(b, 4096, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := full.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	split, err := NewSession(b, 4096, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := split.Sync(prompt[:1024]); err != nil {
		t.Fatal(err)
	}
	if err := split.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(full.Tokens(), split.Tokens()) ||
		!reflect.DeepEqual(full.Logits(), split.Logits()) {
		t.Fatal("aligned split changed the session result")
	}
	var fullBytes, splitBytes bytes.Buffer
	if err := full.Save(&fullBytes); err != nil {
		t.Fatal(err)
	}
	if err := split.Save(&splitBytes); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(fullBytes.Bytes(), splitBytes.Bytes()) {
		t.Fatal("aligned split changed the serialized state")
	}
}

func TestSessionAligned2048SplitInvariance(t *testing.T) {
	b := sessionBackend{}
	prompt := make([]int, 4096)
	for i := range prompt {
		prompt[i] = i + 100
	}
	full, err := NewSession(b, 8192, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := full.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	split, err := NewSession(b, 8192, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := split.Sync(prompt[:2048]); err != nil {
		t.Fatal(err)
	}
	if err := split.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(full.Tokens(), split.Tokens()) ||
		!reflect.DeepEqual(full.Logits(), split.Logits()) {
		t.Fatal("2048-token split changed the session result")
	}
	var fullBytes, splitBytes bytes.Buffer
	if err := full.Save(&fullBytes); err != nil {
		t.Fatal(err)
	}
	if err := split.Save(&splitBytes); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(fullBytes.Bytes(), splitBytes.Bytes()) {
		t.Fatal("2048-token split changed the serialized state")
	}
}

func TestSessionLoadRejectsTrailingBytes(t *testing.T) {
	b := sessionBackend{}
	s, err := NewSession(b, 8, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{1}); err != nil {
		t.Fatal(err)
	}
	var saved bytes.Buffer
	if err := s.Save(&saved); err != nil {
		t.Fatal(err)
	}
	saved.WriteByte(1)
	if err := s.Load(bytes.NewReader(saved.Bytes())); !errors.Is(err, errSessionFormat) {
		t.Fatalf("Load error = %v, want errSessionFormat", err)
	}
	if got := s.Tokens(); !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("tokens after rejected load = %v", got)
	}
}

type zeroSessionWriter struct{}

func (zeroSessionWriter) Write([]byte) (int, error) { return 0, nil }

func TestSessionSaveRejectsShortWrites(t *testing.T) {
	s, err := NewSession(&sessionBackend{}, 16, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{1, 2}); err != nil {
		t.Fatal(err)
	}
	var dst zeroSessionWriter
	if err := s.Save(&dst); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("Save error = %v, want io.ErrShortWrite", err)
	}
}

func TestSessionLoadMutationGate(t *testing.T) {
	b := sessionBackend{}
	source, err := NewSession(b, 128, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := source.Sync([]int{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	var saved bytes.Buffer
	if err := source.Save(&saved); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		mutate func([]byte)
	}{
		{name: "magic", mutate: func(data []byte) { data[0] ^= 1 }},
		{name: "version", mutate: func(data []byte) { data[8]++ }},
		{name: "context", mutate: func(data []byte) { binary.LittleEndian.PutUint32(data[12:16], 127) }},
		{name: "token count", mutate: func(data []byte) { binary.LittleEndian.PutUint32(data[16:20], 128) }},
		{name: "logit count", mutate: func(data []byte) { binary.LittleEndian.PutUint32(data[20:24], maxSessionLogits+1) }},
		{name: "state length", mutate: func(data []byte) { binary.LittleEndian.PutUint64(data[24:32], 0) }},
		{name: "truncated", mutate: func(data []byte) { data = data[:len(data)-1] }},
		{name: "trailing", mutate: func(data []byte) { data = append(data, 1) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := append([]byte(nil), saved.Bytes()...)
			if tt.name == "truncated" {
				data = data[:len(data)-1]
			} else if tt.name == "trailing" {
				data = append(data, 1)
			} else {
				tt.mutate(data)
			}
			destination, err := NewSession(b, 128, NewStopSet(99))
			if err != nil {
				t.Fatal(err)
			}
			if err := destination.Sync([]int{9}); err != nil {
				t.Fatal(err)
			}
			if err := destination.Load(bytes.NewReader(data)); err == nil {
				t.Fatal("Load accepted a mutated session")
			}
			if got := destination.Tokens(); !reflect.DeepEqual(got, []int{9}) {
				t.Fatalf("tokens changed after rejected load: %v", got)
			}
			if got := destination.Logits(); !reflect.DeepEqual(got, []float32{9}) {
				t.Fatalf("logits changed after rejected load: %v", got)
			}
		})
	}
}

func TestStateLoadFailurePreservesState(t *testing.T) {
	state := &sessionState{values: []int{7, 11, 13}}
	var saved bytes.Buffer
	if err := state.Save(&saved); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		data []byte
	}{
		{name: "short header", data: []byte{3}},
		{name: "truncated value", data: []byte{2, 0, 0, 0, 7, 0, 0, 0}},
		{name: "trailing bytes", data: []byte{0, 0, 0, 0, 1}},
		{name: "huge count", data: []byte{0xff, 0xff, 0xff, 0xff}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := state.Load(bytes.NewReader(tt.data)); err == nil {
				t.Fatal("Load accepted malformed state")
			}
			var after bytes.Buffer
			if err := state.Save(&after); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(saved.Bytes(), after.Bytes()) {
				t.Fatalf("failed Load changed state: before=%x after=%x", saved.Bytes(), after.Bytes())
			}
		})
	}
}
