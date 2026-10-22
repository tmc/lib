package dsv4

import (
	"bytes"
	"io"
	"testing"
)

type consistencyBackend struct{ next int }

type consistencyState struct {
	mode   int
	values []int
}

func (b *consistencyBackend) NewState(int) (State, error) {
	b.next++
	return &consistencyState{mode: b.next - 1}, nil
}

func (b *consistencyBackend) Eval(state State, token, position int) error {
	s := state.(*consistencyState)
	if position != len(s.values) {
		return io.ErrUnexpectedEOF
	}
	s.values = append(s.values, token)
	return nil
}

func (b *consistencyBackend) Logits(state State) []float32 {
	s := state.(*consistencyState)
	if len(s.values) == 0 {
		return nil
	}
	last := float32(s.values[len(s.values)-1])
	if s.mode == 0 {
		return []float32{last, 0}
	}
	return []float32{0, last}
}

func (*consistencyBackend) Close() error { return nil }

func (s *consistencyState) Reset() error { s.values = nil; return nil }
func (s *consistencyState) Snapshot() (Snapshot, error) {
	return &consistencySnapshot{values: append([]int(nil), s.values...)}, nil
}
func (s *consistencyState) Restore(snapshot Snapshot) error {
	copyOf, ok := snapshot.(*consistencySnapshot)
	if !ok {
		return io.ErrUnexpectedEOF
	}
	s.values = append(s.values[:0], copyOf.values...)
	return nil
}
func (s *consistencyState) Save(w io.Writer) error {
	_, err := w.Write([]byte{byte(len(s.values))})
	return err
}
func (s *consistencyState) Load(r io.Reader) error {
	b, err := io.ReadAll(r)
	if err != nil || len(b) != 1 {
		return io.ErrUnexpectedEOF
	}
	s.values = make([]int, int(b[0]))
	return nil
}
func (*consistencyState) Close() error { return nil }

type consistencySnapshot struct{ values []int }

func (*consistencySnapshot) Close() error { return nil }

func TestCheckDecodeConsistencyPassesForIndependentStates(t *testing.T) {
	b := &sessionBackend{}
	report, err := CheckDecodeConsistency(b, 16, NewStopSet(99), []int{1, 2}, []int{3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	if report.Steps != 3 || report.Compared != 3 || report.MaxAbs != 0 || !report.Pass(0, 0) {
		t.Fatalf("report=%+v", report)
	}
}

func TestCheckDecodeConsistencyDetectsStateDivergence(t *testing.T) {
	report, err := CheckDecodeConsistency(&consistencyBackend{}, 16, NewStopSet(99), []int{1}, []int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if report.Steps != 2 || report.Top1Mismatches != 2 || report.MaxAbs == 0 || report.Pass(100, 100) {
		t.Fatalf("report=%+v", report)
	}
}

func TestCheckDecodeConsistencyRejectsInvalidInputs(t *testing.T) {
	tests := []struct {
		name      string
		backend   Backend
		context   int
		stop      StopSet
		prompt    []int
		generated []int
	}{
		{"nil backend", nil, 16, NewStopSet(99), []int{1}, []int{2}},
		{"empty prompt", &sessionBackend{}, 16, NewStopSet(99), nil, []int{2}},
		{"empty generation", &sessionBackend{}, 16, NewStopSet(99), []int{1}, nil},
		{"context overflow", &sessionBackend{}, 3, NewStopSet(99), []int{1, 2}, []int{3}},
		{"empty stop set", &sessionBackend{}, 16, nil, []int{1}, []int{2}},
		{"stop token in generation", &sessionBackend{}, 16, NewStopSet(99), []int{1}, []int{99}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := CheckDecodeConsistency(tt.backend, tt.context, tt.stop, tt.prompt, tt.generated)
			if err == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}

	// Keep the State contract in this test: the helper must not require a
	// concrete snapshot format or mutate an input buffer.
	var saved bytes.Buffer
	s, err := NewSession(&sessionBackend{}, 16, NewStopSet(99))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := s.Save(&saved); err != nil {
		t.Fatal(err)
	}
	if saved.Len() == 0 {
		t.Fatal("empty session snapshot")
	}
}
