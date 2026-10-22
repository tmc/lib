package dsv4

import (
	"errors"
	"reflect"
	"strconv"
	"testing"
)

type speculativeSnapshot struct{ values []int }

func (s *speculativeSnapshot) Close() error { return nil }

type speculativeVerifier struct {
	logits     []float32
	values     []int
	tops       []int
	baseLen    int
	snapshots  int
	restores   int
	verifyErr  error
	commitErr  error
	restoreErr error
}

func (v *speculativeVerifier) CurrentLogits() []float32 { return v.logits }
func (v *speculativeVerifier) Snapshot() (Snapshot, error) {
	v.snapshots++
	return &speculativeSnapshot{values: append([]int(nil), v.values...)}, nil
}
func (v *speculativeVerifier) Restore(snapshot Snapshot) error {
	s, ok := snapshot.(*speculativeSnapshot)
	if !ok {
		return errors.New("foreign snapshot")
	}
	v.restores++
	if v.restoreErr != nil {
		return v.restoreErr
	}
	v.values = append(v.values[:0], s.values...)
	return nil
}
func (v *speculativeVerifier) VerifyBatch(_ int, drafts []int, _ int) ([]int, error) {
	v.baseLen = len(v.values)
	v.values = append(v.values, drafts...)
	if v.verifyErr != nil {
		return nil, v.verifyErr
	}
	return append([]int(nil), v.tops...), nil
}
func (v *speculativeVerifier) CommitPrefix(n int) error {
	if v.commitErr != nil {
		return v.commitErr
	}
	if n < 0 || v.baseLen+n > len(v.values) {
		return errors.New("invalid prefix")
	}
	v.values = v.values[:v.baseLen+n]
	return nil
}

func TestSpeculatorRestoresWhenCommitFails(t *testing.T) {
	commitErr := errors.New("commit failed")
	v := &speculativeVerifier{
		logits:    []float32{0, 1},
		values:    []int{8},
		tops:      []int{1},
		commitErr: commitErr,
	}
	s, err := NewSpeculator(v, fixedDrafter{1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Verify(8, 1, 1); !errors.Is(err, commitErr) {
		t.Fatalf("Verify error = %v, want %v", err, commitErr)
	}
	if v.restores != 1 || !reflect.DeepEqual(v.values, []int{8}) {
		t.Fatalf("commit failure did not restore frontier: restores=%d values=%v", v.restores, v.values)
	}
}

type fixedDrafter []int

func (d fixedDrafter) Draft(int, int, int) ([]int, error) { return append([]int(nil), d...), nil }

func TestSpeculatorGreedyAcceptance(t *testing.T) {
	v := &speculativeVerifier{
		logits: []float32{1, 9, 9, 2},
		values: []int{10},
		tops:   []int{1, 2, 4},
	}
	s, err := NewSpeculator(v, fixedDrafter{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	got, err := s.Verify(10, 11, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := SpeculativeResult{Drafts: []int{1, 2, 3}, Tops: []int{1, 2, 4}, Accepted: 2, Committed: 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("result = %+v, want %+v", got, want)
	}
	if !reflect.DeepEqual(v.values, []int{10, 1, 2}) {
		t.Fatalf("committed values = %v, want [10 1 2]", v.values)
	}
	if v.snapshots != 1 || v.restores != 0 {
		t.Fatalf("snapshot/restore = %d/%d, want 1/0", v.snapshots, v.restores)
	}
}

func TestSpeculatorCommitPrefixMatchesEveryScalarPrefix(t *testing.T) {
	seed := []int{10, 20}
	drafts := []int{1, 2, 3, 4, 5, 6}
	for accepted := 0; accepted <= len(drafts); accepted++ {
		t.Run("accepted-"+strconv.Itoa(accepted), func(t *testing.T) {
			tops := append([]int(nil), drafts...)
			if accepted < len(tops) {
				tops[accepted] = 99
			}
			v := &speculativeVerifier{
				logits: []float32{1, 9},
				values: append([]int(nil), seed...),
				tops:   tops,
			}
			s, err := NewSpeculatorWithLimits(v, fixedDrafter(drafts), SpeculativeLimits{MaxDrafts: len(drafts)})
			if err != nil {
				t.Fatal(err)
			}
			got, err := s.Verify(seed[len(seed)-1], len(seed), len(drafts))
			if err != nil {
				t.Fatal(err)
			}
			wantValues := append(append([]int(nil), seed...), drafts[:accepted]...)
			if !reflect.DeepEqual(v.values, wantValues) {
				t.Fatalf("values = %v, want %v", v.values, wantValues)
			}
			if got.Accepted != accepted || got.Committed != accepted {
				t.Fatalf("result = %+v, want accepted/committed %d", got, accepted)
			}
			if accepted == 0 && v.restores != 1 {
				t.Fatalf("rejected prefix restore count = %d, want 1", v.restores)
			}
			if accepted > 0 && v.restores != 0 {
				t.Fatalf("accepted prefix restore count = %d, want 0", v.restores)
			}
		})
	}
}

func TestSpeculatorRejectRestoresSnapshot(t *testing.T) {
	v := &speculativeVerifier{
		logits: []float32{4, 7},
		values: []int{8, 9},
		tops:   []int{3, 3},
	}
	s, err := NewSpeculator(v, fixedDrafter{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	got, err := s.Verify(9, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got.Accepted != 0 || got.Committed != 0 {
		t.Fatalf("rejected result = %+v, want no committed drafts", got)
	}
	if !reflect.DeepEqual(v.values, []int{8, 9}) {
		t.Fatalf("values after rejection = %v, want [8 9]", v.values)
	}
	if v.snapshots != 1 || v.restores != 1 {
		t.Fatalf("snapshot/restore = %d/%d, want 1/1", v.snapshots, v.restores)
	}
}

func TestSpeculatorRejectsExtraVerifierRows(t *testing.T) {
	v := &speculativeVerifier{
		logits: []float32{0, 1},
		values: []int{8},
		tops:   []int{1, 2, 3},
	}
	s, err := NewSpeculator(v, fixedDrafter{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Verify(8, 1, 2); err == nil {
		t.Fatal("Verify accepted an extra verifier row")
	}
	if v.restores != 1 || !reflect.DeepEqual(v.values, []int{8}) {
		t.Fatalf("extra-row failure did not restore frontier: restores=%d values=%v", v.restores, v.values)
	}
}

func TestSpeculatorRestoresWhenVerifierFails(t *testing.T) {
	verifyErr := errors.New("device verification failed")
	v := &speculativeVerifier{
		logits:    []float32{0, 1},
		values:    []int{8},
		verifyErr: verifyErr,
	}
	s, err := NewSpeculator(v, fixedDrafter{1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Verify(8, 1, 1); !errors.Is(err, verifyErr) {
		t.Fatalf("Verify error = %v, want %v", err, verifyErr)
	}
	if v.restores != 1 || !reflect.DeepEqual(v.values, []int{8}) {
		t.Fatalf("verifier failure did not restore frontier: restores=%d values=%v", v.restores, v.values)
	}
}

func TestSpeculatorReportsDesynchronizedWhenRestoreFails(t *testing.T) {
	restoreErr := errors.New("restore failed")
	v := &speculativeVerifier{
		logits:     []float32{0, 1},
		values:     []int{8},
		tops:       []int{1, 2, 3},
		restoreErr: restoreErr,
	}
	s, err := NewSpeculatorWithLimits(v, fixedDrafter{1, 2}, SpeculativeLimits{MaxDrafts: 2})
	if err != nil {
		t.Fatal(err)
	}
	_, err = s.Verify(8, 1, 2)
	if !errors.Is(err, ErrDesynchronized) {
		t.Fatalf("Verify error = %v, want ErrDesynchronized", err)
	}
	if !errors.Is(err, restoreErr) {
		t.Fatalf("Verify error = %v, want restore error", err)
	}
}

func TestSpeculatorRejectsUnsupportedVerifier(t *testing.T) {
	if _, err := NewSpeculator(nil, fixedDrafter{1}); !errors.Is(err, ErrUnimplemented) {
		t.Fatalf("NewSpeculator(nil) = %v, want ErrUnimplemented", err)
	}
	if _, err := NewSpeculatorForBackend(contractBackend{}, contractState{}, fixedDrafter{1}); !errors.Is(err, ErrUnimplemented) {
		t.Fatalf("unsupported backend = %v, want ErrUnimplemented", err)
	}
	var _ Snapshot = (*speculativeSnapshot)(nil)
}

func TestSpeculatorBoundsDraftsAndTokenIDs(t *testing.T) {
	v := &speculativeVerifier{logits: []float32{1, 2}, values: []int{8}, tops: []int{1}}
	if _, err := NewSpeculatorWithLimits(v, fixedDrafter{1}, SpeculativeLimits{}); err == nil {
		t.Fatal("accepted an empty speculative limit")
	}
	s, err := NewSpeculatorWithLimits(v, fixedDrafter{1}, SpeculativeLimits{MaxDrafts: 1, VocabSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Verify(8, 1, 2); err == nil {
		t.Fatal("accepted a draft limit above the configured bound")
	}
	negative, err := NewSpeculatorWithLimits(v, fixedDrafter{-1}, SpeculativeLimits{MaxDrafts: 1, VocabSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := negative.Verify(8, 1, 1); err == nil {
		t.Fatal("accepted a negative draft token")
	}
	badTop := &speculativeVerifier{logits: []float32{1, 2}, values: []int{8}, tops: []int{2}}
	s, err = NewSpeculatorWithLimits(badTop, fixedDrafter{1}, SpeculativeLimits{MaxDrafts: 1, VocabSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Verify(8, 1, 1); err == nil {
		t.Fatal("accepted a verifier token outside the vocabulary")
	}
	if !reflect.DeepEqual(badTop.values, []int{8}) || badTop.restores != 1 {
		t.Fatalf("invalid verifier result did not restore frontier: values=%v restores=%d", badTop.values, badTop.restores)
	}
}
