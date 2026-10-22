package dsv4

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

type preparedSnapshot struct {
	values []int
	hc     int
	owner  *preparedVerifier
}

func (s *preparedSnapshot) Close() error { s.owner.closes++; return s.owner.closeErr }

type preparedVerifier struct {
	speculativeVerifier
	hc                    int
	closes                int
	closeErr, snapshotErr error
	nilSnapshot           bool
}

func (v *preparedVerifier) Snapshot() (Snapshot, error) {
	v.snapshots++
	if v.nilSnapshot {
		return nil, v.snapshotErr
	}
	return &preparedSnapshot{append([]int(nil), v.values...), v.hc, v}, v.snapshotErr
}
func (v *preparedVerifier) Restore(snapshot Snapshot) error {
	v.restores++
	if v.restoreErr != nil {
		return v.restoreErr
	}
	s, ok := snapshot.(*preparedSnapshot)
	if !ok {
		return errors.New("foreign prepared snapshot")
	}
	v.values = append([]int(nil), s.values...)
	v.hc = s.hc
	return nil
}
func (v *preparedVerifier) VerifyBatch(seed int, drafts []int, pos int) ([]int, error) {
	tops, err := v.speculativeVerifier.VerifyBatch(seed, drafts, pos)
	if len(v.values) > 0 {
		v.hc = v.values[len(v.values)-1] * 10
	}
	return tops, err
}
func (v *preparedVerifier) CommitPrefix(n int) error {
	if err := v.speculativeVerifier.CommitPrefix(n); err != nil {
		return err
	}
	v.hc = v.values[len(v.values)-1] * 10
	return nil
}

type preparedDrafter struct {
	active, lookahead, proposal     []int
	seed, position                  int
	historyHash                     string
	drafts, commits, discards       int
	draftErr, commitErr, discardErr error
}

func (d *preparedDrafter) prepare(history []int, hc int) {
	d.seed, d.position = history[len(history)-1], len(history)
	d.historyHash, _ = SpeculativeHistorySHA256(history)
	d.active = append(d.active, hc)
}
func (d *preparedDrafter) Draft(seed, pos, max int) ([]int, error) {
	d.drafts++
	if seed != d.seed || pos != d.position {
		return nil, errors.New("prepared frontier mismatch")
	}
	d.lookahead = append([]int(nil), d.proposal...)
	return append([]int(nil), d.proposal...), d.draftErr
}

func (d *preparedDrafter) PreparedSeed() (int, int, error) { return d.seed, d.position, nil }
func (d *preparedDrafter) CommitDrafts(n int) error {
	d.commits++
	if d.commitErr != nil {
		return d.commitErr
	}
	d.active = append(d.active, d.lookahead[:n]...)
	d.lookahead = nil
	return nil
}
func (d *preparedDrafter) DiscardDrafts() error {
	d.discards++
	if d.discardErr != nil {
		return d.discardErr
	}
	d.lookahead = nil
	return nil
}

func preparedFixture(t *testing.T) (*Speculator, *preparedVerifier, *preparedDrafter) {
	t.Helper()
	v := &preparedVerifier{speculativeVerifier: speculativeVerifier{values: []int{8}, logits: []float32{0, 9, 0, 0, 0}, tops: []int{1, 2, 4}}, hc: 80}
	d := &preparedDrafter{proposal: []int{1, 2, 3}}
	d.prepare(v.values, v.hc)
	s, err := NewSpeculatorWithLimits(v, d, SpeculativeLimits{MaxDrafts: 4, VocabSize: 10})
	if err != nil {
		t.Fatal(err)
	}
	return s, v, d
}

func TestVerifyPreparedTwoFrames(t *testing.T) {
	s, v, d := preparedFixture(t)
	r, err := s.VerifyPrepared(8, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	if r.Declined || r.Accepted != 2 || r.Committed != 2 || !reflect.DeepEqual(v.values, []int{8, 1, 2}) || !reflect.DeepEqual(d.active, []int{80, 1, 2}) || v.hc != 20 {
		t.Fatalf("partial frame=%+v target=%v HC=%d support=%v", r, v.values, v.hc, d.active)
	}
	// Preparation receives fresh terminal target HC and the complete new
	// prefix. Reusing the support HC from the preceding frame is observable.
	d.prepare(v.values, v.hc)
	expectedHash, _ := SpeculativeHistorySHA256([]int{8, 1, 2})
	if d.historyHash != expectedHash || d.position != 3 || d.active[len(d.active)-1] != 20 {
		t.Fatal("second frame has stale HC/history")
	}
	d.proposal = []int{3, 4}
	v.logits = []float32{0, 0, 0, 9, 0}
	v.tops = []int{3, 4}
	r, err = s.VerifyPrepared(2, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if r.Accepted != 2 || r.Committed != 2 || !reflect.DeepEqual(v.values, []int{8, 1, 2, 3, 4}) || !reflect.DeepEqual(d.active, []int{80, 1, 2, 20, 3, 4}) || v.hc != 40 || len(d.lookahead) != 0 {
		t.Fatal("second frame diverged", r, v.values, v.hc, d.active)
	}
	if v.snapshots != 2 || v.closes != 2 || v.restores != 0 || d.commits != 2 || d.discards != 0 {
		t.Fatal("unexpected lifecycle callbacks")
	}
	r.Drafts[0] = 99
	r.Tops[0] = 99
	if v.values[3] != 3 || d.active[4] != 3 || d.proposal[0] != 3 {
		t.Fatal("result aliases active state or proposal")
	}
}

func TestVerifyPreparedDeclines(t *testing.T) {
	for _, tt := range []struct {
		name     string
		proposal []int
	}{{"empty", nil}, {"first miss", []int{2, 3}}} {
		t.Run(tt.name, func(t *testing.T) {
			s, v, d := preparedFixture(t)
			d.proposal = tt.proposal
			r, err := s.VerifyPrepared(8, 1, 3)
			if err != nil || !r.Declined || r.Committed != 0 || r.Accepted != 0 {
				t.Fatal("decline", r, err)
			}
			if v.snapshots != 0 || v.closes != 0 || v.restores != 0 || !reflect.DeepEqual(v.values, []int{8}) || v.hc != 80 || !reflect.DeepEqual(d.active, []int{80}) || len(d.lookahead) != 0 || d.discards != 1 {
				t.Fatal("decline mutated active frontier")
			}
			// A normal target evaluation follows the decline, then a fresh prepare.
			v.values = append(v.values, 1)
			v.hc = 10
			d.prepare(v.values, v.hc)
			d.proposal = []int{1}
			v.tops = []int{1}
			if _, err := s.VerifyPrepared(1, 2, 1); err != nil {
				t.Fatal("decline poisoned ordinary continuation", err)
			}
		})
	}
}

func TestVerifyPreparedFailures(t *testing.T) {
	primary := errors.New("primary failure")
	cleanup := errors.New("cleanup failure")
	restore := errors.New("restore failure")
	closeErr := errors.New("close failure")
	for _, tt := range []struct {
		name             string
		configure        func(*preparedVerifier, *preparedDrafter)
		causes           []error
		poison           bool
		restores, closes int
	}{
		{"draft", func(v *preparedVerifier, d *preparedDrafter) { d.draftErr = primary }, []error{primary}, false, 0, 0},
		{"draft poison", func(v *preparedVerifier, d *preparedDrafter) { d.draftErr = errors.Join(primary, ErrDesynchronized) }, []error{primary}, true, 0, 0},
		{"long proposal", func(v *preparedVerifier, d *preparedDrafter) { d.proposal = []int{1, 2, 3, 4} }, nil, false, 0, 0},
		{"invalid proposal", func(v *preparedVerifier, d *preparedDrafter) { d.proposal = []int{1, -1} }, nil, false, 0, 0},
		{"snapshot", func(v *preparedVerifier, d *preparedDrafter) { v.snapshotErr = primary; v.nilSnapshot = true }, []error{primary}, false, 0, 0},
		{"snapshot with resource", func(v *preparedVerifier, d *preparedDrafter) { v.snapshotErr = primary }, []error{primary}, false, 0, 1},
		{"nil snapshot", func(v *preparedVerifier, d *preparedDrafter) { v.nilSnapshot = true }, nil, false, 0, 0},
		{"batch", func(v *preparedVerifier, d *preparedDrafter) { v.verifyErr = primary }, []error{primary}, false, 1, 1},
		{"row count", func(v *preparedVerifier, d *preparedDrafter) { v.tops = []int{1} }, nil, false, 1, 1},
		{"row token", func(v *preparedVerifier, d *preparedDrafter) { v.tops = []int{1, 2, -1} }, nil, false, 1, 1},
		{"target commit", func(v *preparedVerifier, d *preparedDrafter) { v.commitErr = primary }, []error{primary}, false, 1, 1},
		{"support commit", func(v *preparedVerifier, d *preparedDrafter) { d.commitErr = primary }, []error{primary}, false, 1, 1},
		{"restore", func(v *preparedVerifier, d *preparedDrafter) { v.verifyErr = primary; v.restoreErr = restore }, []error{primary, restore}, true, 1, 1},
		{"discard", func(v *preparedVerifier, d *preparedDrafter) { v.verifyErr = primary; d.discardErr = cleanup }, []error{primary, cleanup}, true, 1, 1},
		{"all cleanup", func(v *preparedVerifier, d *preparedDrafter) {
			v.verifyErr = primary
			v.restoreErr = restore
			d.discardErr = cleanup
			v.closeErr = closeErr
		}, []error{primary, restore, cleanup, closeErr}, true, 1, 1},
		{"snapshot close", func(v *preparedVerifier, d *preparedDrafter) { v.snapshotErr = primary; v.closeErr = closeErr }, []error{primary, closeErr}, true, 0, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s, v, d := preparedFixture(t)
			tt.configure(v, d)
			r, err := s.VerifyPrepared(8, 1, 3)
			if err == nil || !reflect.DeepEqual(r, PreparedResult{}) {
				t.Fatal("failed cycle published success", r, err)
			}
			for _, cause := range tt.causes {
				if !errors.Is(err, cause) {
					t.Fatalf("error=%v missing %v", err, cause)
				}
			}
			if errors.Is(err, ErrDesynchronized) != tt.poison || v.restores != tt.restores || v.closes != tt.closes || d.discards != 1 {
				t.Fatalf("cleanup mismatch err=%v restore=%d close=%d discard=%d", err, v.restores, v.closes, d.discards)
			}
			if v.restoreErr == nil && (!reflect.DeepEqual(v.values, []int{8}) || v.hc != 80) {
				t.Fatal("target rollback lost complete frontier")
			}
			if !reflect.DeepEqual(d.active, []int{80}) {
				t.Fatal("failed operation changed active seed")
			}
			if d.discardErr == nil && len(d.lookahead) != 0 {
				t.Fatal("failed cycle leaked lookahead")
			}
			if tt.poison {
				drafts := d.drafts
				snapshots := v.snapshots
				if _, again := s.VerifyPrepared(8, 1, 3); !errors.Is(again, ErrDesynchronized) || d.drafts != drafts || v.snapshots != snapshots {
					t.Fatal("poisoned request reused")
				}
				if _, again := s.Verify(8, 1, 3); !errors.Is(again, ErrDesynchronized) {
					t.Fatal("legacy entry point bypassed poison")
				}
			} else {
				// After rollback, ordinary target evaluation and fresh
				// preparation can begin the next frame without stale HC.
				v.verifyErr, v.commitErr, v.snapshotErr = nil, nil, nil
				v.nilSnapshot = false
				d.draftErr, d.commitErr = nil, nil
				v.values = append(v.values, 1)
				v.hc = 10
				v.tops = []int{1}
				d.prepare(v.values, v.hc)
				d.proposal = []int{1}
				r, err := s.VerifyPrepared(1, 2, 1)
				if err != nil || r.Committed != 1 || !reflect.DeepEqual(v.values, []int{8, 1, 1}) || !reflect.DeepEqual(d.active, []int{80, 10, 1}) {
					t.Fatal("rollback broke next frame", r, err, v.values, d.active)
				}
			}
		})
	}
}

func TestVerifyPreparedCloseAfterCommit(t *testing.T) {
	s, v, d := preparedFixture(t)
	closeErr := errors.New("close failed")
	v.closeErr = closeErr
	r, err := s.VerifyPrepared(8, 1, 3)
	if !errors.Is(err, closeErr) || !errors.Is(err, ErrDesynchronized) || !reflect.DeepEqual(r, PreparedResult{}) || v.closes != 1 || d.commits != 1 || d.discards != 0 {
		t.Fatal("close failure was lost", r, err)
	}
	if !reflect.DeepEqual(v.values, []int{8, 1, 2}) || !reflect.DeepEqual(d.active, []int{80, 1, 2}) {
		t.Fatal("close error hid inconsistent dual commit")
	}
}

func TestVerifyPreparedDeclineCleanupFailure(t *testing.T) {
	s, v, d := preparedFixture(t)
	d.proposal = nil
	d.discardErr = errors.New("discard failed")
	r, err := s.VerifyPrepared(8, 1, 3)
	if !errors.Is(err, ErrDesynchronized) || !errors.Is(err, d.discardErr) || !reflect.DeepEqual(r, PreparedResult{}) || v.snapshots != 0 || d.discards != 1 {
		t.Fatal("cleanup failure published decline", r, err)
	}
}

func TestVerifyPreparedRequiresLifecycle(t *testing.T) {
	v := &speculativeVerifier{logits: []float32{0, 1}}
	s, _ := NewSpeculator(v, fixedDrafter{1})
	if _, err := s.VerifyPrepared(8, 1, 1); !errors.Is(err, ErrUnimplemented) || v.snapshots != 0 {
		t.Fatal("accepted drafter without lifecycle")
	}
}

func TestVerifyPreparedInvalidInputPreservesPreparation(t *testing.T) {
	for _, tt := range []struct {
		name           string
		seed, pos, max int
	}{{"zero max", 8, 1, 0}, {"large max", 8, 1, 5}, {"negative seed", -1, 1, 3}, {"large seed", 10, 1, 3}, {"negative position", 8, -1, 3}, {"wrong seed", 9, 1, 3}, {"stale position", 8, 0, 3}} {
		t.Run(tt.name, func(t *testing.T) {
			s, v, d := preparedFixture(t)
			if _, err := s.VerifyPrepared(tt.seed, tt.pos, tt.max); err == nil {
				t.Fatal("accepted invalid prepared input")
			}
			if d.drafts != 0 || d.discards != 0 || d.commits != 0 || v.snapshots != 0 || !reflect.DeepEqual(d.active, []int{80}) || !reflect.DeepEqual(v.values, []int{8}) {
				t.Fatal("invalid input consumed preparation")
			}
			if _, err := s.VerifyPrepared(8, 1, 3); err != nil {
				t.Fatal("valid retry cannot reuse preserved preparation", err)
			}
		})
	}
}

func TestVerifyPreparedReportsAllNonfiniteLogits(t *testing.T) {
	s, v, d := preparedFixture(t)
	v.logits = []float32{0, float32(math.NaN()), 2, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	r, err := s.VerifyPrepared(8, 1, 3)
	if err == nil || !strings.Contains(err.Error(), "4 nonfinite values, first at index 1") || !reflect.DeepEqual(r, PreparedResult{}) || v.snapshots != 0 || d.discards != 1 {
		t.Fatal("incomplete nonfinite diagnostic", r, err)
	}
}

func TestVerifyCloseFailureIsReported(t *testing.T) {
	s, v, _ := preparedFixture(t)
	v.closeErr = errors.New("close failed")
	if _, err := s.Verify(8, 1, 3); !errors.Is(err, v.closeErr) || !errors.Is(err, ErrDesynchronized) || v.closes != 1 {
		t.Fatal("Verify ignored snapshot close failure", err)
	}
}

func ExampleDraftLifecycle() {
	d := &preparedDrafter{active: []int{80}, lookahead: []int{1, 2}}
	var lifecycle DraftLifecycle = d
	_ = lifecycle.DiscardDrafts()
	fmt.Println(d.active, len(d.lookahead))
	// Output: [80] 0
}

func ExamplePreparedResult() {
	r := PreparedResult{Declined: true}
	fmt.Println(r.Declined, r.Committed)
	// Output: true 0
}

func ExampleSpeculator_VerifyPrepared() {
	v := &preparedVerifier{speculativeVerifier: speculativeVerifier{values: []int{8}, logits: []float32{0, 1}, tops: []int{1}}, hc: 80}
	d := &preparedDrafter{proposal: []int{1}}
	d.prepare(v.values, v.hc)
	s, _ := NewSpeculator(v, d)
	r, err := s.VerifyPrepared(8, 1, 1)
	fmt.Println(r.Declined, r.Committed, err)
	// Output: false 1 <nil>
}
