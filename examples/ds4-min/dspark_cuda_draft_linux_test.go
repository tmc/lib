//go:build linux

package main

import (
	"strings"
	"testing"
)

// These metadata-only fixtures do not run a support/target forward and cannot
// count as native proposal or accepted-draft receipts.
func dsparkDraftGuardFixture(t *testing.T) *cudaDSparkDrafter {
	t.Helper()
	s := &cudaRequestState{runtime: &cudaRuntime{pos: 3}, generation: 3, lifecycleEpoch: 7}
	j, err := newDSparkFeatureJournal(8, 1)
	if err != nil {
		t.Fatal(err)
	}
	for position, token := range []int{1, 2, 3} {
		if err := j.append(position, token, 1); err != nil {
			t.Fatal(err)
		}
	}
	m := &dsparkModel{sha256: strings.Repeat("a", 64), block: 5}
	f := &cudaDSparkFeatures{request: s, r: s.runtime, published: true, position: 2, token: 3, generation: 2, modelSHA: m.sha256}
	f.history = &cudaDSparkHistory{owner: f, journal: j}
	s.dspark = f
	d := &cudaDSparkDrafter{forward: &cudaDSparkForward{target: s, weights: &cudaDSparkWeights{model: m}}, verifier: &cudaSpeculativeVerifier{state: s}}
	if err := d.prepare([]int{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	return d
}

func TestDSparkNativePreparedHistory(t *testing.T) {
	d := dsparkDraftGuardFixture(t)
	if token, pos, err := d.PreparedSeed(); err != nil || token != 3 || pos != 3 {
		t.Fatalf("seed %d/%d %v", token, pos, err)
	}
	if err := d.prepare([]int{99, 2, 3}); err == nil || !strings.Contains(err.Error(), "history mismatch") {
		t.Fatalf("changed prompt admitted: %v", err)
	}
	d.forward.target.lifecycleEpoch++ // same history/position, later Restore
	if _, _, err := d.PreparedSeed(); err == nil || !strings.Contains(err.Error(), "stale prepared") {
		t.Fatalf("old preparation admitted: %v", err)
	}
}

func TestDSparkNativeDeclineRetry(t *testing.T) {
	for _, pending := range [][]int{nil, {99}} {
		d := dsparkDraftGuardFixture(t)
		d.pending, d.drafted = pending, true
		if err := d.DiscardDrafts(); err != nil || d.drafted || len(d.pending) != 0 {
			t.Fatalf("discard lookahead: %v", err)
		}
		if token, position, err := d.PreparedSeed(); err != nil || token != 3 || position != 3 {
			t.Fatalf("free decline lost prepared frontier: %d/%d/%v", token, position, err)
		}
		if err := d.DiscardDrafts(); err != nil || !d.prepared {
			t.Fatalf("idempotent decline lost preparation: %v", err)
		}
		d.forward.target.lifecycleEpoch++
		if err := d.DiscardDrafts(); err != nil || d.prepared || len(d.history) != 0 || d.hash != "" {
			t.Fatalf("changed target retained stale preparation: %v", err)
		}
	}
}

func TestDSparkNativeCommitLifecycle(t *testing.T) {
	for _, kind := range []string{"valid", "foreign-epoch", "wrong-prefix", "wrong-position"} {
		t.Run(kind, func(t *testing.T) {
			d := dsparkDraftGuardFixture(t)
			d.pending, d.drafted = []int{10, 20}, true
			s := d.forward.target
			s.runtime.pos, s.lifecycleEpoch = 4, 11
			s.dspark.token, s.dspark.position = 10, 3
			if err := s.dspark.history.journal.append(3, 10, 1); err != nil {
				t.Fatal(err)
			}
			want := ""
			switch kind {
			case "foreign-epoch":
				s.lifecycleEpoch++
				want = "foreign target lifecycle"
			case "wrong-prefix":
				s.dspark.history.journal.prefix[0] ^= 1
				want = "accepted target history mismatch"
			case "wrong-position":
				s.runtime.pos++
				want = "foreign target lifecycle"
			}
			err := d.CommitDrafts(1)
			if want == "" {
				if err != nil || d.prepared || d.drafted {
					t.Fatalf("commit did not clear temporary proposal: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("error=%v want %q", err, want)
			}
		})
	}
}
