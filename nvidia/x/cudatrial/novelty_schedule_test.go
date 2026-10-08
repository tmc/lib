package cudatrial

import (
	"crypto/sha256"
	"sync"
	"testing"
)

// mockFilter is an in-memory unique trace filter for testing.
type mockFilter struct {
	mu   sync.Mutex
	seen map[[32]byte]bool
}

func (m *mockFilter) Evaluate(raw []byte) (bool, error) {
	any := false
	for _, b := range raw {
		if b != 0 {
			any = true
			break
		}
	}
	if !any {
		return false, nil
	}
	h := sha256.Sum256(raw)
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.seen == nil {
		m.seen = make(map[[32]byte]bool)
	}
	if m.seen[h] {
		return false, nil
	}
	m.seen[h] = true
	return true, nil
}

func TestNoveltyScheduler_FiltersDuplicateTrials(t *testing.T) {
	filter := &mockFilter{}
	sched, err := NewNoveltyScheduler(filter)
	if err != nil {
		t.Fatal(err)
	}

	b1, _ := NewEdgeBitmap(32)
	b1.counts[16] = 1
	r1 := Receipt{Name: "trial1", Coverage: b1}

	b2, _ := NewEdgeBitmap(32)
	b2.counts[16] = 1 // exact same trace
	r2 := Receipt{Name: "trial2", Coverage: b2}

	admitted1, err := sched.ScheduleTrial(r1)
	if err != nil || !admitted1 {
		t.Fatalf("first trial should be admitted: %v, err=%v", admitted1, err)
	}

	admitted2, err := sched.ScheduleTrial(r2)
	if err != nil || admitted2 {
		t.Fatalf("duplicate trial should be rejected: %v, err=%v", admitted2, err)
	}

	stats := sched.Stats()
	if stats.EvaluatedTrials != 2 || stats.NovelTrials != 1 || stats.DuplicateTrials != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestNoveltyScheduler_AdmitsNovelEdge(t *testing.T) {
	filter := &mockFilter{}
	sched, err := NewNoveltyScheduler(filter)
	if err != nil {
		t.Fatal(err)
	}

	b1, _ := NewEdgeBitmap(32)
	b1.counts[16] = 1
	r1 := Receipt{Name: "trial1", Coverage: b1}

	b2, _ := NewEdgeBitmap(32)
	b2.counts[17] = 1 // new edge
	r2 := Receipt{Name: "trial2", Coverage: b2}

	novel, err := sched.FilterNovel([]Receipt{r1, r1, r2, r1})
	if err != nil {
		t.Fatal(err)
	}
	if len(novel) != 2 {
		t.Fatalf("expected 2 novel receipts, got %d", len(novel))
	}
	if novel[0].Name != "trial1" || novel[1].Name != "trial2" {
		t.Fatalf("unexpected filtered receipts: %+v", novel)
	}
}

func TestNoveltyScheduler_Errors(t *testing.T) {
	if _, err := NewNoveltyScheduler(nil); err == nil {
		t.Fatal("expected error with nil filter")
	}

	sched, _ := NewNoveltyScheduler(&mockFilter{})
	// Receipt without coverage
	if _, err := sched.ScheduleTrial(Receipt{Name: "no-cov"}); err == nil {
		t.Fatal("expected error for receipt without coverage")
	}
}
