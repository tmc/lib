package dsv4bench

import (
	"bytes"
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type memoryState struct {
	benchState
	storage            []byte
	failClose          bool
	snapshotCloseFault bool
	snap               *memorySnapshot
}
type memorySnapshot struct {
	benchSnapshot
	storage   []byte
	failClose bool
}

func (s *memoryState) MemoryStats() (dsv4.ContextMemory, error) {
	if s.closed {
		return dsv4.ContextMemory{}, dsv4.ErrClosed
	}
	return dsv4.ContextMemory{Raw: int64(cap(s.storage))}, nil
}
func (s *memorySnapshot) MemoryStats() (dsv4.ContextMemory, error) {
	if s.closed {
		return dsv4.ContextMemory{}, dsv4.ErrClosed
	}
	return dsv4.ContextMemory{Raw: int64(cap(s.storage))}, nil
}
func (s *memoryState) Snapshot() (dsv4.Snapshot, error) {
	s.snap = &memorySnapshot{benchSnapshot: benchSnapshot{tokens: append([]int(nil), s.tokens...)}, storage: make([]byte, 29), failClose: s.snapshotCloseFault}
	return s.snap, nil
}
func (s *memoryState) Restore(v dsv4.Snapshot) error {
	return s.benchState.Restore(&v.(*memorySnapshot).benchSnapshot)
}
func (s *memoryState) Close() error {
	if s.failClose {
		return errors.New("state close fault")
	}
	s.storage = nil
	return s.benchState.Close()
}
func (s *memorySnapshot) Close() error {
	if s.failClose {
		return errors.New("snapshot close fault")
	}
	s.storage = nil
	return s.benchSnapshot.Close()
}
func (s *memorySnapshot) SnapshotBytes() (int64, error) {
	if s.closed {
		return 0, dsv4.ErrClosed
	}
	return int64(len(s.tokens) * 4), nil
}

type memoryBackend struct {
	benchBackend
	state                               *memoryState
	stateCloseFault, snapshotCloseFault bool
}

func (b *memoryBackend) NewState(int) (dsv4.State, error) {
	b.state = &memoryState{storage: make([]byte, 0, 137), failClose: b.stateCloseFault, snapshotCloseFault: b.snapshotCloseFault}
	return b.state, nil
}
func (b *memoryBackend) Eval(s dsv4.State, token, pos int) error {
	return b.benchBackend.Eval(&s.(*memoryState).benchState, token, pos)
}
func (b *memoryBackend) Logits(s dsv4.State) []float32 {
	return b.benchBackend.Logits(&s.(*memoryState).benchState)
}

func TestMeasuredRunCSVAndRestore(t *testing.T) {
	cfg := runFixture()
	cfg.MeasureMemory = true
	result, err := Run(context.Background(), new(memoryBackend), cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i, m := range result.Measurements {
		if !m.MemoryObserved || m.Row.ContextRawBytes != 137 || m.Row.ContextTotalBytes != 137 || m.Row.PeakRSSBytes <= 0 {
			t.Fatalf("unmeasured row: %+v", m)
		}
		want := int64(0)
		if i == 0 {
			want = 8
		}
		if m.Row.KVCacheBytes != want || m.Row.PrefillTokens != 2 || m.Logits[1] != float32(2+i*2) {
			t.Fatalf("snapshot/restore accounting: %+v", m)
		}
	}
	var out bytes.Buffer
	if err := WriteMeasuredCSV(&out, result); err != nil {
		t.Fatal(err)
	}
	wantHeader := "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes,context_total_bytes,context_raw_bytes,context_compressed_bytes,context_scratch_bytes,peak_rss_bytes\n"
	if !strings.HasPrefix(out.String(), wantHeader) {
		t.Fatal("literal upstream header differs")
	}
	rows, err := ReadCSV(&out)
	if err != nil || len(rows) != 2 {
		t.Fatalf("CSV: %v", err)
	}
	result.Measurements[0].MemoryObserved = false
	if err := WriteMeasuredCSV(&out, result); err == nil {
		t.Fatal("unobserved metadata exported")
	}
	if _, err := Run(context.Background(), new(benchBackend), cfg); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("missing capability: %v", err)
	}
}

func TestPeakRSSUnits(t *testing.T) {
	for _, c := range []struct {
		value int64
		kib   bool
		want  int64
		bad   bool
	}{{17, true, 17408, false}, {17, false, 17, false}, {0, true, 0, false}, {-1, false, 0, true}, {math.MaxInt64, true, 0, true}} {
		got, err := normalizePeakRSS(c.value, c.kib)
		if (err != nil) != c.bad || !c.bad && got != c.want {
			t.Fatalf("RSS %+v: %d %v", c, got, err)
		}
	}
}

func TestCleanupRetainsAccounting(t *testing.T) {
	s := &memoryState{storage: make([]byte, 31), failClose: true}
	p := &memorySnapshot{storage: make([]byte, 29), failClose: true}
	e := &CleanupError{cause: errors.New("work failed"), state: s, snapshots: []dsv4.Snapshot{p}}
	if err := e.Close(); err == nil {
		t.Fatal("close fault hidden")
	}
	m, err := e.MemoryStats()
	if err != nil || m.Raw != 60 {
		t.Fatalf("retained owner bytes=%+v %v", m, err)
	}
	p.failClose = false
	if err := e.Close(); err == nil {
		t.Fatal("state fault hidden")
	}
	m, err = e.MemoryStats()
	if err != nil || m.Raw != 31 {
		t.Fatal("released snapshot still counted or retained state lost")
	}
	s.failClose = false
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	m, err = e.MemoryStats()
	if err != nil || m != (dsv4.ContextMemory{}) {
		t.Fatal("successful cleanup still counted")
	}
}

func TestRunReturnsRetainedCleanup(t *testing.T) {
	for _, snapshot := range []bool{false, true} {
		b := &memoryBackend{stateCloseFault: !snapshot, snapshotCloseFault: snapshot}
		got, err := Run(context.Background(), b, runFixture())
		var owner *CleanupError
		if !errors.As(err, &owner) || len(got.Measurements) != 0 {
			t.Fatalf("cleanup owner/result: %+v %v", got, err)
		}
		m, err := owner.MemoryStats()
		want := int64(137)
		if snapshot {
			want = 29
		}
		if err != nil || m.Raw != want {
			t.Fatalf("retained inventory: %+v %v", m, err)
		}
		b.state.failClose = false
		if b.state.snap != nil {
			b.state.snap.failClose = false
		}
		if err := owner.Close(); err != nil {
			t.Fatal(err)
		}
		if err := owner.Close(); err != nil {
			t.Fatal(err)
		}
		m, err = owner.MemoryStats()
		if err != nil || m != (dsv4.ContextMemory{}) {
			t.Fatal("released inventory remains")
		}
	}
}
