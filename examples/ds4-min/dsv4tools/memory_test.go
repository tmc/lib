package dsv4tools

import (
	"errors"
	"regexp"
	"strings"
	"sync"
	"testing"
)

func TestMemoryStoresExactDSMLAndGeneratesIDs(t *testing.T) {
	m := NewMemory(2)
	const want = `<｜DSML｜tool_calls><｜DSML｜invoke name="x">\n\t<｜DSML｜parameter name="a">1</｜DSML｜parameter>\n</｜DSML｜invoke></｜DSML｜tool_calls>`
	id, err := m.Store(want)
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`^call_[0-9a-f]{32}$`).MatchString(id) {
		t.Fatalf("id = %q", id)
	}
	got, ok := m.Lookup(id)
	if !ok || got != want {
		t.Fatalf("Lookup = %q, %v; want exact %q", got, ok, want)
	}
}

func TestMemoryLRUAndDelete(t *testing.T) {
	m := NewMemory(2)
	a, _ := m.Store("a")
	b, _ := m.Store("b")
	if _, ok := m.Lookup(a); !ok {
		t.Fatal("a missing before eviction")
	}
	c, _ := m.Store("c")
	if _, ok := m.Lookup(b); ok {
		t.Fatal("least-recently-used entry b was retained")
	}
	if _, ok := m.Lookup(a); !ok {
		t.Fatal("lookup did not refresh a")
	}
	if !m.Delete(c) || m.Delete(c) || m.Len() != 1 {
		t.Fatalf("delete/len = %v/%v", m.Delete(c), m.Len())
	}
}

func TestMemoryRejectsMutations(t *testing.T) {
	if _, err := NewMemory(0).Store("x"); !errors.Is(err, ErrMemoryFull) {
		t.Fatalf("zero capacity error = %v", err)
	}
	if _, err := NewMemory(1).Store(""); !errors.Is(err, ErrEmptyDSML) {
		t.Fatalf("empty DSML error = %v", err)
	}
	m := NewMemory(1)
	if _, ok := m.Lookup("call_00000000000000000000000000000000"); ok {
		t.Fatal("unknown ID found")
	}
	if m.Delete("call_00000000000000000000000000000000") {
		t.Fatal("unknown ID deleted")
	}
}

func TestMemoryRecordReplayIsExactAndValidated(t *testing.T) {
	text := `prefix <tool_calls><invoke name="one"><parameter name="x">1</parameter></invoke><invoke name="two"><parameter name="x">2</parameter></invoke></tool_calls> suffix`
	m := NewMemory(2)
	ids, err := m.Record(text)
	if err != nil || len(ids) != 2 {
		t.Fatalf("Record = %v, %v", ids, err)
	}
	for _, id := range ids {
		got, ok := m.Replay(id)
		if !ok || got != text {
			t.Fatalf("Replay(%q) = %q, %v; want exact source", id, got, ok)
		}
	}
	if _, err := NewMemory(2).Record("ordinary text"); !errors.Is(err, ErrNoToolCalls) {
		t.Fatalf("plain text error = %v", err)
	}
	if _, err := NewMemory(2).Record(strings.Replace(text, "</invoke>", "", 1)); err == nil {
		t.Fatal("accepted malformed DSML")
	}
}

func TestMemoryRecordCapacityIsAtomic(t *testing.T) {
	text := `<tool_calls><invoke name="one"></invoke><invoke name="two"></invoke></tool_calls>`
	m := NewMemory(1)
	if _, err := m.Record(text); !errors.Is(err, ErrMemoryFull) {
		t.Fatalf("capacity error = %v", err)
	}
	if m.Len() != 0 {
		t.Fatalf("partial record retained %d entries", m.Len())
	}
}

func TestMemoryConcurrentUse(t *testing.T) {
	m := NewMemory(64)
	const workers = 32
	const rounds = 100
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for round := 0; round < rounds; round++ {
				id, err := m.Store("call-" + string(rune(worker)) + "-" + string(rune(round)))
				if err != nil {
					t.Errorf("Store: %v", err)
					return
				}
				m.Lookup(id)
				if round%3 == 0 {
					m.Delete(id)
				}
			}
		}(worker)
	}
	wg.Wait()
	if got := m.Len(); got < 0 || got > 64 {
		t.Fatalf("Len = %d", got)
	}
}
