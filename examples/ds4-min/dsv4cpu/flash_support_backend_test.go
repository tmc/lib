//go:build ds4cpumodel

package dsv4cpu

import (
	"errors"
	"testing"
)

type supportSourceCloser struct {
	calls int
	fail  bool
}

func (c *supportSourceCloser) Close() error {
	c.calls++
	if c.fail {
		return errors.New("support source close fault")
	}
	return nil
}

func TestFlashSupportRetainedSourceCleanup(t *testing.T) {
	s := supportFixture(t, 0)
	c := &supportSourceCloser{fail: true}
	b := &flashBackend{model: s.model, source: c}
	raw, err := b.NewState(4096)
	if err != nil {
		t.Fatal(err)
	}
	r := raw.(*flashRequest)
	if err := r.state.attachSupport([]int{1, 3}, 4); err != nil {
		t.Fatal(err)
	}
	snapshot, err := r.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err == nil || c.calls != 0 {
		t.Fatal("backend discarded live support snapshot")
	}
	if err := snapshot.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err == nil || b.source != c || b.closed {
		t.Fatal("failed cleanup discarded source owner")
	}
	c.fail = false
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if c.calls != 2 {
		t.Fatal("retained cleanup was not retried exactly")
	}
	s.close()
}

func TestFlashSupportMeasuredBytes(t *testing.T) {
	s := supportFixture(t, 8)
	var count byteCounter
	if err := s.save(&count); err != nil {
		t.Fatal(err)
	}
	if count.n != int64(len(saveFixture(t, s))) {
		t.Fatal("count differs from actual writer")
	}
	attached, err := flashMemory(s)
	if err != nil {
		t.Fatal(err)
	}
	v := s.support
	s.support = nil
	plain, err := flashMemory(s)
	if err != nil {
		t.Fatal(err)
	}
	s.support = v
	if attached.Scratch <= plain.Scratch {
		t.Fatal("support owning allocations not accounted")
	}
	s.close()
}
