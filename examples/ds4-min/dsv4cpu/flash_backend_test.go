//go:build ds4cpumodel

package dsv4cpu

import (
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type cpuCloseCounter struct{ calls int }

func (c *cpuCloseCounter) Close() error { c.calls++; return nil }

// This tests the actual adapter's ownership, not an inferred forward result.
func TestFlashBackendOwnership(t *testing.T) {
	fixture := codecFixture(t)
	defer fixture.close()
	closer := new(cpuCloseCounter)
	b := &flashBackend{model: fixture.model, source: closer}
	raw, err := b.NewState(256)
	if err != nil {
		t.Fatal(err)
	}
	r := raw.(*flashRequest)
	if r.state == fixture || r.state.model != fixture.model {
		t.Fatal("request owner differs")
	}
	if err := b.Close(); err == nil || closer.calls != 0 {
		t.Fatal("model closed with active request")
	}
	other := &flashBackend{model: fixture.model}
	if err := other.Eval(raw, 0, 0); err == nil {
		t.Fatal("foreign backend accepted state")
	}
	if other.Logits(raw) != nil {
		t.Fatal("foreign backend exposed logits")
	}
	snapshot, err := r.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err == nil || closer.calls != 0 {
		t.Fatal("model closed with active snapshot")
	}
	if err := snapshot.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if closer.calls != 1 {
		t.Fatal("model source lifetime differs")
	}
	if _, err := b.NewState(256); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal("closed backend created state")
	}
	if err := raw.Reset(); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal("closed state reset accepted")
	}
}
