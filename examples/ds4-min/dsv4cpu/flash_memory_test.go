//go:build ds4cpumodel

package dsv4cpu

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestFlashMemoryLiveBacking(t *testing.T) {
	f := codecFixture(t)
	defer f.close()
	b := &flashBackend{model: f.model}
	a, err := b.NewState(256)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	c, err := b.NewState(256)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	x := a.(*flashRequest)
	want, err := x.MemoryStats()
	if err != nil {
		t.Fatal(err)
	}
	if want.Raw != 43*128*512*4 || want.Scratch != 129280*4 {
		t.Fatalf("backing bytes: %+v", want)
	}
	other, err := c.(dsv4.MemoryReporter).MemoryStats()
	if err != nil || other != want {
		t.Fatalf("shared weights counted: %+v %v", other, err)
	}
	// Change actual capacity without changing context or any model shape.
	l := &x.state.layers[2]
	old := cap(l.compressed)
	l.compressed = make([]float32, 0, old+113)
	x.state.steering = make([]float32, 7, 19)
	got, err := x.MemoryStats()
	if err != nil {
		t.Fatal(err)
	}
	if got.Raw != want.Raw || got.Compressed-want.Compressed != 113*4 || got.Scratch-want.Scratch != 19*4 {
		t.Fatalf("not actual live capacity: %+v %+v", want, got)
	}
	if err := x.Reset(); err != nil {
		t.Fatal(err)
	}
	reset, err := x.MemoryStats()
	if err != nil || reset != got {
		t.Fatalf("reset lost owned capacity: %+v %v", reset, err)
	}
	if err := x.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := x.MemoryStats(); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal("closed owner reported")
	}
}

func TestFlashSnapshotActualSerializedBytes(t *testing.T) {
	f := codecFixture(t)
	defer f.close()
	b := &flashBackend{model: f.model}
	r, err := b.NewState(256)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	s, err := r.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	x := s.(*flashRequestSnapshot)
	var out bytes.Buffer
	if err := x.snapshot.state.save(&out); err != nil {
		t.Fatal(err)
	}
	n, err := x.SnapshotBytes()
	if err != nil || n != int64(out.Len()) {
		t.Fatalf("size=%d actual=%d err=%v", n, out.Len(), err)
	}
	before, err := x.MemoryStats()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := x.MemoryStats()
	if err != nil || before != after {
		t.Fatal("snapshot inventory borrowed request")
	}
	if err := b.Close(); err == nil {
		t.Fatal("snapshot owner released early")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := x.SnapshotBytes(); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal("closed snapshot sized")
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}
