//go:build darwin && arm64

package main

import (
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

func TestStreamingMetalBatchPreflight(t *testing.T) {
	r := &metalRuntime{ctx: 8, pos: 6}
	r.output.embedding.dimensions = []uint64{1, 32}
	for _, tt := range []struct {
		tokens []int
		pos    int
		fail   bool
	}{
		{[]int{1, 2}, 6, false}, {[]int{1, 2, 3}, 6, true},
		{[]int{1}, -1, true}, {[]int{1}, 8, true},
		{[]int{-1}, 6, true}, {[]int{32}, 6, true},
		{[]int{1<<32 + 1}, 6, true},
	} {
		err := r.validateStreamingBatch(tt.tokens, tt.pos)
		if (err != nil) != tt.fail {
			t.Fatalf("batch admission tokens=%v pos=%d: %v", tt.tokens, tt.pos, err)
		}
		if tt.fail {
			if err := r.evalStreamingPrefill(tt.tokens, tt.pos); err == nil {
				t.Fatal("production prefill ignored admission")
			}
		}
	}
}

func TestStreamingMetalTerminalPoison(t *testing.T) {
	for _, path := range []string{"scalar", "layer-major"} {
		t.Run(path, func(t *testing.T) {
			batch := &metalBatch{done: true}
			x := &metalGroupedExecution{batches: []*metalBatch{batch}, wait: func(*metalBatch) error { return errors.New("pending") }, terminal: func(*metalBatch) bool { return false }}
			r := &metalRuntime{}
			if err := r.finishStreamingExecution(x, nil); err == nil || !errors.Is(r.failed, dsv4.ErrDesynchronized) || r.groupedPending != x {
				t.Fatal("terminal defer stripped nonterminal poison", err, r.failed)
			}
		})
	}
}

func TestStreamingMetalEmptyBatch(t *testing.T) {
	for _, pos := range []int{0, 8} {
		r := &metalRuntime{ctx: 8, pos: pos}
		if err := r.validateStreamingBatch(nil, pos); err != nil {
			t.Fatal("empty frontier refused", err)
		}
		if pos == 8 && r.validateStreamingBatch([]int{1}, pos) == nil {
			t.Fatal("full context token admitted")
		}
	}
}

func TestStreamingMetalScalarFailure(t *testing.T) {
	pending := errors.New("consumer pending")
	selection := &fakeMetalSelection{ready: true, failure: pending}
	w := &metalStreamingWeights{mu: new(sync.Mutex), selection: selection}
	w.mu.Lock()
	w.locked = true
	r := &metalRuntime{streaming: w}
	_, err := r.evalLayerFFN(&metalBatch{}, 0, metalmem.Slice[float32]{}, false)
	if err == nil || !errors.Is(err, pending) || r.failed == nil || !w.locked {
		t.Fatal("scalar error lost retained owner", err)
	}
	if _, err := r.evalPosition(1, 0, true); err == nil {
		t.Fatal("failed cycle admitted next evaluation")
	}
	selection.failure = nil
	if err := r.Close(); err != nil || !selection.released || w.locked {
		t.Fatal("scalar owner retry close failed", err)
	}
}

func TestStreamingMetalSource(t *testing.T) {
	s, err := metalStreamingMSL()
	if err != nil {
		t.Fatal(err)
	}
	// Reversing precisely the address indirection must recover all original
	// arithmetic, including quantized block decoding and reduction order.
	for _, p := range [][2]string{
		{"kernel void matvec_iq2_xxs_pair_selected_addresses(", "kernel void matvec_iq2_xxs_pair_selected_f32("},
		{"kernel void matvec_q2_k_selected_addresses(", "kernel void matvec_q2_k_selected_f32("},
		{"device const ulong *gate [[buffer(0)]], device const ulong *up [[buffer(1)]],", "device const uchar *gate [[buffer(0)]], device const uchar *up [[buffer(1)]],"},
		{"device const ulong *weights [[buffer(0)]],", "device const uchar *weights [[buffer(0)]],"},
		{"const device uchar *selected_gate=reinterpret_cast<device const uchar *>(gate[expert]);", "const device uchar *selected_gate=gate+uint64_t(expert)*expert_stride;"},
		{"const device uchar *selected_up=reinterpret_cast<device const uchar *>(up[expert]);", "const device uchar *selected_up=up+uint64_t(expert)*expert_stride;"},
		{"device const uchar *selected=reinterpret_cast<device const uchar *>(weights[expert]);", "device const uchar *selected=weights+uint64_t(expert)*expert_stride;"},
	} {
		if strings.Count(s, p[0]) != 1 {
			t.Fatalf("address site count: %q", p[0])
		}
		s = strings.Replace(s, p[0], p[1], 1)
	}
	if s != metalPinnedMSL {
		t.Fatal("address route changed existing arithmetic")
	}
}

type fakeMetalSelection struct {
	ready               bool
	failure             error
	released, discarded bool
}

func (s *fakeMetalSelection) Ready() bool { return s.ready }
func (s *fakeMetalSelection) Discard() error {
	if s.failure != nil {
		return s.failure
	}
	s.discarded = true
	return nil
}
func (s *fakeMetalSelection) Release(done dsv4expertcache.Completion) error {
	if s.failure != nil {
		return s.failure
	}
	if err := done.Wait(); err != nil {
		return err
	}
	s.released = true
	return nil
}
func (s *fakeMetalSelection) Slots() ([]int, error) { return []int{0}, nil }
func (s *fakeMetalSelection) Views(int) ([3]metalmem.Slice[byte], error) {
	return [3]metalmem.Slice[byte]{}, nil
}

func TestStreamingMetalRetainedOwner(t *testing.T) {
	for _, ready := range []bool{false, true} {
		s := &fakeMetalSelection{ready: ready, failure: errors.New("pending")}
		w := &metalStreamingWeights{selection: s}
		if err := w.finish(); err == nil || w.selection != s || s.released || s.discarded {
			t.Fatal("failure lost retained owner")
		}
		s.failure = nil
		if err := w.finish(); err != nil || w.selection != nil || (ready && !s.released) || (!ready && !s.discarded) {
			t.Fatal("completion retry failed", err)
		}
	}
}

func TestStreamingMetalRequestOwnership(t *testing.T) {
	foundation := &metalStreamingWeights{mu: new(sync.Mutex)}
	a, b := foundation.request(), foundation.request()
	if a == b || a == foundation || !a.borrowed || a.mu != foundation.mu {
		t.Fatal("request ownership collapsed")
	}
	a.mu.Lock()
	a.locked = true
	if err := foundation.close(); err == nil {
		t.Fatal("active request owner freed")
	}
	if err := a.close(); err != nil || a.locked {
		t.Fatal("request close did not drain", err)
	}
	if err := b.close(); err != nil {
		t.Fatal(err)
	}
	if err := foundation.close(); err != nil {
		t.Fatal("backend close after drain", err)
	}
}

func TestStreamingMetalPrepareSerialization(t *testing.T) {
	w := &metalStreamingWeights{mu: new(sync.Mutex), upload: new(dsv4expertcache.MetalUpload)}
	w.mu.Lock()
	w.locked = true
	done := make(chan error, 1)
	go func() { done <- w.prepare(0, [3]boundMetalTensor{}, nil) }()
	select {
	case <-done:
		w.locked = false
		w.mu.Unlock()
		t.Fatal("prepare bypassed active lease serialization")
	case <-time.After(25 * time.Millisecond):
	}
	w.locked = false
	w.mu.Unlock()
	if err := <-done; err == nil || w.locked {
		t.Fatal("prepare error retained unused lock", err)
	}
}

func TestStreamingMetalTokenSerialization(t *testing.T) {
	foundation := &metalStreamingWeights{mu: new(sync.Mutex), upload: new(dsv4expertcache.MetalUpload)}
	a, b := foundation.request(), foundation.request()
	a.mu.Lock()
	a.locked = true
	done := make(chan error, 1)
	go func() { done <- b.noteToken(0) }()
	select {
	case <-done:
		a.locked = false
		a.mu.Unlock()
		t.Fatal("token aging bypassed another State's cache lease")
	case <-time.After(25 * time.Millisecond):
	}
	if err := a.finish(); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestStreamingMetalServerPartialOwner(t *testing.T) {
	pending := errors.New("pending")
	selection := &fakeMetalSelection{ready: true, failure: pending}
	r := &metalRuntime{streaming: &metalStreamingWeights{mu: new(sync.Mutex), selection: selection}}
	backend, err := openMetalStreamingServer("unused", 2, streamingOptions{enabled: true}, func(string, int, io.Writer, streamingOptions) (*metalRuntime, error) {
		return r, errors.New("construction failed")
	})
	owners := backendErrorOwners(err)
	if backend != nil || err == nil || len(owners) != 1 || r.streaming == nil || selection.released {
		t.Fatal("partial constructor owner lost", backend, err, owners)
	}
	selection.failure = nil
	if err := owners[0].Close(); err != nil || !selection.released || r.streaming != nil {
		t.Fatal("partial constructor cleanup retry failed", err)
	}
}

func TestStreamingMetalMappedFormat(t *testing.T) {
	for _, tt := range []struct {
		gate, up, down uint32
		mapped, fail   bool
	}{
		{typeQ4K, typeQ4K, typeQ4K, true, false},
		{typeIQ2XXS, typeIQ2XXS, typeQ2K, false, false},
		{typeQ4K, typeIQ2XXS, typeQ2K, false, true},
		{typeMXFP4, typeMXFP4, typeMXFP4, false, true},
	} {
		mapped, err := metalStreamingMappedOnly(tt.gate, tt.up, tt.down)
		if mapped != tt.mapped || (err != nil) != tt.fail {
			t.Fatal("mapped expert format changed", tt, mapped, err)
		}
	}
}
