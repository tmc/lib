//go:build ds4cpumodel

package dsv4cpu

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func (r *flashRequest) MemoryStats() (dsv4.ContextMemory, error) {
	if r == nil || r.owner == nil {
		return dsv4.ContextMemory{}, dsv4.ErrClosed
	}
	r.owner.mu.Lock()
	defer r.owner.mu.Unlock()
	if !r.owner.states[r] {
		return dsv4.ContextMemory{}, dsv4.ErrClosed
	}
	return flashMemory(r.state)
}

func (s *flashRequestSnapshot) MemoryStats() (dsv4.ContextMemory, error) {
	if s == nil || s.owner == nil {
		return dsv4.ContextMemory{}, dsv4.ErrClosed
	}
	s.owner.mu.Lock()
	defer s.owner.mu.Unlock()
	if !s.owner.snapshots[s] || s.snapshot == nil {
		return dsv4.ContextMemory{}, dsv4.ErrClosed
	}
	return flashMemory(s.snapshot.state)
}

func (s *flashRequestSnapshot) SnapshotBytes() (int64, error) {
	if s == nil || s.owner == nil {
		return 0, dsv4.ErrClosed
	}
	s.owner.mu.Lock()
	defer s.owner.mu.Unlock()
	if !s.owner.snapshots[s] || s.snapshot == nil || s.snapshot.state == nil {
		return 0, dsv4.ErrClosed
	}
	var w byteCounter
	if err := s.snapshot.state.save(&w); err != nil {
		return 0, err
	}
	return w.n, nil
}

type byteCounter struct{ n int64 }

func (w *byteCounter) Write(p []byte) (int, error) {
	if int64(len(p)) > math.MaxInt64-w.n {
		return 0, fmt.Errorf("snapshot size: overflow")
	}
	w.n += int64(len(p))
	return len(p), nil
}

func flashMemory(s *flashState) (dsv4.ContextMemory, error) {
	var m dsv4.ContextMemory
	if s == nil || s.closed || s.model == nil {
		return m, dsv4.ErrClosed
	}
	add := func(dst *int64, values []float32) error {
		if int64(cap(values)) > (math.MaxInt64-*dst)/4 {
			return fmt.Errorf("CPU memory: overflow")
		}
		*dst += int64(cap(values)) * 4
		return nil
	}
	for _, l := range s.layers {
		if l.raw == nil {
			return dsv4.ContextMemory{}, fmt.Errorf("CPU memory: missing raw owner")
		}
		if err := add(&m.Raw, l.raw.rows); err != nil {
			return dsv4.ContextMemory{}, err
		}
		for _, values := range [][]float32{l.compressed, l.indexRows} {
			if err := add(&m.Compressed, values); err != nil {
				return dsv4.ContextMemory{}, err
			}
		}
		for _, c := range []*Compressor{l.attention, l.index} {
			if c != nil {
				for _, values := range [][]float32{c.kv, c.score} {
					if err := add(&m.Compressed, values); err != nil {
						return dsv4.ContextMemory{}, err
					}
				}
			}
		}
	}
	for _, values := range [][]float32{s.logits, s.steering} {
		if err := add(&m.Scratch, values); err != nil {
			return dsv4.ContextMemory{}, err
		}
	}
	if v := s.support; v != nil {
		if err := v.state.Validate(s.position, 0); err != nil {
			return dsv4.ContextMemory{}, err
		}
		for _, values := range [][]float32{v.state.History, v.staged} {
			if err := add(&m.Scratch, values); err != nil {
				return dsv4.ContextMemory{}, err
			}
		}
		for _, allocation := range []struct {
			count int
			size  uintptr
		}{
			{cap(v.state.Targets), unsafe.Sizeof(int(0))},
			{cap(v.state.Journal.Rows), unsafe.Sizeof(v.state.Journal.Rows[0])},
			{cap(v.captured), unsafe.Sizeof(false)},
		} {
			if uint64(allocation.count) > uint64(math.MaxInt64-m.Scratch)/uint64(allocation.size) {
				return dsv4.ContextMemory{}, fmt.Errorf("CPU support memory: overflow")
			}
			m.Scratch += int64(uint64(allocation.count) * uint64(allocation.size))
		}
	}
	return m, nil
}
