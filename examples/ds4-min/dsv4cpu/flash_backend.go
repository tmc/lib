//go:build ds4cpumodel

package dsv4cpu

import (
	"errors"
	"io"
	"sync"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// flashBackend is private until the actual-model forward gates pass. The
// source remains open until every request and snapshot releases its owner.
type flashBackend struct {
	mu        sync.Mutex
	model     *flashModel
	source    io.Closer
	states    map[*flashRequest]bool
	snapshots map[*flashRequestSnapshot]bool
	closed    bool
}

type flashRequest struct {
	owner *flashBackend
	state *flashState
}
type flashRequestSnapshot struct {
	owner    *flashBackend
	snapshot *flashSnapshot
}

var _ dsv4.Backend = (*flashBackend)(nil)
var _ dsv4.State = (*flashRequest)(nil)
var _ dsv4.Snapshot = (*flashRequestSnapshot)(nil)

func (b *flashBackend) NewState(context int) (dsv4.State, error) {
	if b == nil {
		return nil, dsv4.ErrClosed
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed || b.model == nil {
		return nil, dsv4.ErrClosed
	}
	s, err := b.model.newState(context)
	if err != nil {
		return nil, err
	}
	r := &flashRequest{owner: b, state: s}
	if b.states == nil {
		b.states = make(map[*flashRequest]bool)
	}
	b.states[r] = true
	return r, nil
}

func (b *flashBackend) Eval(state dsv4.State, token, pos int) error {
	r, ok := state.(*flashRequest)
	if b == nil || !ok || r == nil || r.owner != b {
		return errors.New("foreign CPU state")
	}
	if r.state == nil || r.state.closed {
		return dsv4.ErrClosed
	}
	if r.state.failed {
		return dsv4.ErrDesynchronized
	}
	return b.model.eval(r.state, token, pos)
}

func (b *flashBackend) Logits(state dsv4.State) []float32 {
	r, ok := state.(*flashRequest)
	if b == nil || !ok || r == nil || r.owner != b || r.state == nil || r.state.closed || r.state.failed {
		return nil
	}
	return r.state.logits
}

func (b *flashBackend) Close() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	if len(b.states) != 0 || len(b.snapshots) != 0 {
		return errors.New("CPU model still has live states or snapshots")
	}
	if b.source != nil {
		if err := b.source.Close(); err != nil {
			return err
		}
	}
	b.closed = true
	b.model = nil
	return nil
}

func (r *flashRequest) Reset() error {
	if r == nil || r.state == nil {
		return dsv4.ErrClosed
	}
	return r.state.reset()
}

func (r *flashRequest) Snapshot() (dsv4.Snapshot, error) {
	if r == nil || r.state == nil {
		return nil, dsv4.ErrClosed
	}
	s, err := r.state.snapshot()
	if err != nil {
		return nil, err
	}
	out := &flashRequestSnapshot{owner: r.owner, snapshot: s}
	r.owner.mu.Lock()
	defer r.owner.mu.Unlock()
	if r.owner.snapshots == nil {
		r.owner.snapshots = make(map[*flashRequestSnapshot]bool)
	}
	r.owner.snapshots[out] = true
	return out, nil
}

func (r *flashRequest) Restore(snapshot dsv4.Snapshot) error {
	if r == nil || r.state == nil {
		return dsv4.ErrClosed
	}
	s, ok := snapshot.(*flashRequestSnapshot)
	if !ok || s == nil || s.owner != r.owner {
		return errors.New("foreign CPU snapshot")
	}
	if s.snapshot == nil {
		return dsv4.ErrClosed
	}
	return r.state.restore(s.snapshot)
}

func (r *flashRequest) Save(w io.Writer) error {
	if r == nil || r.state == nil {
		return dsv4.ErrClosed
	}
	return r.state.save(w)
}

func (r *flashRequest) Load(rd io.Reader) error {
	if r == nil || r.state == nil {
		return dsv4.ErrClosed
	}
	return r.state.load(rd)
}

func (r *flashRequest) Close() error {
	if r == nil || r.state == nil {
		return nil
	}
	r.state.close()
	r.state = nil
	r.owner.mu.Lock()
	defer r.owner.mu.Unlock()
	delete(r.owner.states, r)
	return nil
}

func (s *flashRequestSnapshot) Close() error {
	if s == nil || s.snapshot == nil {
		return nil
	}
	s.snapshot.close()
	s.snapshot = nil
	s.owner.mu.Lock()
	defer s.owner.mu.Unlock()
	delete(s.owner.snapshots, s)
	return nil
}
