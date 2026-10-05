package cudahotswap

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

// graphRegistry is called under the service mutation gate. The caller owns
// executables, nodes, streams and input buffers, and keeps them live until all
// pending work and bindings close. Calling raw Exec.Launch or modifying nodes
// bypasses these pins and is unsupported. No device association is inferred.
// Operations and owner cleanup callbacks must not reenter this registry.
type graphRegistry struct {
	mu       sync.Mutex
	table    *generationTable
	ops      graphOps
	bindings []*graphBinding
	patch    *graphPatch
	pending  map[*cudagraph.Exec]*graphPending
	closed   bool
}

type graphOps struct {
	patch  func(*cudagraph.Exec, *cudagraph.Node, driver.Function, driver.Dim3, []any) error
	launch func(*cudagraph.Exec, *cuda.Stream) error
	wait   func(context.Context, *cuda.Stream) error
}

func newGraphRegistry(table *generationTable) *graphRegistry {
	return &graphRegistry{table: table, ops: graphOps{
		patch: func(e *cudagraph.Exec, n *cudagraph.Node, f driver.Function, g driver.Dim3, a []any) error {
			return e.HotSwapNode(n, f, g, a...)
		},
		launch: func(e *cudagraph.Exec, s *cuda.Stream) error { return e.Launch(s) },
		wait:   func(ctx context.Context, s *cuda.Stream) error { return s.Synchronize(ctx) },
	}, pending: make(map[*cudagraph.Exec]*graphPending)}
}

type graphBinding struct {
	name   string
	exec   *cudagraph.Exec
	node   *cudagraph.Node
	stream *cuda.Stream
	grid   driver.Dim3
	args   []any
	pin    *generationPin
	failed bool
}

func graphRelease(p *generationPin) error {
	if p.released.Load() {
		p.table.mu.Lock()
		defer p.table.mu.Unlock()
		return p.table.collectLocked()
	}
	return p.release()
}

// bind adopts a pin before attempting installation of the selected function.
// An error may follow an applied driver patch; the returned binding therefore
// remains registered and pinned even when patch reports an error.
func (r *graphRegistry) bind(name string, e *cudagraph.Exec, n *cudagraph.Node, stream *cuda.Stream, grid driver.Dim3, args []any) (*graphBinding, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || r.patch != nil || len(r.pending) != 0 || r.table == nil || e == nil || n == nil || stream == nil {
		return nil, errors.New("bind graph: unavailable registry or owner")
	}
	for _, b := range r.bindings {
		if b.exec == e && (b.node == n || b.stream != stream) {
			return nil, errors.New("bind graph: duplicate node or different owning stream")
		}
	}
	p, _, err := r.table.pin(name)
	if err != nil {
		return nil, err
	}
	b := &graphBinding{name: name, exec: e, node: n, stream: stream, grid: grid, args: append([]any(nil), args...), pin: p}
	r.bindings = append(r.bindings, b)
	err = r.ops.patch(e, n, p.gen.function, grid, b.args)
	b.failed = err != nil
	return b, err
}

// graphPatch owns an unpublished candidate until abort succeeds or publication
// plus commit transfers ownership to the table. publish requires zero candidate
// pins, so the transaction retains its owner rather than prematurely pinning it.
// The caller must not independently close that owner once prepare adopts it.
// A retained-target transaction instead owns an extra pin, never its module.
type graphPatch struct {
	registry    *graphRegistry
	name        string
	candidate   *generation
	bindings    []*graphBinding
	attempted   int
	old         []*generationPin
	committed   bool
	transferred bool
	done        bool
	retained    *generationPin
}

func (r *graphRegistry) prepare(name string, candidate *generation) (*graphPatch, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.prepareLocked(name, candidate, false)
}

// prepareRetained borrows an exact live table-owned version for Select, Revert,
// or rollback after checking a newly published candidate. Abort restores the
// old bindings and releases only this extra pin, never the retained owner.
func (r *graphRegistry) prepareRetained(name string, candidate *generation) (*graphPatch, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.prepareLocked(name, candidate, true)
}

func (r *graphRegistry) prepareLocked(name string, candidate *generation, retained bool) (*graphPatch, error) {
	if r.table == nil || r.closed || r.patch != nil || len(r.pending) != 0 || candidate == nil || candidate.owner == nil || candidate.function == 0 || (!retained && (candidate.retired || candidate.version != 0 || candidate.pins.Load() != 0)) {
		return nil, errors.New("patch graph: unavailable registry or candidate")
	}
	t := &graphPatch{registry: r, name: name, candidate: candidate}
	for _, b := range r.bindings {
		if b.name == name {
			if b.failed {
				return nil, errors.New("patch graph: initial binding failed")
			}
			if err := CheckABI(b.pin.gen.signature, candidate.signature); err != nil {
				return nil, err
			}
			t.bindings = append(t.bindings, b)
		}
	}
	r.table.mu.Lock()
	entry := r.table.entries[name]
	if r.table.closed || (retained && (entry == nil || candidate.version == 0 || entry.versions[candidate.version] != candidate || candidate.retired)) {
		r.table.mu.Unlock()
		return nil, errors.New("patch graph: table is closed or target is not retained")
	}
	if retained {
		candidate.pins.Add(1)
		t.retained = &generationPin{table: r.table, gen: candidate}
	}
	r.table.mu.Unlock()
	r.patch = t
	for i, b := range t.bindings {
		t.attempted = i + 1 // Include a possibly applied failing patch in rollback.
		if err := r.ops.patch(b.exec, b.node, candidate.function, b.grid, b.args); err != nil {
			return t, fmt.Errorf("patch graph node: %w", err)
		}
	}
	return t, nil
}

// commit follows successful table publication under the same mutation gate.
// No graph launch is possible between prepare and commit. Cleanup failure keeps
// the transaction reachable for retry without repeating the pin transfer.
func (t *graphPatch) commit() error {
	r := t.registry
	r.mu.Lock()
	defer r.mu.Unlock()
	if t.done {
		return nil
	}
	if r.patch != t {
		return errors.New("commit graph: transaction is not active")
	}
	if !t.committed {
		r.table.mu.Lock()
		entry := r.table.entries[t.name]
		if !t.transferred || entry == nil || entry.current.Load() != t.candidate || t.candidate.version == 0 || t.candidate.retired {
			r.table.mu.Unlock()
			return errors.New("commit graph: candidate is not published")
		}
		for _, b := range t.bindings {
			t.candidate.pins.Add(1)
			t.old = append(t.old, b.pin)
			b.pin = &generationPin{table: r.table, gen: t.candidate}
		}
		r.table.mu.Unlock()
		if t.retained != nil {
			t.old = append(t.old, t.retained)
		}
		t.committed = true
	}
	for len(t.old) != 0 {
		if err := graphRelease(t.old[0]); err != nil {
			return err
		}
		t.old = t.old[1:]
	}
	r.patch = nil
	t.done = true
	return nil
}

// published must follow table.publish or selectVersion before fallible cleanup. Close must
// never unload a candidate after ownership has transferred to the table.
func (t *graphPatch) published() {
	t.registry.mu.Lock()
	defer t.registry.mu.Unlock()
	t.transferred = true
}

// Close implements the request cleanup contract. Publication and cleanup are
// serialized by the caller's mutation gate; failed cleanup retains this owner.
func (t *graphPatch) Close() error {
	t.registry.mu.Lock()
	transferred := t.transferred
	t.registry.mu.Unlock()
	if transferred {
		return t.commit()
	}
	return t.abort()
}

// abort retries rollback before releasing the candidate. A rollback failure
// leaves old and candidate modules owned and prohibits graph launch.
func (t *graphPatch) abort() error {
	r := t.registry
	r.mu.Lock()
	defer r.mu.Unlock()
	if t.done {
		return nil
	}
	if r.patch != t || t.committed || t.transferred {
		return errors.New("abort graph: transaction is not rollbackable")
	}
	for t.attempted != 0 {
		b := t.bindings[t.attempted-1]
		if err := r.ops.patch(b.exec, b.node, b.pin.gen.function, b.grid, b.args); err != nil {
			return fmt.Errorf("rollback graph node: %w", err)
		}
		t.attempted--
	}
	if t.retained != nil {
		if err := graphRelease(t.retained); err != nil {
			return err
		}
	} else {
		if err := t.candidate.owner.Close(); err != nil {
			return err
		}
	}
	r.patch = nil
	t.done = true
	return nil
}

type graphPending struct {
	registry *graphRegistry
	exec     *cudagraph.Exec
	stream   *cuda.Stream
	pins     []*generationPin
	drained  bool
}

func (r *graphRegistry) launch(e *cudagraph.Exec) (*graphPending, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.table == nil || r.closed || r.patch != nil || r.pending[e] != nil {
		return nil, errors.New("launch graph: registry is unavailable")
	}
	p := &graphPending{registry: r, exec: e}
	for _, b := range r.bindings {
		if b.exec == e && b.failed {
			return nil, errors.New("launch graph: initial binding failed")
		}
	}
	r.table.mu.Lock()
	if r.table.closed {
		r.table.mu.Unlock()
		return nil, errors.New("launch graph: kernel table is closed")
	}
	for _, b := range r.bindings {
		if b.exec == e {
			p.stream = b.stream
			b.pin.gen.pins.Add(1)
			p.pins = append(p.pins, &generationPin{table: r.table, gen: b.pin.gen})
		}
	}
	r.table.mu.Unlock()
	if len(p.pins) == 0 {
		return nil, errors.New("launch graph: executable is not bound")
	}
	r.pending[e] = p // Publish ownership before a possibly partial enqueue.
	return p, r.ops.launch(e, p.stream)
}

func (p *graphPending) wait(ctx context.Context) error {
	if ctx == nil {
		return errors.New("wait graph: nil context")
	}
	r := p.registry
	r.mu.Lock()
	defer r.mu.Unlock()
	if !p.drained {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := r.ops.wait(ctx, p.stream); err != nil {
			return err
		}
		p.drained = true
	}
	for len(p.pins) != 0 {
		if err := graphRelease(p.pins[0]); err != nil {
			return err
		}
		p.pins = p.pins[1:]
	}
	if r.pending[p.exec] == p {
		delete(r.pending, p.exec)
	}
	return nil
}

// close requires pending launches and any patch transaction to finish first.
// It only releases module pins; graph and stream owners remain caller-owned.
func (r *graphRegistry) close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.pending) != 0 || r.patch != nil {
		return errors.New("close graph bindings: work or patch is pending")
	}
	for len(r.bindings) != 0 {
		if err := graphRelease(r.bindings[0].pin); err != nil {
			return err
		}
		r.bindings = r.bindings[1:]
	}
	r.closed = true
	return nil
}

func (r *graphRegistry) Close() error { return r.close() }
