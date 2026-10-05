package cudahotswap

import (
	"context"
	"errors"
	"io"
	"sort"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

// RegisterGraph binds a caller-owned kernel node to the selected version.
// The exec, node and stream must belong to the loader's device. The caller keeps
// these objects and arguments alive and unchanged until Service.Close succeeds,
// and uses LaunchGraph instead of raw Exec.Launch. Grid and the argument slice
// are copied; pointed-to values and buffers are not owned by Service.
//
// A nonzero State with an error means a possibly applied patch retains a module
// pin. Keep the caller-owned objects alive until Service.Close releases it.
// Registration does not change the selected code or its mutation revision.
func (s *Service) RegisterGraph(ctx context.Context, name string, exec *cudagraph.Exec, node *cudagraph.Node, stream *cuda.Stream, grid driver.Dim3, args ...any) (state State, err error) {
	if ctx == nil {
		return State{}, errors.New("register graph: nil context")
	}
	if s == nil || s.loader == nil || s.graphs == nil || exec == nil || node == nil || stream == nil {
		return State{}, errors.New("register graph: service, exec, node and stream are required")
	}
	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	if grid.X == 0 || grid.X > 1<<31-1 || grid.Y == 0 || grid.Y > 65535 || grid.Z == 0 || grid.Z > 65535 {
		return State{}, errors.New("register graph: unsupported grid dimensions")
	}
	if loader, ok := s.loader.(*NVRTCLoader); ok && (loader == nil || loader.dev == nil || stream.Device() != loader.dev) {
		return State{}, errors.New("register graph: stream belongs to another device owner")
	}
	s.requests.RLock()
	defer s.requests.RUnlock()
	s.mutations.Lock()
	defer s.mutations.Unlock()
	if err := s.available(); err != nil {
		return State{}, err
	}
	if err := s.mutationAvailable(name, nil); err != nil {
		return State{}, err
	}
	s.table.mu.RLock()
	entry := s.table.entries[name]
	valid := entry != nil && entry.current.Load() != nil && !s.table.closed
	if valid && len(args) != len(entry.current.Load().signature.Params) {
		valid = false
	}
	s.table.mu.RUnlock()
	if !valid {
		return State{}, errors.New("register graph: kernel or argument count is unavailable")
	}
	lease, err := s.acquire(ctx, name)
	if err != nil {
		return State{}, err
	}
	cleanup := s.retainCleanup(lease)
	defer func() { err = errors.Join(err, cleanup.Close()); s.observeError(err) }()
	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	binding, err := s.graphs.bind(name, exec, node, stream, grid, args)
	if binding != nil {
		s.mu.Lock()
		state = s.states[name]
		s.mu.Unlock()
	}
	return state, err
}

// GraphLaunch owns a possibly submitted graph and its request lease. The caller
// must call Wait even when LaunchGraph also returns an error. Cancellation does
// not establish completion. Its zero value owns no work.
type GraphLaunch struct {
	mu      sync.Mutex
	states  []State
	service *Service
	pending *graphPending
	lease   io.Closer
}

// LaunchGraph submits a registered exec on its registered stream. It pins every
// bound module until Wait confirms completion. A nonnil result with an error
// retains possibly submitted work or a failed lease release for a later Wait.
func (s *Service) LaunchGraph(ctx context.Context, exec *cudagraph.Exec) (*GraphLaunch, error) {
	if ctx == nil {
		return nil, errors.New("launch graph: nil context")
	}
	if s == nil || s.loader == nil || s.graphs == nil || exec == nil {
		return nil, errors.New("launch graph: service and exec are required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.requests.RLock()
	defer s.requests.RUnlock()
	s.mutations.Lock()
	defer s.mutations.Unlock()
	if err := s.available(); err != nil {
		return nil, err
	}
	r := s.graphs
	r.mu.Lock()
	bound := false
	names := make(map[string]bool)
	unavailable := r.closed || r.patch != nil || r.pending[exec] != nil
	for _, b := range r.bindings {
		if b.exec == exec {
			bound = true
			names[b.name] = true
			unavailable = unavailable || b.failed
		}
	}
	r.mu.Unlock()
	if !bound || unavailable {
		return nil, errors.New("launch graph: executable is not ready")
	}
	lease, err := s.acquire(ctx, "graph")
	if err != nil {
		return nil, err
	}
	p := &GraphLaunch{service: s, lease: lease}
	s.mu.Lock()
	if s.graphWork == nil {
		s.graphWork = make(map[*GraphLaunch]struct{})
	}
	s.graphWork[p] = struct{}{} // Retain the lease before a partial enqueue.
	s.mu.Unlock()
	if err = ctx.Err(); err == nil {
		p.pending, err = r.launch(exec)
		if p.pending != nil {
			s.mu.Lock()
			for name := range names {
				p.states = append(p.states, s.states[name])
			}
			s.mu.Unlock()
			sort.Slice(p.states, func(i, j int) bool { return p.states[i].Name < p.states[j].Name })
		}
	}
	s.observeError(err)
	if p.pending == nil {
		cleanupErr := p.Wait(ctx)
		if cleanupErr == nil {
			return nil, err
		}
		return p, errors.Join(err, cleanupErr)
	}
	return p, err
}

// Wait confirms graph completion and releases pins and the request lease.
// Failed or canceled synchronization retains ownership for retry. Once actual
// completion is confirmed, a failed lease release retries without waiting again.
func (p *GraphLaunch) Wait(ctx context.Context) error {
	if ctx == nil {
		return errors.New("wait graph: nil context")
	}
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pending != nil {
		if err := p.pending.wait(ctx); err != nil {
			p.service.observeError(err)
			return err
		}
		p.pending = nil
	}
	if p.lease != nil {
		if err := p.lease.Close(); err != nil {
			if p.service != nil {
				p.service.observeError(err)
			}
			return err
		}
		p.lease = nil
	}
	if p.service != nil {
		p.service.mu.Lock()
		delete(p.service.graphWork, p)
		p.service.mu.Unlock()
	}
	return nil
}
