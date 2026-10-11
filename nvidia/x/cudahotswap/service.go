package cudahotswap

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

// State identifies a kernel mutation and the compiled version it selects.
// Generation increases when code is selected; Version identifies retained code.
// ConfigRevision increases when settings change without selecting new code.
type State struct {
	Name           string
	Kind           string
	Generation     uint64
	Version        uint64
	ConfigRevision uint64
	Label          string
	SourceDigest   string
	CompilerKey    string
}

// Service owns loaded kernel generations and submitted launches. Its zero
// value is not usable. Close must finish successfully before the caller closes
// the loader's device. Raw functions used without a Pin bypass its lifetime
// protection. A Service must not be copied after use.
type Service struct {
	requests    sync.RWMutex
	mutations   sync.RWMutex
	mu          sync.Mutex
	table       generationTable
	loader      Loader
	acquire     func(context.Context, string) (io.Closer, error)
	states      map[string]State
	settings    map[string]kernelSettings
	reports     *reportEvents
	graphs      *graphRegistry
	versionKeys map[*generation]string
	pending     map[*PendingLaunch]struct{}
	graphWork   map[*GraphLaunch]struct{}
	fixtures    map[*Fixture]struct{}
	newReplay   func(string, replaySpec, replayPolicy, []replayDeviceBinding) (replayBackend, error)
	cleanup     map[*requestCleanup]struct{}
	closing     *requestCleanup
	closed      bool
	poison      error
	checking    map[string]*pushCheck
	watch       watchState
	stages      stageRegistry
}

type requestLease struct{ handle *cudalock.Handle }

func (l requestLease) Close() error { return l.handle.Release() }

// requestCleanup retains the lease until every dependent owner has closed.
// Successful steps are not repeated when a later step needs retry.
type requestCleanup struct {
	mu      sync.Mutex
	service *Service
	owners  []io.Closer
	collect func() error
	lease   io.Closer
	drain   func(context.Context) error
}

func (s *Service) retainCleanup(lease io.Closer) *requestCleanup {
	r := &requestCleanup{service: s, lease: lease}
	s.mu.Lock()
	if s.cleanup == nil {
		s.cleanup = make(map[*requestCleanup]struct{})
	}
	s.cleanup[r] = struct{}{}
	s.mu.Unlock()
	return r
}

func (r *requestCleanup) adopt(owner io.Closer) {
	if owner == nil {
		return
	}
	for _, old := range r.owners {
		if reflect.TypeOf(owner).Comparable() && old == owner {
			return
		}
	}
	r.owners = append(r.owners, owner)
}

func (r *requestCleanup) Close() error {
	return r.close(context.Background())
}

func (r *requestCleanup) close(ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.drain != nil {
		if err := r.drain(ctx); err != nil {
			return err
		}
		r.drain = nil
	}
	for i, owner := range r.owners {
		if owner == nil {
			continue
		}
		if err := owner.Close(); err != nil {
			return err
		}
		r.owners[i] = nil
	}
	if r.collect != nil {
		if err := r.collect(); err != nil {
			return err
		}
		r.collect = nil
	}
	if r.lease != nil {
		if err := r.lease.Close(); err != nil {
			return err
		}
		r.lease = nil
	}
	r.service.mu.Lock()
	delete(r.service.cleanup, r)
	r.service.mu.Unlock()
	return nil
}

func (s *Service) collect() error {
	s.table.mu.Lock()
	defer s.table.mu.Unlock()
	return s.table.collectLocked()
}

// NewService creates an idle service. It takes no GPU lease until a request
// performs loading, launching or cleanup.
func NewService(loader Loader) (*Service, error) {
	if loader == nil {
		return nil, fmt.Errorf("create service: nil loader")
	}
	reports, err := newReportEvents(reportEventLimits{ring: 64, queue: 65, subscribers: 32, bytes: 64 << 10})
	if err != nil {
		return nil, fmt.Errorf("create service reports: %w", err)
	}
	s := &Service{loader: loader, states: make(map[string]State), pending: make(map[*PendingLaunch]struct{}), reports: reports}
	s.graphs = newGraphRegistry(&s.table)
	s.versionKeys = make(map[*generation]string)
	s.acquire = func(ctx context.Context, name string) (io.Closer, error) {
		h, err := cudalock.Acquire(ctx, cudalock.WithLabel("cudahotswap:"+name))
		if err != nil {
			return nil, err
		}
		return requestLease{h}, nil
	}
	if native, ok := loader.(*NVRTCLoader); ok && native != nil {
		s.newReplay = func(name string, spec replaySpec, policy replayPolicy, bindings []replayDeviceBinding) (replayBackend, error) {
			b, err := newCUDAReplayBackend(native.dev, spec, policy, bindings)
			if err != nil {
				return nil, err // Construction has not allocated or acquired owners.
			}
			b.acquire = func(ctx context.Context) (io.Closer, error) { return s.replayLease(ctx, name) }
			return b, nil
		}
	}
	return s, nil
}

func (s *Service) available() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return errors.New("service is closed")
	}
	if s.poison != nil {
		return fmt.Errorf("%w: %w", ErrPoisoned, s.poison)
	}
	return nil
}

func (s *Service) observeError(err error) {
	if err == nil {
		return
	}
	// Observation cancellation is not a device fault, but a joined CUDA fault
	// must not disappear merely because the same result includes cancellation.
	deviceFault := serviceDeviceFault(err)
	if !deviceFault && (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)) {
		return
	}
	if outcome, _ := Classify(err); deviceFault || outcome.PoisonsContext() {
		s.mu.Lock()
		if s.poison == nil {
			s.poison = err
		}
		s.mu.Unlock()
	}
}

func serviceDeviceFault(err error) bool {
	if result, ok := err.(driver.Result); ok {
		outcome, _ := Classify(result)
		return outcome.PoisonsContext()
	}
	switch err := err.(type) {
	case interface{ Unwrap() []error }:
		for _, child := range err.Unwrap() {
			if serviceDeviceFault(child) {
				return true
			}
		}
	case interface{ Unwrap() error }:
		return serviceDeviceFault(err.Unwrap())
	}
	return false
}

// Register compiles and registers an original kernel. Compilation and PTX ABI
// validation happen before acquiring a GPU lease. Existing names are refused;
// replacing a registered kernel requires the candidate check path.
func (s *Service) Register(ctx context.Context, name string, source Source, label string) (state State, err error) {
	if ctx == nil {
		return State{}, errors.New("register kernel: nil context")
	}
	if s == nil || s.loader == nil || name == "" {
		return State{}, errors.New("register kernel: service and name are required")
	}
	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	s.requests.RLock()
	defer s.requests.RUnlock()
	if err := s.available(); err != nil {
		return State{}, err
	}
	c, err := s.loader.Compile(ctx, source)
	if err != nil {
		var owner io.Closer
		if errors.As(err, &owner) {
			// Compilation owns no loaded module and takes no GPU lease, but
			// a compiler cleanup failure still needs a reachable retry owner.
			s.mutations.Lock()
			cleanup := s.retainCleanup(nil)
			cleanup.adopt(owner)
			err = errors.Join(err, cleanup.Close())
			s.mutations.Unlock()
		}
		return State{}, err
	}
	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	if c == nil || c.Entry != source.Entry {
		return State{}, errors.New("register kernel: missing or mismatched compilation")
	}
	sig, err := cudalaunch.SignatureFromPTX(c.PTX, c.Entry)
	if err != nil {
		return State{}, err
	}
	if err := CheckABI(sig, sig); err != nil {
		return State{}, err
	}
	s.mutations.Lock()
	defer s.mutations.Unlock()
	if err := s.available(); err != nil {
		return State{}, err
	}
	s.mu.Lock()
	_, exists := s.states[name]
	s.mu.Unlock()
	if exists {
		return State{}, errors.New("register kernel: name already exists")
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
	k, err := s.loader.Load(ctx, c)
	if !nilKernel(k) {
		cleanup.adopt(k)
	}
	if err != nil {
		var owner io.Closer
		if errors.As(err, &owner) {
			cleanup.adopt(owner)
		}
		return State{}, err
	}
	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	if nilKernel(k) || k.Raw() == 0 {
		return State{}, errors.New("register kernel: missing loaded function")
	}
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(source.Code)))
	g := &generation{digest: digest, flags: source.Flags, signature: sig, function: k.Raw(), owner: k}
	stamp, err := s.table.publish(name, g, 0)
	if err != nil {
		return State{}, err
	}
	cleanup.owners = nil // The generation table now owns the kernel.
	state = State{Name: name, Kind: "kernel", Generation: stamp.revision, Version: stamp.version, Label: label, SourceDigest: digest, CompilerKey: c.Key}
	s.mu.Lock()
	s.states[name] = state
	s.versionKeys[g] = c.Key
	s.rememberWatchInputLocked(name, source, c)
	s.mu.Unlock()
	return state, nil
}

// List returns a detached, name-sorted snapshot of registered kernels.
func (s *Service) List() []State {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	states := make([]State, 0, len(s.states))
	for _, state := range s.states {
		states = append(states, state)
	}
	sort.Slice(states, func(i, j int) bool { return states[i].Name < states[j].Name })
	return states
}

// Pin retains a raw function handle until Release. The caller must finish all
// work using Function before Release, including graph replay and forward passes.
type Pin struct {
	mu      sync.Mutex
	state   State
	service *Service
	pin     *generationPin
	cleanup *requestCleanup
}

// Pin obtains the current function without performing GPU work.
func (s *Service) Pin(name string) (*Pin, error) {
	if s == nil {
		return nil, errors.New("pin kernel: nil service")
	}
	s.mutations.RLock()
	defer s.mutations.RUnlock()
	if err := s.available(); err != nil {
		return nil, err
	}
	p, _, err := s.table.pin(name)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	state := s.states[name]
	s.mu.Unlock()
	return &Pin{service: s, pin: p, state: state}, nil
}

// Function returns the retained function, or zero after Release.
func (p *Pin) Function() driver.Function {
	if p == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pin == nil || p.pin.released.Load() {
		return 0
	}
	return p.pin.gen.function
}

// Release releases the pin under a request lease because a retired module may
// now unload. If lease acquisition fails, the pin remains available for retry.
// Once the pin is released, Function returns zero even if collection fails.
// Failed collection retains its lease in the service until Release or Close
// succeeds; do not close the loader's device first.
func (p *Pin) Release(ctx context.Context) error {
	if ctx == nil {
		return errors.New("release pin: nil context")
	}
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pin == nil && p.cleanup == nil {
		return nil
	}
	p.service.mutations.RLock()
	defer p.service.mutations.RUnlock()
	if p.cleanup == nil {
		var lease io.Closer
		// Close may have retained its lease because this generation was
		// pinned. Use that owner while holding mutations instead of trying
		// to acquire a second exclusive lease in the same process.
		if p.service.closing == nil || p.service.closing.lease == nil {
			var err error
			lease, err = p.service.acquire(ctx, "release")
			if err != nil {
				return err
			}
		}
		p.cleanup = p.service.retainCleanup(lease)
		pin := p.pin
		p.cleanup.collect = func() error {
			if !pin.released.Load() {
				return pin.release()
			}
			return p.service.collect()
		}
	}
	if err := p.cleanup.Close(); err != nil {
		return err
	}
	p.pin = nil
	p.cleanup = nil
	return nil
}

// PendingLaunch retains both its module pin and request lease. Wait must
// establish actual completion before either is released. The service also
// retains it, so discarding this result cannot abandon submitted work.
type PendingLaunch struct {
	mu        sync.Mutex
	state     State
	service   *Service
	kernel    Kernel
	pin       *generationPin
	lease     io.Closer
	stream    *cuda.Stream
	completed bool
}

// Launch submits the current generation. A nonnil result remains owned even
// when submission reports an error, since a context-restore error may follow
// successful enqueue. Caller device buffers must remain live until Wait.
func (s *Service) Launch(ctx context.Context, name string, cfg cudalaunch.Config, args ...any) (*PendingLaunch, error) {
	if ctx == nil {
		return nil, errors.New("launch kernel: nil context")
	}
	if s == nil {
		return nil, errors.New("launch kernel: nil service")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mutations.RLock()
	defer s.mutations.RUnlock()
	if err := s.available(); err != nil {
		return nil, err
	}
	p, _, err := s.table.pin(name)
	if err != nil {
		return nil, err
	}
	lease, err := s.acquire(ctx, name)
	if err == nil {
		err = ctx.Err()
	}
	if err != nil {
		// No submission occurred. Avoid GPU cleanup without a lease; future
		// leased collection owns any retired versions.
		s.table.mu.Lock()
		p.released.Store(true)
		p.gen.pins.Add(-1)
		s.table.mu.Unlock()
		if lease != nil {
			cleanup := s.retainCleanup(lease)
			err = errors.Join(err, cleanup.Close())
		}
		return nil, err
	}
	k, ok := p.gen.owner.(Kernel)
	if !ok {
		cleanup := s.retainCleanup(lease)
		cleanup.collect = func() error {
			if !p.released.Load() {
				return p.release()
			}
			return s.collect()
		}
		return nil, errors.Join(errors.New("launch kernel: invalid module owner"), cleanup.Close())
	}
	s.mu.Lock()
	state := s.states[name]
	if geometry := s.settings[name].geometry; geometry != nil {
		cfg.Grid = geometry.Grid
		cfg.Block = geometry.Block
		cfg.SharedMem = geometry.SharedMem
	}
	s.mu.Unlock()
	result := &PendingLaunch{service: s, kernel: k, pin: p, lease: lease, stream: cfg.Stream, state: state}
	s.mu.Lock()
	s.pending[result] = struct{}{}
	s.mu.Unlock()
	err = k.Launch(cfg, args...)
	s.observeError(err)
	return result, err
}

// Wait confirms CUDA completion. Observation cancellation retains the pending
// request for another Wait; it neither unloads the module nor releases its lease.
// Once completion succeeds, a cleanup retry does not wait for the GPU again.
func (p *PendingLaunch) Wait(ctx context.Context) error {
	if ctx == nil {
		return errors.New("wait for launch: nil context")
	}
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pin != nil {
		if !p.completed {
			if err := p.kernel.Wait(ctx, p.stream); err != nil {
				p.service.observeError(err)
				return err
			}
			p.completed = true
		}
		var err error
		if p.pin.released.Load() {
			err = p.service.collect()
		} else {
			err = p.pin.release()
		}
		if err != nil {
			return err
		}
		p.pin = nil
	}
	if p.lease != nil {
		if err := p.lease.Close(); err != nil {
			return err
		}
		p.lease = nil
	}
	p.service.mu.Lock()
	delete(p.service.pending, p)
	p.service.mu.Unlock()
	return nil
}

// Close refuses new requests and drains submitted launches before unloading
// modules. Failed or canceled cleanup retains ownership for an explicit retry.
func (s *Service) Close(ctx context.Context) error {
	if ctx == nil {
		return errors.New("close service: nil context")
	}
	if s == nil {
		return nil
	}
	// Register holds this gate through compilation and adoption of any error
	// cleanup owner. Take it before mutations, matching Register's lock order.
	s.requests.Lock()
	defer s.requests.Unlock()
	s.mutations.Lock()
	defer s.mutations.Unlock()
	s.mu.Lock()
	if len(s.checking) != 0 {
		s.mu.Unlock()
		return errors.New("close service: application check is active")
	}
	s.closed = true
	s.watchChangedLocked()
	var pending []*PendingLaunch
	for p := range s.pending {
		pending = append(pending, p)
	}
	var graphWork []*GraphLaunch
	for p := range s.graphWork {
		graphWork = append(graphWork, p)
	}
	var fixtures []*Fixture
	for f := range s.fixtures {
		fixtures = append(fixtures, f)
	}
	s.mu.Unlock()
	s.stopReports()
	if err := s.drainStages(ctx); err != nil {
		return err
	}
	for _, p := range pending {
		if err := p.Wait(ctx); err != nil {
			return err
		}
	}
	for _, p := range graphWork {
		if err := p.Wait(ctx); err != nil {
			return err
		}
	}
	s.mu.Lock()
	var cleanup []*requestCleanup
	for r := range s.cleanup {
		cleanup = append(cleanup, r)
	}
	s.mu.Unlock()
	for _, r := range cleanup {
		if err := r.close(ctx); err != nil {
			s.observeError(err)
			return err
		}
	}
	// Retry retained exclusive leases before a healthy fixture acquires a new
	// one. A request's explicit drain closes any fixture it still references
	// before releasing its candidate. All fixtures close before the table.
	for _, f := range fixtures {
		if err := f.close(ctx); err != nil {
			return err
		}
	}
	if err := s.closeStages(); err != nil {
		return err
	}
	if s.closing == nil {
		s.table.mu.RLock()
		empty := len(s.table.entries) == 0
		s.table.mu.RUnlock()
		if empty {
			return nil
		}
		lease, err := s.acquire(ctx, "close")
		if err != nil {
			return err
		}
		s.closing = s.retainCleanup(lease)
		s.closing.adopt(s.graphs)
		s.closing.collect = s.table.close
	}
	err := s.closing.Close()
	s.observeError(err)
	return err
}
