package cudahotswap

import (
	"context"
	"errors"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// ReplayPolicy sets finite output, noise and timing comparison limits.
// Rounds must be in [1,64]. Tolerances must be finite and nonnegative.
// Exact compares raw FP32 bits; otherwise MaxULP also bounds each comparison.
type ReplayPolicy replayPolicy

// DefaultReplayPolicy uses five interleaved rounds and exact outputs with no
// baseline noise. Callers must explicitly relax these limits when appropriate.
func DefaultReplayPolicy() ReplayPolicy {
	return ReplayPolicy{Rounds: 5, Exact: true}
}

// FixtureBuffer declares a pointer argument and its complete byte extent.
// Nonstable inputs are copied to scratch; Stable borrows the original address.
// The caller keeps every allocation alive and excludes concurrent writes until
// Fixture.Close succeeds, including when recording returns an error.
type FixtureBuffer struct {
	Arg    int
	Bytes  int
	Buffer cudamem.Slice[byte]
	Stable bool
}

// FixtureOutput declares an output argument. Elem currently supports "f32".
type FixtureOutput struct {
	Arg  int
	Elem string
}

// FixtureConfig describes one real launch. Args contains scalars and nil at
// declared pointer positions; buffers supply the pointer values. Launch.Stream
// must be the owning input stream. Recording synchronizes it before snapshot.
// Geometry overrides configured on Service are applied at record time.
type FixtureConfig struct {
	Args    []any
	Buffers []FixtureBuffer
	Outputs []FixtureOutput
	Launch  cudalaunch.Config
	Policy  ReplayPolicy
}

// Fixture owns a recorded baseline, its module pin and replay allocations.
// Service.Close also closes retained fixtures. The zero value owns nothing.
// A Fixture must not be copied after use.
type Fixture struct {
	mu        sync.Mutex
	service   *Service
	state     State
	replay    *replayFixture
	geometry  *LaunchGeometry
	parent    *requestCleanup
	stageSpec *StageSpec
}

// State identifies the baseline and settings observed at recording.
func (f *Fixture) State() State {
	if f == nil {
		return State{}
	}
	return f.state
}

// RecordFixture snapshots inputs and executes the selected baseline twice,
// then with poisoned outputs. Unchanged outputs or excess baseline noise are
// refused. A nonnil fixture with an error retains all adopted owners for Close;
// cancellation is not permission to free source buffers or close the device.
// Recording changes neither the selected generation nor its config revision.
// At most 32 fixtures, including failed recordings, may be retained at once.
func (s *Service) RecordFixture(ctx context.Context, name string, config FixtureConfig) (*Fixture, error) {
	if ctx == nil {
		return nil, errors.New("record fixture: nil context")
	}
	if s == nil || s.loader == nil {
		return nil, errors.New("record fixture: nil service or loader")
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
	if err := s.mutationAvailable(name, nil); err != nil {
		return nil, err
	}
	if s.newReplay == nil {
		return nil, errors.New("record fixture: loader does not provide native replay")
	}
	s.mu.Lock()
	state, ok := s.states[name]
	settings := s.settings[name]
	full := len(s.fixtures) >= 32
	s.mu.Unlock()
	if full {
		return nil, errors.New("record fixture: retained fixture limit reached")
	}
	if !ok {
		return nil, errors.New("record fixture: kernel is not registered")
	}
	s.table.mu.RLock()
	entry := s.table.entries[name]
	if s.table.closed || entry == nil || entry.current.Load() == nil {
		s.table.mu.RUnlock()
		return nil, errors.New("record fixture: selected kernel is unavailable")
	}
	signature := entry.current.Load().signature
	s.table.mu.RUnlock()
	spec := replaySpec{Signature: signature, Args: config.Args, Config: config.Launch}
	if g := settings.geometry; g != nil {
		spec.Config.Grid, spec.Config.Block, spec.Config.SharedMem = g.Grid, g.Block, g.SharedMem
	}
	var bindings []replayDeviceBinding
	for _, b := range config.Buffers {
		spec.Buffers = append(spec.Buffers, replayBuffer{Arg: b.Arg, Bytes: b.Bytes, Stable: b.Stable})
		bindings = append(bindings, replayDeviceBinding{Arg: b.Arg, Slice: b.Buffer})
	}
	for _, o := range config.Outputs {
		spec.Outputs = append(spec.Outputs, replayOutput{Arg: o.Arg, Elem: o.Elem})
	}
	policy := replayPolicy(config.Policy)
	if err := validateReplay(spec, policy); err != nil {
		return nil, err
	}
	backend, err := s.newReplay(name, spec, policy, bindings)
	if err != nil {
		return nil, err
	}
	if backend == nil {
		return nil, errors.New("record fixture: missing replay backend")
	}
	pin, _, err := s.table.pin(name)
	if err != nil {
		return nil, err
	}
	kernel, ok := pin.gen.owner.(Kernel)
	if !ok || kernel == nil {
		return nil, errors.Join(errors.New("record fixture: selected owner is not a kernel"), pin.release())
	}
	replay, err := recordReplay(ctx, backend, replayArm{Kernel: kernel, Release: pin.release}, spec, policy)
	f := &Fixture{service: s, state: state, replay: replay}
	if settings.geometry != nil {
		g := *settings.geometry
		f.geometry = &g
	}
	s.mu.Lock()
	if s.fixtures == nil {
		s.fixtures = make(map[*Fixture]struct{})
	}
	s.fixtures[f] = struct{}{}
	s.mu.Unlock()
	s.observeError(err)
	return f, err
}

// Close drains outstanding replay work, restores borrowed stable inputs and
// releases pins and allocations under a request lease. Failures retain owners
// for retry. It is valid after Service.Close has begun.
func (f *Fixture) Close(ctx context.Context) error {
	if ctx == nil {
		return errors.New("close fixture: nil context")
	}
	if f == nil || f.service == nil {
		return nil
	}
	f.service.requests.RLock()
	defer f.service.requests.RUnlock()
	f.service.mutations.Lock()
	defer f.service.mutations.Unlock()
	if err := f.service.mutationAvailable(f.state.Name, nil); err != nil {
		return err
	}
	return f.close(ctx)
}

// close is also used by Service.Close, which already holds both request gates.
func (f *Fixture) close(ctx context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.replay == nil {
		return nil
	}
	if f.parent != nil && f.parent.lease != nil {
		ctx = context.WithValue(ctx, pushLeaseKey{}, pushLeaseCapability{f.service, f.state.Name, f.parent})
	}
	if err := f.replay.Close(ctx); err != nil {
		f.service.observeError(err)
		return err
	}
	f.replay = nil
	f.parent = nil
	f.service.mu.Lock()
	delete(f.service.fixtures, f)
	f.service.mu.Unlock()
	return nil
}
