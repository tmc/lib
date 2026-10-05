package cudahotswap

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap/stage"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// StageBuffer names one context pointer and its complete byte extent. Its
// position is the same as FixtureBuffer.Arg and StageLaunch.Buffers.
type StageBuffer struct {
	Name  string
	Bytes int
}

// StageFunction names a context function and the registered Service kernel
// whose actual CUDA generation must remain pinned until stream completion.
type StageFunction struct {
	Name   string
	Kernel string
}

// StageSpec is the immutable pointed-to context schema. HostOnly admits a
// checker-only C stage with no device bindings or timing claims. Native stages
// require an NVRTCLoader device, named buffers and real kernel generations.
type StageSpec struct {
	Buffers   []StageBuffer
	Functions []StageFunction
	HostOnly  bool
}

type stageVersion struct {
	state    State
	spec     StageSpec
	builder  *stage.Builder
	artifact *stage.Artifact
	library  *stage.Library
	source   stage.Source
}

type stageEntry struct {
	current, original *stageVersion
	versions          map[uint64]*stageVersion
}

type stageRegistry struct {
	next     uint64 // Mutation gate protects entries, reservations and counter.
	rollback map[*pushCheck]struct{}
	entries  map[string]*stageEntry
	pending  map[*PendingStage]struct{} // Service.mu protects pending.
}

// canAdvance excludes generations reserved by active application checks.
// The caller holds the mutation gate.
func (r *stageRegistry) canAdvance(n uint64) bool {
	remaining := uint64(math.MaxUint64) - r.next
	reserved := uint64(len(r.rollback))
	return reserved <= remaining && n <= remaining-reserved
}

func validateStageSpec(spec StageSpec) (StageSpec, error) {
	if spec.HostOnly {
		if len(spec.Buffers) != 0 || len(spec.Functions) != 0 {
			return StageSpec{}, errors.New("host-only stage cannot declare device bindings")
		}
		return spec, nil
	}
	if len(spec.Buffers) == 0 || len(spec.Buffers) > 64 || len(spec.Functions) == 0 || len(spec.Functions) > 64 {
		return StageSpec{}, errors.New("stage requires 1..64 named buffers and functions")
	}
	names := make(map[string]bool)
	validName := func(name string) bool {
		if name == "" || len(name) > 128 || strings.ContainsRune(name, 0) || names[name] {
			return false
		}
		names[name] = true
		return true
	}
	total := 0
	for _, b := range spec.Buffers {
		if !validName(b.Name) || b.Bytes <= 0 || b.Bytes > replayMaxBytes-total {
			return StageSpec{}, errors.New("stage has invalid named buffer or bounded extent")
		}
		total += b.Bytes
	}
	for _, f := range spec.Functions {
		if !validName(f.Name) || f.Kernel == "" || len(f.Kernel) > 128 || strings.ContainsRune(f.Kernel, 0) {
			return StageSpec{}, errors.New("stage has invalid named function")
		}
	}
	spec.Buffers = append([]StageBuffer(nil), spec.Buffers...)
	spec.Functions = append([]StageFunction(nil), spec.Functions...)
	return spec, nil
}

func (s *Service) stageDevice(spec StageSpec) (*cuda.Device, error) {
	if spec.HostOnly {
		return nil, nil
	}
	loader, ok := s.loader.(*NVRTCLoader)
	if !ok || loader == nil || loader.dev == nil {
		return nil, errors.New("native stage requires an actual NVRTCLoader device")
	}
	return loader.dev, nil
}

// RegisterStage compiles C host glue before loading under a request lease.
// Artifact layout is checked by the actual compiler witness, not invented PTX.
// Retain builder's generation directory through successful Service.Close.
func (s *Service) RegisterStage(ctx context.Context, name string, builder *stage.Builder, source stage.Source, spec StageSpec, label string) (state State, err error) {
	if ctx == nil || s == nil || s.loader == nil || name == "" || builder == nil {
		return State{}, errors.New("register stage: service, context, name and builder are required")
	}
	spec, err = validateStageSpec(spec)
	if err != nil {
		return State{}, err
	}
	if spec.HostOnly && source.CUDA {
		return State{}, errors.New("host-only stage cannot compile CUDA source")
	}
	s.requests.RLock()
	defer s.requests.RUnlock()
	if err := s.available(); err != nil {
		return State{}, err
	}
	if _, err := s.stageDevice(spec); err != nil {
		return State{}, err
	}
	a, err := builder.Build(ctx, source)
	if err != nil {
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
		return State{}, errors.New("register stage: name already exists")
	}
	if err := s.stageFunctionsAvailable(spec); err != nil {
		return State{}, err
	}
	lease, err := s.acquireStage(ctx, name, spec)
	if err != nil {
		return State{}, err
	}
	cleanup := s.retainCleanup(lease)
	defer func() { err = errors.Join(err, cleanup.Close()); s.observeError(err) }()
	library, err := stage.Load(a)
	cleanup.adopt(library)
	if err != nil {
		return State{}, err
	}
	if !s.stages.canAdvance(1) {
		return State{}, errors.New("stage generation counter exhausted")
	}
	s.stages.next++
	state = State{Name: name, Kind: "stage", Generation: s.stages.next, Version: s.stages.next, Label: label, SourceDigest: a.SourceDigest()}
	source.Flags = append([]string(nil), source.Flags...)
	v := &stageVersion{state: state, spec: spec, builder: builder, artifact: a, library: library, source: source}
	if s.stages.entries == nil {
		s.stages.entries = make(map[string]*stageEntry)
	}
	s.stages.entries[name] = &stageEntry{current: v, original: v, versions: map[uint64]*stageVersion{state.Version: v}}
	s.mu.Lock()
	s.states[name] = state
	s.mu.Unlock()
	cleanup.owners = nil // Typed stage table now owns the actual host library.
	return state, nil
}

type hostStageLease struct{}

func (hostStageLease) Close() error { return nil }

func (s *Service) acquireStage(ctx context.Context, name string, spec StageSpec) (io.Closer, error) {
	if spec.HostOnly {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return hostStageLease{}, nil // Explicit CPU-only mode, never a GPU lease.
	}
	lease, err := s.acquire(ctx, name)
	if err == nil && lease == nil {
		err = errors.New("stage acquire returned no request lease")
	}
	return lease, err
}

// The caller holds the mutation gate, so an inspected kernel cannot disappear
// before its actual pin is obtained by a stage call.
func (s *Service) stageFunctionsAvailable(spec StageSpec) error {
	dev, err := s.stageDevice(spec)
	if err != nil {
		return err
	}
	s.table.mu.RLock()
	defer s.table.mu.RUnlock()
	for _, f := range spec.Functions {
		e := s.table.entries[f.Kernel]
		if s.table.closed || e == nil || e.current.Load() == nil || e.current.Load().retired {
			return fmt.Errorf("stage function %s references an unavailable kernel %s", f.Name, f.Kernel)
		}
		g := e.current.Load()
		owner, ok := g.owner.(*nativeKernel)
		if !ok || owner == nil || owner.dev != dev || g.function == 0 {
			return fmt.Errorf("stage function %s has no actual kernel owner for the device", f.Name)
		}
	}
	return nil
}

// StageLaunch supplies retained named device buffers and the owning stream.
// Buffer order and exact byte extents are fixed by StageSpec. Context itself
// has no launch geometry: stage source chooses the calls it submits.
type StageLaunch struct {
	Buffers []cudamem.Slice[byte]
	Stream  *cuda.Stream
}

// PendingStage retains its host library, named kernel generations and request
// lease. Wait establishes actual stream completion before releasing any owner.
// Cancellation retains the result for retry; Service.Close also drains it.
// The zero value is not usable. Use the owner returned by RunStage.
type PendingStage struct {
	mu       sync.Mutex
	service  *Service
	call     *stageCallable
	cleanup  *requestCleanup
	stream   *cuda.Stream
	complete bool
}

// RunStage invokes the current host glue. A nonnil result with an error still
// owns submitted work; Wait must finish before the buffers or device close.
func (s *Service) RunStage(ctx context.Context, name string, launch StageLaunch) (*PendingStage, error) {
	if ctx == nil || s == nil {
		return nil, errors.New("run stage: service and context are required")
	}
	s.mutations.RLock()
	defer s.mutations.RUnlock()
	if err := s.available(); err != nil {
		return nil, err
	}
	e := s.stages.entries[name]
	if e == nil || e.current == nil {
		return nil, errors.New("run stage: stage is not registered")
	}
	v := e.current
	if len(launch.Buffers) != len(v.spec.Buffers) || v.spec.HostOnly && launch.Stream != nil {
		return nil, errors.New("run stage: context differs from declared schema")
	}
	args := make([]any, len(launch.Buffers))
	for i, b := range launch.Buffers {
		if b.Len() != v.spec.Buffers[i].Bytes || b.Ptr() == 0 {
			return nil, fmt.Errorf("stage buffer %s has %d bytes; requires %d", v.spec.Buffers[i].Name, b.Len(), v.spec.Buffers[i].Bytes)
		}
		args[i] = b.Ptr()
	}
	dev, err := s.stageDevice(v.spec)
	if err != nil {
		return nil, err
	}
	if dev != nil && (launch.Stream == nil || launch.Stream.Device() != dev) {
		return nil, errors.New("run stage: stream must belong to actual stage device")
	}
	s.mu.Lock()
	geometry := s.settings[name].geometry
	s.mu.Unlock()
	if geometry != nil {
		return nil, errors.New("run stage: kernel geometry settings do not describe stage-owned launches")
	}
	lease, err := s.acquireStage(ctx, name, v.spec)
	if err != nil {
		return nil, err
	}
	cleanup := s.retainCleanup(lease)
	for i, b := range launch.Buffers {
		release, err := b.Borrow(dev)
		if err != nil {
			return nil, errors.Join(fmt.Errorf("stage buffer %s: %w", v.spec.Buffers[i].Name, err), cleanup.Close())
		}
		cleanup.adopt(&stageBufferBorrow{release: release})
		args[i] = b.Ptr()
	}
	call, err := s.newStageCallable(v, dev)
	if err != nil {
		return nil, errors.Join(err, cleanup.Close())
	}
	cleanup.adopt(call)
	p := &PendingStage{service: s, call: call, cleanup: cleanup, stream: launch.Stream}
	cleanup.drain = func(ctx context.Context) error { return call.Wait(ctx, launch.Stream) }
	s.mu.Lock()
	if s.stages.pending == nil {
		s.stages.pending = make(map[*PendingStage]struct{})
	}
	s.stages.pending[p] = struct{}{}
	s.mu.Unlock()
	err = call.Launch(cudalaunch.Config{Stream: launch.Stream}, args...)
	s.observeError(err)
	return p, err
}

type stageBufferBorrow struct{ release func() }

func (b *stageBufferBorrow) Close() error { b.release(); return nil }

// Wait confirms completion and retries owner cleanup without resubmitting.
func (p *PendingStage) Wait(ctx context.Context) error {
	if ctx == nil {
		return errors.New("wait for stage: nil context")
	}
	if p == nil {
		return nil
	}
	if p.service == nil || p.cleanup == nil {
		return errors.New("wait for stage: uninitialized owner")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.complete {
		return nil
	}
	if err := p.cleanup.close(ctx); err != nil {
		p.service.observeError(err)
		return err
	}
	p.complete = true
	p.service.mu.Lock()
	delete(p.service.stages.pending, p)
	p.service.mu.Unlock()
	return nil
}

type stageCallable struct {
	mu              sync.Mutex
	service         *Service
	version         *stageVersion
	dev             *cuda.Device
	pin             *stage.Pin
	functions       []*generationPin
	stream          *cuda.Stream
	pending         bool
	completed       bool
	retainFunctions bool
}

func (s *Service) newStageCallable(v *stageVersion, dev *cuda.Device) (*stageCallable, error) {
	p, err := v.library.Pin()
	if err != nil {
		return nil, err
	}
	return &stageCallable{service: s, version: v, dev: dev, pin: p}, nil
}

func (k *nativeKernel) replayDevice() *cuda.Device {
	if k == nil {
		return nil
	}
	return k.dev
}

func (c *stageCallable) replayDevice() *cuda.Device { return c.dev }

func (c *stageCallable) Launch(cfg cudalaunch.Config, args ...any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.pin == nil || c.pending || len(args) != len(c.version.spec.Buffers) {
		return errors.New("stage call is released, pending or has a mismatched schema")
	}
	if c.dev != nil && (cfg.Stream == nil || cfg.Stream.Device() != c.dev) {
		return errors.New("stage call stream belongs to another device")
	}
	var stream driver.Stream
	var names [][]byte
	var buffers []stage.Buffer
	for i, b := range c.version.spec.Buffers {
		ptr, ok := args[i].(driver.DevicePtr)
		if !ok || ptr == 0 {
			return fmt.Errorf("stage buffer %s requires an actual device pointer", b.Name)
		}
		name := append([]byte(b.Name), 0)
		names = append(names, name)
		buffers = append(buffers, stage.Buffer{Name: &name[0], Pointer: uint64(ptr), Bytes: uint64(b.Bytes)})
	}
	var functions []stage.Function
	if len(c.functions) == 0 {
		for _, binding := range c.version.spec.Functions {
			p, _, err := c.service.table.pin(binding.Kernel)
			if err != nil {
				return errors.Join(err, c.releaseFunctions())
			}
			c.functions = append(c.functions, p)
		}
	}
	for i, binding := range c.version.spec.Functions {
		p := c.functions[i]
		owner, ok := p.gen.owner.(*nativeKernel)
		if !ok || owner == nil || owner.dev != c.dev || p.gen.function == 0 {
			return errors.Join(errors.New("stage function is not a pinned kernel of the actual device"), c.releaseFunctions())
		}
		name := append([]byte(binding.Name), 0)
		names = append(names, name)
		functions = append(functions, stage.Function{Name: &name[0], Handle: uintptr(p.gen.function)})
	}
	invoke := func() error {
		context, err := stage.NewContext(buffers, uintptr(stream), functions)
		if err != nil {
			return err
		}
		c.pending, c.completed, c.stream = true, false, cfg.Stream
		return c.pin.Call(context)
	}
	if c.dev == nil {
		return invoke() // Declared CPU-only stage, with no device bindings.
	}
	return c.dev.WithContext(func() error {
		return cfg.Stream.WithHandle(func(h driver.Stream) error { stream = h; return invoke() })
	})
}

func (c *stageCallable) Wait(ctx context.Context, stream *cuda.Stream) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if ctx == nil {
		return errors.New("wait for stage: nil context")
	}
	if c.pending && !c.completed {
		if stream != c.stream {
			return errors.New("wait for stage: different stream")
		}
		if c.dev != nil {
			if err := stream.Synchronize(ctx); err != nil {
				return err
			}
		}
		c.completed = true
	}
	if !c.retainFunctions {
		if err := c.releaseFunctions(); err != nil {
			return err
		}
	}
	c.pending = false
	return nil
}

func (c *stageCallable) releaseFunctions() error {
	for i, p := range c.functions {
		if p == nil {
			continue
		}
		var err error
		if p.released.Load() {
			err = c.service.collect()
		} else {
			err = p.release()
		}
		if err != nil {
			return err
		}
		c.functions[i] = nil
	}
	c.functions = nil
	return nil
}

func (c *stageCallable) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.pending {
		return errors.New("close stage call: completion remains unresolved")
	}
	if err := c.releaseFunctions(); err != nil {
		return err
	}
	if err := c.pin.Release(); err != nil {
		return err
	}
	c.pin = nil
	return nil
}

func (s *Service) drainStages(ctx context.Context) error {
	s.mu.Lock()
	var work []*PendingStage
	for p := range s.stages.pending {
		work = append(work, p)
	}
	s.mu.Unlock()
	for _, p := range work {
		if err := p.Wait(ctx); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) closeStages() error {
	for _, e := range s.stages.entries {
		for _, v := range e.versions {
			if err := v.library.Close(); err != nil {
				return err
			}
		}
	}
	return nil
}

func sameStageSpec(a, b StageSpec) bool { return reflect.DeepEqual(a, b) }

// releaseAfterReplay runs only after the replay engine has drained its actual
// device work under its lease. Wait also releases or retains exact function pins.
func (c *stageCallable) releaseAfterReplay() error {
	c.mu.Lock()
	stream := c.stream
	c.mu.Unlock()
	if err := c.Wait(context.Background(), stream); err != nil {
		return err
	}
	return c.Close()
}
