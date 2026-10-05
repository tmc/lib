package cudahotswap

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// stageReplaySignature is the explicitly declared logical buffer order used by
// the replay engine. It is not PTX or the ABI of int stage(stage_ctx*).
func stageReplaySignature(name string, schema StageSpec) cudalaunch.Signature {
	sig := cudalaunch.Signature{KernelName: name}
	for _, b := range schema.Buffers {
		sig.Params = append(sig.Params, cudalaunch.Param{Name: b.Name, Type: "u64", Size: 8, IsPointer: true})
	}
	return sig
}

// RecordStageFixture records a native stage through the kernel replay engine's
// real scratch buffers, poisoned outputs, native event clock and output policy.
// Arg positions must match the declared names and exact extents in StageSpec.
// Stage source owns launch geometry; Config.Launch selects the owning stream.
func (s *Service) RecordStageFixture(ctx context.Context, name string, config FixtureConfig) (*Fixture, error) {
	if ctx == nil || s == nil || s.loader == nil {
		return nil, errors.New("record stage fixture: service and context are required")
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
	e := s.stages.entries[name]
	if e == nil || e.current == nil || e.current.spec.HostOnly || s.newReplay == nil {
		return nil, errors.New("record stage fixture: actual native stage and replay backend are required")
	}
	v := e.current
	dev, err := s.stageDevice(v.spec)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	state := s.states[name]
	geometry := s.settings[name].geometry
	full := len(s.fixtures) >= 32
	s.mu.Unlock()
	if full || geometry != nil {
		return nil, errors.New("record stage fixture: fixture limit or kernel geometry override is unsupported")
	}
	if len(config.Args) != len(v.spec.Buffers) || len(config.Buffers) != len(v.spec.Buffers) {
		return nil, errors.New("record stage fixture: named buffer schema is incomplete")
	}
	spec := replaySpec{Signature: stageReplaySignature(name, v.spec), Args: config.Args, Config: config.Launch, stageOwnedGeometry: true}
	var bindings []replayDeviceBinding
	seen := make(map[int]bool)
	for _, b := range config.Buffers {
		if b.Arg < 0 || b.Arg >= len(v.spec.Buffers) || seen[b.Arg] || config.Args[b.Arg] != nil || b.Bytes != v.spec.Buffers[b.Arg].Bytes || b.Buffer.Len() != b.Bytes || b.Buffer.Device() != dev {
			return nil, fmt.Errorf("record stage fixture: buffer argument %d differs from its named extent or actual device", b.Arg)
		}
		seen[b.Arg] = true
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
		return nil, errors.New("record stage fixture: missing replay backend")
	}
	call, err := s.newStageCallable(v, dev)
	if err != nil {
		return nil, errors.Join(err, backend.Close(ctx))
	}
	call.retainFunctions = true // Exact named kernel generations stay baseline-owned.
	replay, err := recordReplay(ctx, backend, replayArm{Kernel: call, Release: call.releaseAfterReplay}, spec, policy)
	schema := v.spec
	f := &Fixture{service: s, state: state, replay: replay, stageSpec: &schema}
	s.mu.Lock()
	if s.fixtures == nil {
		s.fixtures = make(map[*Fixture]struct{})
	}
	s.fixtures[f] = struct{}{}
	s.mu.Unlock()
	s.observeError(err)
	return f, err
}

func (s *Service) stageFixtureAvailable(name string, schema StageSpec, settings kernelSettings, config CheckConfig) error {
	f := config.Fixture
	if f == nil {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.service != s || f.state.Name != name || f.stageSpec == nil || !sameStageSpec(*f.stageSpec, schema) || f.replay == nil {
		return errors.New("push stage: recorded fixture is closed or has a different named schema")
	}
	r := f.replay
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.ready || r.closed || r.cleanupStarted || settings.geometry != nil {
		return errors.New("push stage: recorded fixture is unavailable or kernel geometry settings changed")
	}
	policy := r.policy
	if config.Policy != nil {
		policy = replayPolicy(*config.Policy)
	}
	if err := validateReplay(r.spec, policy); err != nil {
		return err
	}
	for _, n := range r.noise {
		if n.NonFinite != 0 || n.MaxAbs == nil || n.RMS == nil || *n.MaxAbs > policy.NoiseMaxAbs || *n.RMS > policy.NoiseRMS {
			return errors.New("push stage: baseline noise exceeds policy")
		}
	}
	return nil
}
