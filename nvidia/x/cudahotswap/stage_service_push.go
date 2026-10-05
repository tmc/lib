package cudahotswap

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahotswap/stage"
)

// PushStage compiles a fresh generation, refuses a changed named context ABI,
// checks native replay before publication, then calls the same application
// Checker as kernels. A failed hook rolls back to the exact prior stage unless
// KeepOnFailure. An unconfigured check never admits a candidate.
func (s *Service) PushStage(ctx context.Context, name string, source stage.Source, schema StageSpec, label string) (result *PushResult, err error) {
	if ctx == nil || s == nil || s.loader == nil {
		return nil, errors.New("push stage: service and context are required")
	}
	schema, err = validateStageSpec(schema)
	if err != nil {
		return nil, err
	}
	s.requests.RLock()
	// Compilation may own a library cleanup later; Close cannot race adoption.
	s.mutations.RLock()
	if err = s.available(); err != nil {
		s.mutations.RUnlock()
		s.requests.RUnlock()
		return nil, err
	}
	if err = s.mutationAvailable(name, nil); err != nil {
		s.mutations.RUnlock()
		s.requests.RUnlock()
		return nil, err
	}
	e := s.stages.entries[name]
	if e == nil || e.current == nil {
		s.mutations.RUnlock()
		s.requests.RUnlock()
		return nil, errors.New("push stage: stage is not registered")
	}
	prior := e.current
	if !sameStageSpec(prior.spec, schema) || source.CUDA != prior.source.CUDA {
		s.mutations.RUnlock()
		s.requests.RUnlock()
		return nil, errors.New("push stage: named context ABI differs from the registered stage")
	}
	s.mu.Lock()
	snapshot := s.states[name]
	settings := s.settings[name]
	settings.flags = append([]string(nil), settings.flags...)
	s.mu.Unlock()
	if settings.flagsSet {
		source.Flags = settings.flags
	} else if source.Flags == nil {
		source.Flags = append([]string(nil), prior.source.Flags...)
	}
	s.mutations.RUnlock()
	result = &PushResult{State: snapshot, Selected: snapshot, Report: Report{Task: name, Candidate: label, Entry: "stage", Outcome: OutcomeOK, Recognized: true, Unmeasured: []string{"application end-to-end timing", "CUPTI kernel activity"}}}
	start := time.Now()
	artifact, buildErr := prior.builder.Build(ctx, source)
	result.Report.Compile = &CompileReport{WallNanos: time.Since(start).Nanoseconds(), SourceBytes: len(source.Code), SourceSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(source.Code)))}
	if artifact != nil {
		result.Report.Compile.SourceSHA256 = artifact.SourceDigest()
	}
	if buildErr != nil {
		result.Report.Compile.Log = buildErr.Error()
		s.requests.RUnlock()
		return result, s.finishPush(result, buildErr, false)
	}
	var config CheckConfig
	var marker *pushCheck
	s.mutations.Lock()
	err = s.pushStageLocked(ctx, name, source, schema, label, snapshot, prior, artifact, result, &config, &marker)
	s.mutations.Unlock()
	s.requests.RUnlock()
	defer func() {
		if marker == nil {
			return
		}
		s.mutations.Lock()
		s.mu.Lock()
		if s.checking[name] == marker {
			delete(s.checking, name)
		}
		delete(s.stages.rollback, marker)
		s.mu.Unlock()
		s.mutations.Unlock()
	}()
	if err != nil {
		return result, s.finishPush(result, err, false)
	}
	if config.Checker != nil {
		err = config.Checker(ctx, result.State)
		if err == nil {
			err = ctx.Err()
		}
		s.observeError(err)
		if err != nil && !config.KeepOnFailure && !serviceDeviceFault(err) {
			selected, rollbackErr := s.rollbackStage(name, prior, result.State.Version, label+": rollback", marker)
			if selected.Generation != 0 {
				result.Selected = selected
				result.RolledBack = true
			}
			err = errors.Join(err, rollbackErr)
		}
	}
	return result, s.finishPush(result, err, config.Checker != nil)
}

// pushStageLocked runs under both request and mutation gates. It releases the
// request lease and all unpublished owners before any application Checker.
func (s *Service) pushStageLocked(ctx context.Context, name string, source stage.Source, schema StageSpec, label string, snapshot State, prior *stageVersion, artifact *stage.Artifact, result *PushResult, config *CheckConfig, marker **pushCheck) (err error) {
	if err := s.available(); err != nil {
		return err
	}
	if err := s.mutationAvailable(name, nil); err != nil {
		return err
	}
	s.mu.Lock()
	state := s.states[name]
	settings := s.settings[name]
	if settings.check != nil {
		*config = *settings.check
	}
	s.mu.Unlock()
	e := s.stages.entries[name]
	if state.Version != snapshot.Version || state.Generation != snapshot.Generation || state.ConfigRevision != snapshot.ConfigRevision || e == nil || e.current != prior {
		return errors.New("push stage: selected stage or settings changed during compilation")
	}
	if config.Fixture == nil && config.Checker == nil {
		return errors.New("push stage: no replay or application check configured")
	}
	if settings.geometry != nil {
		return errors.New("push stage: kernel geometry settings do not describe stage-owned launches")
	}
	if err := s.stageFixtureAvailable(name, schema, settings, *config); err != nil {
		return err
	}
	if err := s.stageFunctionsAvailable(schema); err != nil {
		return err
	}
	requiredGenerations := uint64(1)
	if config.Checker != nil && !config.KeepOnFailure {
		requiredGenerations++
	}
	if !s.stages.canAdvance(requiredGenerations) {
		return errors.New("stage generation counter exhausted")
	}
	*marker = &pushCheck{}
	s.mu.Lock()
	if s.checking == nil {
		s.checking = make(map[string]*pushCheck)
	}
	s.checking[name] = *marker
	s.mu.Unlock()
	lease, err := s.acquireStage(ctx, name, schema)
	cleanup := s.retainCleanup(lease)
	defer func() { err = errors.Join(err, cleanup.close(ctx)); s.observeError(err) }()
	if err != nil {
		return err
	}
	library, err := stage.Load(artifact)
	cleanup.adopt(library)
	if err != nil {
		return err
	}
	source.Flags = append([]string(nil), source.Flags...)
	candidate := &stageVersion{spec: schema, builder: prior.builder, artifact: artifact, library: library, source: source}
	if f := config.Fixture; f != nil {
		dev, err := s.stageDevice(schema)
		if err != nil {
			return err
		}
		call, err := s.newStageCallable(candidate, dev)
		if err != nil {
			return err
		}
		cleanup.adopt(call)
		f.mu.Lock()
		stored := f.replay.policy
		if config.Policy != nil {
			f.replay.policy = replayPolicy(*config.Policy)
		}
		released := false
		replayCtx := context.WithValue(ctx, pushLeaseKey{}, pushLeaseCapability{s, name, cleanup})
		r, checkErr := f.replay.Check(replayCtx, replayArm{Kernel: call, Release: func() error { err := call.releaseAfterReplay(); released = err == nil; return err }})
		f.replay.policy = stored
		if !released || checkErr != nil {
			f.parent = cleanup
			cleanup.drain = f.close
		}
		f.mu.Unlock()
		setPushReplay(&result.Report, r)
		if checkErr != nil {
			return checkErr
		}
		if !r.Pass {
			return errPushReplayMismatch
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if config.Checker != nil && !config.KeepOnFailure {
		if s.stages.rollback == nil {
			s.stages.rollback = make(map[*pushCheck]struct{})
		}
		s.stages.rollback[*marker] = struct{}{}
	}
	s.stages.next++
	selected := State{Name: name, Kind: "stage", Generation: s.stages.next, Version: s.stages.next, ConfigRevision: snapshot.ConfigRevision, Label: label, SourceDigest: artifact.SourceDigest()}
	candidate.state = selected
	e.versions[selected.Version] = candidate
	e.current = candidate
	s.mu.Lock()
	s.states[name] = selected
	s.mu.Unlock()
	cleanup.owners = nil // Table owns library; replay released its callable first.
	result.State, result.Selected, result.Published = selected, selected, true
	return nil
}

// RevertStage selects the retained original stage and advances Generation.
// Stage functions are repinned on each new invocation; an existing fixture
// keeps its exact recorded baseline function generations until it closes.
func (s *Service) RevertStage(ctx context.Context, name, label string) (State, error) {
	if ctx == nil || s == nil {
		return State{}, errors.New("revert stage: service and context are required")
	}
	s.mutations.RLock()
	e := s.stages.entries[name]
	var version uint64
	if e != nil && e.original != nil {
		version = e.original.state.Version
	}
	s.mutations.RUnlock()
	if version == 0 {
		return State{}, errors.New("revert stage: stage is not registered")
	}
	return s.selectStageVersion(ctx, name, version, label, nil)
}

func (s *Service) selectStageVersion(ctx context.Context, name string, version uint64, label string, own *pushCheck) (state State, err error) {
	s.requests.RLock()
	defer s.requests.RUnlock()
	s.mutations.Lock()
	defer s.mutations.Unlock()
	if err := s.available(); err != nil {
		return State{}, err
	}
	if err := s.mutationAvailable(name, own); err != nil {
		return State{}, err
	}
	e := s.stages.entries[name]
	if e == nil || e.versions[version] == nil {
		return State{}, errors.New("select stage: version is not retained")
	}
	if !s.stages.canAdvance(1) {
		return State{}, errors.New("stage generation counter exhausted")
	}
	v := e.versions[version]
	lease, err := s.acquireStage(ctx, name, v.spec)
	if err != nil {
		return State{}, err
	}
	cleanup := s.retainCleanup(lease)
	defer func() { err = errors.Join(err, cleanup.Close()); s.observeError(err) }()
	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	s.stages.next++
	s.mu.Lock()
	old := s.states[name]
	state = v.state
	state.Generation, state.ConfigRevision, state.Label = s.stages.next, old.ConfigRevision, label
	s.states[name] = state
	s.mu.Unlock()
	e.current = v
	return state, nil
}

// rollbackStage restores an already retained owner. It changes only registry
// selection: no CUDA call, owner release, or new lease is needed. The checking
// marker excludes Close and competing mutations until this restoration ends.
// A canceled application context must not prevent authoritative rollback.
func (s *Service) rollbackStage(name string, prior *stageVersion, candidate uint64, label string, own *pushCheck) (State, error) {
	s.requests.RLock()
	defer s.requests.RUnlock()
	s.mutations.Lock()
	defer s.mutations.Unlock()
	if err := s.available(); err != nil {
		return State{}, err
	}
	if err := s.mutationAvailable(name, own); err != nil {
		return State{}, err
	}
	s.mu.Lock()
	marker := s.checking[name]
	s.mu.Unlock()
	e := s.stages.entries[name]
	if own == nil || marker != own || e == nil || e.current == nil || e.current.state.Version != candidate || prior == nil || e.versions[prior.state.Version] != prior {
		return State{}, errors.New("rollback stage: checking owner or retained selection changed")
	}
	if _, reserved := s.stages.rollback[own]; !reserved || s.stages.next == math.MaxUint64 {
		return State{}, errors.New("rollback stage: reserved generation is unavailable")
	}
	delete(s.stages.rollback, own)
	s.stages.next++
	s.mu.Lock()
	old := s.states[name]
	selected := prior.state
	selected.Generation, selected.ConfigRevision, selected.Label = s.stages.next, old.ConfigRevision, label
	s.states[name] = selected
	s.mu.Unlock()
	e.current = prior
	return selected, nil
}
