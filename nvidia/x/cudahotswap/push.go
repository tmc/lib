package cudahotswap

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"reflect"
	"time"

	"github.com/tmc/lib/nvidia/driver"
)

// PushResult identifies the checked candidate and the final selected code.
// State is the candidate's publication stamp, or the preparation stamp when
// publication was refused. Selected includes a rollback's later generation.
// A nonnil result with an error must not be read as a successful check.
type PushResult struct {
	State      State
	Selected   State
	Published  bool
	RolledBack bool
	Report     Report
}

// ReplayOutputReport retains each declared output's measured errors and noise.
type ReplayOutputReport struct {
	Arg       int           `json:"arg"`
	Noise     ReplayMetrics `json:"noise"`
	Candidate ReplayMetrics `json:"candidate"`
	WriteOnly bool          `json:"write_only"`
	Passed    bool          `json:"passed"`
}

// ReplayTrialReport labels one interleaved arm. DeviceNanos is present only
// for complete native CUDA event intervals; fake clocks are never promoted.
type ReplayTrialReport struct {
	Arm         string `json:"arm"`
	Round       int    `json:"round"`
	Complete    bool   `json:"complete"`
	Digest      string `json:"digest,omitempty"`
	DeviceNanos *int64 `json:"device_nanos,omitempty"`
	Error       string `json:"error,omitempty"`
}

// ReplayReport records replay correctness and its exact timing scope. SignP
// is the paired two-sided sign test, present only for native event samples.
type ReplayReport struct {
	Passed        bool                 `json:"passed"`
	Native        bool                 `json:"native"`
	Clock         string               `json:"clock"`
	Outputs       []ReplayOutputReport `json:"outputs"`
	Trials        []ReplayTrialReport  `json:"trials"`
	CandidateWins int                  `json:"candidate_wins"`
	BaselineWins  int                  `json:"baseline_wins"`
	SignP         *float64             `json:"sign_p,omitempty"`
}

type candidateCompileError struct {
	error
	wall int64
}

func (e *candidateCompileError) Unwrap() error { return e.error }

type candidatePreparationError struct {
	state State
	cause error
}

func (e *candidatePreparationError) Error() string { return e.cause.Error() }
func (e *candidatePreparationError) Unwrap() error { return e.cause }

type pushCheck struct {
	prior    *generationPin
	rollback *generationReservation
	custody  *rollbackCustody
}

// mutationAvailable requires the mutation gate. The application callback may
// launch and pin, but cannot invalidate this request's exact rollback target.
func (s *Service) mutationAvailable(name string, own *pushCheck) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if active := s.checking[name]; active != nil && active != own {
		return errors.New("kernel application check is active")
	}
	return nil
}

type pushLeaseKey struct{}
type pushLeaseCapability struct {
	service *Service
	name    string
	owner   *requestCleanup
}
type borrowedPushLease struct{}

func (borrowedPushLease) Close() error { return nil }

// Only Push supplies this capability. End may relinquish the replay's borrow,
// but only its parent cleanup can release the actual lease after every owner.
func (s *Service) replayLease(ctx context.Context, name string) (io.Closer, error) {
	if capability, ok := ctx.Value(pushLeaseKey{}).(pushLeaseCapability); ok {
		if capability.service != s || capability.name != name || capability.owner == nil || capability.owner.lease == nil {
			return nil, errors.New("replay parent lease is unavailable")
		}
		return borrowedPushLease{}, nil
	}
	return s.acquire(ctx, name)
}

// Push compiles source and checks its actual PTX signature without a lease,
// then loads and checks replay under one request lease. Registered graphs are
// patched before atomic publication. A configured application Checker runs
// after publication, with all service gates and the lease released, so it may
// Launch or Pin through the service. A failed hook rolls back to the exact
// prior selected version under a fresh rollback lease unless KeepOnFailure.
// Rollback uses a separate 30-second cooperative acquisition context, preserving
// validated inherited lease custody. CUDA graph patch calls themselves require
// an outer process watchdog; their execution is not bounded by this context.
//
// Same-kernel mutations and Close are refused while the hook is active. The
// callback must finish or retain submitted launch owners, including on failure.
// Failed drain/cleanup retains candidate, fixture and lease for Close retry.
// Unconfigured checks are refused; no timing falls back to host wall time.
func (s *Service) Push(ctx context.Context, name string, source Source, label string) (*PushResult, error) {
	if ctx == nil || s == nil || s.loader == nil {
		return nil, errors.New("push kernel: service and context are required")
	}
	result := &PushResult{Report: Report{Task: name, Candidate: label, Entry: source.Entry, Outcome: OutcomeOK, Recognized: true, Unmeasured: []string{"application end-to-end timing", "CUPTI kernel activity"}}}
	p, err := s.prepareCandidate(ctx, name, source)
	result.Report.Compile = &CompileReport{SourceSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(source.Code))), SourceBytes: len(source.Code)}
	if p == nil {
		var compile *candidateCompileError
		if errors.As(err, &compile) {
			result.Report.Compile.Log = err.Error()
			result.Report.Compile.WallNanos = compile.wall
		}
		result.State, _ = s.Status(name)
		result.Selected = result.State
		var preparation *candidatePreparationError
		if errors.As(err, &preparation) {
			result.State = preparation.state
		}
		result.State.Label = label
		return result, s.finishPush(result, err, false)
	}
	result.State = p.state
	result.State.Label = label
	result.Selected = p.state
	ptxDigest := fmt.Sprintf("%x", sha256.Sum256([]byte(p.compilation.PTX)))
	result.Report.Compile.PTXSHA256, result.Report.Compile.PTXBytes = ptxDigest, len(p.compilation.PTX)
	result.Report.Compile.WallNanos = p.compileWall
	checking, config, err := s.pushGPU(ctx, p, label, result)
	// The hook may panic or exit its goroutine. Release the CPU rollback pin
	// and checking marker after normal reporting and on every unwind path.
	defer s.releasePushCheck(name, checking)
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
			selected, rollbackErr := s.rollbackCheckedKernel(ctx, name, p.state.Version, label+": rollback", checking)
			if selected.Generation != 0 {
				result.Selected = selected
				result.RolledBack = true
			}
			err = errors.Join(err, rollbackErr)
		}
	}
	return result, s.finishPush(result, err, config.Checker != nil)
}

func (s *Service) pushGPU(ctx context.Context, p *preparedCandidate, label string, result *PushResult) (checking *pushCheck, config CheckConfig, err error) {
	s.requests.RLock()
	defer s.requests.RUnlock()
	s.mutations.Lock()
	defer s.mutations.Unlock()
	defer func() {
		if !result.Published {
			s.mu.Lock()
			result.Selected = s.states[p.name]
			s.mu.Unlock()
		}
	}()
	if err := s.validateCandidate(p); err != nil {
		return nil, config, err
	}
	if err := s.mutationAvailable(p.name, nil); err != nil {
		return nil, config, err
	}
	s.mu.Lock()
	settings := s.settings[p.name]
	if settings.check != nil {
		config = *settings.check
	}
	s.mu.Unlock()
	if config.Fixture == nil && config.Checker == nil {
		return nil, config, errors.New("push kernel: no replay or application check configured")
	}
	if err := s.pushFixtureAvailable(p, settings, config); err != nil {
		return nil, config, err
	}
	prior, _, err := s.table.pin(p.name)
	if err != nil {
		return nil, config, err
	}
	checking = &pushCheck{prior: prior}
	if config.Checker != nil && !config.KeepOnFailure {
		checking.rollback, err = s.table.reserveRollback(p.name, prior.gen)
		if err != nil {
			return checking, config, err
		}
	}
	s.mu.Lock()
	if s.checking == nil {
		s.checking = make(map[string]*pushCheck)
	}
	s.checking[p.name] = checking
	s.mu.Unlock()
	lease, err := s.acquire(ctx, p.name)
	cleanup := s.retainCleanup(lease) // A nonnil lease on error remains owned.
	defer func() { err = errors.Join(err, cleanup.close(ctx)); s.observeError(err) }()
	if err != nil {
		return checking, config, err
	}
	if lease == nil {
		return checking, config, errors.New("push kernel: acquire returned no lease")
	}
	checking.custody, err = captureRollbackCustody(lease)
	if err != nil {
		return checking, config, err
	}

	if err := ctx.Err(); err != nil {
		return checking, config, err
	}
	kernel, err := s.loader.Load(ctx, p.compilation)
	if !nilKernel(kernel) {
		cleanup.adopt(kernel)
	}
	if err != nil {
		var owner io.Closer
		if errors.As(err, &owner) {
			cleanup.adopt(owner)
		}
		return checking, config, err
	}
	if nilKernel(kernel) || kernel.Raw() == 0 {
		return checking, config, errors.New("push kernel: missing loaded function")
	}
	if f := config.Fixture; f != nil {
		f.mu.Lock()
		storedPolicy := f.replay.policy
		if config.Policy != nil {
			f.replay.policy = replayPolicy(*config.Policy)
		}
		// Check takes ownership of Release even on a later error. Do not close
		// this unpublished kernel until the engine has established completion.
		released := false
		replayCtx := context.WithValue(ctx, pushLeaseKey{}, pushLeaseCapability{s, p.name, cleanup})
		r, checkErr := f.replay.Check(replayCtx, replayArm{Kernel: kernel, Release: func() error { released = true; return nil }})
		f.replay.policy = storedPolicy
		if !released || checkErr != nil {
			f.parent = cleanup
			cleanup.drain = f.close
		}
		f.mu.Unlock()
		setPushReplay(&result.Report, r)
		if checkErr != nil {
			return checking, config, checkErr
		}
		if !r.Pass {
			return checking, config, errPushReplayMismatch
		}
	}
	if err := ctx.Err(); err != nil {
		return checking, config, err
	}
	if err := s.validateCandidate(p); err != nil {
		return checking, config, err
	}
	candidate := &generation{digest: p.digest, flags: p.source.Flags, signature: p.signature, function: kernel.Raw(), owner: kernel}
	patch, err := s.graphs.prepare(p.name, candidate)
	if patch != nil {
		cleanup.owners = nil // The patch owns the unpublished kernel on abort.
		cleanup.adopt(patch)
	}
	if err != nil {
		return checking, config, err
	}
	stamp, err := s.table.publish(p.name, candidate, p.state.Version)
	if err != nil {
		return checking, config, err
	}
	patch.published() // Immediate ownership marker, before any fallible step.
	state := State{Name: p.name, Kind: "kernel", Generation: stamp.revision, Version: stamp.version, ConfigRevision: p.state.ConfigRevision, Label: label, SourceDigest: p.digest, CompilerKey: p.compilation.Key}
	s.mu.Lock()
	s.states[p.name] = state
	s.versionKeys[candidate] = p.compilation.Key
	s.rememberWatchInputLocked(p.name, p.source, p.compilation)
	s.mu.Unlock()
	result.State, result.Selected, result.Published = state, state, true
	return checking, config, nil
}

func nilKernel(k Kernel) bool {
	if k == nil {
		return true
	}
	v := reflect.ValueOf(k)
	return v.Kind() == reflect.Pointer && v.IsNil()
}

func (s *Service) pushFixtureAvailable(p *preparedCandidate, settings kernelSettings, config CheckConfig) error {
	f := config.Fixture
	if f == nil {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.service != s || f.state.Name != p.name || f.replay == nil {
		return errors.New("push kernel: recorded fixture is closed or belongs elsewhere")
	}
	r := f.replay
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.ready || r.closed || r.cleanupStarted {
		return errors.New("push kernel: recorded fixture is unavailable")
	}
	if err := CheckABI(r.spec.Signature, p.signature); err != nil {
		return err
	}
	if (settings.geometry == nil) != (f.geometry == nil) || (settings.geometry != nil && *settings.geometry != *f.geometry) {
		return errors.New("push kernel: geometry settings changed since fixture recording")
	}
	if g := settings.geometry; g != nil && (g.Grid != r.spec.Config.Grid || g.Block != r.spec.Config.Block || g.SharedMem != r.spec.Config.SharedMem) {
		return errors.New("push kernel: geometry differs from recorded fixture")
	}
	policy := r.policy
	if config.Policy != nil {
		policy = replayPolicy(*config.Policy)
	}
	if err := validateReplay(r.spec, policy); err != nil {
		return err
	}
	for _, noise := range r.noise {
		if noise.NonFinite != 0 || noise.MaxAbs == nil || noise.RMS == nil || *noise.MaxAbs > policy.NoiseMaxAbs || *noise.RMS > policy.NoiseRMS {
			return errors.New("push kernel: baseline noise exceeds configured policy")
		}
	}
	return nil
}

var errPushReplayMismatch = errors.New("push kernel: candidate replay differs from recorded baseline")

func (s *Service) finishPush(result *PushResult, err error, app bool) error {
	s.observeError(err)
	pushReportError(&result.Report, err, app)
	result.Report.Generation = result.State.Generation
	result.Report.ConfigRevision = result.State.ConfigRevision
	_, eventErr := s.publishReport(result.State, &result.Report)
	return errors.Join(err, eventErr)
}

func (s *Service) releasePushCheck(name string, checking *pushCheck) {
	if checking == nil {
		return
	}
	s.mutations.Lock()
	defer s.mutations.Unlock()
	// Dropping this CPU pin does not collect modules outside a GPU lease.
	// Retired modules remain table-owned until later leased collection.
	s.table.releaseRollback(checking.rollback)
	s.table.mu.Lock()
	if !checking.prior.released.Swap(true) {
		checking.prior.gen.pins.Add(-1)
	}
	s.table.mu.Unlock()
	s.mu.Lock()
	if s.checking[name] == checking {
		delete(s.checking, name)
	}
	s.mu.Unlock()
}

func pushReportError(report *Report, err error, app bool) {
	if err == nil {
		return
	}
	if serviceDeviceFault(err) {
		report.Outcome, report.Recognized, report.ContextPoisoned = pushDeviceOutcome(err), true, true
	} else if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		report.Outcome, report.Recognized = OutcomeHarnessError, false
	} else {
		var compile *candidateCompileError
		switch {
		case errors.As(err, &compile):
			report.Outcome, report.Recognized = OutcomeCompileError, true
		case errors.Is(err, errPushReplayMismatch):
			report.Outcome, report.Recognized = OutcomeWrongAnswer, true
		default:
			report.Outcome, report.Recognized = Classify(err)
			if app && !report.Recognized {
				report.Outcome, report.Recognized = OutcomeWrongAnswer, true
			}
		}
	}
	report.Reason = err.Error()
}

func pushDeviceOutcome(err error) Outcome {
	if result, ok := err.(driver.Result); ok {
		outcome, _ := Classify(result)
		if outcome.PoisonsContext() {
			return outcome
		}
	}
	switch err := err.(type) {
	case interface{ Unwrap() []error }:
		for _, child := range err.Unwrap() {
			if serviceDeviceFault(child) {
				return pushDeviceOutcome(child)
			}
		}
	case interface{ Unwrap() error }:
		return pushDeviceOutcome(err.Unwrap())
	}
	return OutcomeDeviceFault
}

func setPushReplay(report *Report, r replayResult) {
	result := &ReplayReport{Passed: r.Pass, Native: r.Native, Clock: string(r.Timing)}
	for _, out := range r.Outputs {
		result.Outputs = append(result.Outputs, ReplayOutputReport{out.Arg, cloneReplayMetrics(out.Noise), cloneReplayMetrics(out.Candidate), out.WriteOnly, out.Pass})
	}
	var base, candidate []int64
	native := r.Native && r.Timing == "cuda_event_interval"
	for _, receipt := range r.Trials {
		trial := ReplayTrialReport{Arm: receipt.Name, Round: receipt.Round, Complete: receipt.Complete, Digest: receipt.Digest}
		if receipt.Err != nil {
			trial.Error = receipt.Err.Error()
		}
		duration, err := receipt.DurationFor(r.Timing)
		if native && receipt.Complete && receipt.Err == nil && err == nil && duration > 0 {
			ns := int64(duration)
			trial.DeviceNanos = &ns
			if receipt.Name == "baseline" {
				base = append(base, ns)
			} else {
				candidate = append(candidate, ns)
			}
		} else {
			native = false
		}
		result.Trials = append(result.Trials, trial)
	}
	if native && len(base) != 0 && len(base) == len(candidate) {
		result.SignP = &r.SignP
		result.CandidateWins, result.BaselineWins = r.CandidateWins, r.BaseWins
		report.Baseline = &TimingReport{Source: "cuda_event_interval", LaunchesPerSample: 1, SampleNanos: base, InputPolicy: "restored fixture; eviction after restore"}
		report.Timing = &TimingReport{Source: "cuda_event_interval", LaunchesPerSample: 1, SampleNanos: candidate, InputPolicy: "restored fixture; eviction after restore"}
		report.Baseline.Summarize()
		report.Timing.Summarize()
		report.Speedup = &SpeedupReport{Method: "alternating cudatrial arms; paired sign test", Ratio: report.Baseline.MedianNanosPerLaunch / report.Timing.MedianNanosPerLaunch, Note: "device-event comparison of these arms only"}
	} else {
		report.Unmeasured = append(report.Unmeasured, "complete native CUDA-event comparison")
	}
	report.Replay = result
}

func cloneReplayMetrics(m ReplayMetrics) ReplayMetrics {
	if m.MaxAbs != nil {
		v := *m.MaxAbs
		m.MaxAbs = &v
	}
	if m.RMS != nil {
		v := *m.RMS
		m.RMS = &v
	}
	if m.MaxULP != nil {
		v := *m.MaxULP
		m.MaxULP = &v
	}
	return m
}

// The detached context only bounds cooperative lease acquisition. Synchronous
// driver graph calls still require the caller's outer process watchdog.
func (s *Service) rollbackCheckedKernel(ctx context.Context, name string, version uint64, label string, checking *pushCheck) (State, error) {
	if checking == nil || checking.rollback == nil {
		return State{}, errors.New("rollback kernel: missing reserved checking owner")
	}
	if err := checking.custody.verifyInherited(); err != nil {
		return State{}, err
	}
	rollbackCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
	defer cancel()
	return s.selectRetainedChecking(rollbackCtx, name, version, false, label, checking)
}
