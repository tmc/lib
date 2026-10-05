package cudahotswap

import (
	"context"
	"errors"
)

// Checker checks the application after a candidate is selected. It receives
// the selected kernel and settings stamp. It may launch through Service using
// ctx; Push releases its GPU lease before calling it. A nil error means the
// application check passed. The hook must finish its submitted work before
// returning, including on errors, or retain the returned launch owners.
type Checker func(ctx context.Context, state State) error

// CheckConfig selects replay, an application check, or both. Replay precedes
// publication; Checker follows publication. At least one check is required.
// Policy overrides the fixture's recorded policy; nil uses its recorded policy.
// KeepOnFailure retains published code when Checker fails. It never admits
// failed replay or suppresses a poisoned CUDA context.
type CheckConfig struct {
	Fixture       *Fixture
	Policy        *ReplayPolicy
	Checker       Checker
	KeepOnFailure bool
}

// SetCheck copies per-kernel check settings. The fixture must belong to this
// service and kernel and must have recorded successfully. The caller keeps it
// open while configured; Push refuses a fixture closed after configuration.
// This changes ConfigRevision and label without changing selected code or
// acquiring a GPU lease. A failed call changes neither settings nor revision.
func (s *Service) SetCheck(name string, config CheckConfig, label string) (State, error) {
	if s == nil || s.loader == nil {
		return State{}, errors.New("set check: nil service or loader")
	}
	if config.Fixture == nil && config.Checker == nil {
		return State{}, errors.New("set check: no replay or application check")
	}
	if config.Fixture == nil && config.Policy != nil {
		return State{}, errors.New("set check: policy without a fixture")
	}
	s.mutations.Lock()
	defer s.mutations.Unlock()
	if err := s.available(); err != nil {
		return State{}, err
	}
	if f := config.Fixture; f != nil {
		f.mu.Lock()
		defer f.mu.Unlock()
		if f.service != s || f.state.Name != name || f.replay == nil {
			return State{}, errors.New("set check: fixture belongs to another service or kernel, or is closed")
		}
		r := f.replay
		r.mu.Lock()
		defer r.mu.Unlock()
		if !r.ready || r.closed || r.cleanupStarted {
			return State{}, errors.New("set check: fixture is unavailable")
		}
		policy := ReplayPolicy(r.policy)
		if config.Policy != nil {
			policy = *config.Policy
		}
		if err := validateReplay(r.spec, replayPolicy(policy)); err != nil {
			return State{}, err
		}
		for _, noise := range r.noise {
			if noise.NonFinite != 0 || noise.MaxAbs == nil || noise.RMS == nil || *noise.MaxAbs > policy.NoiseMaxAbs || *noise.RMS > policy.NoiseRMS {
				return State{}, errors.New("set check: recorded noise exceeds policy")
			}
		}
		config.Policy = &policy
	}
	return s.setSettingsLocked(name, label, func(c *kernelSettings) { c.check = &config })
}
