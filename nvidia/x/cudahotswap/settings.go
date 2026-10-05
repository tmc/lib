package cudahotswap

import (
	"errors"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// LaunchGeometry overrides launch dimensions and dynamic shared memory.
// It does not change the stream. Device-specific limits are checked at launch.
type LaunchGeometry struct {
	Grid      cudashape.Grid3D
	Block     driver.Dim3
	SharedMem uint32
}

type kernelSettings struct {
	flags    []string
	flagsSet bool
	geometry *LaunchGeometry
	check    *CheckConfig
	watch    *WatchPaths
}

// SetCompileFlags copies the flags for subsequent candidate compilations.
// It does not recompile or change the selected kernel. A successful call
// advances ConfigRevision and records label.
func (s *Service) SetCompileFlags(name string, flags []string, label string) (State, error) {
	if len(flags) > 64 {
		return State{}, errors.New("set compile flags: too many flags")
	}
	total := 0
	for _, flag := range flags {
		if flag == "" || strings.ContainsRune(flag, 0) || len(flag) > (16<<10)-total {
			return State{}, errors.New("set compile flags: invalid or oversized flags")
		}
		total += len(flag)
	}
	return s.setSettings(name, label, func(c *kernelSettings) {
		c.flags = append([]string(nil), flags...)
		c.flagsSet = true
	})
}

// SetGeometry copies a launch override. A nil override restores caller-supplied
// geometry. A successful call advances ConfigRevision without changing the
// kernel generation.
func (s *Service) SetGeometry(name string, geometry *LaunchGeometry, label string) (State, error) {
	var copy *LaunchGeometry
	if geometry != nil {
		g := *geometry
		if g.Grid.X == 0 || g.Grid.X > 1<<31-1 || g.Grid.Y == 0 || g.Grid.Y > 65535 || g.Grid.Z == 0 || g.Grid.Z > 65535 || g.Block.X == 0 || g.Block.Y == 0 || g.Block.Z == 0 || uint64(g.Block.X)*uint64(g.Block.Y) > 1024 || uint64(g.Block.X)*uint64(g.Block.Y)*uint64(g.Block.Z) > 1024 {
			return State{}, errors.New("set geometry: unsupported launch dimensions")
		}
		copy = &g
	}
	return s.setSettings(name, label, func(c *kernelSettings) { c.geometry = copy })
}

func (s *Service) setSettings(name, label string, update func(*kernelSettings)) (State, error) {
	if s == nil || s.loader == nil {
		return State{}, errors.New("set settings: nil service or loader")
	}
	s.mutations.Lock()
	defer s.mutations.Unlock()
	return s.setSettingsLocked(name, label, update)
}

// setSettingsLocked requires the mutation lock.
func (s *Service) setSettingsLocked(name, label string, update func(*kernelSettings)) (State, error) {
	if err := s.available(); err != nil {
		return State{}, err
	}
	if err := s.mutationAvailable(name, nil); err != nil {
		return State{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, ok := s.states[name]
	if !ok {
		return State{}, errors.New("set settings: kernel is not registered")
	}
	if state.ConfigRevision == math.MaxUint64 {
		return State{}, errors.New("set settings: config revision exhausted")
	}
	c := s.settings[name]
	update(&c)
	if s.settings == nil {
		s.settings = make(map[string]kernelSettings)
	}
	s.settings[name] = c
	state.ConfigRevision++
	state.Label = label
	s.states[name] = state
	return state, nil
}

// Status returns a detached snapshot of a registered kernel's current state.
func (s *Service) Status(name string) (State, error) {
	if s == nil {
		return State{}, errors.New("kernel status: nil service")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, ok := s.states[name]
	if !ok {
		return State{}, errors.New("kernel status: kernel is not registered")
	}
	return state, nil
}
