package cudahotswap

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// Version identifies retained compiled code. Number is State.Version, not the
// mutation revision in State.Generation. Flags are a detached copy.
type Version struct {
	Number       uint64
	SourceDigest string
	CompilerKey  string
	Flags        []string
	Current      bool
	Original     bool
}

// Versions returns retained code in increasing version order. It takes no GPU
// lease. The returned slice and flags can be changed without changing Service.
func (s *Service) Versions(name string) ([]Version, error) {
	if s == nil || s.loader == nil {
		return nil, errors.New("list versions: nil service or loader")
	}
	s.mutations.RLock()
	defer s.mutations.RUnlock()
	if err := s.available(); err != nil {
		return nil, err
	}
	s.table.mu.RLock()
	defer s.table.mu.RUnlock()
	e := s.table.entries[name]
	if e == nil || s.table.closed {
		return nil, errors.New("list versions: kernel is not registered")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var versions []Version
	for number, g := range e.versions {
		if !g.retired {
			versions = append(versions, Version{Number: number, SourceDigest: g.digest, CompilerKey: s.versionKeys[g], Flags: append([]string(nil), g.flags...), Current: g == e.current.Load(), Original: number == e.original})
		}
	}
	sort.Slice(versions, func(i, j int) bool { return versions[i].Number < versions[j].Number })
	return versions, nil
}

// Select switches to retained compiled code without recompiling. It patches
// registered graph nodes before publication and advances Generation, preserving
// ConfigRevision. A nonzero State with a cleanup error identifies a committed
// selection whose request owner remains retained until Service.Close retries it.
func (s *Service) Select(ctx context.Context, name string, version uint64, label string) (State, error) {
	return s.selectRetained(ctx, name, version, false, label)
}

// Revert selects the original retained code and advances Generation. It does
// not reset settings or ConfigRevision. Its cleanup contract is the same as Select.
func (s *Service) Revert(ctx context.Context, name, label string) (State, error) {
	return s.selectRetained(ctx, name, 0, true, label)
}

func (s *Service) selectRetained(ctx context.Context, name string, version uint64, original bool, label string) (state State, err error) {
	return s.selectRetainedChecking(ctx, name, version, original, label, nil)
}

func (s *Service) selectRetainedChecking(ctx context.Context, name string, version uint64, original bool, label string, checking *pushCheck) (state State, err error) {
	if ctx == nil {
		return State{}, errors.New("select kernel: nil context")
	}
	if s == nil || s.loader == nil || s.graphs == nil {
		return State{}, errors.New("select kernel: nil service or graph owner")
	}
	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	s.requests.RLock()
	defer s.requests.RUnlock()
	s.mutations.Lock()
	defer s.mutations.Unlock()
	if err := s.available(); err != nil {
		return State{}, err
	}
	if err := s.mutationAvailable(name, checking); err != nil {
		return State{}, err
	}
	s.table.mu.RLock()
	e := s.table.entries[name]
	if e == nil || s.table.closed {
		s.table.mu.RUnlock()
		return State{}, errors.New("select kernel: kernel is not registered")
	}
	if original {
		version = e.original
	}
	target := e.versions[version]
	if target == nil || target.retired || target.owner == nil {
		s.table.mu.RUnlock()
		return State{}, errors.New("select kernel: version is not retained")
	}
	s.table.mu.RUnlock()
	if checking != nil {
		if checking.prior == nil || checking.prior.gen != target || checking.rollback == nil {
			return State{}, errors.New("rollback kernel: retained owner changed")
		}
		if err := checking.custody.verifyInherited(); err != nil {
			return State{}, err
		}
	}
	lease, err := s.acquire(ctx, name)
	cleanup := s.retainCleanup(lease)
	defer func() { err = errors.Join(err, cleanup.close(ctx)); s.observeError(err) }()
	if err != nil {
		return State{}, err
	}
	if lease == nil {
		return State{}, errors.New("select kernel: acquire returned no lease")
	}
	if checking != nil {
		if err := checking.custody.verifyLease(lease); err != nil {
			return State{}, err
		}
	}

	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	patch, err := s.graphs.prepareRetained(name, target)
	if patch != nil {
		cleanup.adopt(patch)
	}
	if err != nil {
		return State{}, err
	}
	var stamp generationStamp
	if checking != nil {
		stamp, err = s.table.selectReserved(name, version, checking.rollback)
	} else {
		stamp, err = s.table.selectVersion(name, version)
	}
	if err != nil {
		return State{}, err
	}
	patch.published() // No fallible step may precede the ownership marker.
	s.mu.Lock()
	state = s.states[name]
	state.Generation = stamp.revision
	state.Version = stamp.version
	state.SourceDigest = target.digest
	state.CompilerKey = s.versionKeys[target]
	state.Label = label
	s.states[name] = state
	s.mu.Unlock()
	return state, nil
}

// rollbackCustody binds default request leases across application callbacks.
// A custom test acquisition has no native custody claim.
type rollbackCustody struct {
	dir       string
	owner     cudalock.Owner
	inherited bool
}

func captureRollbackCustody(lease io.Closer) (*rollbackCustody, error) {
	actual, ok := lease.(requestLease)
	if !ok {
		return nil, nil
	}
	if actual.handle == nil {
		return nil, errors.New("rollback custody: missing request handle")
	}
	c := &rollbackCustody{dir: actual.handle.Dir(), owner: actual.handle.Owner()}
	c.inherited = actual.handle.IsNested() && c.owner.PID != os.Getpid()
	if err := c.verifyInherited(); err != nil {
		return nil, err
	}
	return c, nil
}

func sameRollbackOwner(a, b cudalock.Owner) bool {
	return a.PID == b.PID && a.Host == b.Host && a.Epoch == b.Epoch && a.At.Equal(b.At)
}

// Verify the original directory before acquisition so changed inheritance
// cannot cause fallback to an unrelated root lease.
func (c *rollbackCustody) verifyInherited() error {
	if c == nil {
		return nil
	}
	if filepath.Clean(c.dir) != filepath.Clean(cudalock.DefaultDir()) {
		return errors.New("rollback custody: lock directory changed")
	}
	if c.inherited {
		owner, err := cudalock.Inherited(os.Getenv)
		if err != nil {
			return fmt.Errorf("rollback custody: %w", err)
		}
		if !sameRollbackOwner(c.owner, owner) {
			return errors.New("rollback custody: inherited owner changed")
		}
	}
	return nil
}

func (c *rollbackCustody) verifyLease(lease io.Closer) error {
	if c == nil {
		return nil
	}
	if err := c.verifyInherited(); err != nil {
		return err
	}
	actual, ok := lease.(requestLease)
	if !ok || actual.handle == nil || filepath.Clean(actual.handle.Dir()) != filepath.Clean(c.dir) {
		return errors.New("rollback custody: acquired a different request lease")
	}
	if c.inherited {
		if !actual.handle.IsNested() || !sameRollbackOwner(c.owner, actual.handle.Owner()) {
			return errors.New("rollback custody: acquired a different inherited owner")
		}
	} else if actual.handle.Owner().PID != c.owner.PID || actual.handle.Owner().Host != c.owner.Host {
		return errors.New("rollback custody: acquired a different standalone owner")
	}
	return nil
}
