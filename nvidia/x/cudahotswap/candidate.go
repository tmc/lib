package cudahotswap

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// preparedCandidate owns no loaded module. Push must validate its snapshot
// again under the mutation lock before loading or publishing the candidate.
type preparedCandidate struct {
	name        string
	state       State
	source      Source
	digest      string
	signature   cudalaunch.Signature
	compilation *Compilation
	compileWall int64
}

// prepareCandidate does only host work. It snapshots the selected version and
// settings before compilation, then refuses results that have become stale.
// Explicit settings replace source flags, including an explicitly empty set.
// Otherwise nil source flags inherit the selected version's compiler flags.
func (s *Service) prepareCandidate(ctx context.Context, name string, source Source) (prepared *preparedCandidate, err error) {
	if ctx == nil {
		return nil, errors.New("prepare candidate: nil context")
	}
	if s == nil || s.loader == nil {
		return nil, errors.New("prepare candidate: nil service or loader")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.requests.RLock()
	defer s.requests.RUnlock()
	s.mutations.RLock()
	if err := s.available(); err != nil {
		s.mutations.RUnlock()
		return nil, err
	}
	if err := s.mutationAvailable(name, nil); err != nil {
		s.mutations.RUnlock()
		return nil, err
	}
	s.mu.Lock()
	state, ok := s.states[name]
	settings := s.settings[name]
	settings.flags = append([]string(nil), settings.flags...)
	s.mu.Unlock()
	if !ok {
		s.mutations.RUnlock()
		return nil, errors.New("prepare candidate: kernel is not registered")
	}
	defer func() {
		if err != nil {
			err = &candidatePreparationError{state: state, cause: err}
		}
	}()
	s.table.mu.RLock()
	e := s.table.entries[name]
	var current *generation
	if e != nil {
		current = e.current.Load()
	}
	if s.table.closed || current == nil || current.retired || current.version != state.Version || e.revision != state.Generation {
		s.table.mu.RUnlock()
		s.mutations.RUnlock()
		return nil, errors.New("prepare candidate: selected version is unavailable")
	}
	baseline := current.signature
	baseline.Params = append([]cudalaunch.Param(nil), baseline.Params...)
	switch {
	case settings.flagsSet:
		source.Flags = settings.flags
	case source.Flags == nil:
		source.Flags = append([]string(nil), current.flags...)
	default:
		source.Flags = append([]string(nil), source.Flags...)
	}
	s.table.mu.RUnlock()
	s.mutations.RUnlock()

	start := time.Now()
	c, err := s.loader.Compile(ctx, source)
	compileWall := time.Since(start).Nanoseconds()
	if err != nil {
		var owner io.Closer
		if errors.As(err, &owner) {
			// Compilation cleanup needs no GPU lease. Retain a failed owner
			// through the same retry mechanism as Register.
			s.mutations.Lock()
			cleanup := s.retainCleanup(nil)
			cleanup.adopt(owner)
			err = errors.Join(err, cleanup.Close())
			s.mutations.Unlock()
		}
		return nil, &candidateCompileError{error: err, wall: compileWall}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c == nil || c.Entry != source.Entry {
		return nil, errors.New("prepare candidate: missing or mismatched compilation")
	}
	// A successful compiler result defines the current disk source's include
	// closure even when its later PTX/ABI check refuses publication. Header
	// edits must be able to repair that candidate through the watcher.
	s.mu.Lock()
	s.rememberWatchInputLocked(name, source, c)
	s.mu.Unlock()
	signature, err := cudalaunch.SignatureFromPTX(c.PTX, c.Entry)
	if err != nil {
		return nil, err
	}
	if err := CheckABI(baseline, signature); err != nil {
		return nil, err
	}
	p := &preparedCandidate{name: name, state: state, source: source, digest: fmt.Sprintf("%x", sha256.Sum256([]byte(source.Code))), signature: signature, compilation: c, compileWall: compileWall}
	s.mutations.RLock()
	err = s.validateCandidate(p)
	s.mutations.RUnlock()
	if err != nil {
		return nil, err
	}
	return p, nil
}

// validateCandidate requires the caller to hold the mutation lock. It is also
// used immediately before GPU work, because preparation's final check cannot
// prevent a later selection or settings change.
func (s *Service) validateCandidate(p *preparedCandidate) error {
	if err := s.available(); err != nil {
		return err
	}
	if p == nil {
		return errors.New("prepare candidate: missing snapshot")
	}
	s.mu.Lock()
	state, ok := s.states[p.name]
	s.mu.Unlock()
	if !ok || state.Generation != p.state.Generation || state.Version != p.state.Version || state.ConfigRevision != p.state.ConfigRevision {
		return errors.New("prepare candidate: kernel or settings changed during compilation")
	}
	s.table.mu.RLock()
	defer s.table.mu.RUnlock()
	e := s.table.entries[p.name]
	if s.table.closed || e == nil || e.current.Load() == nil || e.current.Load().retired || e.current.Load().version != p.state.Version || e.revision != p.state.Generation {
		return errors.New("prepare candidate: selected version is unavailable")
	}
	return nil
}
