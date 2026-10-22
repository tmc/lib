package dsv4

import (
	"errors"
	"fmt"
	"io"
	"math"
)

// Drafter proposes tokens following token at pos. A Drafter must not mutate
// the verifier state while producing a proposal.
type Drafter interface {
	Draft(token, pos, max int) ([]int, error)
}

// DraftLifecycle commits or discards a prepared drafter's lookahead. The
// prepared seed belongs to the active frontier and survives DiscardDrafts.
// Neither method may change the verifier. A failed method must leave the
// active support frontier unchanged. DiscardDrafts must also work after Draft
// or CommitDrafts fails. A cleanup failure makes the combined request unsafe
// to reuse.
type DraftLifecycle interface {
	// PreparedSeed reports the committed seed token and the next target
	// position bound during preparation. It is read-only and returns an
	// error when no current preparation is available.
	PreparedSeed() (token, pos int, err error)
	CommitDrafts(n int) error
	DiscardDrafts() error
}

// PreparedResult reports an explicitly prepared greedy cycle. Declined is
// true only when an empty proposal or free first-token mismatch skipped
// verification; both active frontiers are unchanged and lookahead is gone.
// The caller may perform ordinary target evaluation and prepare a fresh seed.
// A verified zero-acceptance cycle has Declined=false and Committed=0.
type PreparedResult struct {
	SpeculativeResult
	Declined bool
}

// SpeculativeVerifier is the optional backend capability used by speculative
// decoding. VerifyBatch evaluates seed followed by drafts and returns one
// next-token argmax per evaluated row. Implementations own the row logits and
// the state represented by CommitPrefix.
//
// Backends that do not implement this capability must fail with
// ErrUnimplemented; callers must not silently run a six-row batch as six
// ordinary decodes and call that speculative execution.
type SpeculativeVerifier interface {
	CurrentLogits() []float32
	VerifyBatch(seed int, drafts []int, pos int) ([]int, error)
	CommitPrefix(n int) error
	Snapshot() (Snapshot, error)
	Restore(Snapshot) error
}

// SpeculativeBackend is the optional Backend capability for creating a
// verifier bound to one request State.
type SpeculativeBackend interface {
	NewSpeculativeVerifier(State) (SpeculativeVerifier, error)
}

// SpeculativeResult describes one greedy verification cycle. Accepted is the
// number of draft tokens retained. Committed is the same number when the
// cycle succeeds; it is zero when no draft was accepted and the frontier was
// restored.
type SpeculativeResult struct {
	Drafts    []int
	Tops      []int
	Accepted  int
	Committed int
}

// SpeculativeLimits bounds a verification cycle before a backend can allocate
// a batch or device snapshot. A zero VocabSize leaves token IDs unbounded;
// backends that know their vocabulary should set it.
type SpeculativeLimits struct {
	MaxDrafts int
	VocabSize int
}

// DefaultSpeculativeLimits is the largest batch currently covered by the
// host and CUDA speculative contracts.
var DefaultSpeculativeLimits = SpeculativeLimits{MaxDrafts: 8}

// NewSpeculator binds a drafter to a verifier. It is deliberately separate
// from Backend so unsupported backends cannot accidentally alias mutable
// request state.
// The caller retains ownership of the verifier and drafter; close the
// Speculator before releasing either owner or their underlying request.
func NewSpeculator(verifier SpeculativeVerifier, drafter Drafter) (*Speculator, error) {
	return NewSpeculatorWithLimits(verifier, drafter, DefaultSpeculativeLimits)
}

// NewSpeculatorWithLimits binds a drafter to a verifier with explicit bounds.
// Keeping these checks above the verifier boundary prevents a future device
// implementation from allocating for an unbounded or malformed proposal.
func NewSpeculatorWithLimits(verifier SpeculativeVerifier, drafter Drafter, limits SpeculativeLimits) (*Speculator, error) {
	if verifier == nil || drafter == nil {
		return nil, ErrUnimplemented
	}
	if limits.MaxDrafts <= 0 {
		return nil, errors.New("speculative maximum draft count must be positive")
	}
	if limits.VocabSize < 0 {
		return nil, errors.New("speculative vocabulary size must not be negative")
	}
	return &Speculator{verifier: verifier, drafter: drafter, limits: limits}, nil
}

// NewSpeculatorForBackend obtains the optional verifier from a backend and
// fails closed when the backend has no speculative implementation. In
// particular, it never aliases a normal State or silently falls back to
// ordinary one-token evaluation.
// It owns the created verifier if that verifier implements io.Closer. Close
// releases the frontier before the owned verifier; the caller releases State
// afterward. A constructor cleanup failure returns an error implementing
// io.Closer, which must be retried before State is released.
func NewSpeculatorForBackend(backend Backend, state State, drafter Drafter) (*Speculator, error) {
	if backend == nil || state == nil {
		return nil, ErrUnimplemented
	}
	b, ok := backend.(SpeculativeBackend)
	if !ok {
		return nil, ErrUnimplemented
	}
	verifier, err := b.NewSpeculativeVerifier(state)
	owner, _ := verifier.(io.Closer)
	if err != nil {
		return nil, closeSpeculativeConstructor(owner, fmt.Errorf("new speculative verifier: %w", err))
	}
	s, err := NewSpeculator(verifier, drafter)
	if err != nil {
		return nil, closeSpeculativeConstructor(owner, err)
	}
	s.owned = owner
	return s, nil
}

func closeSpeculativeConstructor(owner io.Closer, cause error) error {
	s := &Speculator{owned: owner, failed: cause}
	if err := s.Close(); err != nil {
		return err
	}
	return cause
}

// Speculator runs one speculative verification cycle. The verifier snapshot
// is taken before the batch. If the first draft is rejected, the snapshot is
// restored and no speculative state remains. For a partial acceptance,
// CommitPrefix is responsible for retaining the accepted prefix and dropping
// the unaccepted suffix.
type Speculator struct {
	verifier SpeculativeVerifier
	drafter  Drafter
	limits   SpeculativeLimits
	failed   error
	pending  Snapshot
	owned    io.Closer
	closed   bool
}

// Close disables verification and releases a snapshot retained after failed
// cleanup. A release error preserves the snapshot for another Close. Neither
// successful cleanup nor repeated Close reopens verification. Close does not
// close a directly supplied verifier, drafter, or underlying request; release
// those afterward. A verifier created by NewSpeculatorForBackend is closed
// after the retained snapshot; a failure retains that verifier for retry.
func (s *Speculator) Close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	if s.pending != nil {
		if err := s.pending.Close(); err != nil {
			return &speculativeCleanupError{cause: errors.Join(s.failed, fmt.Errorf("close retained speculative frontier: %w", err)), owner: s}
		}
		s.pending = nil
	}
	if s.owned != nil {
		if err := s.owned.Close(); err != nil {
			return &speculativeCleanupError{cause: errors.Join(s.failed, ErrDesynchronized, fmt.Errorf("close speculative verifier: %w", err)), owner: s}
		}
		s.owned = nil
	}
	return nil
}

// A failed cycle transfers retryable cleanup through the error as io.Closer.
// Keeping the owner here also covers callers that cannot retain a constructor
// result after a partial snapshot acquisition.
type speculativeCleanupError struct {
	cause error
	owner io.Closer
}

func (e *speculativeCleanupError) Error() string { return e.cause.Error() }
func (e *speculativeCleanupError) Unwrap() error { return e.cause }
func (e *speculativeCleanupError) Close() error  { return e.owner.Close() }

// Verify drafts and greedily accepts the prefix whose verifier argmax agrees
// with the draft. Empty drafts, malformed verifier output, and a proposal
// whose first token is not the current argmax are errors rather than silent
// fallbacks.
func (s *Speculator) Verify(token, pos, max int) (SpeculativeResult, error) {
	r, err := s.verify(token, pos, max, nil, false)
	return r.SpeculativeResult, err
}

// VerifyPrepared verifies a proposal whose seed was prepared from the
// committed target frontier. The drafter must implement DraftLifecycle.
// Empty proposals and free first-token mismatches decline after discarding
// lookahead. Verify retains its stricter error behavior for those cases.
//
// Target and support commits must both succeed. Failure restores the target
// frontier and discards support lookahead. Restore, discard, or snapshot Close
// failure returns ErrDesynchronized and permanently disables this Speculator;
// the caller must close the combined request. Snapshot Close runs exactly
// once during the cycle, and its error is joined with any earlier error. A
// failed release is retained for Close; the returned error also implements
// io.Closer so outer owners can release it before their underlying request.
// Invalid limits, seed IDs, positions, or seed/frontier mismatches leave
// preparation untouched and reusable; Draft and cleanup do not run.
func (s *Speculator) VerifyPrepared(token, pos, max int) (PreparedResult, error) {
	if s == nil {
		return PreparedResult{}, ErrUnimplemented
	}
	if s.closed {
		return PreparedResult{}, ErrClosed
	}
	lifecycle, ok := s.drafter.(DraftLifecycle)
	if !ok {
		return PreparedResult{}, ErrUnimplemented
	}
	return s.verify(token, pos, max, lifecycle, true)
}

func (s *Speculator) verify(token, pos, max int, lifecycle DraftLifecycle, prepared bool) (result PreparedResult, err error) {
	if prepared {
		defer func() {
			if err != nil {
				result = PreparedResult{}
			}
		}()
	}
	if s == nil || s.verifier == nil || s.drafter == nil {
		return PreparedResult{}, ErrUnimplemented
	}
	if s.closed {
		return PreparedResult{}, ErrClosed
	}
	if s.failed != nil {
		return PreparedResult{}, s.failed
	}
	wrap := func(r SpeculativeResult) PreparedResult { return PreparedResult{SpeculativeResult: r} }
	discard := func(cause error) error {
		if lifecycle != nil {
			if cleanup := lifecycle.DiscardDrafts(); cleanup != nil {
				cause = errors.Join(cause, ErrDesynchronized, fmt.Errorf("discard draft lookahead: %w", cleanup))
				s.failed = cause
			}
		}
		if errors.Is(cause, ErrDesynchronized) {
			s.failed = cause
		}
		return cause
	}
	if max <= 0 {
		return PreparedResult{}, errors.New("speculative draft limit must be positive")
	}
	if max > s.limits.MaxDrafts {
		return PreparedResult{}, fmt.Errorf("speculative draft limit %d exceeds maximum %d", max, s.limits.MaxDrafts)
	}
	if prepared {
		if err := s.validateToken(token); err != nil {
			return PreparedResult{}, fmt.Errorf("prepared seed: %w", err)
		}
		if pos < 0 {
			return PreparedResult{}, errors.New("prepared position is negative")
		}
		seed, position, err := lifecycle.PreparedSeed()
		if err != nil {
			if errors.Is(err, ErrDesynchronized) {
				s.failed = err
			}
			return PreparedResult{}, fmt.Errorf("prepared frontier: %w", err)
		}
		if seed != token || position != pos {
			return PreparedResult{}, errors.New("prepared seed does not match committed frontier")
		}
	}
	drafts, err := s.drafter.Draft(token, pos, max)
	if err != nil {
		return PreparedResult{}, discard(fmt.Errorf("draft tokens: %w", err))
	}
	drafts = append([]int(nil), drafts...)
	if len(drafts) == 0 {
		if prepared {
			return PreparedResult{Declined: true}, discard(nil)
		}
		return PreparedResult{}, errors.New("draft returned no tokens")
	}
	if len(drafts) > max {
		return PreparedResult{}, discard(fmt.Errorf("draft returned %d tokens, limit %d", len(drafts), max))
	}
	for i, token := range drafts {
		if err := s.validateToken(token); err != nil {
			return PreparedResult{}, discard(fmt.Errorf("draft token %d: %w", i, err))
		}
	}
	current := s.verifier.CurrentLogits()
	if prepared {
		if len(current) == 0 {
			return PreparedResult{}, discard(errors.New("prepared verifier has no current logits"))
		}
		count, first := 0, -1
		for i, v := range current {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				count++
				if first < 0 {
					first = i
				}
			}
		}
		if count > 0 {
			return PreparedResult{}, discard(fmt.Errorf("prepared verifier current logits have %d nonfinite values, first at index %d", count, first))
		}
	}
	if len(current) == 0 || argmax(current) != drafts[0] {
		if prepared {
			return PreparedResult{SpeculativeResult: SpeculativeResult{Drafts: drafts}, Declined: true}, discard(nil)
		}
		return wrap(SpeculativeResult{Drafts: drafts}), errors.New("first draft does not match current argmax")
	}
	frontier, err := s.verifier.Snapshot()
	if frontier != nil {
		defer func() {
			if closeErr := frontier.Close(); closeErr != nil {
				s.pending = frontier
				err = &speculativeCleanupError{cause: errors.Join(err, ErrDesynchronized, fmt.Errorf("close speculative frontier: %w", closeErr)), owner: s}
				s.failed = err
				result = PreparedResult{}
			}
		}()
	}
	if err != nil {
		return PreparedResult{}, discard(fmt.Errorf("snapshot speculative frontier: %w", err))
	}
	if frontier == nil {
		return PreparedResult{}, discard(errors.New("snapshot returned nil frontier"))
	}
	rollback := func(cause error) error {
		cause = restoreSpeculativeFrontier(s.verifier, frontier, cause)
		cause = discard(cause)
		if errors.Is(cause, ErrDesynchronized) {
			s.failed = cause
		}
		return cause
	}
	tops, err := s.verifier.VerifyBatch(token, drafts, pos)
	if err != nil {
		return PreparedResult{}, fmt.Errorf("verify draft: %w", rollback(err))
	}
	// The row mapping is part of the verifier contract: one top-1 result per
	// draft row, in order. Accepting extra rows would silently ignore a
	// backend's seed/draft indexing mistake and could commit the wrong state.
	if len(tops) != len(drafts) {
		err := fmt.Errorf("verify draft returned %d rows, want %d", len(tops), len(drafts))
		return PreparedResult{}, rollback(err)
	}
	for i, token := range tops {
		if err := s.validateToken(token); err != nil {
			return PreparedResult{}, fmt.Errorf("verifier top %d: %w", i, rollback(err))
		}
	}
	accepted := 0
	for accepted < len(drafts) && tops[accepted] == drafts[accepted] {
		accepted++
	}
	if accepted == 0 {
		if err := rollback(nil); err != nil {
			return PreparedResult{}, fmt.Errorf("restore rejected draft: %w", err)
		}
		return wrap(SpeculativeResult{Drafts: drafts, Tops: append([]int(nil), tops...)}), nil
	}
	if err := s.verifier.CommitPrefix(accepted); err != nil {
		return PreparedResult{}, fmt.Errorf("commit accepted prefix: %w", rollback(err))
	}
	if lifecycle != nil {
		if err := lifecycle.CommitDrafts(accepted); err != nil {
			return PreparedResult{}, fmt.Errorf("commit support prefix: %w", rollback(err))
		}
	}
	return wrap(SpeculativeResult{Drafts: drafts, Tops: append([]int(nil), tops...), Accepted: accepted, Committed: accepted}), nil
}

// restoreSpeculativeFrontier restores the state after a speculative failure.
// A failed restore means that the caller must not reuse the request state: it
// may contain a mixture of the speculative and pre-frontier histories. Keep
// both the operation error and the restore error discoverable to callers.
func restoreSpeculativeFrontier(verifier SpeculativeVerifier, frontier Snapshot, cause error) error {
	if err := verifier.Restore(frontier); err != nil {
		if cause == nil {
			return errors.Join(ErrDesynchronized, err)
		}
		return errors.Join(ErrDesynchronized, cause, err)
	}
	return cause
}

func (s *Speculator) validateToken(token int) error {
	if token < 0 {
		return errors.New("token ID is negative")
	}
	if s.limits.VocabSize > 0 && token >= s.limits.VocabSize {
		return fmt.Errorf("token ID %d is outside vocabulary size %d", token, s.limits.VocabSize)
	}
	return nil
}

// argmax uses the same lowest-index tie rule as the sampler package without
// coupling the dsv4 backend contract to a sampling implementation.
func argmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}
