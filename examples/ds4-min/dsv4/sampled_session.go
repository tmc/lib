package dsv4

import (
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

// SampledVerifier adds full logits to the existing verifier. RowLogits(i)
// returns a detached post-evaluation distribution for draft i, not its top-1.
// Draft zero uses CurrentLogits; draft i>0 uses RowLogits(i-1). CommitPrefix
// takes the number of accepted drafts, excluding the already committed seed,
// installs the corresponding final logits and internally restores/replays
// when prefix slots are unavailable. The verifier must own Session's State.
type SampledVerifier interface {
	SpeculativeVerifier
	// BoundTo checks immutable request identity without mutating either State.
	BoundTo(State) bool
	RowLogits(i int) ([]float32, error)
}

// SampledDrafter binds preparation to the committed target feature frontier.
// Observe must maintain committed support history even on strict/backoff
// cycles; it must not generate proposals. Prepare installs the already
// committed seed at next-position pos. Draft and failed CommitDrafts obey
// DraftLifecycle's unchanged-active-frontier rules.
type SampledDrafter interface {
	Drafter
	DraftLifecycle
	Observe(pos, n int) error
	Prepare(seed, pos int) error
}

// SampledCycle contains the sampled boundary followed by the verified suffix
// or rejection correction. Stop tokens are reported but never evaluated.
// Recoverable suffix failures report the committed boundary only. Irreversible
// observation/cleanup failures report every token already committed and poison
// the request. Accepted excludes correction tokens and stop tokens.
type SampledCycle struct {
	Tokens   []int
	Accepted int
	// Committed counts evaluated tokens, excluding a reported stop token.
	// It remains accurate on an irreversible observation or cleanup failure.
	Committed int
	Stopped   bool
	Mode      SampledMode
	Drafted   bool
}

// SampledSession orchestrates sampling on an existing synchronized Session.
// It owns its RNG copy and policy; neither may be shared concurrently.
// Recoverable suffix failure restores the post-boundary frontier and RNG.
// Failed cleanup or committed feature observation poisons the Session.
type SampledSession struct {
	session  *Session
	verifier SampledVerifier
	drafter  SampledDrafter
	sampler  *sampler.Sampler
	random   sampler.RNG
	params   sampler.Params
	mode     SampledMode
	policy   SampledPolicy
	limit    int
	pending  Snapshot
	owner    io.Closer
	closed   bool
}

// NewSampledSession binds the verifier and drafter to session's existing State.
// The native owner must establish that identity; full logits are mandatory.
// The caller retains ownership of a directly supplied verifier.
func NewSampledSession(session *Session, verifier SampledVerifier, drafter SampledDrafter, params sampler.Params, random sampler.RNG, mode SampledMode, policy SampledPolicy, maxDrafts int) (*SampledSession, error) {
	if session == nil || verifier == nil || drafter == nil {
		return nil, ErrUnimplemented
	}
	if session.closed || session.state == nil {
		return nil, ErrClosed
	}
	if session.broken {
		return nil, ErrDesynchronized
	}
	if !verifier.BoundTo(session.state) {
		return nil, errors.New("sampled verifier belongs to another request")
	}
	if maxDrafts <= 0 || maxDrafts > DefaultSpeculativeLimits.MaxDrafts || mode > OpportunisticSampling {
		return nil, errors.New("invalid sampled session policy")
	}
	s := &SampledSession{session: session, verifier: verifier, drafter: drafter, sampler: sampler.New(len(session.logits)), random: random, params: params, mode: mode, policy: policy, limit: maxDrafts}
	if _, err := s.probabilities(session.logits); err != nil {
		return nil, err
	}
	return s, nil
}

// NewSampledSessionForBackend obtains the verifier from the existing request
// owner. It fails closed when the backend has no full-row sampling capability.
// This is the native integration entry point; it never creates another State.
// It owns the returned verifier when it implements io.Closer. A non-nil result
// returned with an error retains failed cleanup and must be closed before session.
func NewSampledSessionForBackend(session *Session, drafter SampledDrafter, params sampler.Params, random sampler.RNG, mode SampledMode, policy SampledPolicy, maxDrafts int) (*SampledSession, error) {
	if session == nil || session.closed || session.state == nil {
		return nil, ErrClosed
	}
	if session.broken {
		return nil, ErrDesynchronized
	}
	if drafter == nil {
		return nil, ErrUnimplemented
	}
	if maxDrafts <= 0 || maxDrafts > DefaultSpeculativeLimits.MaxDrafts || mode > OpportunisticSampling {
		return nil, errors.New("invalid sampled session policy")
	}
	b, ok := session.backend.(SpeculativeBackend)
	if !ok {
		return nil, ErrUnimplemented
	}
	v, err := b.NewSpeculativeVerifier(session.state)
	owner, _ := v.(io.Closer)
	if err != nil {
		return failedSampledOwner(session, owner, fmt.Errorf("new sampled verifier: %w", err))
	}
	full, ok := v.(SampledVerifier)
	if !ok {
		return failedSampledOwner(session, owner, ErrUnimplemented)
	}
	s, err := NewSampledSession(session, full, drafter, params, random, mode, policy, maxDrafts)
	if err != nil {
		return failedSampledOwner(session, owner, err)
	}
	s.owner = owner
	return s, nil
}

func failedSampledOwner(session *Session, owner io.Closer, cause error) (*SampledSession, error) {
	if owner != nil {
		if err := owner.Close(); err != nil {
			session.broken = true
			return &SampledSession{session: session, owner: owner, closed: true}, errors.Join(cause, ErrDesynchronized, fmt.Errorf("close sampled verifier: %w", err))
		}
	}
	return nil, cause
}

// Random returns the last successfully published RNG frontier. Irreversible
// errors poison the request; Random must not be used to resume such a request.
func (s *SampledSession) Random() sampler.RNG { return s.random }

// Close releases retained snapshots before a factory-owned verifier. It disables Next
// immediately, preserves ownership after a release error, and can be retried.
// The caller still owns the underlying Session and must close it separately.
func (s *SampledSession) Close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	if s.pending != nil {
		if err := s.pending.Close(); err != nil {
			return fmt.Errorf("close retained sampled frontier: %w", err)
		}
		s.pending = nil
	}
	if s.owner != nil {
		if err := s.owner.Close(); err != nil {
			return fmt.Errorf("close sampled verifier: %w", err)
		}
		s.owner = nil
	}
	return nil
}

func (s *SampledSession) probabilities(logits []float32) ([]float32, error) {
	p, err := s.sampler.Probabilities(logits, s.params)
	if err != nil {
		return nil, fmt.Errorf("sampled distribution: %w", err)
	}
	return p, nil
}

type sampledRandom struct{ r *sampler.RNG }

func (r sampledRandom) Float32() float32 { return r.r.Float32() }

// Next emits at most budget tokens. It samples and commits the ordinary
// boundary first, matching 11e8cad's stochastic dispatch. A later error can
// therefore return that boundary as partial progress. Suffix random draws are
// published only after target/support commits and snapshot cleanup succeed.
func (s *SampledSession) Next(budget int) (result SampledCycle, err error) {
	if s == nil || s.session == nil {
		return result, ErrClosed
	}
	if s.closed {
		return result, ErrClosed
	}
	x := s.session
	result.Mode = s.mode
	if x.closed {
		return result, ErrClosed
	}
	if x.broken {
		return result, ErrDesynchronized
	}
	if budget <= 0 || len(x.tokens) == 0 {
		return result, errors.New("sampled cycle requires a positive budget and synchronized prompt")
	}
	budget = min(budget, x.context-len(x.tokens))
	if budget <= 0 {
		return result, errContextOverflow
	}
	boundaryProb, err := s.probabilities(x.logits)
	if err != nil {
		return result, err
	}
	random := s.random
	first := s.sampler.Sample(x.logits, s.params, &random)
	if first < 0 || first >= len(boundaryProb) || boundaryProb[first] <= 0 {
		return result, errors.New("sampled boundary is outside filtered support")
	}
	pos := len(x.tokens)
	stop, err := x.Accept(first)
	if err != nil {
		return result, err
	}
	s.random = random
	result.Tokens = []int{first}
	result.Stopped = stop
	if stop {
		return result, nil
	}
	result.Committed = 1
	poison := func(cause error) error { x.broken = true; return errors.Join(ErrDesynchronized, cause) }
	if err = s.drafter.Observe(pos, 1); err != nil {
		return result, poison(fmt.Errorf("observe boundary: %w", err))
	}
	if !s.policy.Allow(budget) || budget == 1 {
		return result, nil
	}
	pos = len(x.tokens)
	discard := func(cause error) error {
		if e := s.drafter.DiscardDrafts(); e != nil {
			return poison(errors.Join(cause, fmt.Errorf("discard sampled drafts: %w", e)))
		}
		if errors.Is(cause, ErrDesynchronized) {
			return poison(cause)
		}
		return cause
	}
	if err = s.drafter.Prepare(first, pos); err != nil {
		return result, discard(fmt.Errorf("prepare sampled draft: %w", err))
	}
	seed, at, e := s.drafter.PreparedSeed()
	if e != nil || seed != first || at != pos {
		return result, discard(errors.Join(e, errors.New("sampled prepared seed mismatch")))
	}
	maxDrafts := min(s.limit, budget-1, x.context-pos)
	drafts, e := s.drafter.Draft(first, pos, maxDrafts)
	if e != nil {
		return result, discard(fmt.Errorf("sampled draft: %w", e))
	}
	drafts = append([]int(nil), drafts...)
	if len(drafts) > maxDrafts {
		return result, discard(errors.New("sampled proposal exceeds limit"))
	}
	shortProposal := s.mode == ExactSampling && s.params.Temperature > 0 && len(drafts) < 2
	for i, tok := range drafts {
		if tok < 0 || tok >= len(x.logits) {
			return result, discard(errors.New("sampled draft outside vocabulary"))
		}
		if x.stop.contains(tok) {
			drafts = drafts[:i+1]
			break
		}
	}
	if shortProposal {
		drafts = nil
	}
	if len(drafts) == 0 {
		var confidence *float32
		if d, ok := s.drafter.(interface{ DraftConfidence() (float32, bool) }); ok {
			if value, valid := d.DraftConfidence(); valid {
				confidence = &value
			}
		}
		s.policy.Note(0, true, confidence)
		return result, discard(nil)
	}
	result.Drafted = true
	frontier, e := s.verifier.Snapshot()
	if frontier != nil {
		defer func() {
			if frontier != nil {
				if e := frontier.Close(); e != nil {
					s.pending = frontier
					err = poison(errors.Join(err, fmt.Errorf("close sampled frontier: %w", e)))
				}
			}
		}()
	}
	if e != nil || frontier == nil {
		return result, discard(errors.Join(e, errors.New("snapshot sampled frontier failed")))
	}
	rollback := func(cause error) error {
		if e := s.verifier.Restore(frontier); e != nil {
			cause = poison(errors.Join(cause, e))
		}
		return discard(cause)
	}
	current := append([]float32(nil), s.verifier.CurrentLogits()...)
	if len(current) != len(x.logits) {
		return result, rollback(errors.New("sampled frontier vocabulary differs"))
	}
	p, e := s.probabilities(current)
	if e != nil {
		return result, rollback(e)
	}
	firstOK := drafts[0] == argmax(current) && p[drafts[0]] > 0
	firstCorrection := -1
	if s.mode == ExactSampling && s.params.Temperature > 0 {
		firstCorrection, firstOK, e = PointMassVerify(p, drafts[0], sampledRandom{&random})
		if e != nil {
			return result, rollback(e)
		}
	}
	if firstOK && !x.stop.contains(drafts[0]) {
		tops, e := s.verifier.VerifyBatch(first, drafts, pos)
		if e != nil {
			return result, rollback(fmt.Errorf("verify sampled draft: %w", e))
		}
		if len(tops) != len(drafts) {
			return result, rollback(errors.New("sampled verifier row count differs"))
		}
	}
	accepted := 0
	correction := -1
	stopped := false
	for i, draft := range drafts {
		logits := current
		if i > 0 {
			logits, e = s.verifier.RowLogits(i - 1)
			if e != nil {
				return result, rollback(e)
			}
		}
		if len(logits) != len(current) {
			return result, rollback(errors.New("sampled verifier vocabulary differs"))
		}
		p, e := s.probabilities(logits)
		if e != nil {
			return result, rollback(e)
		}
		ok := false
		if i == 0 {
			correction, ok = firstCorrection, firstOK
		} else if s.mode == ExactSampling && s.params.Temperature > 0 {
			correction, ok, e = PointMassVerify(p, draft, sampledRandom{&random})
			if e != nil {
				return result, rollback(e)
			}
		} else {
			ok = draft == argmax(logits) && p[draft] > 0
		}
		if !ok {
			break
		}
		correction = -1
		if x.stop.contains(draft) {
			stopped = true
			break
		}
		accepted++
	}
	if accepted == 0 {
		e = s.verifier.Restore(frontier)
	} else {
		e = s.verifier.CommitPrefix(accepted)
	}
	if e != nil {
		return result, rollback(fmt.Errorf("commit sampled target: %w", e))
	}
	if correction >= 0 && !x.stop.contains(correction) {
		if e = x.backend.Eval(x.state, correction, pos+accepted); e != nil {
			return result, rollback(fmt.Errorf("evaluate sampled correction: %w", e))
		}
	}
	finalLogits := append([]float32(nil), x.backend.Logits(x.state)...)
	if len(finalLogits) != len(current) {
		return result, rollback(errors.New("committed sampled vocabulary differs"))
	}
	if _, e = s.probabilities(finalLogits); e != nil {
		return result, rollback(e)
	}
	if accepted > 0 {
		e = s.drafter.CommitDrafts(accepted)
	} else {
		e = s.drafter.DiscardDrafts()
	}
	if e != nil {
		return result, rollback(fmt.Errorf("commit sampled support: %w", e))
	}
	tokens := append([]int(nil), drafts[:accepted]...)
	committed := accepted
	if correction >= 0 {
		tokens = append(tokens, correction)
		if x.stop.contains(correction) {
			stopped = true
		} else {
			committed++
		}
	}
	if stopped && correction < 0 {
		tokens = append(tokens, drafts[accepted])
	}
	x.tokens = append(x.tokens, tokens[:committed]...)
	x.logits = finalLogits
	result.Tokens = append(result.Tokens, tokens...)
	result.Accepted = accepted
	result.Committed += committed
	result.Stopped = stopped
	if committed > 0 {
		if e = s.drafter.Observe(pos, committed); e != nil {
			return result, poison(fmt.Errorf("observe sampled prefix: %w", e))
		}
	}
	// Publish the suffix RNG only after successful cleanup. On irreversible
	// failure the result above records the committed tokens and request poison.
	e = frontier.Close()
	if e != nil {
		s.pending = frontier
		frontier = nil
		return result, poison(fmt.Errorf("close sampled frontier: %w", e))
	}
	frontier = nil
	s.random = random
	s.policy.Note(accepted, false, nil)
	return result, nil
}
