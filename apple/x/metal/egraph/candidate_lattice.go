package egraph

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/tmc/lib/egraphcore"
)

// CandidateLatticeFinalist is one individually viable bucket finalist at a
// rank. Scores must be finite and nondecreasing. Value must remain immutable
// while the lattice uses it.
type CandidateLatticeFinalist[T any] struct {
	Score float64
	Value T
}

// CandidateLatticeBucket supplies ranked finalists on demand. LoadFinalist is
// called at most once for each rank, in rank order.
type CandidateLatticeBucket[T any] struct {
	Finalists    []CandidateLatticeFinalist[T]
	LoadFinalist func(rank int) (finalist CandidateLatticeFinalist[T], ok bool, err error)
}

// CandidateSearchBucket adapts a completed search to the lattice's lazy
// finalist source. A non-nil validate hard-filters each measured finalist;
// rejected candidates are reported to search and skipped. A nil validate
// accepts every measured finalist. The lattice score is always
// lower-is-better, so descending search scores are negated. The search must
// have exhausted its candidate budgets before the lattice requests a finalist.
// validate must treat the supplied finalist and its expression payloads as
// read-only; changes to the supplied copy are not used by the lattice. Use
// CandidateSearchBucketWithContext when validation needs cooperative cancel.
func CandidateSearchBucket(
	search *CandidateSearch,
	validate func(*CandidateFinalist) error,
) CandidateLatticeBucket[*CandidateFinalist] {
	return CandidateSearchBucketWithContext(context.Background(), search, func(_ context.Context, finalist *CandidateFinalist) error {
		if validate == nil {
			return nil
		}
		return validate(finalist)
	})
}

// CandidateSearchBucketWithContext is CandidateSearchBucket with cooperative
// cancellation for finalist validation. The validation context inherits the
// caller's cancellation and the finalist deadline. Validators must honor the
// context; the adapter reports timeout after return if a validator ignores it.
func CandidateSearchBucketWithContext(
	ctx context.Context,
	search *CandidateSearch,
	validate func(context.Context, *CandidateFinalist) error,
) CandidateLatticeBucket[*CandidateFinalist] {
	nextRank := 0
	return CandidateLatticeBucket[*CandidateFinalist]{
		LoadFinalist: func(rank int) (CandidateLatticeFinalist[*CandidateFinalist], bool, error) {
			if ctx == nil {
				return CandidateLatticeFinalist[*CandidateFinalist]{}, false, fmt.Errorf("egraph: nil finalist validation context")
			}
			if err := ctx.Err(); err != nil {
				return CandidateLatticeFinalist[*CandidateFinalist]{}, false, err
			}
			if search == nil {
				return CandidateLatticeFinalist[*CandidateFinalist]{}, false, fmt.Errorf("egraph: nil candidate search")
			}
			if rank != nextRank {
				return CandidateLatticeFinalist[*CandidateFinalist]{}, false, fmt.Errorf("egraph: finalist rank %d requested, want %d", rank, nextRank)
			}
			for {
				if err := ctx.Err(); err != nil {
					return CandidateLatticeFinalist[*CandidateFinalist]{}, false, err
				}
				finalist, err := search.NextFinalist()
				if err != nil {
					return CandidateLatticeFinalist[*CandidateFinalist]{}, false, err
				}
				if finalist == nil {
					return CandidateLatticeFinalist[*CandidateFinalist]{}, false, nil
				}
				validationErr := func() (err error) {
					validationContext := ctx
					cancel := func() {}
					if !finalist.Deadline.IsZero() {
						validationContext, cancel = context.WithDeadline(ctx, finalist.Deadline)
					}
					defer cancel()
					defer func() {
						if value := recover(); value != nil {
							err = fmt.Errorf("finalist validator panicked: %v", value)
						}
					}()
					if err := validationContext.Err(); err != nil {
						return err
					}
					if validate != nil {
						err = validate(validationContext, cloneCandidateFinalist(finalist))
					}
					if err == nil {
						err = validationContext.Err()
					}
					return err
				}()
				report := CandidateFinalistReport{Accepted: validationErr == nil, TimedOut: errors.Is(validationErr, context.DeadlineExceeded)}
				if validationErr != nil {
					report.Reason = validationErr.Error()
					if report.TimedOut {
						report.Reason = "candidate timeout expired during finalist validation"
					}
					if strings.TrimSpace(report.Reason) == "" {
						report.Reason = "finalist rejected"
					}
				}
				accepted, err := search.reportFinalist(finalist.ID, report)
				if err != nil {
					return CandidateLatticeFinalist[*CandidateFinalist]{}, false, err
				}
				if !accepted {
					continue
				}
				score := finalist.Score
				if search.config.ScoreDirection == CandidateScoreDescending {
					score = -score
				}
				nextRank++
				return CandidateLatticeFinalist[*CandidateFinalist]{
					Score: score,
					Value: cloneCandidateFinalist(finalist),
				}, true, nil
			}
		},
	}
}

// CandidateLatticeSet is one proposed Cartesian product, in bucket order.
// Indices and Finalists slices are copies. Values inside Finalists are shallow
// copies and must remain immutable.
type CandidateLatticeSet[T any] struct {
	Indices   []int
	Finalists []CandidateLatticeFinalist[T]
	Score     float64
	Deadline  time.Time
	serial    uint64
}

// CandidateLattice adapts the backend-neutral egraphcore lattice to Apple's
// public candidate types.
type CandidateLattice[T any] struct {
	core          *egraphcore.CandidateLattice[T]
	pending       *egraphcore.CandidateLatticeSet[T]
	pendingSerial uint64
	serial        uint64
}

// NewCandidateLattice creates a bounded best-first coordinator. maxSets caps
// proposals and maxFinalistsPerBucket caps lazy ranks fetched from each bucket.
// aggregate must be deterministic, leave its input unchanged, and be
// coordinate-monotone: increasing one bucket score must not lower the result.
func NewCandidateLattice[T any](buckets []CandidateLatticeBucket[T], aggregate func([]float64) float64, maxSets, maxFinalistsPerBucket int) (*CandidateLattice[T], error) {
	shared := make([]egraphcore.CandidateLatticeBucket[T], len(buckets))
	for i, bucket := range buckets {
		shared[i].Finalists = make([]egraphcore.CandidateLatticeFinalist[T], len(bucket.Finalists))
		for j, finalist := range bucket.Finalists {
			shared[i].Finalists[j] = egraphcore.CandidateLatticeFinalist[T]{Score: finalist.Score, Value: finalist.Value}
		}
		if bucket.LoadFinalist != nil {
			load := bucket.LoadFinalist
			shared[i].LoadFinalist = func(rank int) (egraphcore.CandidateLatticeFinalist[T], bool, error) {
				finalist, ok, err := load(rank)
				return egraphcore.CandidateLatticeFinalist[T]{Score: finalist.Score, Value: finalist.Value}, ok, err
			}
		}
	}
	core, err := egraphcore.NewCandidateLattice(shared, aggregate, maxSets, maxFinalistsPerBucket)
	if err != nil {
		return nil, err
	}
	return &CandidateLattice[T]{core: core}, nil
}

// SetTimeouts configures optional overall search and per-proposal deadlines.
func (l *CandidateLattice[T]) SetTimeouts(searchTimeout, candidateTimeout time.Duration) error {
	if l == nil || l.core == nil {
		return fmt.Errorf("egraph: nil candidate lattice")
	}
	return l.core.SetTimeouts(searchTimeout, candidateTimeout)
}

// Next returns the lowest-scoring unvisited set. A returned set must be
// accepted or rejected before another call to Next. When ok is false, inspect
// StopReason and LimitReached before treating the search space as exhausted.
func (l *CandidateLattice[T]) Next() (*CandidateLatticeSet[T], bool, error) {
	if l == nil || l.core == nil {
		return nil, false, fmt.Errorf("egraph: nil candidate lattice")
	}
	set, ok, err := l.core.Next()
	if err != nil || !ok {
		return nil, ok, err
	}
	l.serial++
	l.pending = set
	l.pendingSerial = l.serial
	return l.copySet(set, l.pendingSerial), true, nil
}

// Accept resolves the pending proposal as viable.
func (l *CandidateLattice[T]) Accept(set *CandidateLatticeSet[T]) error {
	if err := l.checkPending(set); err != nil {
		return err
	}
	if err := l.core.Accept(l.pending); err != nil {
		return err
	}
	l.pending = nil
	return nil
}

// Reject resolves a proposal that fails aggregate validation and adds its
// one-coordinate slower successors to the frontier.
func (l *CandidateLattice[T]) Reject(set *CandidateLatticeSet[T], reason string) error {
	if err := l.checkPending(set); err != nil {
		return err
	}
	if err := l.core.Reject(l.pending, reason); err != nil {
		return err
	}
	l.pending = nil
	return nil
}

// ValidateNext proposes sets until validation accepts one or the bounded
// search is exhausted.
func (l *CandidateLattice[T]) ValidateNext(validate func(*CandidateLatticeSet[T]) error) (*CandidateLatticeSet[T], bool, error) {
	if validate == nil {
		return nil, false, fmt.Errorf("egraph: nil candidate lattice validator")
	}
	for {
		set, ok, err := l.Next()
		if err != nil || !ok {
			return nil, ok, err
		}
		if l.TimedOut(set) {
			if err := l.Reject(set, "candidate timeout before aggregate validation"); err != nil {
				return nil, false, err
			}
			continue
		}
		validationErr := func() (err error) {
			defer func() {
				if value := recover(); value != nil {
					err = fmt.Errorf("egraph: aggregate finalist validator panicked: %v", value)
				}
			}()
			return validate(set)
		}()
		if l.TimedOut(set) {
			if err := l.Reject(set, "candidate timeout during aggregate validation"); err != nil {
				return nil, false, err
			}
			continue
		}
		if l.core.SearchExpired() {
			if err := l.Reject(set, "search timeout during aggregate validation"); err != nil {
				return nil, false, err
			}
			continue
		}
		if validationErr == nil {
			if err := l.Accept(set); err != nil {
				return nil, false, err
			}
			return set, true, nil
		}
		if err := l.Reject(set, validationErr.Error()); err != nil {
			return nil, false, err
		}
	}
}

// TimedOut reports whether the proposal deadline has elapsed.
func (l *CandidateLattice[T]) TimedOut(set *CandidateLatticeSet[T]) bool {
	if l == nil || set == nil {
		return false
	}
	if l.pending != nil && set.serial == l.pendingSerial {
		return l.core.TimedOut(l.pending)
	}
	return !set.Deadline.IsZero() && !time.Now().Before(set.Deadline)
}

// StopReason reports why Next last returned no proposal. An empty result can
// mean the search space was exhausted, a configured bound was reached, or a
// finalist load timed out.
func (l *CandidateLattice[T]) StopReason() string {
	if l == nil || l.core == nil {
		return ""
	}
	return l.core.StopReason()
}

// LimitReached reports whether a proposal or finalist bound prevented further
// exploration.
func (l *CandidateLattice[T]) LimitReached() bool {
	return l != nil && l.core != nil && l.core.LimitReached()
}

// Attempts reports the number of sets proposed, including rejected sets.
func (l *CandidateLattice[T]) Attempts() int {
	if l == nil || l.core == nil {
		return 0
	}
	return l.core.Attempts()
}

// LastRejection reports the most recent aggregate validation failure.
func (l *CandidateLattice[T]) LastRejection() string {
	if l == nil || l.core == nil {
		return ""
	}
	return l.core.LastRejection()
}

func (l *CandidateLattice[T]) checkPending(set *CandidateLatticeSet[T]) error {
	if l == nil || set == nil || l.pending == nil {
		return fmt.Errorf("egraph: no candidate lattice set awaits resolution")
	}
	if set.serial != l.pendingSerial {
		return fmt.Errorf("egraph: stale candidate lattice set")
	}
	return nil
}

func (l *CandidateLattice[T]) copySet(set *egraphcore.CandidateLatticeSet[T], serial uint64) *CandidateLatticeSet[T] {
	copy := &CandidateLatticeSet[T]{
		Indices:   append([]int(nil), set.Indices...),
		Finalists: make([]CandidateLatticeFinalist[T], len(set.Finalists)),
		Score:     set.Score,
		Deadline:  set.Deadline,
		serial:    serial,
	}
	for i, finalist := range set.Finalists {
		copy.Finalists[i] = CandidateLatticeFinalist[T]{Score: finalist.Score, Value: finalist.Value}
	}
	return copy
}
