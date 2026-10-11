package egraphcore

import (
	"container/heap"
	"fmt"
	"math"
	"strings"
	"time"
)

// CandidateLatticeFinalist is one individually viable bucket finalist at a
// rank. Scores must be finite and nondecreasing within each bucket. Value is
// retained without copying and must remain immutable while the lattice uses
// it.
type CandidateLatticeFinalist[T any] struct {
	Score float64
	Value T
}

// CandidateLatticeBucket supplies individually viable finalists in score
// order. Finalists may contain already-materialized values; LoadFinalist can
// lazily produce ranks beyond Finalists. It is called at most once for each
// rank, in rank order. Return ok=false when the bucket has no more viable
// finalists. Values are retained without copying and must remain immutable.
type CandidateLatticeBucket[T any] struct {
	Finalists    []CandidateLatticeFinalist[T]
	LoadFinalist func(rank int) (finalist CandidateLatticeFinalist[T], ok bool, err error)
}

type candidateLatticeLoad[T any] struct {
	finalist CandidateLatticeFinalist[T]
	ok       bool
	err      error
}

// CandidateLatticeSet is one proposed Cartesian product, in bucket order.
// Indices and Finalists slices are copies and may be retained by the caller.
// Values inside Finalists are shallow copies and must remain immutable.
type CandidateLatticeSet[T any] struct {
	Indices   []int
	Finalists []CandidateLatticeFinalist[T]
	Score     float64
	Deadline  time.Time
	serial    uint64
}

// CandidateLattice walks a bounded Cartesian product of ranked bucket
// finalists from lowest aggregate score to highest. Equal aggregate scores
// are ordered lexicographically by bucket rank. Aggregate must be
// deterministic, must not modify its input, and must be coordinate-monotone:
// increasing one input score while holding the others fixed must not decrease
// the aggregate. Reject opens each one-coordinate slower successor; duplicate
// paths are suppressed.
type CandidateLattice[T any] struct {
	buckets           []CandidateLatticeBucket[T]
	aggregate         func([]float64) float64
	maxSets           int
	maxFinalists      int
	loaded            [][]CandidateLatticeFinalist[T]
	loads             [][]candidateLatticeLoad[T]
	frontier          candidateLatticeHeap[T]
	visited           map[string]struct{}
	proposed          int
	serial            uint64
	pending           *CandidateLatticeSet[T]
	lastRejection     string
	startedAt         time.Time
	searchTimeout     time.Duration
	candidateTime     time.Duration
	stopReason        string
	limitReached      bool
	candidateTimedOut bool
}

// NewCandidateLattice creates a bounded best-first coordinator. maxSets caps
// proposals and maxFinalistsPerBucket caps lazy ranks fetched from each bucket.
// A nonpositive bound is an error. Each bucket must provide at least one
// finalist, either in Finalists or through LoadFinalist.
func NewCandidateLattice[T any](buckets []CandidateLatticeBucket[T], aggregate func([]float64) float64, maxSets, maxFinalistsPerBucket int) (*CandidateLattice[T], error) {
	if len(buckets) == 0 {
		return nil, fmt.Errorf("egraphcore: candidate lattice needs at least one bucket")
	}
	if aggregate == nil {
		return nil, fmt.Errorf("egraphcore: nil candidate lattice aggregate")
	}
	if maxSets <= 0 || maxFinalistsPerBucket <= 0 {
		return nil, fmt.Errorf("egraphcore: candidate lattice bounds must be positive")
	}
	l := &CandidateLattice[T]{
		buckets:      append([]CandidateLatticeBucket[T](nil), buckets...),
		aggregate:    aggregate,
		maxSets:      maxSets,
		maxFinalists: maxFinalistsPerBucket,
		loaded:       make([][]CandidateLatticeFinalist[T], len(buckets)),
		loads:        make([][]candidateLatticeLoad[T], len(buckets)),
		visited:      make(map[string]struct{}),
		startedAt:    time.Now(),
	}
	for i := range l.buckets {
		l.buckets[i].Finalists = append([]CandidateLatticeFinalist[T](nil), buckets[i].Finalists...)
		if len(buckets[i].Finalists) == 0 && buckets[i].LoadFinalist == nil {
			return nil, fmt.Errorf("egraphcore: candidate lattice bucket %d has no finalist source", i)
		}
		for rank, finalist := range l.buckets[i].Finalists {
			if rank >= maxFinalistsPerBucket {
				l.buckets[i].Finalists = l.buckets[i].Finalists[:maxFinalistsPerBucket]
				break
			}
			if !finiteLatticeScore(finalist.Score) {
				return nil, fmt.Errorf("egraphcore: bucket %d finalist %d has non-finite score", i, rank)
			}
			if rank > 0 && finalist.Score < l.buckets[i].Finalists[rank-1].Score {
				return nil, fmt.Errorf("egraphcore: bucket %d finalist scores are not ranked", i)
			}
		}
		l.loaded[i] = l.buckets[i].Finalists
		l.loads[i] = make([]candidateLatticeLoad[T], len(l.loaded[i]))
		for rank, finalist := range l.loaded[i] {
			l.loads[i][rank] = candidateLatticeLoad[T]{finalist: finalist, ok: true}
		}
	}
	return l, nil
}

// SetTimeouts configures optional overall search and per-proposal deadlines.
// Call it before Next. Zero disables the corresponding timeout. Deadlines are
// checked when callbacks return; they do not interrupt a loader or validator.
func (l *CandidateLattice[T]) SetTimeouts(searchTimeout, candidateTimeout time.Duration) error {
	if l == nil {
		return fmt.Errorf("egraphcore: nil candidate lattice")
	}
	if l.proposed != 0 || l.pending != nil {
		return fmt.Errorf("egraphcore: candidate lattice timeouts must be set before proposals")
	}
	if searchTimeout < 0 || candidateTimeout < 0 {
		return fmt.Errorf("egraphcore: candidate lattice timeouts must not be negative")
	}
	l.searchTimeout = searchTimeout
	l.candidateTime = candidateTimeout
	return nil
}

// Next returns the lowest-scoring unvisited set. The caller validates the
// aggregate and either accepts it or passes it to Reject. A returned set must
// be resolved before another call to Next. A false result means no proposal is
// currently available within the configured bounds; inspect StopReason and
// LimitReached before treating the search space as exhausted.
func (l *CandidateLattice[T]) Next() (*CandidateLatticeSet[T], bool, error) {
	if l == nil {
		return nil, false, fmt.Errorf("egraphcore: nil candidate lattice")
	}
	if l.pending != nil {
		return nil, false, fmt.Errorf("egraphcore: candidate lattice set awaits acceptance or rejection")
	}
	if l.searchExpired() {
		l.stopReason = "search timeout during aggregate bucket finalization"
		return nil, false, nil
	}
	if l.proposed >= l.maxSets {
		l.limitReached = true
		l.stopReason = "candidate set limit reached"
		return nil, false, nil
	}
	if len(l.visited) == 0 {
		indices := make([]int, len(l.buckets))
		values := make([]CandidateLatticeFinalist[T], len(l.buckets))
		for i := range l.buckets {
			ok, err := l.ensure(i, 0)
			if err != nil {
				return nil, false, err
			}
			if !ok {
				l.setNoCandidateReason("candidate sources exhausted")
				return nil, false, nil
			}
			values[i] = l.loaded[i][0]
		}
		set, err := l.makeSet(indices, values)
		if err != nil {
			return nil, false, err
		}
		if l.searchExpired() {
			l.stopReason = "search timeout during aggregate bucket finalization"
			return nil, false, nil
		}
		l.add(set)
	}
	if len(l.frontier) == 0 {
		l.setNoCandidateReason("candidate space exhausted")
		return nil, false, nil
	}
	set := heap.Pop(&l.frontier).(*CandidateLatticeSet[T])
	l.proposed++
	l.serial++
	set.serial = l.serial
	if l.candidateTime > 0 && set.Deadline.IsZero() {
		set.Deadline = time.Now().Add(l.candidateTime)
	}
	copySet := cloneLatticeSet(set)
	l.pending = copySet
	return cloneLatticeSet(copySet), true, nil
}

// Accept resolves the pending proposal as viable and permits another Next.
func (l *CandidateLattice[T]) Accept(set *CandidateLatticeSet[T]) error {
	if err := l.checkPending(set); err != nil {
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
	if strings.TrimSpace(reason) == "" {
		return fmt.Errorf("egraphcore: candidate lattice rejection requires a reason")
	}
	l.lastRejection = reason
	set = l.pending
	l.pending = nil
	for bucket := range l.buckets {
		if l.searchExpired() {
			l.stopReason = "search timeout during aggregate bucket finalization"
			break
		}
		indices := append([]int(nil), set.Indices...)
		indices[bucket]++
		key := latticeIndicesKey(indices)
		if _, ok := l.visited[key]; ok {
			continue
		}
		if l.proposed+len(l.frontier) >= l.maxSets {
			l.limitReached = true
			break
		}
		ok, err := l.ensure(bucket, indices[bucket])
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		if l.searchExpired() {
			l.stopReason = "search timeout during aggregate bucket finalization"
			break
		}
		values := make([]CandidateLatticeFinalist[T], len(indices))
		for i, rank := range indices {
			if !l.ensureAlreadyLoaded(i, rank) {
				return fmt.Errorf("egraphcore: missing finalist rank %d for bucket %d", rank, i)
			}
			values[i] = l.loaded[i][rank]
		}
		successor, err := l.makeSet(indices, values)
		if err != nil {
			return err
		}
		if l.searchExpired() {
			l.stopReason = "search timeout during aggregate bucket finalization"
			break
		}
		l.add(successor)
	}
	return nil
}

// ValidateNext proposes sets until validate accepts one or the bounded search
// is exhausted. Rejected sets are expanded automatically.
func (l *CandidateLattice[T]) ValidateNext(validate func(*CandidateLatticeSet[T]) error) (*CandidateLatticeSet[T], bool, error) {
	if validate == nil {
		return nil, false, fmt.Errorf("egraphcore: nil candidate lattice validator")
	}
	for {
		set, ok, err := l.Next()
		if err != nil || !ok {
			return nil, ok, err
		}
		if l.TimedOut(set) {
			if rejectErr := l.Reject(set, "candidate timeout before aggregate validation"); rejectErr != nil {
				return nil, false, rejectErr
			}
			continue
		}
		validationErr := func() (err error) {
			defer func() {
				if value := recover(); value != nil {
					err = fmt.Errorf("egraphcore: aggregate validator panicked: %v", value)
				}
			}()
			return validate(set)
		}()
		if l.TimedOut(set) {
			if rejectErr := l.Reject(set, "candidate timeout during aggregate validation"); rejectErr != nil {
				return nil, false, rejectErr
			}
			continue
		}
		if l.searchExpired() {
			if rejectErr := l.Reject(set, "search timeout during aggregate validation"); rejectErr != nil {
				return nil, false, rejectErr
			}
			continue
		}
		if validationErr == nil {
			if err := l.Accept(set); err != nil {
				return nil, false, err
			}
			return set, true, nil
		} else if rejectErr := l.Reject(set, validationErr.Error()); rejectErr != nil {
			return nil, false, rejectErr
		}
	}
}

// TimedOut reports whether the proposal deadline has elapsed. A zero deadline
// means that proposal timeouts are disabled.
func (l *CandidateLattice[T]) TimedOut(set *CandidateLatticeSet[T]) bool {
	if l == nil || set == nil {
		return false
	}
	deadline := set.Deadline
	if l.pending != nil && l.pending.serial == set.serial {
		deadline = l.pending.Deadline
	}
	return !deadline.IsZero() && !time.Now().Before(deadline)
}

// StopReason reports why Next last returned no proposal. An empty result can
// mean the search space was exhausted, a configured bound was reached, or a
// finalist load timed out; callers must not treat every empty result as proof
// that no unexamined candidate exists.
func (l *CandidateLattice[T]) StopReason() string {
	if l == nil {
		return ""
	}
	return l.stopReason
}

// LimitReached reports whether maxSets or maxFinalistsPerBucket prevented the
// lattice from exploring another candidate set.
func (l *CandidateLattice[T]) LimitReached() bool {
	return l != nil && l.limitReached
}

// SearchExpired reports whether the overall search timeout has elapsed.
func (l *CandidateLattice[T]) SearchExpired() bool {
	return l != nil && l.searchExpired()
}

// Attempts reports the number of sets proposed, including rejected sets.
func (l *CandidateLattice[T]) Attempts() int {
	if l == nil {
		return 0
	}
	return l.proposed
}

// LastRejection reports the most recent aggregate validation failure.
func (l *CandidateLattice[T]) LastRejection() string {
	if l == nil {
		return ""
	}
	return l.lastRejection
}

func (l *CandidateLattice[T]) ensure(bucket, rank int) (bool, error) {
	if rank < len(l.loaded[bucket]) {
		return true, nil
	}
	if rank >= l.maxFinalists || rank != len(l.loaded[bucket]) {
		if rank >= l.maxFinalists {
			l.limitReached = true
		}
		return false, nil
	}
	if rank < len(l.loads[bucket]) {
		load := l.loads[bucket][rank]
		return load.ok, load.err
	}
	load := l.buckets[bucket].LoadFinalist
	if load == nil {
		return false, nil
	}
	startedAt := time.Now()
	finalist, ok, err := load(rank)
	if l.candidateTime > 0 && time.Since(startedAt) >= l.candidateTime {
		l.candidateTimedOut = true
		finalist = CandidateLatticeFinalist[T]{}
		ok = false
		err = nil
	}
	if err == nil && ok {
		if !finiteLatticeScore(finalist.Score) {
			err = fmt.Errorf("egraphcore: bucket %d finalist %d has non-finite score", bucket, rank)
			ok = false
		} else if rank > 0 && finalist.Score < l.loaded[bucket][rank-1].Score {
			err = fmt.Errorf("egraphcore: bucket %d finalist scores are not ranked", bucket)
			ok = false
		}
	}
	l.loads[bucket] = append(l.loads[bucket], candidateLatticeLoad[T]{
		finalist: finalist, ok: ok && err == nil, err: err,
	})
	if err != nil || !ok {
		return ok, err
	}
	l.loaded[bucket] = append(l.loaded[bucket], finalist)
	return true, nil
}

func (l *CandidateLattice[T]) setNoCandidateReason(exhausted string) {
	if l.limitReached {
		l.stopReason = "candidate finalist limit reached"
		return
	}
	if l.candidateTimedOut {
		l.stopReason = "candidate finalist load timed out"
		return
	}
	l.stopReason = exhausted
}

func (l *CandidateLattice[T]) ensureAlreadyLoaded(bucket, rank int) bool {
	return rank >= 0 && rank < len(l.loaded[bucket])
}

func (l *CandidateLattice[T]) makeSet(indices []int, finalists []CandidateLatticeFinalist[T]) (*CandidateLatticeSet[T], error) {
	scores := make([]float64, len(finalists))
	for i := range finalists {
		scores[i] = finalists[i].Score
	}
	score := l.aggregate(scores)
	if !finiteLatticeScore(score) {
		return nil, fmt.Errorf("egraphcore: candidate lattice aggregate returned non-finite score")
	}
	return &CandidateLatticeSet[T]{Indices: append([]int(nil), indices...), Finalists: append([]CandidateLatticeFinalist[T](nil), finalists...), Score: score}, nil
}

func (l *CandidateLattice[T]) add(set *CandidateLatticeSet[T]) {
	l.visited[latticeIndicesKey(set.Indices)] = struct{}{}
	heap.Push(&l.frontier, set)
}

func (l *CandidateLattice[T]) checkPending(set *CandidateLatticeSet[T]) error {
	if l == nil || set == nil || l.pending == nil {
		return fmt.Errorf("egraphcore: no candidate lattice set awaits resolution")
	}
	if set.serial != l.pending.serial {
		return fmt.Errorf("egraphcore: stale candidate lattice set")
	}
	return nil
}

func cloneLatticeSet[T any](set *CandidateLatticeSet[T]) *CandidateLatticeSet[T] {
	copy := *set
	copy.Indices = append([]int(nil), set.Indices...)
	copy.Finalists = append([]CandidateLatticeFinalist[T](nil), set.Finalists...)
	return &copy
}

func finiteLatticeScore(score float64) bool { return !math.IsNaN(score) && !math.IsInf(score, 0) }

type candidateLatticeHeap[T any] []*CandidateLatticeSet[T]

func (h candidateLatticeHeap[T]) Len() int { return len(h) }

func (h candidateLatticeHeap[T]) Less(i, j int) bool {
	return latticeSetLess(h[i], h[j])
}

func (h candidateLatticeHeap[T]) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *candidateLatticeHeap[T]) Push(value any) {
	*h = append(*h, value.(*CandidateLatticeSet[T]))
}

func (h *candidateLatticeHeap[T]) Pop() any {
	old := *h
	last := len(old) - 1
	value := old[last]
	old[last] = nil
	*h = old[:last]
	return value
}

func latticeSetLess[T any](left, right *CandidateLatticeSet[T]) bool {
	if left.Score != right.Score {
		return left.Score < right.Score
	}
	for i, rank := range left.Indices {
		if rank != right.Indices[i] {
			return rank < right.Indices[i]
		}
	}
	return false
}

func (l *CandidateLattice[T]) searchExpired() bool {
	return l != nil && l.searchTimeout > 0 && time.Since(l.startedAt) >= l.searchTimeout
}

func latticeIndicesKey(indices []int) string {
	buf := make([]byte, 0, len(indices)*12)
	for _, index := range indices {
		buf = append(buf, fmt.Sprintf("%d,", index)...)
	}
	return string(buf)
}
