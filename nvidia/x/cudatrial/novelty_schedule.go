package cudatrial

import (
	"fmt"
	"sync"
)

// NoveltyFilter evaluates whether a raw edge counter vector represents novel execution.
type NoveltyFilter interface {
	Evaluate(raw []byte) (bool, error)
}

// SchedulerStats records trial scheduling and novelty progression metrics.
type SchedulerStats struct {
	EvaluatedTrials int
	NovelTrials     int
	DuplicateTrials int
	RejectedTrials  int
}

// NoveltyScheduler connects edge coverage tracking to trial scheduling decisions.
// It uses a NoveltyFilter to determine whether a trial's coverage warrants admission,
// retention, or further rounds.
type NoveltyScheduler struct {
	mu     sync.Mutex
	filter NoveltyFilter
	stats  SchedulerStats
}

// NewNoveltyScheduler creates a scheduler that uses the provided filter.
func NewNoveltyScheduler(filter NoveltyFilter) (*NoveltyScheduler, error) {
	if filter == nil {
		return nil, fmt.Errorf("novelty scheduler: uninitialized filter")
	}
	return &NoveltyScheduler{filter: filter}, nil
}

// ScheduleTrial evaluates a completed trial's coverage and reports whether the trial
// produced novel execution behavior. If receipt does not have coverage or if
// evaluation fails, an error is returned.
func (s *NoveltyScheduler) ScheduleTrial(receipt Receipt) (bool, error) {
	if s == nil || s.filter == nil {
		return false, fmt.Errorf("novelty scheduler: uninitialized scheduler")
	}
	if !receipt.HasCoverage() {
		s.mu.Lock()
		s.stats.EvaluatedTrials++
		s.stats.RejectedTrials++
		s.mu.Unlock()
		return false, fmt.Errorf("novelty scheduler: receipt has no coverage")
	}
	return s.EvaluateTrace(receipt.Coverage.Bytes())
}

// EvaluateTrace evaluates raw coverage counters against the filter and updates scheduler stats.
func (s *NoveltyScheduler) EvaluateTrace(raw []byte) (bool, error) {
	if s == nil || s.filter == nil {
		return false, fmt.Errorf("novelty scheduler: uninitialized scheduler")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stats.EvaluatedTrials++
	novel, err := s.filter.Evaluate(raw)
	if err != nil {
		s.stats.RejectedTrials++
		return false, err
	}
	if novel {
		s.stats.NovelTrials++
	} else {
		s.stats.DuplicateTrials++
	}
	return novel, nil
}

// FilterNovel returns a subset of receipts that achieved novel edge coverage, in original order.
func (s *NoveltyScheduler) FilterNovel(receipts []Receipt) ([]Receipt, error) {
	if s == nil || s.filter == nil {
		return nil, fmt.Errorf("novelty scheduler: uninitialized scheduler")
	}
	var novel []Receipt
	for _, r := range receipts {
		if !r.HasCoverage() {
			continue
		}
		admitted, err := s.EvaluateTrace(r.Coverage.Bytes())
		if err != nil {
			return nil, err
		}
		if admitted {
			novel = append(novel, r)
		}
	}
	return novel, nil
}

// Stats returns a snapshot of scheduling metrics.
func (s *NoveltyScheduler) Stats() SchedulerStats {
	if s == nil {
		return SchedulerStats{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stats
}
