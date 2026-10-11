package egraphcore

import (
	"fmt"
	"math"
	"sort"
)

// MeasuredCandidate is a candidate selected by a caller's measured score.
// Value is opaque to egraphcore; the backend owns candidate construction.
type MeasuredCandidate[T any] struct {
	ID    string
	Score float64
	// Value is retained and returned without copying. Callers must treat it
	// as immutable after Record.
	Value T
}

// MeasuredSearchConfig controls the score-driven part of an evolutionary
// search. Candidate generation and evaluation remain with the caller.
type MeasuredSearchConfig struct {
	KeepBest          int
	GenerationSize    int
	RestartStagnation int
	Descending        bool
}

// MeasuredSearch tracks measured parents and generation-level progress for a
// caller that generates and evaluates candidates. It is backend-neutral.
type MeasuredSearch[T any] struct {
	config            MeasuredSearchConfig
	parents           []MeasuredCandidate[T]
	measured          []MeasuredCandidate[T]
	generationParents []MeasuredCandidate[T]
	generation        int
	budget            int
	improved          bool
	viable            bool
	stagnant          int
	resample          bool
}

// NewMeasuredSearch creates an empty measured population. Zero KeepBest and
// GenerationSize default to one and ten, respectively.
func NewMeasuredSearch[T any](config MeasuredSearchConfig) (*MeasuredSearch[T], error) {
	if config.KeepBest < 0 || config.GenerationSize < 0 || config.RestartStagnation < 0 {
		return nil, fmt.Errorf("egraphcore: invalid measured search bounds")
	}
	if config.KeepBest == 0 {
		config.KeepBest = 1
	}
	if config.GenerationSize == 0 {
		config.GenerationSize = 10
	}
	return &MeasuredSearch[T]{config: config}, nil
}

// Parents reports whether a measured parent is available.
func (s *MeasuredSearch[T]) Parents() bool { return s != nil && len(s.parents) != 0 }

// Resample reports whether the next candidate should be generated without a
// parent, due to an unviable or stagnant generation.
func (s *MeasuredSearch[T]) Resample() bool { return s != nil && s.resample }

// Best returns the current best measured candidate.
func (s *MeasuredSearch[T]) Best() (MeasuredCandidate[T], bool) {
	if s == nil || len(s.parents) == 0 {
		var zero MeasuredCandidate[T]
		return zero, false
	}
	return s.parents[0], true
}

// Ranking returns all measured candidates in score order. Equal scores retain
// encounter order. The returned slice is independent of the search state;
// values are retained without copying and must remain immutable after Record.
func (s *MeasuredSearch[T]) Ranking() []MeasuredCandidate[T] {
	if s == nil || len(s.measured) == 0 {
		return nil
	}
	ranking := append([]MeasuredCandidate[T](nil), s.measured...)
	sort.SliceStable(ranking, func(i, j int) bool {
		return scoreBetter(ranking[i].Score, ranking[j].Score, s.config.Descending)
	})
	return ranking
}

// ParentFor returns the parent assigned to a zero-based candidate position in
// this generation. It uses deterministic, balanced parent allocation.
func (s *MeasuredSearch[T]) ParentFor(position int) (MeasuredCandidate[T], bool) {
	if s == nil || len(s.parents) == 0 || s.resample || position < 0 {
		var zero MeasuredCandidate[T]
		return zero, false
	}
	parents := s.generationParents
	if len(parents) == 0 {
		parents = s.parents
	}
	perParent := 1
	if s.budget > 0 {
		perParent = 1 + (s.budget-1)/len(parents)
	}
	index := position / perParent
	if index >= len(parents) {
		index = len(parents) - 1
	}
	return parents[index], true
}

// MutationLimit returns the configured mutation ceiling after stagnation
// escalation. The caller chooses a count in [1, limit].
func (s *MeasuredSearch[T]) MutationLimit(base int) int {
	if s == nil || base <= 0 || s.config.RestartStagnation == 0 || s.stagnant < s.config.RestartStagnation {
		return base
	}
	kick := 1 + s.stagnant - s.config.RestartStagnation
	if kick > 16 {
		kick = 16
	}
	if base > int(^uint(0)>>1)/kick {
		return int(^uint(0) >> 1)
	}
	return base * kick
}

// Record adds a finite measured candidate to the parent pool and reports
// whether its score improves the incumbent. Equal scores retain encounter
// order, so a full pool is not displaced by a later tie.
func (s *MeasuredSearch[T]) Record(candidate MeasuredCandidate[T]) (bool, error) {
	if s == nil {
		return false, fmt.Errorf("egraphcore: nil measured search")
	}
	if math.IsNaN(candidate.Score) || math.IsInf(candidate.Score, 0) {
		return false, fmt.Errorf("egraphcore: measured score must be finite")
	}
	improved := len(s.parents) == 0 || scoreBetter(candidate.Score, s.parents[0].Score, s.config.Descending)
	s.measured = append(s.measured, candidate)
	s.parents = append(s.parents, candidate)
	sort.SliceStable(s.parents, func(i, j int) bool {
		return scoreBetter(s.parents[i].Score, s.parents[j].Score, s.config.Descending)
	})
	if len(s.parents) > s.config.KeepBest {
		s.parents = s.parents[:s.config.KeepBest]
	}
	s.viable = true
	if improved {
		s.improved = true
	}
	return improved, nil
}

// CloseGeneration advances generation state and sets the next candidate
// budget using remaining measured-candidate capacity.
func (s *MeasuredSearch[T]) CloseGeneration(remaining int) {
	if s == nil {
		return
	}
	if remaining < 0 {
		remaining = 0
	}
	if s.improved {
		s.stagnant = 0
	} else {
		s.stagnant++
	}
	stagnationResample := s.config.RestartStagnation > 0 &&
		s.stagnant >= s.config.RestartStagnation && s.stagnant%2 == 0
	s.resample = !s.viable || stagnationResample
	s.generation++
	s.generationParents = append(s.generationParents[:0], s.parents...)
	s.budget = remaining
	if s.budget > s.config.GenerationSize {
		s.budget = s.config.GenerationSize
	}
	s.improved = false
	s.viable = false
}

// Generation reports the number of completed generations.
func (s *MeasuredSearch[T]) Generation() int {
	if s == nil {
		return 0
	}
	return s.generation
}

func scoreBetter(a, b float64, descending bool) bool {
	if descending {
		return a > b
	}
	return a < b
}
