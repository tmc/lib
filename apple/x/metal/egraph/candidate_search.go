package egraph

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/egraphcore"
)

// CandidateSearchConfig bounds one caller-driven search over a snapshot.
// MaxAttempts counts proposed whole-class choices, including choices that
// cannot be materialized or that duplicate an earlier candidate. MaxCandidates
// limits measured candidates and invalid evaluations; rejected candidates do
// not consume this budget. The initial seed counts only after measurement.
type CandidateSearchConfig struct {
	// MutationPolicy selects the alternatives that candidate search may use for
	// each reachable class. It is called once per class during construction with
	// a defensive copy of that class's alternatives. It must be deterministic and
	// must not depend on later mutation of the graph or its nodes. A nil policy
	// allows every alternative.
	MutationPolicy CandidateMutationPolicy
	Seed           int64
	MaxAttempts    int
	MaxCandidates  int
	Finalists      int
	ScoreDirection CandidateScoreDirection
	// GenerationSize is the maximum number of offspring in each generation.
	// Zero defaults to 10, matching Luminal.
	GenerationSize int
	// KeepBest is the number of best measured candidates used as parents.
	// Zero defaults to one, matching Luminal.
	KeepBest int
	// Mutations is the base upper bound on class mutations per child. Zero
	// defaults to 10; each child selects uniformly from 1..Mutations, multiplied
	// by the stagnation kick when enabled.
	Mutations int
	// RestartStagnation enables mutation escalation and periodic random
	// resampling after this many generations without a new best score. Zero
	// disables the policy.
	RestartStagnation int
	// SearchTimeout sets an overall deadline from search creation. Initial seed
	// retries continue until a measured seed is found; evolution then stops at
	// the deadline. Zero disables the deadline.
	SearchTimeout time.Duration
	// CandidateTimeout limits caller evaluation from candidate handout through
	// Report. Zero disables the deadline.
	CandidateTimeout time.Duration
	// EarlyStopFactor supplies an optional hint for stopping partial evaluation
	// when a candidate cannot beat the current best. Zero disables the hint.
	EarlyStopFactor float64
}

// CandidateMutationPolicy returns the alternative indices that candidate
// search may select for class. Indices refer to the deterministic alternatives
// order in CandidateSnapshot.Alternatives. The supplied alternatives and their
// child slices are defensive copies. Node payloads must remain immutable. The
// policy should be deterministic; its result is evaluated once and cached when
// NewCandidateSearch is called. An empty result pins the class to alternative
// zero for initial choices and excludes it from mutation.
type CandidateMutationPolicy func(class Id, alternatives []ENode) ([]int, error)

// CandidateScoreDirection controls which finite measured score ranks first.
type CandidateScoreDirection uint8

const (
	// CandidateScoreAscending ranks lower scores first. It is the zero value.
	CandidateScoreAscending CandidateScoreDirection = iota
	// CandidateScoreDescending ranks higher scores first.
	CandidateScoreDescending
)

// CandidateProvenance records how a candidate choice was generated.
type CandidateProvenance struct {
	Kind       string // "initial" or "mutation"
	Attempt    int
	Generation int
	Seed       int64
	ParentID   string
}

// CandidateEarlyStopHint gives the caller a measured incumbent and the
// configured factor for deciding whether to stop partial candidate work.
// The caller owns the stopping decision and reports the partial score normally.
type CandidateEarlyStopHint struct {
	BestScore      float64
	Factor         float64
	ScoreDirection CandidateScoreDirection
}

// SearchCandidate is one materialized candidate awaiting exactly one report.
type SearchCandidate struct {
	ID         string
	Expr       *Expr
	Choice     CandidateChoice
	Provenance CandidateProvenance
	// Deadline is the evaluation deadline, or zero when CandidateTimeout is
	// disabled. Callers can use it to cancel external work.
	Deadline time.Time
	// EarlyStop is nil until there is a measured incumbent or when
	// EarlyStopFactor is disabled. It is advisory; callers decide whether to
	// stop partial work and report its score.
	EarlyStop *CandidateEarlyStopHint
}

// CandidateReportKind identifies the caller's outcome for a yielded candidate.
type CandidateReportKind string

const (
	CandidateMeasured CandidateReportKind = "measured"
	CandidateRejected CandidateReportKind = "rejected"
	CandidateInvalid  CandidateReportKind = "invalid"
	// CandidateTimedOut records a measured outcome received after the candidate
	// deadline. Invalid and explicit CandidateRejected outcomes retain their
	// reported kind, matching Luminal's outcome accounting.
	CandidateTimedOut CandidateReportKind = "timed_out"
)

// CandidateReport contains exactly one outcome. Score must be finite for a
// measured candidate; non-measured outcomes require a non-empty reason.
type CandidateReport struct {
	Kind   CandidateReportKind
	Score  float64
	Reason string
}

// CandidateSearchEvaluator evaluates one generated candidate. It must honor
// ctx because Run cannot interrupt it. Returning an error rejects that
// candidate; context cancellation and deadlines also stop the driver after it
// reports the pending candidate.
type CandidateSearchEvaluator func(context.Context, *SearchCandidate) (CandidateReport, error)

// CandidateResult retains the report and generation metadata for a candidate.
type CandidateResult struct {
	ID         string
	Choice     CandidateChoice
	Provenance CandidateProvenance
	Report     CandidateReport
}

// CandidateFinalist is a measured candidate re-materialized from the search
// snapshot for hard validation. Rank is its 1-based position in the frozen
// measured ranking.
type CandidateFinalist struct {
	ID     string
	Rank   int
	Score  float64
	Choice CandidateChoice
	Expr   *Expr
	// Deadline covers materialization and caller hard validation, or zero when
	// CandidateTimeout is disabled.
	Deadline time.Time
}

// CandidateFinalistReport records the caller's hard-validation verdict.
type CandidateFinalistReport struct {
	Accepted bool
	Reason   string
	TimedOut bool
}

// CandidateFinalistResult retains a finalist's hard-validation outcome.
type CandidateFinalistResult struct {
	ID     string
	Rank   int
	Score  float64
	Report CandidateFinalistReport
}

// CandidateSearchStats separates generated attempts from caller-reported
// candidates and their measured outcomes.
type CandidateSearchStats struct {
	Attempts                int
	Yielded                 int
	Evaluated               int
	Measured                int
	Rejected                int
	Invalid                 int
	Unmaterialized          int
	Duplicates              int
	ProgramDuplicates       int
	ChoiceDigestCollisions  int
	Generations             int
	FinalistsYielded        int
	FinalistsAccepted       int
	FinalistsRejected       int
	FinalistsUnmaterialized int
	TimedOut                int
	FinalistsTimedOut       int
	EvaluatorPanics         int
	StopReason              string
}

// CandidateSearch runs a deterministic, budgeted search over one immutable
// CandidateSnapshot. It yields one candidate at a time and requires a report
// before advancing. The caller owns evaluation and scoring.
type CandidateSearch struct {
	snapshot          *CandidateSnapshot
	materializer      *candidateMaterializer
	mutationPools     map[Id][]int
	config            CandidateSearchConfig
	rng               *rand.Rand
	stats             CandidateSearchStats
	pending           *SearchCandidate
	last              *SearchCandidate
	seen              map[string][]seenCandidate
	programs          map[string][]*Expr
	results           []CandidateResult
	generation        int
	generationYielded int
	generationBudget  int
	finalistRanking   []CandidateResult
	finalistIndex     int
	finalistMode      bool
	finalistPending   *CandidateFinalist
	finalistResults   []CandidateFinalistResult
	searchDeadline    time.Time
	population        *egraphcore.MeasuredSearch[CandidateResult]
}

type seenCandidate struct {
	id   string
	expr *Expr
}

// NewCandidateSearch creates a seeded search. Attempt and candidate budgets
// must be positive; timeout durations may be zero to disable their deadlines.
// Finalists may be zero to disable finalist retention.
func NewCandidateSearch(snapshot *CandidateSnapshot, config CandidateSearchConfig) (*CandidateSearch, error) {
	if snapshot == nil {
		return nil, fmt.Errorf("egraph: nil candidate snapshot")
	}
	if err := snapshot.checkCurrent(); err != nil {
		return nil, err
	}
	if config.MaxAttempts <= 0 || config.MaxCandidates <= 0 {
		return nil, fmt.Errorf("egraph: candidate search budgets must be positive")
	}
	if config.Finalists < 0 {
		return nil, fmt.Errorf("egraph: negative finalist limit")
	}
	if config.GenerationSize < 0 || config.KeepBest < 0 || config.Mutations < 0 || config.RestartStagnation < 0 || config.SearchTimeout < 0 || config.CandidateTimeout < 0 {
		return nil, fmt.Errorf("egraph: negative candidate search option")
	}
	if config.EarlyStopFactor < 0 || math.IsNaN(config.EarlyStopFactor) || math.IsInf(config.EarlyStopFactor, 0) {
		return nil, fmt.Errorf("egraph: early-stop factor must be zero or positive and finite")
	}
	if config.GenerationSize == 0 {
		config.GenerationSize = 10
	}
	if config.KeepBest == 0 {
		config.KeepBest = 1
	}
	if config.Mutations == 0 {
		config.Mutations = 10
	}
	if config.ScoreDirection != CandidateScoreAscending && config.ScoreDirection != CandidateScoreDescending {
		return nil, fmt.Errorf("egraph: invalid candidate score direction %d", config.ScoreDirection)
	}
	mutationPools := make(map[Id][]int, len(snapshot.classes))
	for _, class := range snapshot.classes {
		nodes := snapshot.nodes[class]
		pool := make([]int, 0, len(nodes))
		if config.MutationPolicy == nil {
			for index := range nodes {
				pool = append(pool, index)
			}
		} else {
			alternatives := make([]ENode, len(nodes))
			for index, node := range nodes {
				alternatives[index] = node
				alternatives[index].Children = append([]Id(nil), node.Children...)
			}
			selected, err := config.MutationPolicy(class, alternatives)
			if err != nil {
				return nil, fmt.Errorf("egraph: candidate mutation policy for class %d: %w", class, err)
			}
			seen := make(map[int]bool, len(selected))
			for _, index := range selected {
				if index < 0 || index >= len(nodes) {
					return nil, fmt.Errorf("egraph: candidate mutation policy index %d out of range for class %d", index, class)
				}
				if seen[index] {
					return nil, fmt.Errorf("egraph: candidate mutation policy repeats index %d for class %d", index, class)
				}
				seen[index] = true
				pool = append(pool, index)
			}
		}
		// An empty pool pins this class to its first alternative. This lets a
		// caller exclude a class from mutation without making candidates invalid.
		mutationPools[class] = pool
	}
	materializer, err := newCandidateMaterializer(snapshot)
	if err != nil {
		return nil, err
	}
	startedAt := time.Now()
	searchDeadline := time.Time{}
	if config.SearchTimeout > 0 {
		searchDeadline = startedAt.Add(config.SearchTimeout)
	}
	population, err := egraphcore.NewMeasuredSearch[CandidateResult](egraphcore.MeasuredSearchConfig{
		KeepBest: config.KeepBest, GenerationSize: config.GenerationSize,
		RestartStagnation: config.RestartStagnation,
		Descending:        config.ScoreDirection == CandidateScoreDescending,
	})
	if err != nil {
		return nil, fmt.Errorf("egraph: create measured population: %w", err)
	}
	return &CandidateSearch{
		snapshot:       snapshot,
		materializer:   materializer,
		mutationPools:  mutationPools,
		config:         config,
		rng:            rand.New(rand.NewSource(config.Seed)),
		seen:           make(map[string][]seenCandidate),
		programs:       make(map[string][]*Expr),
		searchDeadline: searchDeadline,
		population:     population,
	}, nil
}

// Next returns the next candidate and waits for its report before another
// candidate can be requested. It returns (nil, nil) when a budget or deadline
// stops the search. Snapshot mutation is reported as an error.
func (s *CandidateSearch) Next() (*SearchCandidate, error) {
	if s == nil || s.snapshot == nil {
		return nil, fmt.Errorf("egraph: nil candidate search")
	}
	if err := s.snapshot.checkCurrent(); err != nil {
		return nil, err
	}
	if s.finalistMode {
		return nil, fmt.Errorf("egraph: candidate generation is frozen after finalist extraction begins")
	}
	if s.pending != nil {
		return nil, fmt.Errorf("egraph: candidate %s awaits a report", s.pending.ID)
	}
	if s.population.Parents() && s.searchExpired() {
		s.stop("search timeout")
		return nil, nil
	}
	for s.stats.Attempts < s.config.MaxAttempts && s.stats.Evaluated < s.config.MaxCandidates {
		if s.population.Parents() && s.searchExpired() {
			s.stop("search timeout")
			return nil, nil
		}
		s.stats.Attempts++
		choice, kind, parent, err := s.nextChoice()
		if err != nil {
			return nil, err
		}
		choice, err = s.repairPolicyCycles(choice)
		if err != nil {
			s.stats.Unmaterialized++
			continue
		}
		programHash, err := s.materializer.fingerprint(choice)
		if err != nil {
			s.stats.Unmaterialized++
			continue
		}
		reachable, err := s.materializer.reachableClasses(choice)
		if err != nil {
			s.stats.Unmaterialized++
			continue
		}
		id := candidateChoiceID(s.snapshot, choice, reachable)
		programDuplicate := false
		// ValueFingerprint is conservative: when it cannot fingerprint a payload,
		// it uses a type-only bucket. Deeply equal programs therefore always
		// share this bucket, and equality only needs to scan its collision list.
		for _, seenExpr := range s.programs[programHash] {
			if candidateChoiceEqualExpr(s.snapshot, choice, seenExpr) {
				programDuplicate = true
				break
			}
		}
		if programDuplicate {
			s.stats.Duplicates++
			s.stats.ProgramDuplicates++
			continue
		}
		baseID := id
		bucket := s.seen[baseID]
		duplicate := false
		for _, seen := range bucket {
			if candidateChoiceEqualExpr(s.snapshot, choice, seen.expr) {
				duplicate = true
				break
			}
		}
		if duplicate {
			s.stats.Duplicates++
			continue
		}
		expr, err := s.materializer.materialize(choice)
		if err != nil {
			s.stats.Unmaterialized++
			continue
		}
		if s.population.Parents() && s.searchExpired() {
			s.stop("search timeout")
			return nil, nil
		}
		if len(bucket) != 0 {
			s.stats.ChoiceDigestCollisions++
			id = fmt.Sprintf("%s-%d", id, len(bucket))
		}
		s.seen[baseID] = append(bucket, seenCandidate{id: id, expr: expr})
		s.programs[programHash] = append(s.programs[programHash], expr)
		deadline := time.Time{}
		if s.config.CandidateTimeout > 0 {
			deadline = time.Now().Add(s.config.CandidateTimeout)
		}
		candidate := &SearchCandidate{
			ID: id, Expr: expr, Choice: cloneCandidateChoice(choice),
			Provenance: CandidateProvenance{Kind: kind, Attempt: s.stats.Attempts, Generation: s.generation, Seed: s.config.Seed, ParentID: parent},
			Deadline:   deadline,
		}
		if best, ok := s.population.Best(); s.config.EarlyStopFactor > 0 && ok {
			hint := CandidateEarlyStopHint{
				BestScore:      best.Score,
				Factor:         s.config.EarlyStopFactor,
				ScoreDirection: s.config.ScoreDirection,
			}
			candidate.EarlyStop = &hint
		}
		s.pending = candidate
		s.last = candidate
		s.stats.Yielded++
		s.generationYielded++
		return cloneSearchCandidate(candidate), nil
	}
	if s.stats.StopReason == "" {
		s.stop("search budgets exhausted")
	}
	return nil, nil
}

// Run evaluates candidates until the search is exhausted, the context stops,
// or the evaluator returns a context cancellation error. It requires that no
// candidate is pending and finalist extraction has not started. Evaluator
// panics become rejected candidates and do not strand the search state.
func (s *CandidateSearch) Run(ctx context.Context, evaluate CandidateSearchEvaluator) error {
	if s == nil || s.snapshot == nil {
		return fmt.Errorf("egraph: nil candidate search")
	}
	if ctx == nil {
		return fmt.Errorf("egraph: nil candidate search context")
	}
	if evaluate == nil {
		return fmt.Errorf("egraph: nil candidate search evaluator")
	}
	if s.pending != nil {
		return fmt.Errorf("egraph: candidate %s awaits a report", s.pending.ID)
	}
	if s.finalistMode {
		return fmt.Errorf("egraph: candidate generation is frozen after finalist extraction begins")
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		candidate, err := s.Next()
		if err != nil || candidate == nil {
			return err
		}
		candidateID := candidate.ID
		report, evalErr, panicValue := evaluateCandidateSearch(ctx, evaluate, candidate)
		if panicValue != nil {
			s.stats.EvaluatorPanics++
			report = CandidateReport{Kind: CandidateRejected, Reason: fmt.Sprintf("candidate evaluator panicked: %v", panicValue)}
		} else if evalErr != nil {
			report = CandidateReport{Kind: CandidateRejected, Reason: "candidate evaluator: " + evalErr.Error()}
		}
		stopErr := error(nil)
		if errors.Is(evalErr, context.Canceled) || errors.Is(evalErr, context.DeadlineExceeded) {
			stopErr = evalErr
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			report = CandidateReport{Kind: CandidateRejected, Reason: "candidate search context: " + ctxErr.Error()}
			stopErr = ctxErr
		}
		if reportErr := s.Report(candidateID, report); reportErr != nil {
			fallback := CandidateReport{Kind: CandidateInvalid, Reason: "candidate evaluator returned invalid report: " + reportErr.Error()}
			if fallbackErr := s.Report(candidateID, fallback); fallbackErr != nil {
				return errors.Join(reportErr, fallbackErr)
			}
		}
		if stopErr != nil {
			return stopErr
		}
	}
}

func evaluateCandidateSearch(ctx context.Context, evaluate CandidateSearchEvaluator, candidate *SearchCandidate) (report CandidateReport, err error, panicValue any) {
	defer func() {
		if value := recover(); value != nil {
			report = CandidateReport{}
			err = nil
			panicValue = value
		}
	}()
	report, err = evaluate(ctx, candidate)
	return report, err, nil
}

// NextFinalist freezes the measured ranking on its first call, then lazily
// re-materializes candidates in score order for caller hard validation.
// Candidates that cannot be materialized are recorded and skipped. A search
// candidate must not be awaiting a report when finalist extraction begins.
func (s *CandidateSearch) NextFinalist() (*CandidateFinalist, error) {
	if s == nil || s.snapshot == nil {
		return nil, fmt.Errorf("egraph: nil candidate search")
	}
	if err := s.snapshot.checkCurrent(); err != nil {
		return nil, err
	}
	if s.pending != nil {
		return nil, fmt.Errorf("egraph: candidate %s awaits a report", s.pending.ID)
	}
	if s.finalistPending != nil {
		return nil, fmt.Errorf("egraph: finalist %s awaits a report", s.finalistPending.ID)
	}
	if s.finalistMode && s.finalistIndex > 0 && s.searchExpired() {
		s.stop(fmt.Sprintf("search timeout before finalist rank %d", s.finalistIndex+1))
		return nil, nil
	}
	if s.config.Finalists == 0 {
		return nil, nil
	}
	searchBudgetOpen := s.stats.Attempts < s.config.MaxAttempts && s.stats.Evaluated < s.config.MaxCandidates
	if !s.finalistMode && searchBudgetOpen && (!s.searchExpired() || !s.population.Parents()) {
		return nil, fmt.Errorf("egraph: candidate search budgets are not exhausted")
	}
	if !s.finalistMode {
		s.finalistMode = true
		s.finalistRanking = s.Ranking()
		if s.config.Finalists > 0 && len(s.finalistRanking) > s.config.Finalists {
			s.finalistRanking = s.finalistRanking[:s.config.Finalists]
		}
	}
	for s.finalistIndex < len(s.finalistRanking) {
		if s.finalistIndex > 0 && s.searchExpired() {
			s.stop(fmt.Sprintf("search timeout before finalist rank %d", s.finalistIndex+1))
			return nil, nil
		}
		result := s.finalistRanking[s.finalistIndex]
		s.finalistIndex++
		rank := s.finalistIndex
		startedAt := time.Now()
		deadline := time.Time{}
		if s.config.CandidateTimeout > 0 {
			deadline = startedAt.Add(s.config.CandidateTimeout)
		}
		expr, err := s.materializer.materialize(result.Choice)
		if err != nil {
			s.stats.FinalistsUnmaterialized++
			s.stats.FinalistsRejected++
			s.finalistResults = append(s.finalistResults, CandidateFinalistResult{
				ID: result.ID, Rank: rank, Score: result.Report.Score,
				Report: CandidateFinalistReport{Reason: "materialization failed: " + err.Error()},
			})
			continue
		}
		if expired(deadline, time.Now()) {
			s.stats.FinalistsTimedOut++
			s.stats.FinalistsRejected++
			s.finalistResults = append(s.finalistResults, CandidateFinalistResult{
				ID: result.ID, Rank: rank, Score: result.Report.Score,
				Report: CandidateFinalistReport{TimedOut: true, Reason: "candidate timeout expired during finalist materialization"},
			})
			continue
		}
		finalist := &CandidateFinalist{
			ID: result.ID, Rank: rank, Score: result.Report.Score,
			Choice: cloneCandidateChoice(result.Choice), Expr: expr, Deadline: deadline,
		}
		s.finalistPending = finalist
		s.stats.FinalistsYielded++
		return cloneCandidateFinalist(finalist), nil
	}
	return nil, nil
}

// RestartCandidateTimer resets the pending candidate's evaluation deadline.
// Callers can use it after setup work that should not count against the
// candidate's measurement timeout. A zero return deadline means timeouts are
// disabled.
func (s *CandidateSearch) RestartCandidateTimer(candidateID string) (time.Time, error) {
	if s == nil || s.snapshot == nil {
		return time.Time{}, fmt.Errorf("egraph: nil candidate search")
	}
	if err := s.snapshot.checkCurrent(); err != nil {
		return time.Time{}, err
	}
	if s.pending == nil {
		return time.Time{}, fmt.Errorf("egraph: no candidate awaits a report")
	}
	if candidateID != s.pending.ID {
		return time.Time{}, fmt.Errorf("egraph: timer candidate ID %q does not match pending candidate %q", candidateID, s.pending.ID)
	}
	if s.config.CandidateTimeout == 0 {
		s.pending.Deadline = time.Time{}
		return time.Time{}, nil
	}
	s.pending.Deadline = time.Now().Add(s.config.CandidateTimeout)
	return s.pending.Deadline, nil
}

// ReportFinalist records exactly one hard-validation outcome for the pending
// finalist. Rejections require a reason; acceptance must not include one.
func (s *CandidateSearch) ReportFinalist(finalistID string, report CandidateFinalistReport) error {
	_, err := s.reportFinalist(finalistID, report)
	return err
}

func (s *CandidateSearch) reportFinalist(finalistID string, report CandidateFinalistReport) (bool, error) {
	if s == nil || s.snapshot == nil {
		return false, fmt.Errorf("egraph: nil candidate search")
	}
	if err := s.snapshot.checkCurrent(); err != nil {
		return false, err
	}
	if s.finalistPending == nil {
		return false, fmt.Errorf("egraph: no finalist awaits a report")
	}
	if finalistID != s.finalistPending.ID {
		return false, fmt.Errorf("egraph: finalist report ID %q does not match pending finalist %q", finalistID, s.finalistPending.ID)
	}
	if expired(s.finalistPending.Deadline, time.Now()) {
		report.Accepted = false
		report.TimedOut = true
		report.Reason = "candidate timeout expired during finalist validation"
	}
	if report.Accepted {
		if report.Reason != "" || report.TimedOut {
			return false, fmt.Errorf("egraph: accepted finalist report cannot have a reason")
		}
		s.stats.FinalistsAccepted++
	} else {
		if strings.TrimSpace(report.Reason) == "" {
			return false, fmt.Errorf("egraph: rejected finalist report requires a reason")
		}
		s.stats.FinalistsRejected++
		if report.TimedOut {
			s.stats.FinalistsTimedOut++
		}
	}
	finalist := s.finalistPending
	s.finalistResults = append(s.finalistResults, CandidateFinalistResult{
		ID: finalist.ID, Rank: finalist.Rank, Score: finalist.Score, Report: report,
	})
	s.finalistPending = nil
	return report.Accepted, nil
}

// FinalistResults returns hard-validation outcomes in report order, including
// automatic materialization failures.
func (s *CandidateSearch) FinalistResults() []CandidateFinalistResult {
	if s == nil {
		return nil
	}
	return append([]CandidateFinalistResult(nil), s.finalistResults...)
}

// Report records exactly one outcome for candidateID, which must identify the
// pending candidate. Score order is controlled by
// CandidateSearchConfig.ScoreDirection. A second, stale, or mismatched report
// is an error.
func (s *CandidateSearch) Report(candidateID string, report CandidateReport) error {
	if s == nil || s.snapshot == nil {
		return fmt.Errorf("egraph: nil candidate search")
	}
	if err := s.snapshot.checkCurrent(); err != nil {
		return err
	}
	if s.pending == nil {
		return fmt.Errorf("egraph: no candidate awaits a report")
	}
	if candidateID != s.pending.ID {
		return fmt.Errorf("egraph: report candidate ID %q does not match pending candidate %q", candidateID, s.pending.ID)
	}
	switch report.Kind {
	case CandidateMeasured:
		if !math.IsNaN(report.Score) && !math.IsInf(report.Score, 0) {
			if report.Reason != "" {
				return fmt.Errorf("egraph: measured candidate report has a reason")
			}
		} else {
			return fmt.Errorf("egraph: measured candidate score must be finite")
		}
	case CandidateRejected, CandidateInvalid, CandidateTimedOut:
		if strings.TrimSpace(report.Reason) == "" {
			return fmt.Errorf("egraph: %s candidate report requires a reason", report.Kind)
		}
		if report.Score != 0 {
			return fmt.Errorf("egraph: %s candidate report cannot have a score", report.Kind)
		}
	default:
		return fmt.Errorf("egraph: unknown candidate report kind %q", report.Kind)
	}
	candidate := s.pending
	if expired(candidate.Deadline, time.Now()) && report.Kind == CandidateMeasured {
		report = CandidateReport{Kind: CandidateTimedOut, Reason: "candidate timeout expired"}
	}
	s.results = append(s.results, CandidateResult{
		ID: candidate.ID, Choice: cloneCandidateChoice(candidate.Choice),
		Provenance: candidate.Provenance, Report: report,
	})
	switch report.Kind {
	case CandidateMeasured:
		s.stats.Evaluated++
		s.stats.Measured++
		_, err := s.population.Record(egraphcore.MeasuredCandidate[CandidateResult]{
			ID: candidate.ID, Score: report.Score, Value: cloneCandidateResult(s.results[len(s.results)-1]),
		})
		if err != nil {
			return err
		}
	case CandidateRejected:
		s.stats.Rejected++
	case CandidateInvalid:
		s.stats.Invalid++
		if s.population.Parents() {
			s.stats.Evaluated++
		}
	case CandidateTimedOut:
		s.stats.TimedOut++
		if s.population.Parents() {
			s.stats.Evaluated++
		}
	}
	s.pending = nil
	if s.population.Parents() && (s.generation == 0 || s.generationYielded >= s.generationBudget) {
		s.closeGeneration()
	}
	return nil
}

// Stats returns a copy of the search counters.
func (s *CandidateSearch) Stats() CandidateSearchStats {
	if s == nil {
		return CandidateSearchStats{}
	}
	return s.stats
}

// Results returns all caller-reported outcomes in report order.
func (s *CandidateSearch) Results() []CandidateResult {
	if s == nil {
		return nil
	}
	results := make([]CandidateResult, len(s.results))
	for i, result := range s.results {
		results[i] = cloneCandidateResult(result)
	}
	return results
}

// Ranking returns all measured outcomes in configured score order. Equal
// scores retain report order. Rejected and invalid candidates are omitted.
func (s *CandidateSearch) Ranking() []CandidateResult {
	if s == nil {
		return nil
	}
	measured := make([]CandidateResult, 0, s.stats.Measured)
	for _, result := range s.results {
		if result.Report.Kind == CandidateMeasured {
			measured = append(measured, cloneCandidateResult(result))
		}
	}
	sort.SliceStable(measured, func(i, j int) bool {
		if s.config.ScoreDirection == CandidateScoreDescending {
			return measured[i].Report.Score > measured[j].Report.Score
		}
		return measured[i].Report.Score < measured[j].Report.Score
	})
	return measured
}

// Finalists returns up to the configured number of best measured outcomes.
func (s *CandidateSearch) Finalists() []CandidateResult {
	if s == nil || s.config.Finalists == 0 {
		return nil
	}
	ranked := s.Ranking()
	if len(ranked) > s.config.Finalists {
		ranked = ranked[:s.config.Finalists]
	}
	return ranked
}

func (s *CandidateSearch) searchExpired() bool {
	return s != nil && expired(s.searchDeadline, time.Now())
}

func (s *CandidateSearch) stop(reason string) {
	if s != nil && s.stats.StopReason == "" {
		s.stats.StopReason = reason
	}
}

func expired(deadline, now time.Time) bool {
	return !deadline.IsZero() && !now.Before(deadline)
}

func (s *CandidateSearch) nextChoice() (CandidateChoice, string, string, error) {
	if !s.population.Parents() || s.population.Resample() {
		choice, err := s.randomPolicyChoice()
		if err != nil {
			return nil, "", "", err
		}
		return choice, "initial", "", nil
	}
	parent, ok := s.population.ParentFor(s.generationYielded)
	if !ok {
		return nil, "", "", fmt.Errorf("egraph: measured population has no parent")
	}
	choice := cloneCandidateChoice(parent.Value.Choice)
	reachable, err := s.materializer.reachableClasses(choice)
	if err != nil {
		return nil, "", "", err
	}
	genome := make([]int, len(reachable))
	genes := make([]egraphcore.DiscreteChoiceGene, 0, len(reachable))
	for index, class := range reachable {
		genome[index] = choice[class]
		if len(s.mutationPools[class]) > 1 {
			genes = append(genes, egraphcore.DiscreteChoiceGene{
				Index: index, Alleles: append([]int(nil), s.mutationPools[class]...),
			})
		}
	}
	mutations := s.population.MutationLimit(s.config.Mutations)
	if mutations > 1 {
		mutations = s.rng.Intn(mutations) + 1
	}
	genome, err = egraphcore.MutateDiscreteChoice(genome, genes, mutations, s.rng)
	if err != nil {
		return nil, "", "", fmt.Errorf("egraph: mutate candidate choice: %w", err)
	}
	for index, class := range reachable {
		choice[class] = genome[index]
	}
	return choice, "mutation", parent.ID, nil
}

func (s *CandidateSearch) randomPolicyChoice() (CandidateChoice, error) {
	if err := s.snapshot.checkCurrent(); err != nil {
		return nil, err
	}
	choice, err := s.snapshot.DefaultChoice()
	if err != nil {
		return nil, err
	}
	for _, class := range s.snapshot.classes {
		pool := s.mutationPools[class]
		if len(pool) != 0 {
			choice[class] = pool[s.rng.Intn(len(pool))]
		}
	}
	return s.repairPolicyCycles(choice)
}

// repairPolicyCycles mirrors snapshot cycle repair while restricting every
// replacement to the cached policy pool.
func (s *CandidateSearch) repairPolicyCycles(choice CandidateChoice) (CandidateChoice, error) {
	allowed := make(map[Id]map[int]bool, len(s.mutationPools))
	for class, pool := range s.mutationPools {
		indices := make(map[int]bool, len(pool))
		for _, index := range pool {
			indices[index] = true
		}
		allowed[class] = indices
	}
	repaired, err := egraphcore.RepairChoiceCycles(
		s.snapshot.root,
		map[Id]int(choice),
		func(class Id) []ENode { return s.snapshot.nodes[class] },
		func(node ENode) []Id { return node.Children },
		func(class Id, index int) bool { return allowed[class][index] },
		s.rng.Intn,
		128,
	)
	if err != nil {
		return nil, fmt.Errorf("egraph: repair candidate cycles within mutation policy: %w", err)
	}
	return CandidateChoice(repaired), nil
}

func (s *CandidateSearch) closeGeneration() {
	remaining := s.config.MaxCandidates - s.stats.Evaluated
	s.population.CloseGeneration(remaining)
	s.generation = s.population.Generation()
	s.stats.Generations = s.generation
	s.generationYielded = 0
	s.generationBudget = remaining
	if s.generationBudget > s.config.GenerationSize {
		s.generationBudget = s.config.GenerationSize
	}
}

func candidateChoiceID(snapshot *CandidateSnapshot, choice CandidateChoice, classes []Id) string {
	h := sha256.New()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(snapshot.root))
	_, _ = h.Write(buf[:])
	binary.LittleEndian.PutUint64(buf[:], snapshot.revision)
	_, _ = h.Write(buf[:])
	for _, class := range classes {
		index := choice[class]
		node := snapshot.nodes[class][index]
		binary.LittleEndian.PutUint64(buf[:], uint64(class))
		_, _ = h.Write(buf[:])
		key := node.Key()
		binary.LittleEndian.PutUint64(buf[:], uint64(len(key)))
		_, _ = h.Write(buf[:])
		_, _ = h.Write([]byte(key))
	}
	return hex.EncodeToString(h.Sum(nil)[:16])
}

func cloneCandidateChoice(choice CandidateChoice) CandidateChoice {
	clone := make(CandidateChoice, len(choice))
	for class, index := range choice {
		clone[class] = index
	}
	return clone
}

func cloneSearchCandidate(candidate *SearchCandidate) *SearchCandidate {
	if candidate == nil {
		return nil
	}
	copy := *candidate
	copy.Expr = cloneExpr(candidate.Expr, make(map[*Expr]*Expr))
	copy.Choice = cloneCandidateChoice(candidate.Choice)
	if candidate.EarlyStop != nil {
		hint := *candidate.EarlyStop
		copy.EarlyStop = &hint
	}
	return &copy
}

func cloneCandidateFinalist(finalist *CandidateFinalist) *CandidateFinalist {
	if finalist == nil {
		return nil
	}
	copy := *finalist
	copy.Expr = cloneExpr(finalist.Expr, make(map[*Expr]*Expr))
	copy.Choice = cloneCandidateChoice(finalist.Choice)
	return &copy
}

func cloneExpr(expr *Expr, clones map[*Expr]*Expr) *Expr {
	if expr == nil {
		return nil
	}
	if clone, ok := clones[expr]; ok {
		return clone
	}
	clone := &Expr{Op: expr.Op, Sort: expr.Sort, Data: expr.Data, Children: make([]*Expr, len(expr.Children))}
	clones[expr] = clone
	for i, child := range expr.Children {
		clone.Children[i] = cloneExpr(child, clones)
	}
	return clone
}

func cloneCandidateResult(result CandidateResult) CandidateResult {
	result.Choice = cloneCandidateChoice(result.Choice)
	return result
}
