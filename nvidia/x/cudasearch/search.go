package cudasearch

import (
	"context"
	"crypto/sha256"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// Evaluator abstracts candidate code generation, compilation, execution, and timing.
type Evaluator interface {
	// Evaluate extracts and evaluates a candidate from the given search state root.
	Evaluate(ctx context.Context, state *egraph.SearchState, root egraph.Id) (*CandidatePlan, error)
}

// SearchResult records the outcome of a search exploration run.
type SearchResult struct {
	PolicyID        cudacache.SearchPolicyID
	PolicyComplete  bool  // False when policy inputs cannot be identified completely.
	Seed            int64 // Effective per-search RNG seed, or zero for a deterministic strategy.
	ProfileGuidance *cudaprof.SearchGuidance
	ProfileFallback string // Reason configured evidence could not guide exploration.
	ProfileDecision *ProfileDecision
	MemoryDecision  *MemoryDecision // Frozen exploration priors; never an objective measurement.
	// StopReason records why exploration stopped early. A nonnil result may
	// accompany an error and still contain the best already validated plan.
	StopReason       error
	Rejected         []CandidateRejection
	Strategy         string
	Winner           *CandidatePlan
	AllCandidates    []*CandidatePlan
	StatesVisited    int // Initial state plus attempted transitions; excludes enumeration's internal probes.
	Evaluations      int
	Extractions      int
	ExtractionWork   int
	Compilations     int
	WorkAccounted    bool // False means evaluator extraction/compile counts are unknown.
	ActionsAttempted int  // Includes unsuccessful and no-op transitions.
	AccountedBytes   int64
	Duration         time.Duration
}

// CandidateRejection retains a failed evaluation and its exact transition path.
type CandidateRejection struct {
	Derivation []egraph.Action
	Err        error
	Evidence   *EvaluationEvidence // Nil when the evaluator exposes no executable evidence.
}

// MockEvaluator estimates AST cost and produces synthetic timing receipts for
// host strategy tests. Its admission flag does not qualify executable outputs;
// use ExecutableEvaluator for numerical verification. Search freezes Scoring
// and Evidence before its first evaluation; changes apply to the next search.
// Concurrent mutation is not supported.
type MockEvaluator struct {
	CostModel     egraph.CostModel
	Scoring       ScoringModel
	Evidence      map[string]*ProfileEvidence
	TimingDelay   time.Duration
	FailCandidate func(expr *egraph.Expr) bool
	Workload      WorkloadBindings
	ResetCount    int
}

// Evaluate extracts the best AST from root and scores it.
func (m *MockEvaluator) Evaluate(ctx context.Context, state *egraph.SearchState, root egraph.Id) (*CandidatePlan, error) {
	scoring := mockScoring(ctx, m)
	if state == nil || state.Graph == nil {
		return nil, fmt.Errorf("cudasearch: nil state in evaluate")
	}

	// Mutable-state reset between candidates
	if m.Workload.ResetState != nil {
		if err := m.Workload.ResetState(); err != nil {
			return nil, fmt.Errorf("cudasearch: state reset failed: %w", err)
		}
		m.ResetCount++
	}

	if err := chargeWork(ctx, false); err != nil {
		return nil, err
	}
	expr, astCost, err := extractSearch(ctx, state, root, m.CostModel)
	if err != nil {
		return nil, fmt.Errorf("cudasearch: extraction failed: %w", err)
	}

	if m.FailCandidate != nil && m.FailCandidate(expr) {
		return nil, fmt.Errorf("cudasearch: candidate failed correctness validation")
	}

	// This identifies the extracted plan, not its floating-point outputs.
	h := sha256.Sum256([]byte(expr.Fingerprint()))
	digest := fmt.Sprintf("%x", h)

	// Build evaluation receipt
	dur := time.Duration(astCost*1000) * time.Microsecond
	if dur <= 0 {
		dur = time.Microsecond
	}
	r := &cudatrial.Receipt{
		Name:         expr.Op,
		Complete:     true,
		Started:      time.Now(),
		Finished:     time.Now().Add(dur),
		TimingKind:   cudatrial.TimingSummedKernel,
		SummedKernel: dur,
		WallElapsed:  dur + 5*time.Microsecond,
	}
	_ = r.RecordTiming(cudatrial.TimingSummedKernel, dur)
	_ = r.RecordTiming(cudatrial.TimingWallElapsed, dur+5*time.Microsecond)

	totalCost, scoreErr := scoring.model.Score(r, scoring.evidence[expr.Fingerprint()])
	if scoreErr != nil {
		return nil, scoreErr
	}

	cand := &CandidatePlan{
		ID:           fmt.Sprintf("cand-%s-%d", expr.Op, len(state.Derivation)),
		Expr:         expr,
		Derivation:   append([]egraph.Action(nil), state.Derivation...),
		Cost:         totalCost,
		CostSource:   CostEstimated,
		Receipt:      r,
		Generation:   1,
		ScoringEpoch: scoring.model.Epoch,
		Validated:    true,
		Digest:       digest,
	}
	if evidence := scoring.evidence[expr.Fingerprint()]; scoring.model.acceptsEvidence(evidence) {
		copy := *evidence
		cand.Profile = &copy
	}

	return cand, nil
}
