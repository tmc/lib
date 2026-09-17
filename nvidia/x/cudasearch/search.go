package cudasearch

import (
	"context"
	"crypto/sha256"
	"fmt"
	"time"

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
	Strategy       string
	Winner         *CandidatePlan
	AllCandidates  []*CandidatePlan
	StatesVisited  int
	Evaluations    int
	AccountedBytes int64
	Duration       time.Duration
}

// MockEvaluator is a test and host evaluation engine that estimates AST cost
// and optionally simulates execution timing receipts.
type MockEvaluator struct {
	CostModel    egraph.CostModel
	Scoring      ScoringModel
	Evidence     map[string]*ProfileEvidence
	TimingDelay  time.Duration
	FailCandidate func(expr *egraph.Expr) bool
	ComputeDigest func(expr *egraph.Expr) string
	Workload     WorkloadBindings
	ResetCount   int
}

// Evaluate extracts the best AST from root and scores it.
func (m *MockEvaluator) Evaluate(ctx context.Context, state *egraph.SearchState, root egraph.Id) (*CandidatePlan, error) {
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

	ext := egraph.NewExtractor(state.Graph, m.CostModel)
	expr, astCost, err := ext.Extract(root)
	if err != nil {
		return nil, fmt.Errorf("cudasearch: extraction failed: %w", err)
	}

	if m.FailCandidate != nil && m.FailCandidate(expr) {
		return nil, fmt.Errorf("cudasearch: candidate failed correctness validation")
	}

	digest := ""
	if m.ComputeDigest != nil {
		digest = m.ComputeDigest(expr)
	} else {
		h := sha256.Sum256([]byte(expr.Fingerprint()))
		digest = fmt.Sprintf("%x", h)
	}

	// Verify against expected digest if present
	if len(m.Workload.ExpectedDigests) > 0 {
		// Verify default token length or first key
		for _, want := range m.Workload.ExpectedDigests {
			if want != "" && digest != want {
				return nil, fmt.Errorf("%w: got digest %s, want %s", ErrVerificationFailed, digest, want)
			}
			break
		}
	}

	// Build evaluation receipt
	dur := time.Duration(astCost*1000) * time.Microsecond
	if dur <= 0 {
		dur = time.Microsecond
	}
	r := &cudatrial.Receipt{
		Name:         expr.Op,
		Complete:     true,
		Digest:       digest,
		Started:      time.Now(),
		Finished:     time.Now().Add(dur),
		TimingKind:   cudatrial.TimingSummedKernel,
		SummedKernel: dur,
		WallElapsed:  dur + 5*time.Microsecond,
	}
	_ = r.RecordTiming(cudatrial.TimingSummedKernel, dur)
	_ = r.RecordTiming(cudatrial.TimingWallElapsed, dur+5*time.Microsecond)

	var evidence *ProfileEvidence
	if m.Evidence != nil {
		evidence = m.Evidence[expr.Fingerprint()]
	}

	totalCost, scoreErr := m.Scoring.Score(r, evidence)
	if scoreErr != nil {
		totalCost = astCost
	}

	cand := &CandidatePlan{
		ID:          fmt.Sprintf("cand-%s-%d", expr.Op, len(state.Derivation)),
		Expr:        expr,
		Derivation:  append([]egraph.Action(nil), state.Derivation...),
		Cost:        totalCost,
		Receipt:     r,
		Generation:  1,
		Validated:   true,
		Digest:      digest,
		ProfileCost: totalCost - float64(dur.Nanoseconds()),
	}

	return cand, nil
}
