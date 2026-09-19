package cudasearch

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// SearchBudget configures limits on search exploration.
type SearchBudget struct {
	MaxStates      int           // Maximum total search states visited
	MaxDepth       int           // Maximum transition depth from root
	MaxActions     int           // Maximum candidate actions evaluated per state
	MaxMemoryBytes int64         // Maximum accounted search state memory (or 0 for unlimited)
	MaxEvaluations int           // Maximum number of compiler/evaluator calls
	Timeout        time.Duration // Maximum wall time for search
}

// DefaultBudget provides sensible conservative limits.
func DefaultBudget() SearchBudget {
	return SearchBudget{
		MaxStates:      200,
		MaxDepth:       10,
		MaxActions:     50,
		MaxMemoryBytes: 64 * 1024 * 1024, // 64 MB
		MaxEvaluations: 50,
		Timeout:        30 * time.Second,
	}
}

// BeamConfig specifies parameters for deterministic bounded beam search.
type BeamConfig struct {
	BeamWidth int          // Number of top candidate states retained per depth level
	Budget    SearchBudget // Search resource budget
	RootName  string       // E-graph named root to extract/evaluate (default: "root")
}

// BeamSearch executes deterministic beam exploration over e-graph rewrite transitions.
type BeamSearch struct {
	cfg BeamConfig
}

// NewBeamSearch creates a new BeamSearch engine with the given configuration.
func NewBeamSearch(cfg BeamConfig) *BeamSearch {
	if cfg.BeamWidth <= 0 {
		cfg.BeamWidth = 4
	}
	if cfg.RootName == "" {
		cfg.RootName = "root"
	}
	if cfg.Budget.MaxStates <= 0 {
		cfg.Budget.MaxStates = 200
	}
	if cfg.Budget.MaxDepth <= 0 {
		cfg.Budget.MaxDepth = 10
	}
	return &BeamSearch{cfg: cfg}
}

type beamItem struct {
	state      *egraph.SearchState
	cost       float64
	actionKey  string
	derivation []egraph.Action
	candidate  *CandidatePlan
}

// Search executes beam exploration starting from initial, applying rules, and evaluating
// candidates via evaluator.
func (b *BeamSearch) Search(ctx context.Context, initial *egraph.SearchState, rules []*egraph.Rewrite, eval Evaluator) (*SearchResult, error) {
	if initial == nil || initial.Graph == nil {
		return nil, fmt.Errorf("cudasearch: nil initial search state")
	}
	if eval == nil {
		return nil, fmt.Errorf("cudasearch: nil evaluator")
	}

	startTime := time.Now()
	budget := b.cfg.Budget
	if budget.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, budget.Timeout)
		defer cancel()
	}

	stateCount := 0
	evalCount := 0
	var accountedMemory int64

	rootID, ok := initial.CanonicalRoot(b.cfg.RootName)
	if !ok {
		// Try any available root
		for _, v := range initial.Roots {
			rootID = v
			ok = true
			break
		}
		if !ok {
			return nil, fmt.Errorf("cudasearch: no root e-class in initial state")
		}
	}

	// Evaluate initial state candidate
	initCand, err := eval.Evaluate(ctx, initial, rootID)
	evalCount++
	if err != nil {
		return nil, fmt.Errorf("cudasearch: failed evaluating initial state: %w", err)
	}

	initCost := initCand.Cost
	initialItem := beamItem{
		state:      initial.Clone(),
		cost:       initCost,
		actionKey:  "",
		derivation: nil,
		candidate:  initCand,
	}

	accountedMemory += estimateStateMemory(initialItem.state)
	if budget.MaxMemoryBytes > 0 && accountedMemory > budget.MaxMemoryBytes {
		return nil, ErrMemoryBudgetExceeded
	}

	currentBeam := []beamItem{initialItem}
	var bestGlobal *CandidatePlan = initCand
	var allEvaluated []*CandidatePlan
	allEvaluated = append(allEvaluated, initCand)

	seenStates := make(map[string]bool)
	seenStates[stateFingerprint(initial)] = true

	for depth := 1; depth <= budget.MaxDepth; depth++ {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrSearchCancelled, err)
		}

		var nextFrontier []beamItem

		for _, item := range currentBeam {
			if err := ctx.Err(); err != nil {
				return nil, fmt.Errorf("%w: %v", ErrSearchCancelled, err)
			}
			if stateCount >= budget.MaxStates {
				break
			}

			// Bounded action enumeration
			actionLimit := budget.MaxActions
			if actionLimit <= 0 {
				actionLimit = 50
			}
			actions := egraph.EnumerateActionsBounded(item.state.Graph, rules, actionLimit)

			for _, act := range actions {
				if err := ctx.Err(); err != nil {
					return nil, fmt.Errorf("%w: %v", ErrSearchCancelled, err)
				}
				if stateCount >= budget.MaxStates {
					break
				}
				stateCount++

				res, applyErr := item.state.ApplyAction(act)
				if applyErr != nil || res.NoOp {
					continue
				}

				fp := stateFingerprint(res.State)
				if seenStates[fp] {
					continue
				}
				seenStates[fp] = true

				mem := estimateStateMemory(res.State)
				accountedMemory += mem
				if budget.MaxMemoryBytes > 0 && accountedMemory > budget.MaxMemoryBytes {
					return nil, ErrMemoryBudgetExceeded
				}

				// Extract and evaluate candidate from this state
				candRoot, found := res.State.CanonicalRoot(b.cfg.RootName)
				if !found {
					continue
				}

				if evalCount >= budget.MaxEvaluations {
					// Stop expanding further evaluations
					break
				}

				cand, evalErr := eval.Evaluate(ctx, res.State, candRoot)
				evalCount++
				if evalErr != nil {
					// Discard failing candidate
					continue
				}

				allEvaluated = append(allEvaluated, cand)

				actKey := act.ActionKey()
				nextFrontier = append(nextFrontier, beamItem{
					state:      res.State,
					cost:       cand.Cost,
					actionKey:  actKey,
					derivation: res.State.Derivation,
					candidate:  cand,
				})

				// Update best global independently of beam pruning
				if cand.Cost < bestGlobal.Cost {
					bestGlobal = cand
				}
			}
		}

		if len(nextFrontier) == 0 {
			// No more productive transitions found
			break
		}

		// Sort nextFrontier with stable tie-breaking:
		// Primary: Cost (ascending, lower is better)
		// Secondary: Action key derivation lexicographical order
		sort.SliceStable(nextFrontier, func(i, j int) bool {
			if nextFrontier[i].cost != nextFrontier[j].cost {
				return nextFrontier[i].cost < nextFrontier[j].cost
			}
			return nextFrontier[i].actionKey < nextFrontier[j].actionKey
		})

		// Retain up to BeamWidth items
		if len(nextFrontier) > b.cfg.BeamWidth {
			currentBeam = nextFrontier[:b.cfg.BeamWidth]
		} else {
			currentBeam = nextFrontier
		}
	}

	if bestGlobal == nil {
		return nil, ErrNoAdmittedPlan
	}

	return &SearchResult{
		Strategy:       "beam",
		Winner:         bestGlobal,
		AllCandidates:  allEvaluated,
		StatesVisited:  stateCount,
		Evaluations:    evalCount,
		AccountedBytes: accountedMemory,
		Duration:       time.Since(startTime),
	}, nil
}

func stateFingerprint(s *egraph.SearchState) string {
	if s == nil || s.Graph == nil {
		return "<nil>"
	}
	var sb strings.Builder
	for k, v := range s.Roots {
		sb.WriteString(fmt.Sprintf("%s:%d;", k, s.Graph.Find(v)))
	}
	for _, a := range s.Derivation {
		sb.WriteString(a.Key())
		sb.WriteString(";")
	}
	return sb.String()
}

func estimateStateMemory(s *egraph.SearchState) int64 {
	if s == nil || s.Graph == nil {
		return 0
	}
	// Conservative estimate: ~512 bytes per class + ~128 bytes per derivation step + 2048 base
	classes := len(s.Graph.Classes())
	return int64(2048 + (classes * 512) + (len(s.Derivation) * 128))
}
