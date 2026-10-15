package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// SearchBudget configures limits on search exploration. Negative limits are
// rejected by Search. Zero state, depth, action and evaluation limits select
// defaults; zero memory, extraction, compilation and timeout limits are unlimited.
type SearchBudget struct {
	MaxStates          int // Initial state plus attempted transitions, including failures/no-ops.
	MaxNodesPerState   int // Maximum admitted e-nodes per state; zero is unlimited.
	MaxClassesPerState int // Maximum admitted e-classes per state; zero is unlimited.
	MaxDepth           int // Maximum transition depth from root
	MaxActions         int // Maximum candidate actions evaluated per state
	MaxTotalActions    int // Maximum attempted transitions across the search (default 1000)
	// MaxMemoryBytes limits accounted storage: reachable state and payload
	// storage, retained candidate data, declared callback workspace, and device
	// buffers reserved by built-in compilers. Zero is unlimited. Accounting uses
	// Go value sizes, slice capacities, strings and live map entries; it excludes
	// allocator/map bucket overhead, runtime/driver/compiler internals and caller
	// captures. Matching/extraction scratch and transition growth are not yet
	// reserved before allocation, so this is not a complete owned-allocation cap.
	MaxMemoryBytes int64
	// CallbackMemoryBytes declares the maximum simultaneously owned storage of
	// opaque evaluator/compiler callbacks, including opaque executable workspace.
	// With a memory limit, zero refuses callbacks lacking built-in accounting.
	// The caller must uphold this declaration; it is not a sandbox or measured
	// peak. A zero-allocation callback still needs a conservative positive
	// declaration because zero means undeclared. Built-in compiler device buffers
	// are reserved separately.
	CallbackMemoryBytes int64
	MaxEvaluations      int           // Maximum number of compiler/evaluator calls
	MaxExtractions      int           // Built-in evaluator extraction attempts; zero is unlimited.
	MaxExtractionWork   int           // Total built-in extraction work units; zero is unlimited.
	MaxCompilations     int           // Built-in evaluator compile attempts; zero is unlimited.
	Timeout             time.Duration // Maximum wall time for search
}

// DefaultBudget provides sensible conservative limits.
func DefaultBudget() SearchBudget {
	return SearchBudget{
		MaxStates:          200,
		MaxNodesPerState:   4096,
		MaxClassesPerState: 2048,
		MaxDepth:           10,
		MaxActions:         50,
		MaxTotalActions:    1000,
		MaxMemoryBytes:     64 * 1024 * 1024, // 64 MB
		MaxEvaluations:     50,
		Timeout:            30 * time.Second,
	}
}

// BeamConfig specifies parameters for deterministic bounded beam search.
type BeamConfig struct {
	EvaluatorID string // Caller-qualified evaluator/callback policy identity, if needed.
	Profile     *ProfileActions
	Memory      *SearchMemory // Optional frozen catalog and cross-task exploration policy.
	BeamWidth   int           // Number of top candidate states retained per depth level
	Budget      SearchBudget  // Search resource budget
	RootName    string        // E-graph named root to extract/evaluate (default: "root")
}

// BeamSearch executes deterministic beam exploration over e-graph rewrite transitions.
type BeamSearch struct {
	cfg       BeamConfig
	configErr error
}

// NewBeamSearch creates a new BeamSearch engine with the given configuration.
func NewBeamSearch(cfg BeamConfig) *BeamSearch {
	if err := validateSearchBudget(cfg.Budget); err != nil {
		return &BeamSearch{configErr: err}
	}
	if cfg.BeamWidth < 0 {
		return &BeamSearch{configErr: fmt.Errorf("cudasearch: negative beam width")}
	}
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
	if cfg.Budget.MaxEvaluations <= 0 {
		cfg.Budget.MaxEvaluations = 50
	}
	if cfg.Budget.MaxActions == 0 {
		cfg.Budget.MaxActions = DefaultBudget().MaxActions
	}
	if cfg.Budget.MaxTotalActions <= 0 {
		cfg.Budget.MaxTotalActions = DefaultBudget().MaxTotalActions
	}
	return &BeamSearch{cfg: cfg}
}

type beamItem struct {
	state      *egraph.SearchState
	cost       float64
	derivation []egraph.Action
	candidate  *CandidatePlan
}

// Search executes beam exploration starting from initial, applying rules, and evaluating
// candidates via evaluator.
func (b *BeamSearch) Search(ctx context.Context, initial *egraph.SearchState, rules []*egraph.Rewrite, eval Evaluator) (*SearchResult, error) {
	if b.configErr != nil {
		return nil, b.configErr
	}
	if initial == nil || initial.Graph == nil {
		return nil, fmt.Errorf("cudasearch: nil initial search state")
	}
	if eval == nil {
		return nil, fmt.Errorf("cudasearch: nil evaluator")
	}

	startTime := time.Now()
	budget := b.cfg.Budget
	profile := freezeProfileActions(b.cfg.Profile)
	if err := validateMemoryEvaluator(b.cfg.Memory, eval); err != nil {
		return nil, err
	}
	if err := freezeSearchMemory(b.cfg.Memory, &profile); err != nil {
		return nil, err
	}
	ctx, work, err := searchWorkBudget(ctx, budget, eval)
	if err != nil {
		return nil, err
	}
	policy, policyComplete, err := makeSearchPolicy("beam", b.cfg.RootName, budget, map[string]string{"beam_width": fmt.Sprint(b.cfg.BeamWidth), "tie_order": "full-replay-path-v1"}, 0, rules, eval, work, profile, b.cfg.EvaluatorID)
	if err != nil {
		return nil, err
	}
	if budget.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, budget.Timeout)
		defer cancel()
	}

	stateCount := 0
	evalCount := 0
	actionCount := 0
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrSearchCancelled, err)
	}

	rootID, ok := initial.CanonicalRoot(b.cfg.RootName)
	if !ok {
		return nil, fmt.Errorf("cudasearch: missing search root %q", b.cfg.RootName)
	}
	if err := admitGraphBudget(initial, budget); err != nil {
		return nil, err
	}
	if err := retainSearchValue(ctx, initial); err != nil {
		return nil, err
	}

	// Evaluate initial state candidate
	initCand, err := evaluateSearch(ctx, eval, initial, rootID)
	evalCount++
	if err == nil {
		err = admitSearchCandidate(initCand)
	}
	if err != nil {
		failure := fmt.Errorf("cudasearch: failed evaluating initial state: %w", err)
		rejectedInitial, memoryErr := retainSearchRejection(ctx, initial, eval, err)
		var initialRejections []CandidateRejection
		if memoryErr == nil {
			initialRejections = []CandidateRejection{rejectedInitial}
		} else {
			failure = errors.Join(failure, memoryErr)
		}
		return &SearchResult{PolicyID: policy, PolicyComplete: policyComplete, Strategy: "beam",
			MemoryDecision: profile.memory, ProfileDecision: profile.decision, ProfileGuidance: profile.guidance, ProfileFallback: profile.fallback,
			Rejected: initialRejections, StopReason: failure,
			StatesVisited: 1, Evaluations: 1, Extractions: work.extractions, ExtractionWork: work.extractionWork,
			Compilations: work.compilations, WorkAccounted: work.known, AccountedBytes: work.memory.accountedBytes(),
			Duration: time.Since(startTime)}, failure
	}

	initCost := initCand.Cost
	initialItem := beamItem{
		state:      initial,
		cost:       initCost,
		derivation: initial.Derivation,
		candidate:  initCand,
	}

	stateCount = 1
	currentBeam := []beamItem{initialItem}
	var bestGlobal *CandidatePlan = initCand
	bestPath := initial.Derivation
	var allEvaluated []*CandidatePlan
	allEvaluated = append(allEvaluated, initCand)
	var rejected []CandidateRejection
	var stopReason error
	finish := func(err error) (*SearchResult, error) {
		if work.exhausted {
			stopReason = ErrBudgetExhausted
		}
		if err != nil {
			stopReason = err
		}
		return &SearchResult{MemoryDecision: profile.memory, PolicyID: policy, PolicyComplete: policyComplete, ProfileDecision: profile.decision, ProfileGuidance: profile.guidance, ProfileFallback: profile.fallback, Strategy: "beam", Winner: bestGlobal, AllCandidates: allEvaluated, Rejected: rejected, StopReason: stopReason, StatesVisited: stateCount, ActionsAttempted: actionCount, Evaluations: evalCount, Extractions: work.extractions, ExtractionWork: work.extractionWork, Compilations: work.compilations, WorkAccounted: work.known, AccountedBytes: work.memory.accountedBytes(), Duration: time.Since(startTime)}, err
	}

	cloned, err := cloneSearchState(ctx, initial)
	if err != nil {
		return finish(err)
	}
	currentBeam[0].state = cloned
	seenStates := make(map[string]bool)
	seenStates[stateFingerprint(initial)] = true

	for depth := 1; depth <= budget.MaxDepth; depth++ {
		if err := ctx.Err(); err != nil {
			return finish(fmt.Errorf("%w: %w", ErrSearchCancelled, err))
		}

		var nextFrontier []beamItem

		for _, item := range currentBeam {
			if actionCount >= budget.MaxTotalActions {
				stopReason = ErrBudgetExhausted
				return finish(nil)
			}
			if err := ctx.Err(); err != nil {
				return finish(fmt.Errorf("%w: %w", ErrSearchCancelled, err))
			}
			if stateCount >= budget.MaxStates {
				stopReason = ErrBudgetExhausted
				break
			}
			if evalCount >= budget.MaxEvaluations {
				stopReason = ErrBudgetExhausted
				break
			}

			// Bounded action enumeration
			actionLimit := budget.MaxActions
			if actionLimit <= 0 {
				actionLimit = 50
			}
			actions, err := enumerateSearchActions(ctx, item.state, rules, actionLimit)
			if err != nil {
				return finish(err)
			}
			profile.order(actions)

			for _, act := range actions {
				if actionCount >= budget.MaxTotalActions {
					stopReason = ErrBudgetExhausted
					return finish(nil)
				}
				if err := ctx.Err(); err != nil {
					return finish(fmt.Errorf("%w: %w", ErrSearchCancelled, err))
				}
				if stateCount >= budget.MaxStates {
					stopReason = ErrBudgetExhausted
					break
				}
				if evalCount >= budget.MaxEvaluations {
					stopReason = ErrBudgetExhausted
					break
				}
				stateCount++

				actionCount++
				res, applyErr := applySearchAction(ctx, item.state, act)
				if errors.Is(applyErr, ErrMemoryBudgetExceeded) || errors.Is(applyErr, ErrMemoryAccountingUnavailable) {
					return finish(applyErr)
				}
				if applyErr != nil || res.NoOp {
					continue
				}
				if err := admitGraphBudget(res.State, budget); err != nil {
					stopReason = err
					return finish(nil)
				}

				fp := stateFingerprint(res.State)
				if seenStates[fp] {
					continue
				}
				if err := retainSearchValue(ctx, fp); err != nil {
					return finish(err)
				}
				seenStates[fp] = true

				// Extract and evaluate candidate from this state
				candRoot, found := res.State.CanonicalRoot(b.cfg.RootName)
				if !found {
					continue
				}

				if evalCount >= budget.MaxEvaluations {
					// Stop expanding further evaluations
					break
				}

				cand, evalErr := evaluateSearch(ctx, eval, res.State, candRoot)
				evalCount++
				if evalErr == nil {
					evalErr = admitComparableCandidate(cand, initCand)
				}
				if evalErr != nil {
					r, memoryErr := retainSearchRejection(ctx, res.State, eval, evalErr)
					if memoryErr != nil {
						return finish(memoryErr)
					}
					rejected = append(rejected, r)
					if errors.Is(evalErr, ErrMemoryBudgetExceeded) || errors.Is(evalErr, ErrMemoryAccountingUnavailable) {
						return finish(evalErr)
					}
					if work.exhausted {
						return finish(nil)
					}
					continue
				}

				allEvaluated = append(allEvaluated, cand)

				nextFrontier = append(nextFrontier, beamItem{
					state:      res.State,
					cost:       cand.Cost,
					derivation: res.State.Derivation,
					candidate:  cand,
				})

				// Update best global independently of beam pruning
				if cand.Cost < bestGlobal.Cost || (cand.Cost == bestGlobal.Cost && compareDerivations(res.State.Derivation, bestPath) < 0) {
					bestGlobal = cand
					bestPath = res.State.Derivation
				}
			}
		}

		if len(nextFrontier) == 0 {
			// No more productive transitions found
			break
		}

		// Sort nextFrontier with stable tie-breaking:
		// Primary: Cost (ascending, lower is better)
		// Secondary: full replay-path lexicographical order
		sort.SliceStable(nextFrontier, func(i, j int) bool {
			if nextFrontier[i].cost != nextFrontier[j].cost {
				return nextFrontier[i].cost < nextFrontier[j].cost
			}
			return compareDerivations(nextFrontier[i].derivation, nextFrontier[j].derivation) < 0
		})

		// Retain up to BeamWidth items
		if len(nextFrontier) > b.cfg.BeamWidth {
			currentBeam = nextFrontier[:b.cfg.BeamWidth]
		} else {
			currentBeam = nextFrontier
		}
	}

	if err := ctx.Err(); err != nil {
		return finish(fmt.Errorf("%w: %w", ErrSearchCancelled, err))
	}
	if bestGlobal == nil {
		return nil, ErrNoAdmittedPlan
	}

	return finish(nil)
}

func admitSearchCandidate(c *CandidatePlan) error {
	if c == nil || !c.Validated || math.IsNaN(c.Cost) || math.IsInf(c.Cost, 0) {
		return fmt.Errorf("%w: evaluator returned no validated finite-cost candidate", ErrNoAdmittedPlan)
	}
	switch c.CostSource {
	case "", CostEstimated:
	case CostMeasured:
		if c.Receipt == nil || !c.Receipt.Complete || c.Receipt.Err != nil {
			return fmt.Errorf("%w: measured candidate has no complete receipt", ErrNoAdmittedPlan)
		}
		kind := c.Receipt.TimingKind
		if kind != cudatrial.TimingSummedKernel && kind != cudatrial.TimingGPUBusy {
			return ErrTimingKindMismatch
		}
		if duration, err := c.Receipt.DurationFor(kind); err != nil || duration <= 0 {
			return fmt.Errorf("%w: measured candidate has no positive device duration", ErrNoAdmittedPlan)
		}
	default:
		return fmt.Errorf("%w: unknown cost source %q", ErrNoAdmittedPlan, c.CostSource)
	}
	return nil
}

func admitComparableCandidate(c, baseline *CandidatePlan) error {
	if err := admitSearchCandidate(c); err != nil {
		return err
	}
	if c.CostSource != baseline.CostSource || c.ScoringEpoch != baseline.ScoringEpoch {
		return fmt.Errorf("%w: candidate cost source or scoring epoch differs", ErrNoAdmittedPlan)
	}
	if c.CostSource == CostMeasured && c.Receipt.TimingKind != baseline.Receipt.TimingKind {
		return ErrTimingKindMismatch
	}
	return nil
}

// stateFingerprint identifies a root mapping and exact replay path within one
// search's initial graph. Different derivations remain distinct even when they
// extract the same program; this is not a semantic graph-equivalence test.
func stateFingerprint(s *egraph.SearchState) string {
	if s == nil || s.Graph == nil {
		return "<nil>"
	}
	var sb strings.Builder
	writeString := func(value string) {
		fmt.Fprintf(&sb, "%d:", len(value))
		sb.WriteString(value)
	}
	keys := make([]string, 0, len(s.Roots))
	for key := range s.Roots {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	fmt.Fprintf(&sb, "%d;", len(keys))
	for _, key := range keys {
		writeString(key)
		fmt.Fprintf(&sb, "%d;", s.Graph.Find(s.Roots[key]))
	}
	fmt.Fprintf(&sb, "%d;", len(s.Derivation))
	for _, a := range s.Derivation {
		writeString(a.RuleName)
		writeString(a.RuleID)
		writeString(a.Source)
		writeString(a.RHSNode)
		fmt.Fprintf(&sb, "%d;%d;", a.ClassID, a.StepIndex)
		keys = keys[:0]
		for key := range a.Subst {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		fmt.Fprintf(&sb, "%d;", len(keys))
		for _, key := range keys {
			writeString(key)
			fmt.Fprintf(&sb, "%d;", a.Subst[key])
		}
	}
	return sb.String()
}

func estimateStateMemory(s *egraph.SearchState) int64 {
	bytes, err := storageBytes(s)
	if err != nil {
		return math.MaxInt64
	}
	return bytes
}
