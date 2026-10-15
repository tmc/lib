package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// MCTSConfig specifies parameters for Monte Carlo Tree Search.
type MCTSConfig struct {
	EvaluatorID  string // Caller-qualified evaluator/callback policy identity, if needed.
	Profile      *ProfileActions
	Memory       *SearchMemory // Optional frozen catalog and cross-task exploration policy.
	Iterations   int           // Number of MCTS simulation iterations
	ExplorationC float64       // UCT exploration constant (e.g. math.Sqrt(2))
	Seed         int64         // Seed for each Search invocation; zero selects 42.
	RolloutDepth int           // Maximum simulation rollout depth
	Budget       SearchBudget  // Resource limits
	RootName     string        // E-graph named root (default: "root")
}

// DefaultMCTSConfig returns standard MCTS configuration.
func DefaultMCTSConfig() MCTSConfig {
	return MCTSConfig{
		Iterations:   50,
		ExplorationC: 1.41421356, // sqrt(2)
		Seed:         42,
		RolloutDepth: 4,
		Budget:       DefaultBudget(),
		RootName:     "root",
	}
}

// mctsNode represents a tree node in MCTS exploration.
type mctsNode struct {
	state          *egraph.SearchState
	parent         *mctsNode
	children       []*mctsNode
	actionTaken    *egraph.ActionCandidate
	untriedActions []egraph.ActionCandidate
	visits         int
	totalReward    float64
	bestCandidate  *CandidatePlan
	isTerminal     bool
}

// MCTSSearchEngine implements seeded, bounded Monte Carlo Tree Search over rewrite transitions.
type MCTSSearchEngine struct {
	cfg       MCTSConfig
	configErr error
}

// NewMCTSSearch creates an MCTS search engine.
func NewMCTSSearch(cfg MCTSConfig) *MCTSSearchEngine {
	if err := validateSearchBudget(cfg.Budget); err != nil {
		return &MCTSSearchEngine{configErr: err}
	}
	if cfg.Iterations < 0 || cfg.RolloutDepth < 0 || cfg.ExplorationC < 0 || math.IsNaN(cfg.ExplorationC) || math.IsInf(cfg.ExplorationC, 0) {
		return &MCTSSearchEngine{configErr: fmt.Errorf("cudasearch: invalid MCTS configuration")}
	}
	if cfg.Iterations <= 0 {
		cfg.Iterations = 50
	}
	if cfg.ExplorationC <= 0 {
		cfg.ExplorationC = 1.41421356
	}
	if cfg.RolloutDepth <= 0 {
		cfg.RolloutDepth = 4
	}
	if cfg.RootName == "" {
		cfg.RootName = "root"
	}
	defaults := DefaultBudget()
	if cfg.Budget.MaxTotalActions <= 0 {
		cfg.Budget.MaxTotalActions = defaults.MaxTotalActions
	}
	if cfg.Budget.MaxStates <= 0 {
		cfg.Budget.MaxStates = defaults.MaxStates
	}
	if cfg.Budget.MaxDepth <= 0 {
		cfg.Budget.MaxDepth = defaults.MaxDepth
	}
	if cfg.Budget.MaxActions <= 0 {
		cfg.Budget.MaxActions = defaults.MaxActions
	}
	if cfg.Budget.MaxEvaluations <= 0 {
		cfg.Budget.MaxEvaluations = defaults.MaxEvaluations
	}
	if cfg.Seed == 0 {
		cfg.Seed = 42
	}
	return &MCTSSearchEngine{
		cfg: cfg,
	}
}

// Search executes MCTS exploration starting from initial.
func (m *MCTSSearchEngine) Search(ctx context.Context, initial *egraph.SearchState, rules []*egraph.Rewrite, eval Evaluator) (*SearchResult, error) {
	if m.configErr != nil {
		return nil, m.configErr
	}
	if initial == nil || initial.Graph == nil {
		return nil, fmt.Errorf("cudasearch: nil initial search state")
	}
	if eval == nil {
		return nil, fmt.Errorf("cudasearch: nil evaluator")
	}

	startTime := time.Now()
	rng := rand.New(rand.NewSource(m.cfg.Seed))
	budget := m.cfg.Budget
	profile := freezeProfileActions(m.cfg.Profile)
	if err := validateMemoryEvaluator(m.cfg.Memory, eval); err != nil {
		return nil, err
	}
	if err := freezeSearchMemory(m.cfg.Memory, &profile); err != nil {
		return nil, err
	}
	ctx, work, err := searchWorkBudget(ctx, budget, eval)
	if err != nil {
		return nil, err
	}
	policy, policyComplete, err := makeSearchPolicy("mcts", m.cfg.RootName, budget, map[string]string{"iterations": fmt.Sprint(m.cfg.Iterations), "rollout_depth": fmt.Sprint(m.cfg.RolloutDepth), "exploration": fmt.Sprint(m.cfg.ExplorationC)}, m.cfg.Seed, rules, eval, work, profile, m.cfg.EvaluatorID)
	if err != nil {
		return nil, err
	}
	if budget.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, budget.Timeout)
		defer cancel()
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrSearchCancelled, err)
	}
	rootID, ok := initial.CanonicalRoot(m.cfg.RootName)
	if !ok {
		return nil, fmt.Errorf("cudasearch: missing search root %q", m.cfg.RootName)
	}
	if err := admitGraphBudget(initial, budget); err != nil {
		return nil, err
	}
	if err := retainSearchValue(ctx, initial); err != nil {
		return nil, err
	}

	// Evaluate initial state to establish baseline cost
	initCand, err := evaluateSearch(ctx, eval, initial, rootID)
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
		return &SearchResult{PolicyID: policy, PolicyComplete: policyComplete, Strategy: "mcts", Seed: m.cfg.Seed,
			MemoryDecision: profile.memory, ProfileDecision: profile.decision, ProfileGuidance: profile.guidance, ProfileFallback: profile.fallback,
			Rejected: initialRejections, StopReason: failure,
			StatesVisited: 1, Evaluations: 1, Extractions: work.extractions, ExtractionWork: work.extractionWork,
			Compilations: work.compilations, WorkAccounted: work.known, AccountedBytes: work.memory.accountedBytes(),
			Duration: time.Since(startTime)}, failure
	}

	baselineCost := initCand.Cost
	if baselineCost <= 0 {
		baselineCost = 1.0
	}

	var initialActions []egraph.ActionCandidate

	rootNode := &mctsNode{
		state:          initial,
		untriedActions: initialActions,
		bestCandidate:  initCand,
		isTerminal:     len(initialActions) == 0,
	}

	var bestGlobal *CandidatePlan = initCand
	var allEvaluated []*CandidatePlan
	allEvaluated = append(allEvaluated, initCand)

	stateCount := 1
	evalCount := 1
	actionCount := 0
	var rejected []CandidateRejection
	var stopReason error
	finish := func(err error) (*SearchResult, error) {
		if work.exhausted {
			stopReason = ErrBudgetExhausted
		}
		if err != nil {
			stopReason = err
		}
		return &SearchResult{MemoryDecision: profile.memory, PolicyID: policy, PolicyComplete: policyComplete, Seed: m.cfg.Seed, ProfileDecision: profile.decision, ProfileGuidance: profile.guidance, ProfileFallback: profile.fallback, Strategy: "mcts", Winner: bestGlobal, AllCandidates: allEvaluated, Rejected: rejected, StopReason: stopReason, StatesVisited: stateCount, ActionsAttempted: actionCount, Evaluations: evalCount, Extractions: work.extractions, ExtractionWork: work.extractionWork, Compilations: work.compilations, WorkAccounted: work.known, AccountedBytes: work.memory.accountedBytes(), Duration: time.Since(startTime)}, err
	}

	cloned, err := cloneSearchState(ctx, initial)
	if err != nil {
		return finish(err)
	}
	rootNode.state = cloned
	initialOwned, err := enumerateSearchActions(ctx, rootNode.state, rules, budget.MaxActions)
	if err != nil {
		return finish(err)
	}
	defer initialOwned.Close()
	initialActions = initialOwned.Actions
	rootNode.untriedActions = initialActions
	rootNode.isTerminal = len(initialActions) == 0

	for iter := 0; iter < m.cfg.Iterations; iter++ {
		if err := ctx.Err(); err != nil {
			return finish(fmt.Errorf("%w: %w", ErrSearchCancelled, err))
		}
		if stateCount >= budget.MaxStates || evalCount >= budget.MaxEvaluations || actionCount >= budget.MaxTotalActions {
			stopReason = ErrBudgetExhausted
			break
		}
		if rootNode.isTerminal {
			break
		}

		// 1. Selection
		node := rootNode
		depth := 0
		for len(node.untriedActions) == 0 && len(node.children) > 0 && depth < budget.MaxDepth {
			child := m.selectChild(node)
			if child == nil {
				node.isTerminal = true
				break
			}
			node = child
			depth++
		}

		// 2. Expansion
		if len(node.untriedActions) > 0 && depth < budget.MaxDepth && stateCount < budget.MaxStates {
			// Pop an untried action deterministically (or seeded random)
			idx := profile.choose(node.untriedActions, rng)
			act := node.untriedActions[idx]
			// Remove from untried
			node.untriedActions = append(node.untriedActions[:idx], node.untriedActions[idx+1:]...)

			actionCount++
			stateCount++
			res, applyErr := applySearchAction(ctx, node.state, act)
			if errors.Is(applyErr, ErrMemoryBudgetExceeded) || errors.Is(applyErr, ErrMemoryAccountingUnavailable) {
				return finish(applyErr)
			}
			if applyErr == nil && !res.NoOp {
				if err := admitGraphBudget(res.State, budget); err != nil {
					stopReason = err
					return finish(nil)
				}

				nextOwned, err := enumerateSearchActions(ctx, res.State, rules, budget.MaxActions)
				if err != nil {
					return finish(err)
				}
				defer nextOwned.Close()
				nextActions := nextOwned.Actions
				childNode := &mctsNode{
					state:          res.State,
					parent:         node,
					actionTaken:    &act,
					untriedActions: nextActions,
					isTerminal:     len(nextActions) == 0 || depth+1 >= budget.MaxDepth,
				}

				candRoot, found := res.State.CanonicalRoot(m.cfg.RootName)
				if found && evalCount < budget.MaxEvaluations {
					cand, evalErr := evaluateSearch(ctx, eval, res.State, candRoot)
					evalCount++
					if evalErr == nil {
						evalErr = admitComparableCandidate(cand, initCand)
					}
					if evalErr == nil {
						childNode.bestCandidate = cand
						allEvaluated = append(allEvaluated, cand)
						if cand.Cost < bestGlobal.Cost {
							bestGlobal = cand
						}
					} else {
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
					}
				}

				node.children = append(node.children, childNode)
				node = childNode
				depth++
			}
		}

		// 3. Evaluate rollout states using the same counters as tree expansion.
		// ApplyAction produces a branch; no unaccounted initial clone is needed.
		currState := node.state
		rollCandidate := node.bestCandidate
		for step := 0; step < m.cfg.RolloutDepth && depth < budget.MaxDepth && !node.isTerminal; step++ {
			if err := ctx.Err(); err != nil {
				return finish(fmt.Errorf("%w: %w", ErrSearchCancelled, err))
			}
			if stateCount >= budget.MaxStates || evalCount >= budget.MaxEvaluations {
				stopReason = ErrBudgetExhausted
				break
			}
			ownedActions, err := enumerateSearchActions(ctx, currState, rules, budget.MaxActions)
			if err != nil {
				return finish(err)
			}
			defer ownedActions.Close()
			actions := ownedActions.Actions
			var next *egraph.SearchState
			for len(actions) > 0 {
				if err := ctx.Err(); err != nil {
					return finish(fmt.Errorf("%w: %w", ErrSearchCancelled, err))
				}
				if actionCount >= budget.MaxTotalActions || stateCount >= budget.MaxStates {
					stopReason = ErrBudgetExhausted
					return finish(nil)
				}
				idx := profile.choose(actions, rng)
				act := actions[idx]
				actions = append(actions[:idx], actions[idx+1:]...)
				actionCount++
				stateCount++
				res, err := applySearchAction(ctx, currState, act)
				if errors.Is(err, ErrMemoryBudgetExceeded) || errors.Is(err, ErrMemoryAccountingUnavailable) {
					return finish(err)
				}
				if err == nil && !res.NoOp {
					next = res.State
					break
				}
			}
			if next == nil {
				if currState == node.state && len(node.children) == 0 {
					node.isTerminal = true
				}
				break
			}
			depth++
			if err := admitGraphBudget(next, budget); err != nil {
				stopReason = err
				return finish(nil)
			}

			currState = next
			id, ok := currState.CanonicalRoot(m.cfg.RootName)
			if !ok {
				rollCandidate = nil
				break
			}
			cand, err := evaluateSearch(ctx, eval, currState, id)
			evalCount++
			if err == nil {
				err = admitComparableCandidate(cand, initCand)
			}
			if err != nil {
				r, memoryErr := retainSearchRejection(ctx, currState, eval, err)
				if memoryErr != nil {
					return finish(memoryErr)
				}
				rejected = append(rejected, r)
				if work.exhausted {
					return finish(nil)
				}
				rollCandidate = nil
			} else {
				rollCandidate = cand
				allEvaluated = append(allEvaluated, cand)
				if cand.Cost < bestGlobal.Cost {
					bestGlobal = cand
				}
			}
		}
		reward := mctsReward(rollCandidate, baselineCost)

		// 4. Backpropagation
		curr := node
		for curr != nil {
			curr.visits++
			curr.totalReward += reward
			curr = curr.parent
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

// selectChild selects the child maximizing UCT formula.
func (m *MCTSSearchEngine) selectChild(parent *mctsNode) *mctsNode {
	var bestChild *mctsNode
	bestScore := -math.MaxFloat64

	logParentVisits := math.Log(float64(parent.visits + 1))

	for _, child := range parent.children {
		if child.isTerminal {
			continue
		}
		if child.visits == 0 {
			// Prioritize exploring unvisited children
			return child
		}

		// Exploitation term: average normalized reward [0, 1]
		exploit := child.totalReward / float64(child.visits)

		// Exploration term: C * sqrt(ln(N_parent) / N_child)
		explore := m.cfg.ExplorationC * math.Sqrt(logParentVisits/float64(child.visits))

		score := exploit + explore
		if score > bestScore {
			bestScore = score
			bestChild = child
		}
	}

	return bestChild
}

func mctsReward(candidate *CandidatePlan, baselineCost float64) float64 {
	// Compute reward: bounded improvement over baseline
	// Higher reward for lower cost. Normalized in [0, 1].
	if candidate != nil {
		cost := candidate.Cost
		if cost <= 0 {
			return 1.0
		}
		// Relative speedup/improvement: baselineCost / cost
		// Clamped to [0, 1] range: reward = 1.0 - (cost / (baselineCost * 2))
		ratio := cost / baselineCost
		reward := 1.0 - (ratio / 2.0)
		if reward < 0.0 {
			reward = 0.0
		}
		if reward > 1.0 {
			reward = 1.0
		}
		return reward
	}

	return 0 // A failed evaluation is not neutral evidence of a useful branch.
}
