package cudasearch

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// MCTSConfig specifies parameters for Monte Carlo Tree Search.
type MCTSConfig struct {
	Iterations   int          // Number of MCTS simulation iterations
	ExplorationC float64      // UCT exploration constant (e.g. math.Sqrt(2))
	Seed         int64        // Seed for local deterministic pseudo-random generator
	RolloutDepth int          // Maximum simulation rollout depth
	Budget       SearchBudget // Resource limits
	RootName     string       // E-graph named root (default: "root")
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
	cfg MCTSConfig
	rng *rand.Rand
}

// NewMCTSSearch creates an MCTS search engine.
func NewMCTSSearch(cfg MCTSConfig) *MCTSSearchEngine {
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
	seed := cfg.Seed
	if seed == 0 {
		seed = 42
	}
	return &MCTSSearchEngine{
		cfg: cfg,
		rng: rand.New(rand.NewSource(seed)),
	}
}

// Search executes MCTS exploration starting from initial.
func (m *MCTSSearchEngine) Search(ctx context.Context, initial *egraph.SearchState, rules []*egraph.Rewrite, eval Evaluator) (*SearchResult, error) {
	if initial == nil || initial.Graph == nil {
		return nil, fmt.Errorf("cudasearch: nil initial search state")
	}
	if eval == nil {
		return nil, fmt.Errorf("cudasearch: nil evaluator")
	}

	startTime := time.Now()
	budget := m.cfg.Budget
	if budget.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, budget.Timeout)
		defer cancel()
	}

	rootID, ok := initial.CanonicalRoot(m.cfg.RootName)
	if !ok {
		for _, v := range initial.Roots {
			rootID = v
			ok = true
			break
		}
		if !ok {
			return nil, fmt.Errorf("cudasearch: no root e-class in initial state")
		}
	}

	// Evaluate initial state to establish baseline cost
	initCand, err := eval.Evaluate(ctx, initial, rootID)
	if err != nil {
		return nil, fmt.Errorf("cudasearch: failed evaluating initial state: %w", err)
	}

	baselineCost := initCand.Cost
	if baselineCost <= 0 {
		baselineCost = 1.0
	}

	initialActions := egraph.EnumerateActionsBounded(initial.Graph, rules, budget.MaxActions)

	rootNode := &mctsNode{
		state:          initial.Clone(),
		untriedActions: initialActions,
		bestCandidate:  initCand,
	}

	var bestGlobal *CandidatePlan = initCand
	var allEvaluated []*CandidatePlan
	allEvaluated = append(allEvaluated, initCand)

	stateCount := 1
	evalCount := 1
	var accountedMemory int64 = estimateStateMemory(initial)

	for iter := 0; iter < m.cfg.Iterations; iter++ {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrSearchCancelled, err)
		}
		if stateCount >= budget.MaxStates || evalCount >= budget.MaxEvaluations {
			break
		}

		// 1. Selection
		node := rootNode
		depth := 0
		for len(node.untriedActions) == 0 && len(node.children) > 0 && depth < budget.MaxDepth {
			node = m.selectChild(node)
			depth++
		}

		// 2. Expansion
		if len(node.untriedActions) > 0 && depth < budget.MaxDepth && stateCount < budget.MaxStates {
			// Pop an untried action deterministically (or seeded random)
			idx := m.rng.Intn(len(node.untriedActions))
			act := node.untriedActions[idx]
			// Remove from untried
			node.untriedActions = append(node.untriedActions[:idx], node.untriedActions[idx+1:]...)

			res, applyErr := node.state.ApplyAction(act)
			if applyErr == nil && !res.NoOp {
				stateCount++
				mem := estimateStateMemory(res.State)
				accountedMemory += mem
				if budget.MaxMemoryBytes > 0 && accountedMemory > budget.MaxMemoryBytes {
					return nil, ErrMemoryBudgetExceeded
				}

				nextActions := egraph.EnumerateActionsBounded(res.State.Graph, rules, budget.MaxActions)
				childNode := &mctsNode{
					state:          res.State,
					parent:         node,
					actionTaken:    &act,
					untriedActions: nextActions,
				}

				candRoot, found := res.State.CanonicalRoot(m.cfg.RootName)
				if found && evalCount < budget.MaxEvaluations {
					cand, evalErr := eval.Evaluate(ctx, res.State, candRoot)
					evalCount++
					if evalErr == nil {
						childNode.bestCandidate = cand
						allEvaluated = append(allEvaluated, cand)
						if cand.Cost < bestGlobal.Cost {
							bestGlobal = cand
						}
					}
				}

				node.children = append(node.children, childNode)
				node = childNode
			}
		}

		// 3. Rollout (Simulation)
		reward := m.rollout(ctx, node, rules, baselineCost)

		// 4. Backpropagation
		curr := node
		for curr != nil {
			curr.visits++
			curr.totalReward += reward
			curr = curr.parent
		}
	}

	if bestGlobal == nil {
		return nil, ErrNoAdmittedPlan
	}

	return &SearchResult{
		Strategy:       "mcts",
		Winner:         bestGlobal,
		AllCandidates:  allEvaluated,
		StatesVisited:  stateCount,
		Evaluations:    evalCount,
		AccountedBytes: accountedMemory,
		Duration:       time.Since(startTime),
	}, nil
}

// selectChild selects the child maximizing UCT formula.
func (m *MCTSSearchEngine) selectChild(parent *mctsNode) *mctsNode {
	var bestChild *mctsNode
	bestScore := -math.MaxFloat64

	logParentVisits := math.Log(float64(parent.visits + 1))

	for _, child := range parent.children {
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

	if bestChild == nil && len(parent.children) > 0 {
		return parent.children[0]
	}
	return bestChild
}

// rollout performs a bounded pseudo-random simulation from node.
func (m *MCTSSearchEngine) rollout(ctx context.Context, start *mctsNode, rules []*egraph.Rewrite, baselineCost float64) float64 {
	curr := start.state.Clone()
	depth := 0

	for depth < m.cfg.RolloutDepth {
		if err := ctx.Err(); err != nil {
			return 0.0 // Cancelled rollout receives zero reward
		}

		actions := egraph.EnumerateActionsBounded(curr.Graph, rules, 10)
		if len(actions) == 0 {
			break
		}

		// Random selection from legal actions
		act := actions[m.rng.Intn(len(actions))]
		res, err := curr.ApplyAction(act)
		if err != nil || res.NoOp {
			break
		}
		curr = res.State
		depth++
	}

	// Compute reward: bounded improvement over baseline
	// Higher reward for lower cost. Normalized in [0, 1].
	if start.bestCandidate != nil {
		cost := start.bestCandidate.Cost
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

	return 0.5 // Default neutral reward
}
