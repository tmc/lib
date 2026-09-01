package egraph

import (
	"fmt"
	"math"
	"strings"
)

// Expr represents a concrete extracted AST tree from an E-Graph.
type Expr struct {
	Op       string
	Data     any
	Children []*Expr
}

func (e *Expr) String() string {
	if len(e.Children) == 0 {
		return e.Op
	}
	var sb strings.Builder
	sb.WriteString("(")
	sb.WriteString(e.Op)
	for _, c := range e.Children {
		sb.WriteString(" ")
		sb.WriteString(c.String())
	}
	sb.WriteString(")")
	return sb.String()
}

// CostModel calculates the cost of an ENode given the best costs of its child e-classes.
type CostModel interface {
	NodeCost(n ENode, childCosts []float64) float64
}

// AstSizeCost is a default cost model where every operator node has cost 1.0.
type AstSizeCost struct{}

func (AstSizeCost) NodeCost(n ENode, childCosts []float64) float64 {
	cost := 1.0
	for _, c := range childCosts {
		cost += c
	}
	return cost
}

// FusedMegakernelCost favors fused multi-operation kernels over discrete memory roundtrips.
type FusedMegakernelCost struct{}

func (FusedMegakernelCost) NodeCost(n ENode, childCosts []float64) float64 {
	baseCost := 10.0
	switch n.Op {
	case "fused_rmsnorm_gemv":
		baseCost = 1.0
	case "fused_rope_store_kv":
		baseCost = 1.5
	case "fused_swiglu_gemv":
		baseCost = 2.0
	case "fused_batched_qkv":
		baseCost = 1.8
	case "rmsnorm":
		baseCost = 8.0
	case "gemv", "matmul":
		baseCost = 12.0
	case "silu", "mul", "add":
		baseCost = 4.0
	case "rope", "store_kv":
		baseCost = 6.0
	default:
		if len(n.Children) == 0 {
			baseCost = 0.1 // Leaf constants/variables
		}
	}

	for _, c := range childCosts {
		baseCost += c
	}
	return baseCost
}

// Extractor computes the optimal expression for any e-class in the E-Graph according to a cost model.
type Extractor struct {
	g         *EGraph
	costModel CostModel
	costs     map[Id]float64
	bestNodes map[Id]ENode
}

// NewExtractor initializes extraction tables and finds the minimum cost for each e-class.
func NewExtractor(g *EGraph, costModel CostModel) *Extractor {
	if costModel == nil {
		costModel = AstSizeCost{}
	}
	ext := &Extractor{
		g:         g,
		costModel: costModel,
		costs:     make(map[Id]float64),
		bestNodes: make(map[Id]ENode),
	}
	ext.computeCosts()
	return ext
}

func (ext *Extractor) computeCosts() {
	changed := true
	for changed {
		changed = false
		for _, eclass := range ext.g.Classes() {
			canonicalID := ext.g.Find(eclass.ID)

			for _, node := range eclass.Nodes {
				allChildrenKnown := true
				childCosts := make([]float64, len(node.Children))
				for i, c := range node.Children {
					childID := ext.g.Find(c)
					cost, ok := ext.costs[childID]
					if !ok {
						allChildrenKnown = false
						break
					}
					childCosts[i] = cost
				}

				if !allChildrenKnown {
					continue
				}

				nodeCost := ext.costModel.NodeCost(node, childCosts)
				currentBest, exists := ext.costs[canonicalID]
				if !exists || nodeCost < currentBest {
					ext.costs[canonicalID] = nodeCost
					ext.bestNodes[canonicalID] = node
					changed = true
				}
			}
		}
	}
}

// Extract returns the optimal AST Expr and its calculated cost for a root e-class ID.
func (ext *Extractor) Extract(rootId Id) (*Expr, float64, error) {
	canonicalRoot := ext.g.Find(rootId)
	cost, ok := ext.costs[canonicalRoot]
	if !ok {
		return nil, math.Inf(1), fmt.Errorf("egraph: cannot extract e-class %d (unreachable)", rootId)
	}

	expr := ext.buildExpr(canonicalRoot)
	return expr, cost, nil
}

func (ext *Extractor) buildExpr(id Id) *Expr {
	canonicalID := ext.g.Find(id)
	bestNode, ok := ext.bestNodes[canonicalID]
	if !ok {
		return &Expr{Op: fmt.Sprintf("unresolved_%d", canonicalID)}
	}

	childExprs := make([]*Expr, len(bestNode.Children))
	for i, c := range bestNode.Children {
		childExprs[i] = ext.buildExpr(c)
	}

	return &Expr{
		Op:       bestNode.Op,
		Data:     bestNode.Data,
		Children: childExprs,
	}
}

// ExtractBest is a helper function that constructs an extractor and returns the optimal AST Expr.
func ExtractBest(g *EGraph, rootId Id, costModel CostModel) (*Expr, float64, error) {
	ext := NewExtractor(g, costModel)
	return ext.Extract(rootId)
}

// CandidateAST represents an extracted AST candidate along with its calculated heuristic cost.
type CandidateAST struct {
	Expr *Expr
	Cost float64
}

// ExtractTopK extracts up to k structurally unique candidate ASTs from the root e-class,
// ordered by cost.
func (ext *Extractor) ExtractTopK(rootId Id, k int) ([]CandidateAST, error) {
	if k <= 0 {
		k = 1
	}
	canonicalRoot := ext.g.Find(rootId)
	eclass, ok := ext.g.classes[canonicalRoot]
	if !ok || len(eclass.Nodes) == 0 {
		return nil, fmt.Errorf("egraph: cannot extract e-class %d (not found)", rootId)
	}

	type nodeCandidate struct {
		expr *Expr
		cost float64
	}
	var candidates []nodeCandidate
	seen := make(map[string]bool)

	for _, node := range eclass.Nodes {
		childCosts := make([]float64, len(node.Children))
		childExprs := make([]*Expr, len(node.Children))
		valid := true
		for i, c := range node.Children {
			childID := ext.g.Find(c)
			cost, ok := ext.costs[childID]
			if !ok {
				valid = false
				break
			}
			childCosts[i] = cost
			childExprs[i] = ext.buildExpr(childID)
		}
		if !valid {
			continue
		}
		expr := &Expr{
			Op:       node.Op,
			Data:     node.Data,
			Children: childExprs,
		}
		exprStr := expr.String()
		if seen[exprStr] {
			continue
		}
		seen[exprStr] = true

		cost := ext.costModel.NodeCost(node, childCosts)
		candidates = append(candidates, nodeCandidate{expr: expr, cost: cost})
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("egraph: no valid candidate expressions found for e-class %d", rootId)
	}

	// Sort candidates by cost ascending
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].cost < candidates[i].cost {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	if len(candidates) > k {
		candidates = candidates[:k]
	}

	result := make([]CandidateAST, len(candidates))
	for i, c := range candidates {
		result[i] = CandidateAST{
			Expr: c.expr,
			Cost: c.cost,
		}
	}
	return result, nil
}

// ExtractTopK is a package helper to extract the top-K candidate ASTs.
func ExtractTopK(g *EGraph, rootId Id, k int, costModel CostModel) ([]CandidateAST, error) {
	ext := NewExtractor(g, costModel)
	return ext.ExtractTopK(rootId, k)
}
