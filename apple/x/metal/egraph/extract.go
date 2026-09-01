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
