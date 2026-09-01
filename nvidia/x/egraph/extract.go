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
	// Fixed-point iteration to compute optimal bottom-up costs
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

// Extract returns the minimum-cost AST Expr and its total cost for the specified e-class root.
func (ext *Extractor) Extract(root Id) (*Expr, float64, error) {
	canonicalRoot := ext.g.Find(root)
	cost, ok := ext.costs[canonicalRoot]
	if !ok {
		return nil, 0, fmt.Errorf("egraph: unable to extract e-class %d (cyclic or ungrounded)", root)
	}

	expr := ext.buildExpr(canonicalRoot)
	return expr, cost, nil
}

func (ext *Extractor) buildExpr(id Id) *Expr {
	canonicalID := ext.g.Find(id)
	bestNode, ok := ext.bestNodes[canonicalID]
	if !ok {
		return &Expr{Op: fmt.Sprintf("<unextracted:%d>", canonicalID)}
	}

	children := make([]*Expr, len(bestNode.Children))
	for i, childID := range bestNode.Children {
		children[i] = ext.buildExpr(childID)
	}

	return &Expr{
		Op:       bestNode.Op,
		Data:     bestNode.Data,
		Children: children,
	}
}

// ExtractBest is a convenience function that initializes an Extractor and extracts the best expression.
func ExtractBest(g *EGraph, root Id, costModel CostModel) (*Expr, float64, error) {
	ext := NewExtractor(g, costModel)
	return ext.Extract(root)
}

// FusedMegakernelCost is a specialized cost model designed for CUDA megakernel fusion.
// It assigns low cost to monolithic fused blocks and high cost to separate memory roundtrips.
type FusedMegakernelCost struct{}

func (FusedMegakernelCost) NodeCost(n ENode, childCosts []float64) float64 {
	baseCost := 1.0
	switch {
	case strings.HasPrefix(n.Op, "fused_"):
		// Rewarded: Mega-kernel executing in registers/shared-mem
		baseCost = 0.2
	case n.Op == "gemv" || n.Op == "matmul":
		baseCost = 2.0
	case n.Op == "rmsnorm" || n.Op == "rope" || n.Op == "silu":
		baseCost = 1.5
	case n.Op == "store_kv" || n.Op == "load_kv":
		baseCost = 1.8
	}

	total := baseCost
	for _, c := range childCosts {
		total += c
	}
	return math.Round(total*100) / 100
}
