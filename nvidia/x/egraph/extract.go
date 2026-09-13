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
	case n.Op == "tile_load" || n.Op == "tile_store" || n.Op == "load_strided" || n.Op == "partition_view":
		// Penalized: Unaligned / unoptimized memory access patterns
		baseCost = 4.0
	case strings.HasPrefix(n.Op, "tma_") || strings.HasPrefix(n.Op, "assume_") || strings.HasPrefix(n.Op, "coalesce_") || n.Op == "mmaf_scaled" || n.Op == "tile_mma_aligned":
		// Rewarded: Hardware TMA / Tensor Core hardware-aligned acceleration
		baseCost = 0.1
	case strings.HasPrefix(n.Op, "fused_") || (strings.HasPrefix(n.Op, "tile_") && n.Op != "tile_mma"):
		// Rewarded: Mega-kernel or declarative Tile IR executing in registers/shared-mem
		baseCost = 0.2
	case n.Op == "tile_mma":
		// Rewarded over GEMV/Matmul, but tile_mma_aligned is preferred for hardware alignment
		baseCost = 0.3
	case strings.HasPrefix(n.Op, "ptx_"):
		// Rewarded: In-register bit-unswizzling
		baseCost = 0.4
	case strings.HasPrefix(n.Op, "pdl_") || strings.HasPrefix(n.Op, "pdl_pipeline"):
		// Rewarded: Programmatic Dependent Launch (PDL) overlapping sequential stages
		// with hardware overlap factor (alpha ~ 0.85) eliminating inter-kernel dispatch bubbles.
		baseCost = 0.6
	case n.Op == "cublaslt_gemm":
		// Rewarded: High-throughput Tensor Core GEMM for batched matrix multiply
		baseCost = 0.5
	case n.Op == "gemv" || n.Op == "matmul":
		baseCost = 2.0
	case n.Op == "rmsnorm" || n.Op == "rope" || n.Op == "silu" || n.Op == "gelu" || n.Op == "swish":
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

// CandidateAST represents an extracted AST candidate along with its calculated heuristic cost.
type CandidateAST struct {
	Expr *Expr
	Cost float64
}

// ExtractTopK extracts up to k structurally unique candidate ASTs from the root e-class,
// ordered by cost.
func (ext *Extractor) ExtractTopK(root Id, k int) ([]CandidateAST, error) {
	if k <= 0 {
		k = 1
	}
	canonicalRoot := ext.g.Find(root)
	eclass, ok := ext.g.classes[canonicalRoot]
	if !ok || len(eclass.Nodes) == 0 {
		return nil, fmt.Errorf("egraph: cannot extract e-class %d (not found)", root)
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
		return nil, fmt.Errorf("egraph: no valid candidate expressions found for e-class %d", root)
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
func ExtractTopK(g *EGraph, root Id, k int, costModel CostModel) ([]CandidateAST, error) {
	ext := NewExtractor(g, costModel)
	return ext.ExtractTopK(root, k)
}
