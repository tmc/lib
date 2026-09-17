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
		if e.Data != nil && fmt.Sprint(e.Data) != e.Op {
			return fmt.Sprintf("%s{%v}", e.Op, e.Data)
		}
		return e.Op
	}
	var sb strings.Builder
	sb.WriteString("(")
	sb.WriteString(e.Op)
	if e.Data != nil && fmt.Sprint(e.Data) != e.Op {
		sb.WriteString(fmt.Sprintf("{%v}", e.Data))
	}
	for _, c := range e.Children {
		sb.WriteString(" ")
		sb.WriteString(c.String())
	}
	sb.WriteString(")")
	return sb.String()
}

// Fingerprint returns an unambiguous structural representation preserving Op, Data, and child structure.
func (e *Expr) Fingerprint() string {
	if e == nil {
		return "<nil>"
	}
	if len(e.Children) == 0 {
		if e.Data != nil {
			return fmt.Sprintf("%s{%T:%v}", e.Op, e.Data, e.Data)
		}
		return e.Op
	}
	var sb strings.Builder
	sb.WriteString("(")
	sb.WriteString(e.Op)
	if e.Data != nil {
		sb.WriteString(fmt.Sprintf("{%T:%v}", e.Data, e.Data))
	}
	for _, c := range e.Children {
		sb.WriteString(" ")
		sb.WriteString(c.Fingerprint())
	}
	sb.WriteString(")")
	return sb.String()
}

// Equal returns true if e and other have identical tree structure, operations, and data values.
func (e *Expr) Equal(other *Expr) bool {
	if e == other {
		return true
	}
	if e == nil || other == nil {
		return false
	}
	if e.Op != other.Op || e.Data != other.Data || len(e.Children) != len(other.Children) {
		return false
	}
	for i := range e.Children {
		if !e.Children[i].Equal(other.Children[i]) {
			return false
		}
	}
	return true
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
	// Fixed-point iteration to compute optimal bottom-up costs.
	// Bounded iteration limit: at most len(classes) + 2 passes to prevent infinite loop on cyclic dependencies.
	classes := ext.g.Classes()
	maxIters := len(classes) + 2
	if maxIters < 10 {
		maxIters = 10
	}

	changed := true
	for iter := 0; iter < maxIters && changed; iter++ {
		changed = false
		for _, eclass := range classes {
			canonicalID := ext.g.Find(eclass.ID)

			for _, node := range eclass.Nodes {
				allChildrenKnown := true
				childCosts := make([]float64, len(node.Children))
				for i, c := range node.Children {
					childID := ext.g.Find(c)
					cost, ok := ext.costs[childID]
					if !ok || math.IsNaN(cost) || math.IsInf(cost, 0) {
						allChildrenKnown = false
						break
					}
					childCosts[i] = cost
				}

				if !allChildrenKnown {
					continue
				}

				nodeCost := ext.costModel.NodeCost(node, childCosts)
				// Reject non-finite costs (NaN or +/-Inf)
				if math.IsNaN(nodeCost) || math.IsInf(nodeCost, 0) {
					continue
				}

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
	if !ok || math.IsNaN(cost) || math.IsInf(cost, 0) {
		return nil, 0, fmt.Errorf("egraph: unable to extract e-class %d (cyclic, ungrounded, or nonfinite cost)", root)
	}

	visited := make(map[Id]bool)
	expr, err := ext.buildExprChecked(canonicalRoot, visited)
	if err != nil {
		return nil, 0, fmt.Errorf("egraph: extract e-class %d: %w", root, err)
	}
	return expr, cost, nil
}

func (ext *Extractor) buildExpr(id Id) *Expr {
	expr, _ := ext.buildExprChecked(id, make(map[Id]bool))
	if expr == nil {
		return &Expr{Op: fmt.Sprintf("<unextracted:%d>", ext.g.Find(id))}
	}
	return expr
}

func (ext *Extractor) buildExprChecked(id Id, visited map[Id]bool) (*Expr, error) {
	canonicalID := ext.g.Find(id)
	if visited[canonicalID] {
		return nil, fmt.Errorf("cyclic extraction detected at e-class %d", canonicalID)
	}
	bestNode, ok := ext.bestNodes[canonicalID]
	if !ok {
		return nil, fmt.Errorf("no extracted node for e-class %d", canonicalID)
	}

	visited[canonicalID] = true
	defer delete(visited, canonicalID)

	children := make([]*Expr, len(bestNode.Children))
	for i, childID := range bestNode.Children {
		child, err := ext.buildExprChecked(childID, visited)
		if err != nil {
			return nil, err
		}
		children[i] = child
	}

	return &Expr{
		Op:       bestNode.Op,
		Data:     bestNode.Data,
		Children: children,
	}, nil
}

// ExtractBest is a convenience function that initializes an Extractor and extracts the best expression.
func ExtractBest(g *EGraph, root Id, costModel CostModel) (*Expr, float64, error) {
	ext := NewExtractor(g, costModel)
	return ext.Extract(root)
}

// TerminalOps are operators with direct CUDA C++ code generation support in GenerateCUDA.
var TerminalOps = []string{
	"fused_swiglu_gemv",
	"tile_swiglu",
	"fused_swiglu_q8_0",
	"fused_rmsnorm_gemv",
	"tile_rmsnorm_qkv",
	"tile_rmsnorm_proj",
	"fused_gemv_pipeline",
	"q8_0_matvec",
	"u2_simd",
	"u2_scalar",
	"q2_k_matvec",
	"iq2_xxs_matvec",
	"fused_rope_store_kv",
	"add",
	"pdl_split",
	"pdl_gemv_pipeline",
	"pdl_pipeline",
}

// PlaceholderOps are high-level mathematical primitives and intermediate memory
// staging nodes recognized by equality saturation rewrite rules and priced by
// cost models. They represent un-fused operations that must be rewritten into
// terminal operators prior to CUDA codegen.
var PlaceholderOps = []string{
	"matmul",
	"gemv",
	"rmsnorm",
	"rope",
	"silu",
	"gelu",
	"swish",
	"store_kv",
	"load_kv",
	"cublaslt_gemm",
	"tile_mma",
	"tile_mma_aligned",
	"tile_load",
	"tile_store",
	"load_strided",
	"partition_view",
	"mmaf_scaled",
}

// IsTerminalOp reports whether op is a terminal operator directly lowerable by GenerateCUDA.
func IsTerminalOp(op string) bool {
	if strings.HasPrefix(op, "pdl_") {
		return true
	}
	for _, top := range TerminalOps {
		if op == top {
			return true
		}
	}
	return false
}

// IsPlaceholderOp reports whether op is an intermediate or placeholder operator.
func IsPlaceholderOp(op string) bool {
	for _, pop := range PlaceholderOps {
		if op == pop {
			return true
		}
	}
	return false
}

// FusedMegakernelCost is a specialized cost model designed for CUDA megakernel fusion.
// It assigns low cost to monolithic fused blocks and high cost to separate memory roundtrips.
type FusedMegakernelCost struct{}

func (FusedMegakernelCost) NodeCost(n ENode, childCosts []float64) float64 {
	baseCost := 1.0
	switch {
	case n.Op == "tile_load" || n.Op == "tile_store" || n.Op == "load_strided" || n.Op == "partition_view" || n.Op == "cublaslt_gemm":
		// Penalized: Unaligned memory access patterns or unlowerable host library operators
		baseCost = 4.0
	case n.Op == "tile_mma_aligned" || n.Op == "mmaf_scaled" ||
		n.Op == "tile_tma_load" || n.Op == "tile_tma_store" ||
		n.Op == "tma_box" || n.Op == "tma_stride_1" || n.Op == "tma_partition_view" ||
		n.Op == "assume_aligned" || n.Op == "coalesce_contiguous":
		// Rewarded: Hardware TMA / Tensor Core hardware-aligned acceleration
		baseCost = 0.1
	case strings.HasPrefix(n.Op, "fused_") || (strings.HasPrefix(n.Op, "tile_") && n.Op != "tile_mma"):
		// Rewarded only if lowerable: Mega-kernel or declarative Tile IR executing in registers/shared-mem.
		if IsLowerableOp(n.Op) {
			baseCost = 0.2
		} else {
			// Ghost unlowerable operator: heavily penalized so extractor avoids it.
			baseCost = 10.0
		}
	case n.Op == "tile_mma":
		// Rewarded over GEMV/Matmul, but tile_mma_aligned is preferred for hardware alignment.
		if IsLowerableOp(n.Op) {
			baseCost = 0.3
		} else {
			baseCost = 10.0
		}
	case strings.HasPrefix(n.Op, "ptx_"):
		// Rewarded: In-register bit-unswizzling
		baseCost = 0.4
	case strings.HasPrefix(n.Op, "pdl_"):
		if hasPDLPreambleWork(n, childCosts) && IsLowerableOp(n.Op) {
			// Rewarded: Programmatic Dependent Launch (PDL) with independent preamble
			// work (weight fetches, bias loads, RoPE indexing, dequantization) executed
			// before cudaGridDependencySynchronize(), achieving ~20 µs (~29%) kernel span
			// overlap (alpha ~ 0.71).
			baseCost = 0.6
		} else {
			// Unearned or unlowerable: Entry-syncing kernels without independent preamble
			// work or unlowerable ghost PDL ops yield no overlap.
			baseCost = 2.0
		}
	case n.Op == "gemv" || n.Op == "matmul":
		baseCost = 2.0
	case n.Op == "rmsnorm" || n.Op == "rope" || n.Op == "silu" || n.Op == "gelu" || n.Op == "swish":
		baseCost = 1.5
	case n.Op == "store_kv" || n.Op == "load_kv":
		baseCost = 1.8
	default:
		if !IsLowerableOp(n.Op) && len(childCosts) > 0 {
			baseCost = 5.0
		}
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
// exploring combinations across both root-level alternatives and nested child alternatives, ordered by cost.
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

	// Memoized table for top child alternatives per e-class (limit branching per child e-class to min(k, 4))
	childLimit := k
	if childLimit > 4 {
		childLimit = 4
	}
	childCandMemo := make(map[Id][]nodeCandidate)

	var getChildCands func(cid Id, depth int) []nodeCandidate
	getChildCands = func(cid Id, depth int) []nodeCandidate {
		cCanon := ext.g.Find(cid)
		if cached, ok := childCandMemo[cCanon]; ok {
			return cached
		}
		cClass, ok := ext.g.classes[cCanon]
		if !ok || len(cClass.Nodes) == 0 || depth > 4 {
			// Fallback to best single expression
			if _, hasCost := ext.costs[cCanon]; hasCost {
				expr := ext.buildExpr(cCanon)
				return []nodeCandidate{{expr: expr, cost: ext.costs[cCanon]}}
			}
			return nil
		}

		var cands []nodeCandidate
		cSeen := make(map[string]bool)
		for _, n := range cClass.Nodes {
			valid := true
			childCosts := make([]float64, len(n.Children))
			childExprs := make([]*Expr, len(n.Children))
			for i, ch := range n.Children {
				chID := ext.g.Find(ch)
				cCost, ok := ext.costs[chID]
				if !ok {
					valid = false
					break
				}
				childCosts[i] = cCost
				childExprs[i] = ext.buildExpr(chID)
			}
			if !valid {
				continue
			}
			expr := &Expr{Op: n.Op, Data: n.Data, Children: childExprs}
			fp := expr.Fingerprint()
			if cSeen[fp] {
				continue
			}
			cSeen[fp] = true
			cost := ext.costModel.NodeCost(n, childCosts)
			cands = append(cands, nodeCandidate{expr: expr, cost: cost})
		}
		// Sort ascending
		for i := 0; i < len(cands); i++ {
			for j := i + 1; j < len(cands); j++ {
				if cands[j].cost < cands[i].cost {
					cands[i], cands[j] = cands[j], cands[i]
				}
			}
		}
		if len(cands) > childLimit {
			cands = cands[:childLimit]
		}
		childCandMemo[cCanon] = cands
		return cands
	}

	for _, node := range eclass.Nodes {
		if len(node.Children) == 0 {
			cost := ext.costModel.NodeCost(node, nil)
			expr := &Expr{Op: node.Op, Data: node.Data}
			fp := expr.Fingerprint()
			if !seen[fp] {
				seen[fp] = true
				candidates = append(candidates, nodeCandidate{expr: expr, cost: cost})
			}
			continue
		}

		// Enumerate child candidate combinations for this node
		childOptions := make([][]nodeCandidate, len(node.Children))
		hasValidOptions := true
		for i, c := range node.Children {
			opts := getChildCands(c, 0)
			if len(opts) == 0 {
				hasValidOptions = false
				break
			}
			childOptions[i] = opts
		}
		if !hasValidOptions {
			continue
		}

		// Cartesian product over child options (bounded)
		var generateCombos func(idx int, currentExprs []*Expr, currentCosts []float64)
		generateCombos = func(idx int, currentExprs []*Expr, currentCosts []float64) {
			if idx == len(childOptions) {
				expr := &Expr{
					Op:       node.Op,
					Data:     node.Data,
					Children: append([]*Expr(nil), currentExprs...),
				}
				fp := expr.Fingerprint()
				if !seen[fp] {
					seen[fp] = true
					cost := ext.costModel.NodeCost(node, currentCosts)
					candidates = append(candidates, nodeCandidate{expr: expr, cost: cost})
				}
				return
			}
			for _, opt := range childOptions[idx] {
				generateCombos(idx+1, append(currentExprs, opt.expr), append(currentCosts, opt.cost))
				if len(candidates) >= k*4 {
					return
				}
			}
		}

		generateCombos(0, nil, nil)
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
