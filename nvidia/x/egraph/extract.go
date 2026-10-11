package egraph

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"strings"

	"github.com/tmc/lib/egraphcore"
)

// Expr represents a concrete extracted AST tree from an E-Graph.
type Expr struct {
	// Class and Origin retain extraction attribution for emitted source.
	Class    Id
	Origin   string
	Op       string
	Data     any
	Children []*Expr
}

// String formats the expression as an S-expression. Nil children and cycles
// are written as <nil> and <cycle>.
func (e *Expr) String() string { return formatExpr(e, false) }

// Fingerprint returns a structural bucket key preserving Op, Data, and child
// structure. Keys may collide; use Equal to confirm structural equality.
func (e *Expr) Fingerprint() string { return formatExpr(e, true) }

type exprFormatEvent struct {
	expr  *Expr
	text  string
	leave bool
}

func formatExpr(root *Expr, fingerprint bool) string {
	var out strings.Builder
	active := make(map[*Expr]bool)
	events := []exprFormatEvent{{expr: root}}
	for len(events) != 0 {
		last := len(events) - 1
		event := events[last]
		events[last] = exprFormatEvent{}
		events = events[:last]
		if event.text != "" {
			out.WriteString(event.text)
			continue
		}
		if event.leave {
			delete(active, event.expr)
			continue
		}
		node := event.expr
		if node == nil {
			out.WriteString("<nil>")
			continue
		}
		if active[node] {
			out.WriteString("<cycle>")
			continue
		}
		active[node] = true
		if len(node.Children) == 0 {
			out.WriteString(node.Op)
			if node.Data != nil {
				if fingerprint {
					fmt.Fprintf(&out, "{%T:%v}", node.Data, node.Data)
				} else if fmt.Sprint(node.Data) != node.Op {
					fmt.Fprintf(&out, "{%v}", node.Data)
				}
			}
			delete(active, node)
			continue
		}
		out.WriteByte('(')
		out.WriteString(node.Op)
		if node.Data != nil {
			if fingerprint {
				fmt.Fprintf(&out, "{%T:%v}", node.Data, node.Data)
			} else if fmt.Sprint(node.Data) != node.Op {
				fmt.Fprintf(&out, "{%v}", node.Data)
			}
		}
		events = append(events, exprFormatEvent{expr: node, leave: true}, exprFormatEvent{text: ")"})
		for i := len(node.Children) - 1; i >= 0; i-- {
			events = append(events, exprFormatEvent{expr: node.Children[i]}, exprFormatEvent{text: " "})
		}
	}
	return out.String()
}

// Equal returns true if e and other have identical tree structure, operations, and data values.
func (e *Expr) Equal(other *Expr) bool {
	type pair struct{ left, right *Expr }
	stack := []pair{{left: e, right: other}}
	seen := make(map[pair]bool)
	for len(stack) != 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if current.left == current.right {
			continue
		}
		if current.left == nil || current.right == nil {
			return false
		}
		if seen[current] {
			continue
		}
		seen[current] = true
		if current.left.Op != current.right.Op || !reflect.DeepEqual(current.left.Data, current.right.Data) || len(current.left.Children) != len(current.right.Children) {
			return false
		}
		for i := range current.left.Children {
			stack = append(stack, pair{left: current.left.Children[i], right: current.right.Children[i]})
		}
	}
	return true
}

// CostModel calculates the cost of an ENode given the best costs of its child
// e-classes. It must be deterministic, must not mutate the node or its
// payload, and must return finite, non-negative costs that are monotone in
// every child cost and no smaller than any child cost. Use CandidateSearch to
// rank complete candidates by a caller-measured whole-graph metric.
type CostModel interface {
	egraphcore.CostModel[ENode]
}

// AdditiveCostModel assigns a local cost to one node. The selected expression
// cost is the sum of each node's local cost, counting repeated child uses.
type AdditiveCostModel interface {
	NodeLocalCost(ENode) float64
}

// AstSizeCost is a default cost model where every operator node has cost 1.0.
type AstSizeCost struct{}

func (AstSizeCost) NodeCost(_ ENode, childCosts []float64) float64 {
	return egraphcore.ASTSizeCost(childCosts)
}

// AstSizeAdditiveCost assigns local cost 1 to every node.
type AstSizeAdditiveCost struct{}

func (AstSizeAdditiveCost) NodeLocalCost(ENode) float64 { return 1 }

func evaluateCostModel(model CostModel, node ENode, childCosts []float64) float64 {
	return egraphcore.EvaluateNodeCost(model, node, childCosts, cloneCostNode)
}

func cloneCostNode(node ENode) ENode {
	node.Children = append([]Id(nil), node.Children...)
	return node
}

// Extractor computes a minimum-cost expression for each e-class according to
// a cost model. Equal-cost expressions prefer the smaller unfolded tree, with
// graph order breaking any remaining tie. Its cached tables are bound to the
// graph revision at creation and extraction methods return ErrStaleExtractor
// after graph mutation.
type Extractor struct {
	g                 *EGraph
	costModel         CostModel
	additiveCostModel AdditiveCostModel
	additive          bool
	costs             map[Id]float64
	bestNodes         map[Id]ENode
	topKNodes         map[Id][]topKNode
	revision          uint64
	err               error
}

type topKNode struct {
	node ENode
}

// ErrStaleExtractor reports use of extraction tables after graph mutation.
var ErrStaleExtractor = errors.New("egraph: extractor is stale after graph mutation")

// ErrTopKAdditiveExtractor reports that top-k extraction is unavailable for
// an extractor initialized with additive local costs.
var ErrTopKAdditiveExtractor = errors.New("egraph: top-k extraction is unavailable for additive extractors")

// NewExtractor initializes extraction tables and finds the minimum cost for
// each e-class. A nil graph is retained as a constructor error and reported by
// extraction methods.
func NewExtractor(g *EGraph, costModel CostModel) *Extractor {
	ext, err := NewExtractorChecked(g, costModel)
	if err != nil {
		return &Extractor{err: err}
	}
	return ext
}

// NewExtractorChecked rebuilds pending graph changes, then initializes
// extraction tables. It reports invalid graphs, rebuild failures, exhausted
// work budgets, and cost models that do not converge within the relaxation
// guard.
func NewExtractorChecked(g *EGraph, costModel CostModel) (*Extractor, error) {
	if g == nil {
		return nil, fmt.Errorf("egraph: nil graph")
	}
	if g.store == nil || !g.store.Initialized() {
		return nil, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("egraph: rebuild before extraction: %w", err)
	}
	if costModel == nil {
		costModel = AstSizeCost{}
	}
	ext := &Extractor{
		g:         g,
		costModel: costModel,
		costs:     make(map[Id]float64),
		bestNodes: make(map[Id]ENode),
		topKNodes: make(map[Id][]topKNode),
		revision:  g.revision,
	}
	if err := ext.computeCostsBounded(nil); err != nil {
		return nil, fmt.Errorf("egraph: initialize extractor: %w", err)
	}
	return ext, nil
}

// NewAdditiveExtractorChecked rebuilds pending graph changes, then computes a
// minimum-cost acyclic expression for every class reachable from any class in
// the graph. Classes without a finite acyclic derivation are omitted.
func NewAdditiveExtractorChecked(g *EGraph, costModel AdditiveCostModel) (*Extractor, error) {
	if g == nil {
		return nil, fmt.Errorf("egraph: nil graph")
	}
	if g.store == nil || !g.store.Initialized() {
		return nil, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("egraph: rebuild before additive extraction: %w", err)
	}
	if costModel == nil {
		costModel = AstSizeAdditiveCost{}
	}
	roots := make([]Id, 0, g.store.ClassCount())
	for _, class := range g.classes() {
		roots = append(roots, class.ID)
	}
	return newAdditiveExtractor(g, roots, costModel)
}

// NewAdditiveExtractorForRootChecked rebuilds pending graph changes, validates
// root, and computes minimum-cost acyclic expressions reachable from root.
// Other graph classes are not considered unless reachable from root.
func NewAdditiveExtractorForRootChecked(g *EGraph, root Id, costModel AdditiveCostModel) (*Extractor, error) {
	if g == nil {
		return nil, fmt.Errorf("egraph: nil graph")
	}
	if g.store == nil || !g.store.Initialized() {
		return nil, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("egraph: rebuild before additive extraction: %w", err)
	}
	if uint64(root) >= uint64(g.store.Len()) {
		return nil, fmt.Errorf("egraph: invalid extraction root %d", root)
	}
	canonicalRoot := g.Find(root)
	if g.class(canonicalRoot) == nil {
		return nil, fmt.Errorf("egraph: missing extraction root %d", canonicalRoot)
	}
	if costModel == nil {
		costModel = AstSizeAdditiveCost{}
	}
	return newAdditiveExtractor(g, []Id{canonicalRoot}, costModel)
}

func newAdditiveExtractor(g *EGraph, roots []Id, costModel AdditiveCostModel) (*Extractor, error) {
	ext := &Extractor{
		g:                 g,
		additiveCostModel: costModel,
		additive:          true,
		costs:             make(map[Id]float64),
		bestNodes:         make(map[Id]ENode),
		topKNodes:         make(map[Id][]topKNode),
		revision:          g.revision,
	}
	if len(roots) == 0 {
		return ext, nil
	}
	selected, _, err := egraphcore.ExtractBestAdditiveFromRoots(
		roots,
		func(id Id) []ENode {
			class := g.class(id)
			if class == nil {
				return nil
			}
			return class.Nodes
		},
		g.Find,
		func(node ENode) []Id { return node.Children },
		func(node ENode) float64 { return costModel.NodeLocalCost(cloneCostNode(node)) },
		func(node ENode) bool { return !node.Subsumed },
		int(^uint(0)>>1),
	)
	if err != nil {
		return nil, fmt.Errorf("egraph: initialize additive extractor: %w", err)
	}
	for classID, result := range selected {
		ext.costs[classID] = result.Cost
		ext.bestNodes[classID] = result.Node
	}
	return ext, nil
}

func (ext *Extractor) computeCostsBounded(budget *extractionBudget) error {
	if budget != nil {
		// Bound enumeration before allocating class state. Node setup below
		// charges each class's nodes and edges, so do not charge them twice.
		if ext.g.storage().ClassCount() > budget.limits.MaxWork-budget.work {
			budget.workExhausted = true
			return ErrExtractionLimit
		}
		if err := budget.take(0); err != nil {
			return err
		}
	}
	classes := ext.g.classes()
	classIDs := make([]Id, len(classes))
	for i, class := range classes {
		classIDs[i] = class.ID
	}
	maxVisits := int(^uint(0) >> 1)
	budgetLimitedVisits := false
	if budget != nil {
		remaining := budget.limits.MaxWork - budget.work
		maxVisits = remaining
		budgetLimitedVisits = true
		if maxVisits < 1 {
			maxVisits = 1
		}
	} else {
		maxIters := len(classes) + 2
		if maxIters < 10 {
			maxIters = 10
		}
		totalNodes := 0
		for _, class := range classes {
			if len(class.Nodes) > int(^uint(0)>>1)-totalNodes {
				totalNodes = int(^uint(0) >> 1)
			} else {
				totalNodes += len(class.Nodes)
			}
		}
		if totalNodes > 0 && maxIters <= maxVisits/totalNodes {
			maxVisits = maxIters * totalNodes
		}
	}
	best, _, err := egraphcore.RelaxBestCostsWorklist(
		classIDs,
		func(classID Id) []ENode {
			class := ext.g.class(classID)
			if class == nil {
				return nil
			}
			return class.Nodes
		},
		ext.g.Find,
		func(node ENode) []Id { return node.Children },
		ext.g.Find,
		func(node ENode) error { return budget.take(1 + len(node.Children)) },
		func(node ENode) error { return budget.take(1 + len(node.Children)) },
		func(node ENode, childStates []egraphcore.RankedBestCostNode[ENode, *big.Int], allChildrenKnown bool) (float64, *big.Int, bool, error) {
			if node.Subsumed || !allChildrenKnown {
				return 0, nil, false, nil
			}
			childCosts := make([]float64, len(childStates))
			nodeSize := big.NewInt(1)
			for i, child := range childStates {
				childCosts[i] = child.Cost
				nodeSize.Add(nodeSize, child.Rank)
			}
			nodeCost := evaluateCostModel(ext.costModel, node, childCosts)
			if err := budget.take(0); err != nil {
				return 0, nil, false, err
			}
			// An unavailable candidate must not prevent selection of a finite
			// alternative in the same equivalence class.
			if math.IsNaN(nodeCost) || math.IsInf(nodeCost, 0) {
				return 0, nil, false, nil
			}
			if err := egraphcore.ValidateMonotoneCost(nodeCost, childCosts); err != nil {
				return 0, nil, false, fmt.Errorf("egraph: cost for %s: %w", node.Op, err)
			}
			return nodeCost, nodeSize, true, nil
		},
		func(candidate, current egraphcore.RankedBestCostNode[ENode, *big.Int]) bool {
			return candidate.Cost < current.Cost || candidate.Cost == current.Cost && candidate.Rank.Cmp(current.Rank) < 0
		},
		func(classID Id, candidate egraphcore.RankedBestCostNode[ENode, *big.Int], lookup egraphcore.RankedBestCostLookup[Id, ENode, *big.Int]) (bool, error) {
			cycle, err := egraphcore.WouldCreateCycleWithVisit(
				classID,
				candidate.Node.Children,
				ext.g.Find,
				func(id Id) (ENode, bool) {
					state, ok := lookup(id)
					return state.Node, ok
				},
				func(node ENode) []Id { return node.Children },
				func(_ Id, work int) error { return budget.take(work) },
			)
			return !cycle, err
		},
		maxVisits,
	)
	if errors.Is(err, egraphcore.ErrCostRelaxationLimit) {
		if budgetLimitedVisits {
			budget.workExhausted = true
			return ErrExtractionLimit
		}
		return fmt.Errorf("egraph: cost relaxation did not converge within %d node visits: %w", maxVisits, err)
	}
	if err != nil {
		return err
	}
	for classID, result := range best {
		ext.costs[classID] = result.Cost
		ext.bestNodes[classID] = result.Node
	}
	return nil
}

// Extract returns the minimum-cost AST Expr and its total cost for the specified e-class root.
func (ext *Extractor) Extract(root Id) (*Expr, float64, error) {
	if err := ext.checkCurrent(); err != nil {
		return nil, 0, err
	}
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

func (ext *Extractor) buildExprChecked(id Id, visited map[Id]bool) (*Expr, error) {
	return ext.buildExprBounded(id, visited, nil, 1)
}

func (ext *Extractor) buildExprBounded(id Id, visited map[Id]bool, budget *extractionBudget, depth int) (*Expr, error) {
	type frame struct {
		id        Id
		node      ENode
		expr      *Expr
		nextChild int
		depth     int
	}
	newFrame := func(id Id, depth int) (frame, error) {
		if err := budget.node(depth); err != nil {
			return frame{}, err
		}
		id = ext.g.Find(id)
		if visited[id] {
			return frame{}, fmt.Errorf("cyclic extraction detected at e-class %d", id)
		}
		node, ok := ext.bestNodes[id]
		if !ok {
			return frame{}, fmt.Errorf("no extracted node for e-class %d", id)
		}
		if budget != nil && len(node.Children) > budget.limits.MaxNodes-budget.nodes {
			return frame{}, ErrExtractionLimit
		}
		visited[id] = true
		return frame{
			id:    id,
			node:  node,
			expr:  &Expr{Class: id, Origin: ext.g.nodeOrigin(node), Op: node.Op, Data: cloneData(node.Data), Children: make([]*Expr, len(node.Children))},
			depth: depth,
		}, nil
	}
	root, err := newFrame(id, depth)
	if err != nil {
		return nil, err
	}
	stack := []frame{root}
	defer func() {
		for _, pending := range stack {
			delete(visited, pending.id)
		}
	}()
	for len(stack) != 0 {
		last := len(stack) - 1
		current := &stack[last]
		if current.nextChild < len(current.node.Children) {
			childID := current.node.Children[current.nextChild]
			child, err := newFrame(childID, current.depth+1)
			if err != nil {
				return nil, err
			}
			stack = append(stack, child)
			continue
		}

		completed := current.expr
		delete(visited, current.id)
		stack[last] = frame{}
		stack = stack[:last]
		if len(stack) == 0 {
			return completed, nil
		}
		parent := &stack[len(stack)-1]
		parent.expr.Children[parent.nextChild] = completed
		parent.nextChild++
	}
	return nil, fmt.Errorf("egraph: failed to materialize e-class %d", id)
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

// ExtractTopK extracts up to k structurally unique candidate ASTs from the
// root e-class, including combinations of nested child alternatives. The cost
// model must be finite and monotone nondecreasing in each child cost for the
// best-first combination search to return the true top k. The extractor
// reports nonfinite costs and costs below a selected child cost; it cannot
// prove that a custom model is monotone for all possible child costs. A
// nonpositive k returns no candidates.
func (ext *Extractor) ExtractTopK(root Id, k int) ([]CandidateAST, error) {
	return ext.ExtractTopKWithBudget(root, k, nil)
}

// ExtractTopKWithBudget extracts up to k unique candidates using a shared
// egraphcore work budget. A nonpositive k returns no candidates. A nil budget
// is unlimited. Budget exhaustion returns no partial result.
func (ext *Extractor) ExtractTopKWithBudget(root Id, k int, budget *egraphcore.WorkBudget) ([]CandidateAST, error) {
	if ext != nil && ext.additive {
		return nil, ErrTopKAdditiveExtractor
	}
	if err := ext.checkCurrent(); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, nil
	}
	canonicalRoot := ext.g.Find(root)
	eclass := ext.g.storage().Class(canonicalRoot)
	if eclass == nil || len(eclass.Nodes) == 0 {
		return nil, fmt.Errorf("egraph: cannot extract e-class %d (not found)", root)
	}
	// The core enumerates child-choice combinations best-first. This is correct
	// when parent cost is monotone in every child cost. The callback checks that
	// contract for every candidate it builds, and rejects nonfinite costs.
	type topKValue struct {
		expr        *Expr
		fingerprint string
	}
	less := func(a, b egraphcore.TopKCandidate[topKValue]) bool {
		return a.Cost < b.Cost
	}
	// Building an Expr preserves its operator, payload, and every child; Expr
	// equality therefore preserves distinctions among child candidates.
	candidates, err := egraphcore.ExtractTopKWithChildPreservationAndHash[Id, topKNode, topKValue](
		canonicalRoot,
		k,
		func(classID Id) []topKNode {
			classID = ext.g.Find(classID)
			if nodes, ok := ext.topKNodes[classID]; ok {
				return nodes
			}
			class := ext.g.storage().Class(classID)
			if class == nil {
				return nil
			}
			nodes := make([]topKNode, 0, len(class.Nodes))
			for _, node := range class.Nodes {
				if node.Subsumed {
					continue
				}
				nodes = append(nodes, topKNode{node: node})
			}
			ext.topKNodes[classID] = nodes
			return nodes
		},
		ext.g.Find,
		func(node topKNode) []Id { return node.node.Children },
		func(classID Id, node topKNode, children []egraphcore.TopKCandidate[topKValue]) (egraphcore.TopKCandidate[topKValue], error) {
			childExprs := make([]*Expr, len(children))
			childCosts := make([]float64, len(children))
			childFingerprints := make([]string, len(children))
			for i, child := range children {
				childExprs[i] = child.Value.expr
				childCosts[i] = child.Cost
				childFingerprints[i] = child.Value.fingerprint
			}
			cost := evaluateCostModel(ext.costModel, node.node, childCosts)
			if err := egraphcore.ValidateMonotoneCost(cost, childCosts); err != nil {
				return egraphcore.TopKCandidate[topKValue]{}, fmt.Errorf("egraph: top-k cost for e-class %d node %q: %w", classID, node.node.Op, err)
			}
			fingerprint := egraphcore.StructuralFingerprint(node.node.Op, "", node.node.Data, childFingerprints)
			expr := &Expr{Class: ext.g.Find(classID), Origin: ext.g.nodeOrigin(node.node), Op: node.node.Op, Data: cloneData(node.node.Data), Children: childExprs}
			return egraphcore.TopKCandidate[topKValue]{Value: topKValue{
				expr:        expr,
				fingerprint: fingerprint,
			}, Cost: cost}, nil
		},
		nil,
		less,
		nil,
		func(a, b topKValue) bool { return reflect.DeepEqual(a.expr, b.expr) },
		func(value topKValue) string { return value.fingerprint },
		budget,
	)
	if err != nil {
		return nil, fmt.Errorf("egraph: extract top-k from e-class %d: %w", root, err)
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("egraph: no valid candidate expressions found for e-class %d", root)
	}
	result := make([]CandidateAST, len(candidates))
	for i, candidate := range candidates {
		result[i] = CandidateAST{Expr: candidate.Value.expr, Cost: candidate.Cost}
	}
	return result, nil
}

func (ext *Extractor) checkCurrent() error {
	if ext == nil {
		return fmt.Errorf("egraph: nil extractor")
	}
	if ext.err != nil {
		return ext.err
	}
	if ext.g == nil {
		return fmt.Errorf("egraph: nil extractor graph")
	}
	if ext.g.revision != ext.revision {
		return ErrStaleExtractor
	}
	return nil
}

// ExtractTopK is a package helper to extract the top-K candidate ASTs.
func ExtractTopK(g *EGraph, root Id, k int, costModel CostModel) ([]CandidateAST, error) {
	return ExtractTopKWithBudget(g, root, k, costModel, nil)
}

// ExtractTopKWithBudget is the package helper form of ExtractTopKWithBudget.
func ExtractTopKWithBudget(g *EGraph, root Id, k int, costModel CostModel, budget *egraphcore.WorkBudget) ([]CandidateAST, error) {
	ext := NewExtractor(g, costModel)
	return ext.ExtractTopKWithBudget(root, k, budget)
}
