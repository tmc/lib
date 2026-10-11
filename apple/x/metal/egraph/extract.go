package egraph

import (
	"fmt"
	"math"
	"math/big"
	"reflect"
	"sort"
	"strings"

	"github.com/tmc/lib/egraphcore"
)

// Expr represents an extracted expression graph. Repeated uses of an
// equivalence class may point to the same Expr value.
type Expr struct {
	Op       string
	Sort     string
	Data     any // Shared with the graph; payloads must remain immutable.
	Children []*Expr
}

// Equal reports whether two extracted expressions have deeply equal
// operators, payloads, and children. Pointer sharing does not affect equality.
func (e *Expr) Equal(other *Expr) bool {
	return reflect.DeepEqual(e, other)
}

// String formats an expression as an S-expression. Shared or cyclic nodes use
// references of the form #n=definition and #n; ordinary trees retain the
// unreferenced S-expression form.
func (e *Expr) String() string {
	if e == nil {
		return "<nil>"
	}
	counts := map[*Expr]int{e: 1}
	visited := map[*Expr]bool{e: true}
	stack := []*Expr{e}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for i := len(node.Children) - 1; i >= 0; i-- {
			child := node.Children[i]
			if child == nil {
				continue
			}
			counts[child]++
			if !visited[child] {
				visited[child] = true
				stack = append(stack, child)
			}
		}
	}

	labels := make(map[*Expr]int)
	visited = map[*Expr]bool{e: true}
	stack = []*Expr{e}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if counts[node] > 1 {
			labels[node] = len(labels)
		}
		for i := len(node.Children) - 1; i >= 0; i-- {
			child := node.Children[i]
			if child != nil && !visited[child] {
				visited[child] = true
				stack = append(stack, child)
			}
		}
	}

	var sb strings.Builder
	emitted := make(map[*Expr]bool, len(labels))
	type event struct {
		node   *Expr
		text   string
		isText bool
	}
	events := []event{{node: e}}
	for len(events) > 0 {
		next := events[len(events)-1]
		events[len(events)-1] = event{}
		events = events[:len(events)-1]
		if next.isText {
			sb.WriteString(next.text)
			continue
		}
		node := next.node
		if node == nil {
			sb.WriteString("<nil>")
			continue
		}
		if label, ok := labels[node]; ok {
			if emitted[node] {
				fmt.Fprintf(&sb, "#%d", label)
				continue
			}
			emitted[node] = true
			fmt.Fprintf(&sb, "#%d=", label)
		}
		if len(node.Children) == 0 {
			if node.Sort != "" {
				sb.WriteString(node.Sort)
				sb.WriteByte(':')
			}
			if node.Data != nil {
				fmt.Fprintf(&sb, "%s{%T:%#v}", node.Op, node.Data, node.Data)
			} else {
				sb.WriteString(node.Op)
			}
			continue
		}
		sb.WriteByte('(')
		if node.Sort != "" {
			sb.WriteString(node.Sort)
			sb.WriteByte(':')
		}
		sb.WriteString(node.Op)
		if node.Data != nil {
			fmt.Fprintf(&sb, "{%T:%#v}", node.Data, node.Data)
		}
		events = append(events, event{text: ")", isText: true})
		for i := len(node.Children) - 1; i >= 0; i-- {
			events = append(events, event{node: node.Children[i]})
			events = append(events, event{text: " ", isText: true})
		}
	}
	return sb.String()
}

// CostModel calculates the total cost of an ENode given the best costs of its
// child e-classes. Costs are recursive per child use, so a shared child is
// charged once for each parent edge. Use CandidateSearch to rank complete
// candidates by a caller-measured whole-graph metric. CostModel must be
// deterministic for the same node and child costs,
// and must not mutate the node payload. Costs must be finite, non-negative,
// non-decreasing in each child cost, and at least as large as every child cost.
// The extractor passes copies of the node's child IDs and costs.
type CostModel interface {
	egraphcore.CostModel[ENode]
}

// LocalCostModel assigns a local cost to a node. Additive extraction adds that
// cost to the cost of each child use. Local costs and resulting expression
// costs may be negative. The model must be deterministic and must not mutate
// node payloads. Additive extraction compares the float64 local costs as exact
// binary rationals and rejects reachable negative-cost cycles. It returns the
// rounded float64 total for each selected expression.
type LocalCostModel interface {
	NodeLocalCost(n ENode) float64
}

func evaluateCostModel(model CostModel, node ENode, childCosts []float64) float64 {
	return egraphcore.EvaluateNodeCost(model, node, childCosts, cloneCostNode)
}

func cloneCostNode(node ENode) ENode {
	node.Children = append([]Id(nil), node.Children...)
	return node
}

func validateNodeCost(node ENode, cost float64, childCosts []float64) error {
	if err := egraphcore.ValidateMonotoneCost(cost, childCosts); err != nil {
		return fmt.Errorf("egraph: cost for %s: %w", node.Op, err)
	}
	return nil
}

// AstSizeCost is a default cost model where every operator node has cost 1.0.
type AstSizeCost struct{}

func (AstSizeCost) NodeCost(_ ENode, childCosts []float64) float64 {
	return egraphcore.ASTSizeCost(childCosts)
}

// FusedMegakernelCost favors fused multi-operation kernels over discrete memory roundtrips.
type FusedMegakernelCost struct{}

func (FusedMegakernelCost) NodeCost(n ENode, childCosts []float64) float64 {
	baseCost := 10.0
	switch n.Op {
	case "fused_rmsnorm_gemv":
		baseCost = 1.0
	case "fused_rmsnorm_qkv":
		baseCost = 1.2
	case "fused_rope_store_kv":
		baseCost = 1.5
	case "fused_swiglu_gemv":
		baseCost = 2.0
	case "fused_swiglu_down":
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
// Equal-cost derivations prefer the expression with fewer nodes.
// It is a snapshot of the graph at construction time and becomes stale after
// any graph mutation.
type Extractor struct {
	g              *EGraph
	costModel      CostModel
	localCostModel LocalCostModel
	costs          map[Id]float64
	bestNodes      map[Id]ENode
	scope          map[Id]struct{}
	root           Id
	rootScoped     bool
	revision       uint64
	err            error
}

// NewExtractor initializes extraction tables and finds the minimum cost for each e-class.
func NewExtractor(g *EGraph, costModel CostModel) *Extractor {
	ext, err := NewExtractorChecked(g, costModel)
	if err != nil {
		return &Extractor{err: err}
	}
	return ext
}

// NewExtractorChecked initializes extraction tables and reports invalid graphs
// or costs.
func NewExtractorChecked(g *EGraph, costModel CostModel) (*Extractor, error) {
	return newExtractorChecked(g, costModel, nil, InvalidId, false)
}

// NewExtractorForRootChecked initializes an extractor for the e-classes that
// can contribute to rootId. It avoids evaluating unrelated e-classes in large
// graphs. It reports an invalid graph, root ID, or cost. Extract may be called
// only for rootId and its reachable classes.
func NewExtractorForRootChecked(g *EGraph, rootId Id, costModel CostModel) (*Extractor, error) {
	return newExtractorChecked(g, costModel, nil, rootId, true)
}

// NewAdditiveExtractorChecked creates a best-only extractor for all classes.
// LocalCostModel can assign a node a cost below its children.
func NewAdditiveExtractorChecked(g *EGraph, costModel LocalCostModel) (*Extractor, error) {
	return newAdditiveExtractorChecked(g, costModel, InvalidId, false)
}

// NewAdditiveExtractorForRootChecked creates a best-only extractor scoped to
// classes reachable from rootId. It supports negative local costs and reports
// a reachable negative-cost cycle. Cycle detection is exact for the float64
// local-cost values returned by the model.
func NewAdditiveExtractorForRootChecked(g *EGraph, rootId Id, costModel LocalCostModel) (*Extractor, error) {
	return newAdditiveExtractorChecked(g, costModel, rootId, true)
}

func newAdditiveExtractorChecked(g *EGraph, costModel LocalCostModel, rootId Id, scoped bool) (*Extractor, error) {
	if costModel == nil {
		return nil, fmt.Errorf("egraph: nil additive cost model")
	}
	return newExtractorChecked(g, nil, costModel, rootId, scoped)
}

func newExtractorChecked(g *EGraph, costModel CostModel, localCostModel LocalCostModel, rootId Id, scoped bool) (*Extractor, error) {
	if g == nil {
		return nil, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.opIndex == nil {
		return nil, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if costModel == nil && localCostModel == nil {
		costModel = AstSizeCost{}
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("egraph: rebuild before extraction: %w", err)
	}
	var scope map[Id]struct{}
	var root Id
	if scoped {
		var err error
		root, err = g.FindChecked(rootId)
		if err != nil {
			return nil, err
		}
		scope = reachableClasses(g, root)
	}
	ext := &Extractor{
		g:              g,
		costModel:      costModel,
		localCostModel: localCostModel,
		costs:          make(map[Id]float64),
		bestNodes:      make(map[Id]ENode),
		scope:          scope,
		root:           root,
		rootScoped:     scoped,
		revision:       g.revision,
	}
	if err := ext.computeCosts(); err != nil {
		return nil, err
	}
	return ext, nil
}

func (ext *Extractor) computeCosts() error {
	if ext.localCostModel != nil {
		return ext.computeAdditiveCosts()
	}
	var classes []*EClass
	if ext.scope == nil {
		classes = ext.g.classesList()
	} else {
		classes = make([]*EClass, 0, len(ext.scope))
		for id := range ext.scope {
			if class := ext.g.store.Class(id); class != nil {
				classes = append(classes, class)
			}
		}
		sort.Slice(classes, func(i, j int) bool { return classes[i].ID < classes[j].ID })
	}
	classIDs := make([]Id, 0, len(classes))
	for _, class := range classes {
		classIDs = append(classIDs, class.ID)
	}
	best, _, err := egraphcore.RelaxBestCostsWorklist(
		classIDs,
		func(id Id) []ENode {
			class := ext.g.store.Class(id)
			if class == nil {
				return nil
			}
			return class.Nodes
		},
		ext.g.Find,
		func(node ENode) []Id { return node.Children },
		ext.g.Find,
		nil,
		nil,
		func(node ENode, childStates []egraphcore.RankedBestCostNode[ENode, *big.Int], allKnown bool) (float64, *big.Int, bool, error) {
			if node.Subsumed || !allKnown {
				return 0, nil, false, nil
			}
			childCosts := make([]float64, len(childStates))
			nodeSize := big.NewInt(1)
			for i, child := range childStates {
				childCosts[i] = child.Cost
				nodeSize.Add(nodeSize, child.Rank)
			}
			nodeCost := evaluateCostModel(ext.costModel, node, childCosts)
			if err := validateNodeCost(node, nodeCost, childCosts); err != nil {
				return 0, nil, false, err
			}
			return nodeCost, nodeSize, true, nil
		},
		func(candidate, current egraphcore.RankedBestCostNode[ENode, *big.Int]) bool {
			return candidate.Cost < current.Cost || candidate.Cost == current.Cost && candidate.Rank.Cmp(current.Rank) < 0
		},
		func(classID Id, candidate egraphcore.RankedBestCostNode[ENode, *big.Int], lookup egraphcore.RankedBestCostLookup[Id, ENode, *big.Int]) (bool, error) {
			return !egraphcore.WouldCreateCycle(
				classID,
				candidate.Node.Children,
				ext.g.Find,
				func(id Id) (ENode, bool) {
					state, ok := lookup(id)
					return state.Node, ok
				},
				func(node ENode) []Id { return node.Children },
			), nil
		},
		math.MaxInt,
	)
	if err != nil {
		return err
	}
	for classID, result := range best {
		ext.costs[classID] = result.Cost
		ext.bestNodes[classID] = result.Node
	}
	return nil
}

func evaluateLocalCostModel(model LocalCostModel, node ENode) float64 {
	node.Children = append([]Id(nil), node.Children...)
	return model.NodeLocalCost(node)
}

func (ext *Extractor) computeAdditiveCosts() error {
	var roots []Id
	if ext.scope == nil {
		for _, class := range ext.g.classesList() {
			roots = append(roots, ext.g.Find(class.ID))
		}
	} else {
		roots = []Id{ext.root}
	}
	sort.Slice(roots, func(i, j int) bool { return roots[i] < roots[j] })
	if len(roots) == 0 {
		return nil
	}
	best, _, err := egraphcore.ExtractBestAdditiveFromRoots[Id, ENode](
		roots,
		func(id Id) []ENode {
			class := ext.g.store.Class(ext.g.Find(id))
			if class == nil {
				return nil
			}
			return class.Nodes
		},
		ext.g.Find,
		func(node ENode) []Id { return append([]Id(nil), node.Children...) },
		func(node ENode) float64 { return evaluateLocalCostModel(ext.localCostModel, node) },
		func(node ENode) bool { return !node.Subsumed },
		math.MaxInt,
	)
	if err != nil {
		return fmt.Errorf("egraph: additive extraction: %w", err)
	}
	if ext.rootScoped {
		if _, ok := best[ext.root]; !ok {
			return fmt.Errorf("egraph: additive extraction could not resolve a derivation for e-class %d", ext.root)
		}
	}
	for id, result := range best {
		ext.costs[id] = result.Cost
		ext.bestNodes[id] = cloneCostNode(result.Node)
	}
	return nil
}

func reachableClasses(g *EGraph, root Id) map[Id]struct{} {
	reachable := make(map[Id]struct{})
	stack := []Id{root}
	for len(stack) > 0 {
		id := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		id = g.Find(id)
		if id == InvalidId {
			continue
		}
		if _, ok := reachable[id]; ok {
			continue
		}
		class := g.store.Class(id)
		if class == nil {
			continue
		}
		reachable[id] = struct{}{}
		for _, node := range class.Nodes {
			if node.Subsumed {
				continue
			}
			stack = append(stack, node.Children...)
		}
	}
	return reachable
}

// Extract returns the lowest-cost expression graph for a root e-class ID.
// Shared e-classes share Expr pointers; the returned cost still counts each
// child use separately, as defined by CostModel.
func (ext *Extractor) Extract(rootId Id) (*Expr, float64, error) {
	if ext == nil {
		return nil, math.Inf(1), fmt.Errorf("egraph: nil extractor")
	}
	if ext.err != nil {
		return nil, math.Inf(1), ext.err
	}
	if ext.g.revision != ext.revision {
		return nil, math.Inf(1), fmt.Errorf("egraph: extractor is stale after graph mutation")
	}
	canonicalRoot := ext.g.Find(rootId)
	if ext.scope != nil {
		if _, ok := ext.scope[canonicalRoot]; !ok {
			return nil, math.Inf(1), fmt.Errorf("egraph: e-class %d is outside extractor root scope", rootId)
		}
	}
	cost, ok := ext.costs[canonicalRoot]
	if !ok {
		return nil, math.Inf(1), fmt.Errorf("egraph: cannot extract e-class %d (unreachable)", rootId)
	}

	expr, err := ext.buildExpr(canonicalRoot)
	if err != nil {
		return nil, math.Inf(1), err
	}
	return expr, cost, nil
}

func (ext *Extractor) buildExpr(root Id) (*Expr, error) {
	type frame struct {
		id         Id
		node       ENode
		nextChild  int
		childExprs []*Expr
	}

	newFrame := func(id Id) (frame, error) {
		id = ext.g.Find(id)
		node, ok := ext.bestNodes[id]
		if !ok {
			return frame{}, fmt.Errorf("egraph: no selected node for e-class %d", id)
		}
		return frame{
			id:         id,
			node:       node,
			childExprs: make([]*Expr, len(node.Children)),
		}, nil
	}

	rootFrame, err := newFrame(root)
	if err != nil {
		return nil, err
	}
	active := map[Id]bool{rootFrame.id: true}
	stack := []frame{rootFrame}
	memo := make(map[Id]*Expr)
	for len(stack) > 0 {
		current := &stack[len(stack)-1]
		if current.nextChild < len(current.node.Children) {
			childID := ext.g.Find(current.node.Children[current.nextChild])
			if childExpr, ok := memo[childID]; ok {
				current.childExprs[current.nextChild] = childExpr
				current.nextChild++
				continue
			}
			if active[childID] {
				return nil, fmt.Errorf("egraph: selected extraction contains a cycle at e-class %d", childID)
			}
			childFrame, err := newFrame(childID)
			if err != nil {
				return nil, err
			}
			active[childID] = true
			stack = append(stack, childFrame)
			continue
		}

		expr := &Expr{
			Op:       current.node.Op,
			Sort:     current.node.Sort,
			Data:     current.node.Data,
			Children: current.childExprs,
		}
		memo[current.id] = expr
		delete(active, current.id)
		stack = stack[:len(stack)-1]
		if len(stack) == 0 {
			return expr, nil
		}
		parent := &stack[len(stack)-1]
		parent.childExprs[parent.nextChild] = expr
		parent.nextChild++
	}
	return nil, fmt.Errorf("egraph: failed to materialize e-class %d", root)
}

// ExtractBest is a helper function that constructs an extractor and returns the optimal AST Expr.
func ExtractBest(g *EGraph, rootId Id, costModel CostModel) (*Expr, float64, error) {
	ext, err := NewExtractorForRootChecked(g, rootId, costModel)
	if err != nil {
		return nil, math.Inf(1), err
	}
	return ext.Extract(rootId)
}

// CandidateAST represents an extracted expression candidate and its calculated cost.
type CandidateAST struct {
	Expr *Expr
	Cost float64
}

// ExtractTopK extracts up to k top-ranked structurally unique, acyclic
// candidate ASTs from the root e-class. Exactness relies on CostModel's
// documented nondecreasing-child-cost contract; the extractor checks costs as
// candidates are built but cannot prove that a custom model is monotone for
// every possible child cost. A nonpositive k returns no candidates.
func (ext *Extractor) ExtractTopK(rootId Id, k int) ([]CandidateAST, error) {
	return ext.ExtractTopKWithBudget(rootId, k, nil)
}

// ExtractTopKWithBudget extracts up to k exact top-ranked unique candidates
// using a shared egraphcore work budget. Exactness relies on CostModel's
// documented nondecreasing-child-cost contract. The extractor checks each
// candidate's cost but cannot prove that a custom model is monotone for every
// possible child cost. A nonpositive k returns no candidates. A nil budget is
// unlimited. Budget exhaustion returns no partial result.
func (ext *Extractor) ExtractTopKWithBudget(rootId Id, k int, budget *egraphcore.WorkBudget) ([]CandidateAST, error) {
	if ext == nil {
		return nil, fmt.Errorf("egraph: nil extractor")
	}
	if ext.localCostModel != nil {
		return nil, fmt.Errorf("egraph: top-k extraction does not support additive cost models")
	}
	if ext.err != nil {
		return nil, ext.err
	}
	if ext.g.revision != ext.revision {
		return nil, fmt.Errorf("egraph: extractor is stale after graph mutation")
	}
	if k <= 0 {
		return nil, nil
	}
	canonicalRoot := ext.g.Find(rootId)
	if ext.scope != nil {
		if _, ok := ext.scope[canonicalRoot]; !ok {
			return nil, fmt.Errorf("egraph: e-class %d is outside extractor root scope", rootId)
		}
	}
	eclass := ext.g.store.Class(canonicalRoot)
	if eclass == nil || len(eclass.Nodes) == 0 {
		return nil, fmt.Errorf("egraph: cannot extract e-class %d (not found)", rootId)
	}

	type topKValue struct {
		candidate   CandidateAST
		fingerprint string
	}
	build := func(_ Id, node ENode, children []egraphcore.TopKCandidate[topKValue]) (egraphcore.TopKCandidate[topKValue], error) {
		childExprs := make([]*Expr, len(children))
		childCosts := make([]float64, len(children))
		childFingerprints := make([]string, len(children))
		for i, child := range children {
			childExprs[i] = child.Value.candidate.Expr
			childCosts[i] = child.Cost
			childFingerprints[i] = child.Value.fingerprint
		}
		expr := &Expr{Op: node.Op, Sort: node.Sort, Data: node.Data, Children: childExprs}
		cost := evaluateCostModel(ext.costModel, node, childCosts)
		if err := validateNodeCost(node, cost, childCosts); err != nil {
			return egraphcore.TopKCandidate[topKValue]{}, err
		}
		candidate := CandidateAST{Expr: expr, Cost: cost}
		fingerprint := egraphcore.StructuralFingerprint(node.Op, node.Sort, node.Data, childFingerprints)
		return egraphcore.TopKCandidate[topKValue]{Value: topKValue{
			candidate:   candidate,
			fingerprint: fingerprint,
		}, Cost: cost}, nil
	}
	nodes := func(id Id) []ENode {
		class := ext.g.store.Class(id)
		if class == nil {
			return nil
		}
		return class.Nodes
	}
	// Expr construction retains each selected child, and Expr.Equal compares
	// the complete tree, so this builder preserves child distinctions.
	coreCandidates, err := egraphcore.ExtractTopKWithChildPreservationAndHash(
		canonicalRoot,
		k,
		nodes,
		ext.g.Find,
		func(node ENode) []Id { return node.Children },
		build,
		func(_ Id, node ENode) bool { return !node.Subsumed },
		func(a, b egraphcore.TopKCandidate[topKValue]) bool { return a.Cost < b.Cost },
		nil,
		func(a, b topKValue) bool { return a.candidate.Expr.Equal(b.candidate.Expr) },
		func(value topKValue) string { return value.fingerprint },
		budget,
	)
	if err != nil {
		return nil, err
	}
	if len(coreCandidates) == 0 {
		return nil, fmt.Errorf("egraph: no valid candidate expressions found for e-class %d", rootId)
	}
	candidates := make([]CandidateAST, len(coreCandidates))
	for i, candidate := range coreCandidates {
		candidates[i] = candidate.Value.candidate
	}
	return candidates, nil
}

// ExtractTopK is a package helper to extract the top-K candidate ASTs.
func ExtractTopK(g *EGraph, rootId Id, k int, costModel CostModel) ([]CandidateAST, error) {
	return ExtractTopKWithBudget(g, rootId, k, costModel, nil)
}

// ExtractTopKWithBudget is the package helper form of ExtractTopKWithBudget.
func ExtractTopKWithBudget(g *EGraph, rootId Id, k int, costModel CostModel, budget *egraphcore.WorkBudget) ([]CandidateAST, error) {
	ext, err := NewExtractorForRootChecked(g, rootId, costModel)
	if err != nil {
		return nil, err
	}
	return ext.ExtractTopKWithBudget(rootId, k, budget)
}
