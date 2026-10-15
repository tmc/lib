package egraph

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/bits"
	"reflect"
	"strconv"

	"github.com/tmc/lib/egraphcore"
)

// MemoryCostModel explicitly accounts callback workspace during bounded
// extraction. NodeCostWithMemory reserves before creating owned storage. Its
// scope is borrowed and must not be closed or retained by the callback. Node
// data and child costs must not be mutated or retained. Captures and runtime
// internals are outside this capability. Existing CostModel alone is not a
// declaration of bounded callback storage.
type MemoryCostModel interface {
	CostModel
	NodeCostWithMemory(node ENode, childCosts []float64, memory *egraphcore.MemoryScope) (float64, error)
}

// Extraction owns an extracted tree's reservations. Expr and its mutable data
// are usable until Close, including through aliases. Scratch reservations have
// already been released on return. Immutable strings may share source storage.
// MemoryScope defines the accounting units and allocator overhead boundary.
// Close must not race with use or another Close.
type Extraction struct {
	Expr   *Expr
	Cost   float64
	Stats  ExtractionStats
	memory *egraphcore.MemoryScope
}

// Close releases the tree's reservations once and invalidates all aliases of
// the owned expression. It never closes or mutates the borrowed source graph.
func (x *Extraction) Close() {
	if x == nil {
		return
	}
	x.Expr = nil
	x.memory.Close()
}

// ExtractBoundedWithMemory reserves extraction collections, rank backing
// storage and supported result payload copies before allocation. The caller
// supplies a clean graph and closes the returned owner after its last use.
// AstSizeCost and FusedMegakernelCost are built in; other models must implement
// MemoryCostModel. Unsupported payload ownership or provenance formatting is
// refused with ErrMemoryOperation. A nil memory capability uses the existing
// ExtractBoundedWithStats behavior, including arbitrary legacy cost models.
// On failure a nonnil result contains usage statistics only: its expression is
// nil and all reservations have already been released.
func ExtractBoundedWithMemory(ctx context.Context, g *EGraph, root Id, model CostModel, limits ExtractionLimits, memory egraphcore.Memory) (owned *Extraction, err error) {
	if memory == nil {
		expr, cost, stats, err := ExtractBoundedWithStats(ctx, g, root, model, limits)
		return &Extraction{Expr: expr, Cost: cost, Stats: stats}, err
	}
	if ctx == nil || g == nil || g.store == nil || !g.store.Initialized() || limits.MaxWork <= 0 || limits.MaxNodes <= 0 || limits.MaxDepth <= 0 {
		return nil, egraphcore.ErrInvalidCostRelaxation
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(g.worklist) != 0 {
		return nil, ErrPendingGraphChanges
	}
	if model == nil {
		model = AstSizeCost{}
	}
	switch model.(type) {
	case AstSizeCost, FusedMegakernelCost, MemoryCostModel:
	default:
		return nil, ErrMemoryOperation
	}
	scratch, err := egraphcore.NewMemoryScope(memory)
	if err != nil {
		return nil, err
	}
	defer scratch.Close()
	work, err := egraphcore.NewWorkBudgetWithMemory(0, scratch)
	if err != nil {
		return nil, err
	}
	result, err := egraphcore.NewMemoryScope(memory)
	if err != nil {
		return nil, err
	}
	output, err := egraphcore.NewWorkBudgetWithMemory(0, result)
	if err != nil {
		result.Close()
		return nil, err
	}
	if err := graphSlots[Extraction](output, 1); err != nil {
		result.Close()
		return nil, err
	}
	owner := &Extraction{memory: result}
	complete := false
	defer func() {
		if !complete {
			owner.Close()
		}
	}()
	if err := graphSlots[extractionBudget](work, 1); err != nil {
		return nil, err
	}
	b := &extractionBudget{ctx: ctx, limits: limits}
	defer func() {
		owner.Stats = ExtractionStats{Work: b.work, Nodes: b.nodes, WorkExhausted: b.workExhausted}
		if !complete {
			owned = owner
		}
	}()
	allInput := true
	for _, origin := range g.origins {
		if err := b.take(0); err != nil {
			return nil, err
		}
		if origin != "input" {
			allInput = false
			break
		}
	}
	best, err := memoryBestNodes(g, model, b, work)
	if err != nil {
		return nil, err
	}
	id := g.Find(root)
	selected, ok := best[id]
	if !ok || math.IsNaN(selected.Cost) || math.IsInf(selected.Cost, 0) {
		return nil, fmt.Errorf("egraph: no finite expression for e-class %d", root)
	}
	expr, err := memoryBuildExpr(g, id, best, allInput, b, work, output)
	if err != nil {
		return nil, err
	}
	owner.Expr, owner.Cost = expr, selected.Cost
	owner.Stats = ExtractionStats{Work: b.work, Nodes: b.nodes, WorkExhausted: b.workExhausted}
	complete = true
	return owner, nil
}

func memoryBestNodes(g *EGraph, model CostModel, b *extractionBudget, work *egraphcore.WorkBudget) (map[Id]egraphcore.RankedBestCostNode[ENode, *big.Int], error) {
	if g.store.ClassCount() > b.limits.MaxWork-b.work {
		b.workExhausted = true
		return nil, ErrExtractionLimit
	}
	classes, err := matchingRoots(g, work)
	if err != nil {
		return nil, err
	}
	maxVisits := max(1, b.limits.MaxWork-b.work)
	best, _, err := egraphcore.RelaxBestCostsWorklistWithBudget(
		classes,
		func(id Id) []ENode {
			if class := g.class(id); class != nil {
				return class.Nodes
			}
			return nil
		},
		g.Find, func(node ENode) []Id { return node.Children }, g.Find,
		func(node ENode) error { return b.take(1 + len(node.Children)) },
		func(node ENode) error { return b.take(1 + len(node.Children)) },
		func(node ENode, children []egraphcore.RankedBestCostNode[ENode, *big.Int], known bool) (float64, *big.Int, bool, error) {
			if node.Subsumed || !known {
				return 0, nil, false, nil
			}
			costs, err := graphSlice[float64](work, len(children), len(children))
			if err != nil {
				return 0, nil, false, err
			}
			rank, err := memoryTreeRank(children, work)
			if err != nil {
				return 0, nil, false, err
			}
			for i, child := range children {
				costs[i] = child.Cost
			}
			// Preserve the existing callback's detached child arrays.
			costChildren, err := graphSlice[Id](work, len(node.Children), len(node.Children))
			if err != nil {
				return 0, nil, false, err
			}
			copy(costChildren, node.Children)
			node.Children = costChildren
			modelCosts, err := graphSlice[float64](work, len(costs), len(costs))
			if err != nil {
				return 0, nil, false, err
			}
			copy(modelCosts, costs)
			var cost float64
			if guarded, ok := model.(MemoryCostModel); ok {
				cost, err = guarded.NodeCostWithMemory(node, modelCosts, work.MemoryScope())
			} else {
				cost = model.NodeCost(node, modelCosts)
			}
			if err != nil {
				return 0, nil, false, err
			}
			if err := work.MemoryError(); err != nil {
				return 0, nil, false, err
			}
			if err := b.take(0); err != nil {
				return 0, nil, false, err
			}
			if math.IsNaN(cost) || math.IsInf(cost, 0) {
				return 0, nil, false, nil
			}
			if err := egraphcore.ValidateMonotoneCost(cost, costs); err != nil {
				return 0, nil, false, err
			}
			return cost, rank, true, nil
		},
		func(candidate, current egraphcore.RankedBestCostNode[ENode, *big.Int]) bool {
			return candidate.Cost < current.Cost || candidate.Cost == current.Cost && candidate.Rank.Cmp(current.Rank) < 0
		},
		func(id Id, candidate egraphcore.RankedBestCostNode[ENode, *big.Int], lookup egraphcore.RankedBestCostLookup[Id, ENode, *big.Int]) (bool, error) {
			cycle, err := egraphcore.WouldCreateCycleWithBudget(id, candidate.Node.Children, g.Find,
				func(id Id) (ENode, bool) { state, ok := lookup(id); return state.Node, ok },
				func(node ENode) []Id { return node.Children }, func(_ Id, units int) error { return b.take(units) }, work)
			return !cycle, err
		}, maxVisits, work)
	if memoryErr := work.MemoryError(); memoryErr != nil {
		return nil, memoryErr
	}
	if errors.Is(err, egraphcore.ErrCostRelaxationLimit) {
		b.workExhausted = true
		return nil, ErrExtractionLimit
	}
	return best, err
}

// The complete positive sum needs at most the largest operand's word count
// plus ceil(log2(children+1)/wordBits) words. One extra word covers big.Int's
// Add carry slot. Preallocated SetBits storage is reused by every Add.
func memoryTreeRank(children []egraphcore.RankedBestCostNode[ENode, *big.Int], work *egraphcore.WorkBudget) (*big.Int, error) {
	words := 1
	for _, child := range children {
		words = max(words, len(child.Rank.Bits()))
	}
	extra := (bits.Len(uint(len(children)))+bits.UintSize-1)/bits.UintSize + 1
	if words > math.MaxInt-extra {
		return nil, egraphcore.ErrMemorySize
	}
	storage, err := graphSlice[big.Word](work, 1, words+extra)
	if err != nil {
		return nil, err
	}
	if err := graphSlots[big.Int](work, 1); err != nil {
		return nil, err
	}
	storage[0] = 1
	rank := new(big.Int).SetBits(storage)
	for _, child := range children {
		rank.Add(rank, child.Rank)
	}
	return rank, nil
}

func memoryBuildExpr(g *EGraph, root Id, best map[Id]egraphcore.RankedBestCostNode[ENode, *big.Int], allInput bool, b *extractionBudget, work, output *egraphcore.WorkBudget) (*Expr, error) {
	type frame struct {
		id          Id
		node        ENode
		expr        *Expr
		next, depth int
	}
	visited, err := graphMap[Id, bool](work, 0)
	if err != nil {
		return nil, err
	}
	newFrame := func(id Id, depth int) (frame, error) {
		if err := b.node(depth); err != nil {
			return frame{}, err
		}
		id = g.Find(id)
		if visited[id] {
			return frame{}, ErrMemoryOperation
		}
		state, ok := best[id]
		if !ok {
			return frame{}, ErrMemoryOperation
		}
		node := state.Node
		if len(node.Children) > b.limits.MaxNodes-b.nodes {
			return frame{}, ErrExtractionLimit
		}
		if err := graphPut(work, visited, id, true); err != nil {
			return frame{}, err
		}
		origin := "input"
		if !allInput {
			origin, err = memoryNodeOrigin(g, node, b, work)
			if err != nil {
				return frame{}, err
			}
		}
		data, err := cloneDataWithMemory(node.Data, b, work, output)
		if err != nil {
			return frame{}, err
		}
		children, err := graphSlice[*Expr](output, len(node.Children), len(node.Children))
		if err != nil {
			return frame{}, err
		}
		if err := graphSlots[Expr](output, 1); err != nil {
			return frame{}, err
		}
		expr := &Expr{Class: id, Op: node.Op, Origin: origin, Data: data, Children: children}
		return frame{id: id, node: node, expr: expr, depth: depth}, nil
	}
	first, err := newFrame(root, 1)
	if err != nil {
		return nil, err
	}
	stack, err := graphSlice[frame](work, 1, 1)
	if err != nil {
		return nil, err
	}
	stack[0] = first
	for len(stack) > 0 {
		last := len(stack) - 1
		current := &stack[last]
		if current.next < len(current.node.Children) {
			child, err := newFrame(current.node.Children[current.next], current.depth+1)
			if err != nil {
				return nil, err
			}
			stack, err = graphAppend(work, stack, child)
			if err != nil {
				return nil, err
			}
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
		parent.expr.Children[parent.next] = completed
		parent.next++
	}
	return nil, ErrMemoryOperation
}

func memoryNodeOrigin(g *EGraph, node ENode, b *extractionBudget, work *egraphcore.WorkBudget) (string, error) {
	// If every possible key with this operator/type prefix is either input
	// or absent, the existing lookup and its fallback both return input.
	// This needs no formatting of mutable payloads or custom callbacks.
	if node.Data != nil && !primitivePayload(reflect.ValueOf(node.Data)) {
		typ := reflect.TypeOf(node.Data).String()
		onlyInput := true
		for key, origin := range g.origins {
			if err := b.take(0); err != nil {
				return "", err
			}
			start := len(node.Op)
			if len(key) <= start+len(typ)+2 || key[:start] != node.Op || key[start] != '{' {
				continue
			}
			if key[start+1:start+1+len(typ)] == typ && key[start+1+len(typ)] == ':' && origin != "input" {
				onlyInput = false
				break
			}
		}
		if onlyInput {
			return "input", nil
		}
		return "", ErrMemoryOperation
	}
	key, err := primitiveNodeKey(node, work)
	if err != nil {
		return "", err
	}
	if origin, ok := g.origins[key]; ok {
		return origin, nil
	}
	children, err := graphSlice[Id](work, len(node.Children), len(node.Children))
	if err != nil {
		return "", err
	}
	for i, child := range node.Children {
		children[i] = g.Find(child)
	}
	node.Children = children
	key, err = primitiveNodeKey(node, work)
	if err != nil {
		return "", err
	}
	if origin, ok := g.origins[key]; ok {
		return origin, nil
	}
	return "input", nil
}

func primitiveNodeKey(node ENode, work *egraphcore.WorkBudget) (string, error) {
	bound := int64(len(node.Op)) + 256
	var value reflect.Value
	if node.Data != nil {
		value = reflect.ValueOf(node.Data)
		if !primitivePayload(value) {
			return "", ErrMemoryOperation
		}
		if int64(len(value.Type().String())) > math.MaxInt64-bound {
			return "", egraphcore.ErrMemorySize
		}
		bound += int64(len(value.Type().String()))
		if value.Kind() == reflect.String {
			if int64(value.Len()) > (math.MaxInt64-bound)/6 {
				return "", egraphcore.ErrMemorySize
			}
			bound += 6 * int64(value.Len())
		}
	}
	if int64(len(node.Children)) > (math.MaxInt64-bound)/11 {
		return "", egraphcore.ErrMemorySize
	}
	bound += 11 * int64(len(node.Children))
	if bound > int64(math.MaxInt) {
		return "", egraphcore.ErrMemorySize
	}
	data, err := graphSlice[byte](work, 0, int(bound))
	if err != nil {
		return "", err
	}
	data = append(data, node.Op...)
	if value.IsValid() {
		data = append(data, '{')
		data = append(data, value.Type().String()...)
		data = append(data, ':')
		switch value.Kind() {
		case reflect.String:
			data = strconv.AppendQuote(data, value.String())
		case reflect.Bool:
			data = strconv.AppendBool(data, value.Bool())
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			data = strconv.AppendInt(data, value.Int(), 10)
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			data = append(data, "0x"...)
			data = strconv.AppendUint(data, value.Uint(), 16)
		case reflect.Float32, reflect.Float64:
			data = strconv.AppendFloat(data, value.Float(), 'g', -1, value.Type().Bits())
		default:
			return "", ErrMemoryOperation
		}
		data = append(data, '}')
	}
	if len(node.Children) > 0 {
		data = append(data, '(')
		for i, child := range node.Children {
			if i > 0 {
				data = append(data, ',')
			}
			data = strconv.AppendUint(data, uint64(child), 10)
		}
		data = append(data, ')')
	}
	if err := graphSlots[byte](work, len(data)); err != nil {
		return "", err
	}
	return string(data), nil
}
