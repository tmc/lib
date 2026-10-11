package egraphcore

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// TopKCandidate is one expression produced while enumerating an e-class.
// Value is owned by the backend; the core only combines candidates and reads
// Cost when constructing candidates through the supplied callback.
type TopKCandidate[Value any] struct {
	Value Value
	Cost  float64
}

// TopKBuild combines a node with one candidate for each child, or returns a
// backend validation error.
type TopKBuild[ID ~uint32, Node, Value any] func(
	class ID,
	node Node,
	children []TopKCandidate[Value],
) (TopKCandidate[Value], error)

// TopKNodeEligible may exclude a node before its child alternatives are
// enumerated. A nil callback accepts every node.
type TopKNodeEligible[ID ~uint32, Node any] func(class ID, node Node) bool

// ErrInvalidTopKExtraction reports missing callbacks.
var ErrInvalidTopKExtraction = errors.New("egraphcore: invalid top-k extraction")

// ErrTopKWorkBudget reports that top-k enumeration exhausted its work budget
// before finishing.
var ErrTopKWorkBudget = errors.New("egraphcore: top-k work budget exceeded")

// ErrTopKExpansionLimit reports that extraction could not represent a larger
// saturated child result limit while the result might still be truncated.
var ErrTopKExpansionLimit = errors.New("egraphcore: top-k expansion limit exceeded")

type topKVector[Value any] struct {
	indices []int
	value   TopKCandidate[Value]
}

type topKFrame[ID ~uint32, Node, Value any] struct {
	id           ID
	memoKey      string
	nodes        []Node
	nodeIndex    int
	result       []TopKCandidate[Value]
	seen         []Value
	seenByHash   map[string][]Value
	node         Node
	childIDs     []ID
	options      [][]TopKCandidate[Value]
	childIndex   int
	waitingChild bool
	phase        uint8
}

type topKVectorHeap[Value any] struct {
	items []topKVector[Value]
	less  func(TopKCandidate[Value], TopKCandidate[Value]) bool
}

func (h topKVectorHeap[Value]) Len() int { return len(h.items) }
func (h topKVectorHeap[Value]) Less(i, j int) bool {
	return h.less(h.items[i].value, h.items[j].value)
}
func (h topKVectorHeap[Value]) Swap(i, j int) { h.items[i], h.items[j] = h.items[j], h.items[i] }
func (h *topKVectorHeap[Value]) Push(v any)   { h.items = append(h.items, v.(topKVector[Value])) }
func (h *topKVectorHeap[Value]) Pop() any {
	n := len(h.items)
	v := h.items[n-1]
	h.items[n-1] = topKVector[Value]{}
	h.items = h.items[:n-1]
	return v
}

func topKSortWork(n int) int {
	if n <= 1 {
		return n
	}
	levels := 0
	for width := n; width > 1; width = width/2 + width%2 {
		levels++
	}
	maxInt := int(^uint(0) >> 1)
	if n > maxInt/levels {
		return maxInt
	}
	return n * levels
}

// ExtractTopK enumerates acyclic expressions from root and returns exact top-k
// results for arbitrary build and equal callbacks. If child alternatives may
// collapse to the same parent value, it expands capped child result sets until
// the result is complete. This can enumerate the full finite acyclic search
// space; use a work budget to bound it, or use ExtractTopKWithChildPreservation
// when the build callback has the documented preservation property. Retries
// share the same work budget. Graph and ordering callbacks must return stable
// results across retries. The core owns traversal, cycle rejection,
// ancestor-sensitive memoization, and a bounded best-first heap for each
// node's child-choice combinations. Backend callbacks define graph access and
// expression semantics. build receives selected child candidates, allowing
// the backend to calculate and validate cost. less must provide a deterministic
// strict weak ordering. equal reports structural equality for backend values;
// equal candidates must have the same rank under less and comboLess. comboLess
// may order candidates that compare equal under less, but the combined order
// must also be monotone in each child: replacing one child with a later (no
// better) candidate must not produce a better parent candidate. This applies
// to the full tie-break order, not just cost. Nodes and children are visited in
// the order supplied by the backend. Backends are responsible for this
// cost-model property. A nil
// result with no error means there were no valid acyclic candidates. A nil
// result is also returned when k is nonpositive. A nil budget is unlimited. A
// non-nil budget charges graph visits, ancestor-sensitive
// memo keys, node and child scans, candidate builds, heap operations, duplicate
// comparisons, hash-key callback calls, and result sorting. Hash callback work
// is charged as one unit per call; its internal cost is backend-defined.
// Exceeding it returns ErrTopKWorkBudget without partial results. If the limit
// cannot be represented while expanding a result that may be truncated,
// ErrTopKExpansionLimit is returned.
// Traversal uses an explicit stack and is not limited by Go call-stack depth.
func ExtractTopK[ID ~uint32, Node, Value any](
	root ID,
	k int,
	nodes func(ID) []Node,
	canonical func(ID) ID,
	children func(Node) []ID,
	build TopKBuild[ID, Node, Value],
	nodeEligible TopKNodeEligible[ID, Node],
	less func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	comboLess func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	equal func(Value, Value) bool,
	budget *WorkBudget,
) ([]TopKCandidate[Value], error) {
	return extractTopK(root, k, nodes, canonical, children, build, nodeEligible, less, comboLess, equal, nil, budget, false)
}

// ExtractTopKWithHash is ExtractTopK with hash-based duplicate bucketing.
// hash must return the same key for values considered equal by equal. Hashes
// only select a bucket; equal remains the authority when resolving collisions.
func ExtractTopKWithHash[ID ~uint32, Node, Value any](
	root ID, k int, nodes func(ID) []Node, canonical func(ID) ID, children func(Node) []ID,
	build TopKBuild[ID, Node, Value], nodeEligible TopKNodeEligible[ID, Node],
	less func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	comboLess func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	equal func(Value, Value) bool, hash func(Value) string, budget *WorkBudget,
) ([]TopKCandidate[Value], error) {
	if hash == nil {
		return nil, ErrInvalidTopKExtraction
	}
	return extractTopK(root, k, nodes, canonical, children, build, nodeEligible, less, comboLess, equal, hash, budget, false)
}

// ExtractTopKWithChildPreservation uses the bounded top-k search when build
// preserves child distinctions: for a fixed class and node, two child tuples
// that differ under equal must build values that also differ under equal. In
// addition to the ordering requirements of ExtractTopK, this property lets
// each child contribute at most k alternatives without hiding a distinct
// parent result. Callers must ensure the property; it cannot be checked for
// child alternatives that have not been enumerated.
func ExtractTopKWithChildPreservation[ID ~uint32, Node, Value any](
	root ID,
	k int,
	nodes func(ID) []Node,
	canonical func(ID) ID,
	children func(Node) []ID,
	build TopKBuild[ID, Node, Value],
	nodeEligible TopKNodeEligible[ID, Node],
	less func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	comboLess func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	equal func(Value, Value) bool,
	budget *WorkBudget,
) ([]TopKCandidate[Value], error) {
	return extractTopK(root, k, nodes, canonical, children, build, nodeEligible, less, comboLess, equal, nil, budget, true)
}

// ExtractTopKWithChildPreservationAndHash combines bounded child-preserving
// extraction with hash-based duplicate bucketing. hash must be consistent
// with equal; collisions are always checked with equal.
func ExtractTopKWithChildPreservationAndHash[ID ~uint32, Node, Value any](
	root ID, k int, nodes func(ID) []Node, canonical func(ID) ID, children func(Node) []ID,
	build TopKBuild[ID, Node, Value], nodeEligible TopKNodeEligible[ID, Node],
	less func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	comboLess func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	equal func(Value, Value) bool, hash func(Value) string, budget *WorkBudget,
) ([]TopKCandidate[Value], error) {
	if hash == nil {
		return nil, ErrInvalidTopKExtraction
	}
	return extractTopK(root, k, nodes, canonical, children, build, nodeEligible, less, comboLess, equal, hash, budget, true)
}

func extractTopK[ID ~uint32, Node, Value any](
	root ID,
	k int,
	nodes func(ID) []Node,
	canonical func(ID) ID,
	children func(Node) []ID,
	build TopKBuild[ID, Node, Value],
	nodeEligible TopKNodeEligible[ID, Node],
	less func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	comboLess func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	equal func(Value, Value) bool,
	hash func(Value) string,
	budget *WorkBudget,
	childPreservation bool,
) ([]TopKCandidate[Value], error) {
	if k <= 0 {
		return nil, nil
	}
	limit := k
	for {
		truncated := false
		candidates, err := extractTopKPass(root, limit, nodes, canonical, children, build, nodeEligible, less, comboLess, equal, hash, budget, &truncated)
		if err != nil {
			return nil, err
		}
		if childPreservation && len(candidates) >= k {
			return candidates[:k], nil
		}
		if !truncated {
			if len(candidates) > k {
				candidates = candidates[:k]
			}
			return candidates, nil
		}
		if limit > math.MaxInt/2 {
			return nil, ErrTopKExpansionLimit
		}
		limit *= 2
	}
}

func extractTopKPass[ID ~uint32, Node, Value any](
	root ID,
	k int,
	nodes func(ID) []Node,
	canonical func(ID) ID,
	children func(Node) []ID,
	build TopKBuild[ID, Node, Value],
	nodeEligible TopKNodeEligible[ID, Node],
	less func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	comboLess func(TopKCandidate[Value], TopKCandidate[Value]) bool,
	equal func(Value, Value) bool,
	hash func(Value) string,
	budget *WorkBudget,
	truncated *bool,
) ([]TopKCandidate[Value], error) {
	if nodes == nil || canonical == nil || children == nil || build == nil || less == nil || equal == nil {
		return nil, ErrInvalidTopKExtraction
	}
	if comboLess == nil {
		comboLess = less
	}
	// Use the same complete ordering for final results and Cartesian-product
	// traversal. Both less and its tie-break extension must be monotone in each
	// child for stopping after k results to be exact.
	combinationLess := func(a, b TopKCandidate[Value]) bool {
		if less(a, b) {
			return true
		}
		if less(b, a) {
			return false
		}
		return comboLess(a, b)
	}
	spend := func(units int) bool { return budget == nil || budget.TrySpend(units) }
	memo := make(map[string][]TopKCandidate[Value])
	active := make(map[ID]bool)
	stack := make([]topKFrame[ID, Node, Value], 0)
	defer func() {
		// Errors can leave several active ancestors on the explicit stack.
		for i := range stack {
			delete(active, stack[i].id)
		}
	}()
	push := func(id ID) ([]TopKCandidate[Value], bool, error) {
		if !spend(1) {
			return nil, false, ErrTopKWorkBudget
		}
		id = canonical(id)
		if active[id] {
			return nil, false, nil
		}
		if !spend(topKSortWork(len(active))) {
			return nil, false, ErrTopKWorkBudget
		}
		memoKey := topKMemoKey(id, active)
		if result, ok := memo[memoKey]; ok {
			return result, false, nil
		}
		active[id] = true
		frame := topKFrame[ID, Node, Value]{id: id, memoKey: memoKey, nodes: nodes(id)}
		if hash != nil {
			frame.seenByHash = make(map[string][]Value)
		}
		stack = append(stack, frame)
		return nil, true, nil
	}
	acceptChild := func(frame *topKFrame[ID, Node, Value], options []TopKCandidate[Value]) error {
		if len(options) == 0 {
			frame.nodeIndex++
			frame.phase = 0
			return nil
		}
		if !spend(topKSortWork(len(options))) {
			return ErrTopKWorkBudget
		}
		axis := append([]TopKCandidate[Value](nil), options...)
		sort.SliceStable(axis, func(a, b int) bool {
			return combinationLess(axis[a], axis[b])
		})
		frame.options[frame.childIndex] = axis
		frame.childIndex++
		return nil
	}
	if _, _, err := push(root); err != nil {
		return nil, err
	}
	var returned []TopKCandidate[Value]
	for len(stack) > 0 {
		frame := &stack[len(stack)-1]
		switch frame.phase {
		case 0: // Select the next node in this class.
			if frame.nodeIndex == len(frame.nodes) {
				if !spend(topKSortWork(len(frame.result))) {
					return nil, ErrTopKWorkBudget
				}
				if len(frame.result) > k {
					*truncated = true
				}
				sort.SliceStable(frame.result, func(i, j int) bool {
					return combinationLess(frame.result[i], frame.result[j])
				})
				if len(frame.result) > k {
					frame.result = frame.result[:k]
				}
				memo[frame.memoKey] = frame.result
				returned = frame.result
				delete(active, frame.id)
				stack = stack[:len(stack)-1]
				if len(stack) == 0 {
					return returned, nil
				}
				parent := &stack[len(stack)-1]
				parent.waitingChild = false
				if err := acceptChild(parent, returned); err != nil {
					return nil, err
				}
				continue
			}
			if !spend(1) {
				return nil, ErrTopKWorkBudget
			}
			node := frame.nodes[frame.nodeIndex]
			if nodeEligible != nil && !nodeEligible(frame.id, node) {
				frame.nodeIndex++
				continue
			}
			frame.node = node
			frame.childIDs = children(node)
			frame.options = make([][]TopKCandidate[Value], len(frame.childIDs))
			frame.childIndex = 0
			frame.phase = 1
		case 1: // Resolve child alternatives from left to right.
			if frame.waitingChild {
				// The child frame is above this one and will publish its result
				// when it completes.
				continue
			}
			if frame.childIndex == len(frame.options) {
				frame.phase = 2
				continue
			}
			if !spend(1) {
				return nil, ErrTopKWorkBudget
			}
			child := frame.childIDs[frame.childIndex]
			childOptions, pushed, err := push(child)
			if err != nil {
				return nil, err
			}
			if pushed {
				stack[len(stack)-2].waitingChild = true
				continue
			}
			if err := acceptChild(frame, childOptions); err != nil {
				return nil, err
			}
		case 2: // Enumerate this node's ordered Cartesian product.
			valid := true
			for _, option := range frame.options {
				if len(option) == 0 {
					valid = false
					break
				}
			}
			if !valid {
				frame.nodeIndex++
				frame.phase = 0
				continue
			}
			id, node, options := frame.id, frame.node, frame.options
			makeCandidate := func(indices []int) (TopKCandidate[Value], error) {
				if !spend(1) || !spend(len(indices)) {
					return TopKCandidate[Value]{}, ErrTopKWorkBudget
				}
				selected := make([]TopKCandidate[Value], len(options))
				for i := range options {
					selected[i] = options[i][indices[i]]
				}
				return build(id, node, selected)
			}
			indicesKey := func(indices []int) string {
				// Vector keys grow with the number of children.
				var b strings.Builder
				for _, index := range indices {
					fmt.Fprintf(&b, "%d,", index)
				}
				return b.String()
			}
			zero := make([]int, len(options))
			first, err := makeCandidate(zero)
			if err != nil {
				return nil, err
			}
			queue := &topKVectorHeap[Value]{less: combinationLess}
			if !spend(1) || !spend(len(zero)) {
				return nil, ErrTopKWorkBudget
			}
			heap.Push(queue, topKVector[Value]{indices: zero, value: first})
			queued := map[string]struct{}{indicesKey(zero): {}}
			nodeResults := 0
			for queue.Len() > 0 && nodeResults < k {
				if !spend(1) {
					return nil, ErrTopKWorkBudget
				}
				current := heap.Pop(queue).(topKVector[Value])
				duplicate := false
				bucket := frame.seen
				hashKey := ""
				if hash != nil {
					if !spend(1) {
						return nil, ErrTopKWorkBudget
					}
					hashKey = hash(current.value.Value)
					bucket = frame.seenByHash[hashKey]
				}
				for _, value := range bucket {
					if !spend(1) {
						return nil, ErrTopKWorkBudget
					}
					if equal(value, current.value.Value) {
						duplicate = true
						break
					}
				}
				if !duplicate {
					if hash == nil {
						frame.seen = append(frame.seen, current.value.Value)
					} else {
						frame.seenByHash[hashKey] = append(bucket, current.value.Value)
					}
					frame.result = append(frame.result, current.value)
					nodeResults++
				}
				for i := range options {
					if !spend(1) {
						return nil, ErrTopKWorkBudget
					}
					next := append([]int(nil), current.indices...)
					next[i]++
					if next[i] >= len(options[i]) {
						continue
					}
					if !spend(len(next)) {
						return nil, ErrTopKWorkBudget
					}
					vectorKey := indicesKey(next)
					if _, ok := queued[vectorKey]; ok {
						continue
					}
					queued[vectorKey] = struct{}{}
					candidate, err := makeCandidate(next)
					if err != nil {
						return nil, err
					}
					if !spend(1) {
						return nil, ErrTopKWorkBudget
					}
					heap.Push(queue, topKVector[Value]{indices: next, value: candidate})
				}
			}
			if nodeResults >= k && queue.Len() > 0 {
				*truncated = true
			}
			frame.nodeIndex++
			frame.phase = 0
		}
	}
	return returned, nil
}

func topKMemoKey[ID ~uint32](id ID, active map[ID]bool) string {
	ancestors := make([]ID, 0, len(active))
	for ancestor := range active {
		ancestors = append(ancestors, ancestor)
	}
	sort.Slice(ancestors, func(i, j int) bool { return ancestors[i] < ancestors[j] })
	var b strings.Builder
	fmt.Fprintf(&b, "%d:", id)
	for _, ancestor := range ancestors {
		fmt.Fprintf(&b, "%d,", ancestor)
	}
	return b.String()
}
