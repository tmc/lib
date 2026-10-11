package egraphcore

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// ErrChoiceFingerprintWorkBudget reports exhausted work budget during hashing.
var ErrChoiceFingerprintWorkBudget = errors.New("egraphcore: choice fingerprint work budget exceeded")

// ErrChoiceFingerprintLimit reports that a selected node's fingerprint input is too large.
var ErrChoiceFingerprintLimit = errors.New("egraphcore: choice fingerprint size limit exceeded")

// ChoiceFingerprint is a fixed-size structural digest for one selected class.
type ChoiceFingerprint [32]byte

// ChoiceFingerprintBuild combines a selected node with its child fingerprints.
// Child fingerprints are in the node's child order.
type ChoiceFingerprintBuild[ID ~uint32, Node any] func(ID, Node, []ChoiceFingerprint) ChoiceFingerprint

// ChoiceFingerprinter caches subtree fingerprints for one immutable choice
// space. Cache entries depend on the selected alternative and child
// fingerprints, so changing one part of a choice only invalidates ancestors
// whose selected child fingerprints change. Do not reuse it with a different
// graph or callbacks that return different alternatives or fingerprints. It
// is not safe for concurrent use.
type ChoiceFingerprinter[ID ~uint32, Node any] struct {
	cache map[ID]map[string]ChoiceFingerprint
}

// NewChoiceFingerprinter creates an empty cache for an immutable choice space.
func NewChoiceFingerprinter[ID ~uint32, Node any]() *ChoiceFingerprinter[ID, Node] {
	return &ChoiceFingerprinter[ID, Node]{cache: make(map[ID]map[string]ChoiceFingerprint)}
}

// Fingerprint computes the selected root's structural fingerprint without
// materializing backend expression values. It rejects missing choices,
// invalid alternatives, and cycles. A nil budget is unlimited; otherwise the
// budget charges selected classes, child edges, and cache-key construction.
func (f *ChoiceFingerprinter[ID, Node]) Fingerprint(
	root ID,
	choice map[ID]int,
	alternatives func(ID) []Node,
	children func(Node) []ID,
	build ChoiceFingerprintBuild[ID, Node],
	budget *WorkBudget,
) (ChoiceFingerprint, error) {
	if f == nil || alternatives == nil || children == nil || build == nil {
		return ChoiceFingerprint{}, ErrInvalidChoiceGraph
	}
	if f.cache == nil {
		f.cache = make(map[ID]map[string]ChoiceFingerprint)
	}
	spend := func(units int) bool { return budget == nil || budget.TrySpend(units) }
	type frame struct {
		class    ID
		index    int
		node     Node
		children []ID
		next     int
	}
	selectNode := func(class ID) (int, Node, error) {
		var node Node
		if !spend(1) {
			return 0, node, ErrChoiceFingerprintWorkBudget
		}
		nodes := alternatives(class)
		index, ok := choice[class]
		if !ok {
			return 0, node, fmt.Errorf("%w: missing choice for class %d", ErrInvalidChoiceGraph, class)
		}
		if index < 0 || index >= len(nodes) {
			return 0, node, fmt.Errorf("%w: alternative %d out of range for class %d", ErrInvalidChoiceGraph, index, class)
		}
		return index, nodes[index], nil
	}
	rootIndex, rootNode, err := selectNode(root)
	if err != nil {
		return ChoiceFingerprint{}, err
	}
	state := make(map[ID]uint8, len(choice))
	hashes := make(map[ID]ChoiceFingerprint, len(choice))
	state[root] = 1
	stack := []frame{{class: root, index: rootIndex, node: rootNode, children: append([]ID(nil), children(rootNode)...)}}
	for len(stack) != 0 {
		last := len(stack) - 1
		current := &stack[last]
		if current.next < len(current.children) {
			if !spend(1) {
				return ChoiceFingerprint{}, ErrChoiceFingerprintWorkBudget
			}
			child := current.children[current.next]
			switch state[child] {
			case 1:
				return ChoiceFingerprint{}, fmt.Errorf("%w: cycle at class %d", ErrInvalidChoiceGraph, child)
			case 2:
				current.next++
			default:
				index, node, err := selectNode(child)
				if err != nil {
					return ChoiceFingerprint{}, err
				}
				state[child] = 1
				stack = append(stack, frame{class: child, index: index, node: node, children: append([]ID(nil), children(node)...)})
			}
			continue
		}
		maxInt := int(^uint(0) >> 1)
		if len(current.children) > (maxInt-8)/len(ChoiceFingerprint{}) {
			return ChoiceFingerprint{}, ErrChoiceFingerprintLimit
		}
		if !spend(1 + len(current.children)) {
			return ChoiceFingerprint{}, ErrChoiceFingerprintWorkBudget
		}
		keyBytes := make([]byte, 8+len(current.children)*len(ChoiceFingerprint{}))
		binary.LittleEndian.PutUint64(keyBytes, uint64(current.index))
		childHashes := make([]ChoiceFingerprint, len(current.children))
		for i, child := range current.children {
			childHashes[i] = hashes[child]
			copy(keyBytes[8+i*len(ChoiceFingerprint{}):], childHashes[i][:])
		}
		key := string(keyBytes)
		fingerprint, ok := f.cache[current.class][key]
		if !ok {
			fingerprint = build(current.class, current.node, childHashes)
			variants := f.cache[current.class]
			if variants == nil {
				variants = make(map[string]ChoiceFingerprint)
				f.cache[current.class] = variants
			}
			variants[key] = fingerprint
		}
		hashes[current.class] = fingerprint
		state[current.class] = 2
		stack[last] = frame{}
		stack = stack[:last]
	}
	return hashes[root], nil
}
