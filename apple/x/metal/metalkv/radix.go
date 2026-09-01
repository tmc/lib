// SPDX-License-Identifier: Apache-2.0

package metalkv

import (
	"sync"
	"time"
)

// RadixNode represents a branch or token segment within the Radix-Tree Prefix Cache.
type RadixNode struct {
	Tokens     []int32
	BlockIDs   []int
	Children   map[int32]*RadixNode
	Parent     *RadixNode
	LastAccess int64
	RefCount   int
}

// RadixTree manages hierarchical shared prefix caches for zero-recomputation multi-turn inference.
type RadixTree struct {
	Root     *RadixNode
	NodeMap  map[int]*RadixNode // blockID -> node
	mu       sync.RWMutex
	maxNodes int
}

// NewRadixTree initializes an empty Radix-Tree prefix cache.
func NewRadixTree(maxNodes int) *RadixTree {
	root := &RadixNode{
		Tokens:   make([]int32, 0),
		Children: make(map[int32]*RadixNode),
	}
	return &RadixTree{
		Root:     root,
		NodeMap:  make(map[int]*RadixNode),
		maxNodes: maxNodes,
	}
}

// MatchPrefix finds the longest matching token prefix in the Radix Tree.
// Returns the number of matched tokens, associated physical block IDs, and the matched node.
func (r *RadixTree) MatchPrefix(tokens []int32) (int, []int, *RadixNode) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	curr := r.Root
	matchedTokens := 0
	matchedBlocks := make([]int, 0)

	for matchedTokens < len(tokens) {
		firstTok := tokens[matchedTokens]
		child, exists := curr.Children[firstTok]
		if !exists {
			break
		}

		// Verify matching segment
		segLen := len(child.Tokens)
		remaining := len(tokens) - matchedTokens
		checkLen := segLen
		if remaining < checkLen {
			checkLen = remaining
		}

		matched := true
		for i := 0; i < checkLen; i++ {
			if child.Tokens[i] != tokens[matchedTokens+i] {
				matched = false
				break
			}
		}

		if !matched {
			break
		}

		matchedTokens += checkLen
		matchedBlocks = append(matchedBlocks, child.BlockIDs...)
		child.LastAccess = time.Now().UnixNano()

		if checkLen < segLen {
			// Partial segment match
			return matchedTokens, matchedBlocks, child
		}

		curr = child
	}

	return matchedTokens, matchedBlocks, curr
}

// Insert adds a sequence of token IDs and their allocated block IDs into the tree.
func (r *RadixTree) Insert(tokens []int32, blockIDs []int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(tokens) == 0 {
		return
	}

	firstTok := tokens[0]
	node := &RadixNode{
		Tokens:     tokens,
		BlockIDs:   blockIDs,
		Children:   make(map[int32]*RadixNode),
		Parent:     r.Root,
		LastAccess: time.Now().UnixNano(),
	}
	r.Root.Children[firstTok] = node
	for _, b := range blockIDs {
		r.NodeMap[b] = node
	}
}
