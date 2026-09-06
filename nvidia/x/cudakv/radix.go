// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
)

// pkgName prefixes this package's errors.
const pkgName = "cudakv"

// RadixNode is one segment of a cached token sequence.
//
// Tokens holds the segment; Blocks holds the cache blocks covering it, one per
// TokensPerBlock tokens. A node's segment is always a whole number of blocks,
// so a match never ends in the middle of a block that only partly matches.
type RadixNode struct {
	Tokens []int32
	Blocks []int

	// Children are keyed by the first block of the child's segment, so two
	// sequences that agree on a whole block share a node and diverge at a
	// block boundary. Keying by a single token could not distinguish two
	// blocks that begin with the same token.
	Children map[string]*RadixNode
	Parent   *RadixNode

	refs int    // live holders of this node or of one below it
	used uint64 // tree clock reading of the last match
}

// RadixTree is a prefix cache: it maps token sequences to the cache blocks
// holding their computed keys and values, so a request sharing a prefix with an
// earlier one reuses those blocks instead of recomputing them.
//
// The tree is a radix tree over blocks of tokens. Inserting a sequence that
// shares a prefix with a stored one splits the stored node at the divergence
// point, so the shared prefix is stored once however many sequences pass
// through it.
type RadixTree struct {
	Root *RadixNode

	// NodeMap indexes nodes by the blocks they hold, so a caller that knows
	// a block id can find what is cached in it.
	NodeMap map[int]*RadixNode

	// TokensPerBlock is how many tokens one cache block covers. Matches are
	// reported in whole blocks: a caller reuses blocks, so a match of half a
	// block would name tokens it cannot in fact reuse.
	TokensPerBlock int

	mu       sync.Mutex
	maxNodes int
	nodes    int
	clock    uint64
}

// NewRadixTree creates an empty prefix cache holding at most maxNodes segments,
// each block covering tokensPerBlock tokens.
//
// Inserting into a full tree evicts the least recently matched unpinned leaves.
func NewRadixTree(maxNodes, tokensPerBlock int) *RadixTree {
	if maxNodes < 1 {
		maxNodes = 1
	}
	if tokensPerBlock < 1 {
		tokensPerBlock = 1
	}
	return &RadixTree{
		Root:           &RadixNode{Children: make(map[string]*RadixNode)},
		NodeMap:        make(map[int]*RadixNode),
		TokensPerBlock: tokensPerBlock,
		maxNodes:       maxNodes,
	}
}

// blockKey names a block by the tokens in it.
func blockKey(tokens []int32) string {
	var b strings.Builder
	for i, t := range tokens {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatInt(int64(t), 10))
	}
	return b.String()
}

// MatchPrefix returns the longest stored prefix of tokens.
//
// The result is the number of tokens matched, the blocks holding them in order,
// and the node the match ended at. The match is a whole number of blocks and
// stops at the first block that is not stored, so a caller can reuse exactly
// the blocks returned and must compute the rest.
func (r *RadixTree) MatchPrefix(tokens []int32) (int, []int, *RadixNode) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.match(tokens)
}

// match walks the tree; the caller holds the lock.
func (r *RadixTree) match(tokens []int32) (int, []int, *RadixNode) {
	tpb := r.TokensPerBlock
	curr := r.Root
	matched := 0
	blocks := []int{}

	for matched+tpb <= len(tokens) {
		child, ok := curr.Children[blockKey(tokens[matched:matched+tpb])]
		if !ok {
			break
		}
		n := commonBlocks(child.Tokens, tokens[matched:], tpb)
		r.clock++
		child.used = r.clock
		matched += n
		blocks = append(blocks, child.Blocks[:n/tpb]...)
		curr = child
		if n < len(child.Tokens) {
			// The query leaves this node's segment partway through, so
			// nothing below it can match.
			break
		}
	}
	return matched, blocks, curr
}

// commonBlocks returns the length of the common prefix of a and b, rounded down
// to a whole number of blocks.
func commonBlocks(a, b []int32, tpb int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i - i%tpb
}

// Insert stores tokens and the blocks holding them, sharing the parts already
// stored.
//
// len(tokens) must be a whole number of blocks and blocks must hold one entry
// per block of tokens. Inserting a sequence already stored refreshes it and
// keeps the stored blocks: the caller's blocks for the shared part are then
// redundant, which MatchPrefix would have told it before it computed them.
func (r *RadixTree) Insert(tokens []int32, blocks []int) error {
	tpb := r.TokensPerBlock
	if len(tokens) == 0 {
		return fmt.Errorf("%s: inserting an empty sequence", pkgName)
	}
	if len(tokens)%tpb != 0 {
		return fmt.Errorf("%s: %d tokens is not a whole number of %d-token blocks", pkgName, len(tokens), tpb)
	}
	if len(blocks) != len(tokens)/tpb {
		return fmt.Errorf("%s: %d tokens need %d blocks, got %d", pkgName, len(tokens), len(tokens)/tpb, len(blocks))
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	curr := r.Root
	pos := 0
	for pos < len(tokens) {
		key := blockKey(tokens[pos : pos+tpb])
		child, ok := curr.Children[key]
		if !ok {
			r.pin(curr)
			r.evictFor(1)
			r.unpin(curr)
			node := &RadixNode{
				Tokens:   append([]int32(nil), tokens[pos:]...),
				Blocks:   append([]int(nil), blocks[pos/tpb:]...),
				Children: make(map[string]*RadixNode),
				Parent:   curr,
			}
			r.clock++
			node.used = r.clock
			curr.Children[key] = node
			r.nodes++
			for _, b := range node.Blocks {
				r.NodeMap[b] = node
			}
			return nil
		}

		n := commonBlocks(child.Tokens, tokens[pos:], tpb)
		r.clock++
		child.used = r.clock
		if n < len(child.Tokens) {
			// The stored segment continues past where this sequence
			// leaves it. Split it so the shared part is one node.
			r.pin(child)
			r.evictFor(1)
			r.unpin(child)
			curr = r.split(child, n)
		} else {
			curr = child
		}
		pos += n
	}
	return nil
}

// split divides node after its first n tokens. It inserts a new parent holding
// those n tokens and leaves node holding the rest, and returns the new parent.
//
// The prefix is the new node rather than node itself so that a caller holding
// node keeps holding the end of the segment it acquired. Splitting the other
// way around would leave the held blocks on a node the caller has no pointer
// to, and Release, which walks upwards, could never reach them.
func (r *RadixTree) split(node *RadixNode, n int) *RadixNode {
	tpb := r.TokensPerBlock
	prefix := &RadixNode{
		Tokens:   append([]int32(nil), node.Tokens[:n]...),
		Blocks:   append([]int(nil), node.Blocks[:n/tpb]...),
		Children: map[string]*RadixNode{blockKey(node.Tokens[n : n+tpb]): node},
		Parent:   node.Parent,
		refs:     node.refs,
		used:     node.used,
	}
	prefix.Parent.Children[blockKey(prefix.Tokens[:tpb])] = prefix
	node.Tokens = append([]int32(nil), node.Tokens[n:]...)
	node.Blocks = append([]int(nil), node.Blocks[n/tpb:]...)
	node.Parent = prefix
	r.nodes++
	for _, b := range prefix.Blocks {
		r.NodeMap[b] = prefix
	}
	return prefix
}

// pin marks node and its ancestors as held, which keeps them from being
// evicted. The caller holds the lock.
func (r *RadixTree) pin(n *RadixNode) {
	for ; n != nil && n != r.Root; n = n.Parent {
		n.refs++
	}
}

// unpin undoes one pin. The caller holds the lock.
func (r *RadixTree) unpin(n *RadixNode) {
	for ; n != nil && n != r.Root; n = n.Parent {
		if n.refs > 0 {
			n.refs--
		}
	}
}

// Acquire matches tokens and pins the matched path, so the blocks it returns
// are not evicted until the returned node is released. It returns a nil node
// when nothing matched, which needs no release.
func (r *RadixTree) Acquire(tokens []int32) (int, []int, *RadixNode) {
	r.mu.Lock()
	defer r.mu.Unlock()

	matched, blocks, node := r.match(tokens)
	if node == r.Root {
		return matched, blocks, nil
	}
	r.pin(node)
	return matched, blocks, node
}

// Release undoes one Acquire. Releasing a nil node does nothing.
func (r *RadixTree) Release(node *RadixNode) {
	if node == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.unpin(node)
}

// Evict removes up to n least recently matched segments and returns the blocks
// they held, for the caller to hand back to its allocator.
//
// Only leaves are evicted, and only unpinned ones, so a segment is never
// dropped while a sequence below it is still cached or in use. Evict returns
// fewer blocks than asked, possibly none, when everything left is pinned or has
// cached continuations.
func (r *RadixTree) Evict(n int) []int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.evict(n)
}

func (r *RadixTree) evict(n int) []int {
	freed := []int{}
	for i := 0; i < n; i++ {
		victim := r.lruLeaf()
		if victim == nil {
			break
		}
		freed = append(freed, victim.Blocks...)
		for _, b := range victim.Blocks {
			if r.NodeMap[b] == victim {
				delete(r.NodeMap, b)
			}
		}
		delete(victim.Parent.Children, blockKey(victim.Tokens[:r.TokensPerBlock]))
		victim.Parent = nil
		r.nodes--
	}
	return freed
}

// evictFor makes room for n more nodes.
func (r *RadixTree) evictFor(n int) {
	if r.nodes+n > r.maxNodes {
		r.evict(r.nodes + n - r.maxNodes)
	}
}

// lruLeaf returns the least recently matched evictable node: a leaf that
// nothing holds.
func (r *RadixTree) lruLeaf() *RadixNode {
	var best *RadixNode
	var walk func(*RadixNode)
	walk = func(n *RadixNode) {
		for _, c := range n.Children {
			walk(c)
		}
		if n == r.Root || len(n.Children) > 0 || n.refs > 0 {
			return
		}
		if best == nil || n.used < best.used {
			best = n
		}
	}
	walk(r.Root)
	return best
}

// Len returns the number of stored segments.
func (r *RadixTree) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.nodes
}
