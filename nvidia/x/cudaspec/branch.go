// SPDX-License-Identifier: Apache-2.0

package cudaspec

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
)

// Common errors for branch and cache operations.
var (
	ErrInvalidBranchID      = errors.New("cudaspec: invalid branch identity")
	ErrBranchAlreadyExists  = errors.New("cudaspec: branch identity already exists")
	ErrBranchNotFound       = errors.New("cudaspec: branch not found")
	ErrBranchNotActive      = errors.New("cudaspec: branch is not active")
	ErrIncorrectTokenAccept = errors.New("cudaspec: cannot accept mismatched or invalid token")
	ErrCacheKeyMismatch     = errors.New("cudaspec: cache key mismatch or invalidated")
	ErrKVHoldersConflict    = errors.New("cudaspec: KV block retains holders conflict")
	ErrNilKVAllocator       = errors.New("cudaspec: nil KV block allocator")
)

// BranchID uniquely identifies a speculative branch in a proposal tree.
type BranchID int64

// ProposalState tracks the lifecycle of a speculative proposal branch.
type ProposalState int

const (
	ProposalProposed ProposalState = iota
	ProposalCommitted
	ProposalRolledBack
)

func (s ProposalState) String() string {
	switch s {
	case ProposalProposed:
		return "proposed"
	case ProposalCommitted:
		return "committed"
	case ProposalRolledBack:
		return "rolled_back"
	default:
		return fmt.Sprintf("state(%d)", int(s))
	}
}

// Branch represents a speculative candidate branch with explicit token sequence,
// tree node mappings, KV block tracking, and proposal lifecycle state.
type Branch struct {
	ID           BranchID
	ParentBranch BranchID // -1 for root
	Tokens       []int32  // Candidate tokens in this branch
	NodeIDs      []int    // Node IDs corresponding to tokens in DraftTree
	State        ProposalState
	KVBlocks     []int // Physical KV blocks owned specifically by this branch
	SharedPrefix []int // Physical KV blocks shared with parent/prefix
	SequenceID   int64 // Sequence ID in cudakv if allocated
}

// BranchTree coordinates speculative branches, explicit tree masks, verification,
// and KV-block reference counting during commit and rollback cycles.
type BranchTree struct {
	mu           sync.Mutex
	rootSeqID    int64
	sharedPrefix []int
	branches     map[BranchID]*Branch
	nextBranchID BranchID
	activeBranch BranchID // Currently admitted or target branch
	draftTree    *DraftTree
}

// NewBranchTree creates a branch coordinator rooted on sharedPrefix KV blocks.
func NewBranchTree(rootSeqID int64, sharedPrefix []int, rootToken int32) *BranchTree {
	prefixCopy := append([]int(nil), sharedPrefix...)
	dt := NewDraftTree(rootToken)
	bt := &BranchTree{
		rootSeqID:    rootSeqID,
		sharedPrefix: prefixCopy,
		branches:     make(map[BranchID]*Branch),
		nextBranchID: 1,
		activeBranch: 0,
		draftTree:    dt,
	}

	// Create root branch 0
	rootBranch := &Branch{
		ID:           0,
		ParentBranch: -1,
		Tokens:       []int32{rootToken},
		NodeIDs:      []int{0},
		State:        ProposalCommitted,
		SharedPrefix: prefixCopy,
		SequenceID:   rootSeqID,
	}
	bt.branches[0] = rootBranch
	return bt
}

// DraftTree returns the underlying DraftTree.
func (bt *BranchTree) DraftTree() *DraftTree {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	return bt.draftTree
}

// ProposeBranch attaches a new speculative candidate branch to parentID.
func (bt *BranchTree) ProposeBranch(parentID BranchID, tokens []int32, kvBlocks []int) (*Branch, error) {
	bt.mu.Lock()
	defer bt.mu.Unlock()

	parent, ok := bt.branches[parentID]
	if !ok {
		return nil, fmt.Errorf("%w: parent %d", ErrBranchNotFound, parentID)
	}
	if parent.State == ProposalRolledBack {
		return nil, fmt.Errorf("%w: parent %d rolled back", ErrBranchNotActive, parentID)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("%w: empty proposal tokens", ErrInvalidBranchID)
	}

	bID := bt.nextBranchID
	bt.nextBranchID++

	// Map into draft tree
	var currNodeID int
	if len(parent.NodeIDs) > 0 {
		currNodeID = parent.NodeIDs[len(parent.NodeIDs)-1]
	} else {
		currNodeID = 0
	}

	nodeIDs := make([]int, 0, len(tokens))
	for _, tok := range tokens {
		node := bt.draftTree.AddNode(currNodeID, tok)
		if node == nil {
			return nil, fmt.Errorf("cudaspec: failed to add node for token %d under parent node %d", tok, currNodeID)
		}
		currNodeID = node.ID
		nodeIDs = append(nodeIDs, node.ID)
	}

	// Inherit shared blocks from parent + parent's private blocks
	shared := make([]int, 0, len(parent.SharedPrefix)+len(parent.KVBlocks))
	shared = append(shared, parent.SharedPrefix...)
	shared = append(shared, parent.KVBlocks...)

	b := &Branch{
		ID:           bID,
		ParentBranch: parentID,
		Tokens:       append([]int32(nil), tokens...),
		NodeIDs:      nodeIDs,
		State:        ProposalProposed,
		KVBlocks:     append([]int(nil), kvBlocks...),
		SharedPrefix: shared,
	}
	bt.branches[bID] = b
	return b, nil
}

// ActiveBranch returns the currently active committed branch ID.
func (bt *BranchTree) ActiveBranch() BranchID {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	return bt.activeBranch
}

// PathTokens returns the full sequence of tokens from root to the given branch.
func (bt *BranchTree) PathTokens(id BranchID) ([]int32, error) {
	bt.mu.Lock()
	defer bt.mu.Unlock()

	var branchPath []*Branch
	curr := id
	for curr != -1 {
		b, ok := bt.branches[curr]
		if !ok {
			return nil, fmt.Errorf("%w: %d", ErrBranchNotFound, curr)
		}
		branchPath = append(branchPath, b)
		curr = b.ParentBranch
	}

	var totalTokens []int32
	// Traverse from root (end of slice) down to leaf
	for i := len(branchPath) - 1; i >= 0; i-- {
		totalTokens = append(totalTokens, branchPath[i].Tokens...)
	}
	return totalTokens, nil
}

// GetBranch retrieves a branch by ID.
func (bt *BranchTree) GetBranch(id BranchID) (*Branch, error) {
	bt.mu.Lock()
	defer bt.mu.Unlock()
	b, ok := bt.branches[id]
	if !ok {
		return nil, fmt.Errorf("%w: %d", ErrBranchNotFound, id)
	}
	return b, nil
}

// KVBlockReleaser abstracts KV block release and retain methods from cudakv.PagedCache or cudakv.BlockAllocator.
type KVBlockReleaser interface {
	Release(blocks []int) error
}

// CommitBranch marks the branch as accepted/committed and rolls back all competing
// sibling branches under the same parent. If releaser is supplied, private KV blocks
// of rejected branches are released, while shared prefix blocks remain held.
func (bt *BranchTree) CommitBranch(acceptedID BranchID, expectedTokens []int32, releaser KVBlockReleaser) error {
	bt.mu.Lock()
	defer bt.mu.Unlock()

	b, ok := bt.branches[acceptedID]
	if !ok {
		return fmt.Errorf("%w: %d", ErrBranchNotFound, acceptedID)
	}
	if b.State != ProposalProposed {
		return fmt.Errorf("%w: branch %d is already %s", ErrBranchNotActive, acceptedID, b.State)
	}

	// Verify token match if expectedTokens provided
	if expectedTokens != nil {
		if len(expectedTokens) != len(b.Tokens) {
			return fmt.Errorf("%w: expected %d tokens, branch has %d", ErrIncorrectTokenAccept, len(expectedTokens), len(b.Tokens))
		}
		for i, tok := range expectedTokens {
			if b.Tokens[i] != tok {
				return fmt.Errorf("%w: token at %d (%d != %d)", ErrIncorrectTokenAccept, i, b.Tokens[i], tok)
			}
		}
	}

	b.State = ProposalCommitted
	bt.activeBranch = acceptedID

	// Roll back competing sibling branches under the same parent
	for id, other := range bt.branches {
		if id != acceptedID && other.ParentBranch == b.ParentBranch && other.State == ProposalProposed {
			other.State = ProposalRolledBack
			if releaser != nil && len(other.KVBlocks) > 0 {
				if err := releaser.Release(other.KVBlocks); err != nil {
					return fmt.Errorf("cudaspec: release rejected branch %d KV: %w", id, err)
				}
				other.KVBlocks = nil
			}
		}
	}

	return nil
}

// RollbackBranch rolls back a branch and releases its private KV blocks.
func (bt *BranchTree) RollbackBranch(id BranchID, releaser KVBlockReleaser) error {
	bt.mu.Lock()
	defer bt.mu.Unlock()

	b, ok := bt.branches[id]
	if !ok {
		return fmt.Errorf("%w: %d", ErrBranchNotFound, id)
	}
	if b.State == ProposalRolledBack {
		return nil // idempotent
	}

	b.State = ProposalRolledBack
	if releaser != nil && len(b.KVBlocks) > 0 {
		if err := releaser.Release(b.KVBlocks); err != nil {
			return fmt.Errorf("cudaspec: rollback release branch %d KV: %w", id, err)
		}
		b.KVBlocks = nil
	}

	// Also recursively roll back any child branches
	for _, other := range bt.branches {
		if other.ParentBranch == id && other.State == ProposalProposed {
			other.State = ProposalRolledBack
			if releaser != nil && len(other.KVBlocks) > 0 {
				_ = releaser.Release(other.KVBlocks)
				other.KVBlocks = nil
			}
		}
	}

	return nil
}

// DenseReferenceMask computes the dense reference mask for prefix/tree/glue operations.
// Given numPrefixTokens prefix tokens and draftTree of N nodes, the resulting mask is
// [numPrefixTokens + N, numPrefixTokens + N] boolean matrix where:
// - Prefix tokens attend to earlier prefix tokens (causal).
// - Tree node i (offset by numPrefixTokens) attends to all prefix tokens.
// - Tree node i attends to tree node j iff node j is an ancestor of node i in the draft tree.
// - If fullyMaskedNodes contains a node index, its entire row is false (fully masked row).
func DenseReferenceMask(numPrefixTokens int, tree *DraftTree, fullyMaskedNodes []int) [][]bool {
	numNodes := 0
	if tree != nil {
		numNodes = len(tree.Nodes)
	}
	total := numPrefixTokens + numNodes
	mask := make([][]bool, total)
	for i := 0; i < total; i++ {
		mask[i] = make([]bool, total)
	}

	// 1. Prefix causal attention: row i attends to 0..i
	for i := 0; i < numPrefixTokens; i++ {
		for j := 0; j <= i; j++ {
			mask[i][j] = true
		}
	}

	// Set of fully masked nodes
	fullyMasked := make(map[int]bool)
	for _, fn := range fullyMaskedNodes {
		fullyMasked[fn] = true
	}

	// 2. Tree nodes attention:
	if tree != nil {
		treeMask := tree.BuildCausalTreeMask()
		for i := 0; i < numNodes; i++ {
			rowIdx := numPrefixTokens + i
			if fullyMasked[i] {
				// Fully masked row: contributes zero / all false
				continue
			}
			// Tree node attends to all prefix tokens
			for j := 0; j < numPrefixTokens; j++ {
				mask[rowIdx][j] = true
			}
			// Tree node attends to ancestors in the tree
			for j := 0; j < numNodes; j++ {
				if treeMask[i][j] {
					mask[rowIdx][numPrefixTokens+j] = true
				}
			}
		}
	}

	return mask
}

// SpeculationCacheKey binds weight digest, prefix tokens, sequence positions,
// numerical mode, and generation number to prevent stale or cross-branch reuse.
type SpeculationCacheKey struct {
	ModelDigest   [32]byte
	PrefixHash    [32]byte
	BranchID      BranchID
	StartPosition int
	NumTokens     int
	NumericalMode string // e.g. "FP32", "FP16", "BF16"
	Generation    uint64
}

// ComputeSpeculationKey constructs an authenticated SpeculationCacheKey.
func ComputeSpeculationKey(modelDigest []byte, prefixTokens []int32, branchID BranchID, startPos, numTokens int, numMode string, generation uint64) SpeculationCacheKey {
	var mDigest [32]byte
	if len(modelDigest) >= 32 {
		copy(mDigest[:], modelDigest[:32])
	} else {
		h := sha256.Sum256(modelDigest)
		copy(mDigest[:], h[:])
	}

	h := sha256.New()
	for _, tok := range prefixTokens {
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], uint32(tok))
		h.Write(buf[:])
	}
	var pHash [32]byte
	copy(pHash[:], h.Sum(nil))

	return SpeculationCacheKey{
		ModelDigest:   mDigest,
		PrefixHash:    pHash,
		BranchID:      branchID,
		StartPosition: startPos,
		NumTokens:     numTokens,
		NumericalMode: numMode,
		Generation:    generation,
	}
}

// SpeculationCacheEntry stores cached draft proposals or verification logits.
type SpeculationCacheEntry struct {
	Key        SpeculationCacheKey
	DraftPath  []int32
	Logits     [][]float32
	BonusToken int32
}

// SpeculationCache is a bounded cache for speculation outputs keyed by SpeculationCacheKey.
type SpeculationCache struct {
	mu       sync.RWMutex
	capacity int
	entries  map[SpeculationCacheKey]*SpeculationCacheEntry
	order    []SpeculationCacheKey
}

// NewSpeculationCache creates a bounded speculation cache with maximum entries capacity.
func NewSpeculationCache(capacity int) *SpeculationCache {
	if capacity <= 0 {
		capacity = 64
	}
	return &SpeculationCache{
		capacity: capacity,
		entries:  make(map[SpeculationCacheKey]*SpeculationCacheEntry),
		order:    make([]SpeculationCacheKey, 0, capacity),
	}
}

// Get retrieves an entry from the cache matching the exact key.
func (c *SpeculationCache) Get(key SpeculationCacheKey) (*SpeculationCacheEntry, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, ok := c.entries[key]
	return entry, ok
}

// Put adds or replaces an entry in the cache. Evicts oldest when capacity is reached.
func (c *SpeculationCache) Put(entry *SpeculationCacheEntry) {
	if entry == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	key := entry.Key
	if _, exists := c.entries[key]; !exists {
		if len(c.order) >= c.capacity {
			// Evict oldest
			oldest := c.order[0]
			c.order = c.order[1:]
			delete(c.entries, oldest)
		}
		c.order = append(c.order, key)
	}
	c.entries[key] = entry
}

// InvalidateGeneration purges entries matching a specific generation or earlier.
func (c *SpeculationCache) InvalidateGeneration(generation uint64) int {
	c.mu.Lock()
	defer c.mu.Unlock()

	purged := 0
	newOrder := make([]SpeculationCacheKey, 0, len(c.order))
	for _, k := range c.order {
		if k.Generation <= generation {
			delete(c.entries, k)
			purged++
		} else {
			newOrder = append(newOrder, k)
		}
	}
	c.order = newOrder
	return purged
}

// InvalidateModel purges all entries not matching the active modelDigest.
func (c *SpeculationCache) InvalidateModel(activeModelDigest [32]byte) int {
	c.mu.Lock()
	defer c.mu.Unlock()

	purged := 0
	newOrder := make([]SpeculationCacheKey, 0, len(c.order))
	for _, k := range c.order {
		if k.ModelDigest != activeModelDigest {
			delete(c.entries, k)
			purged++
		} else {
			newOrder = append(newOrder, k)
		}
	}
	c.order = newOrder
	return purged
}
