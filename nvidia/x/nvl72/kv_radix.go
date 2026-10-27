package nvl72

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"slices"
	"sync"
)

type pageTier uint8

const (
	tierNone pageTier = iota
	tierHBM
	tierC2C
)

type radixNode struct {
	hash       uint64
	tokens     []int32
	children   map[uint64][]*radixNode
	tier       pageTier
	hbmPageID  int
	c2cPageID  int
	refCount   int
	lastAccess uint64
}

// AllocationReceipt reports the per-request Tier 1 HBM3e and Tier 2 Grace
// LPDDR5X NVLink-C2C page allocation and promotion metrics from
// [RadixPageAllocator.MatchOrAllocatePrefix].
type AllocationReceipt struct {
	ReqID                  string
	TotalPages             int
	HBMHitPages            int
	C2CPromotedPages       int
	NewlyAllocatedPages    int
	EvictedToC2CPages      int
	C2CPromoteBytes        int64
	C2CPromoteLatencyUs    float64
	PageIDs                []int
	PrefillTokensSaved     int
	PrefillTokensToCompute int
}

// RadixPageAllocator manages a two-tier prefix-tree (radix) paged KV/MLA cache
// pool on a GB200/GB300 node:
//   - Tier 1 (Hot HBM3e): HBMPageCapacity pages of PageSize tokens,
//   - Tier 2 (Warm Grace LPDDR5X via NVLink-C2C at 450 GB/s): C2CPageCapacity pages.
//
// Zero-refcount pages evicted from Tier 1 HBM3e are demoted to Tier 2 Grace
// LPDDR5X over NVLink-C2C rather than discarded, and are promoted back to
// Tier 1 HBM3e on subsequent prefix hits without recomputing prefill attention.
type RadixPageAllocator struct {
	PageSize        int // tokens per page (defaults to 16)
	PageBytes       int // bytes per page across layers (defaults to PageSize * 640)
	HBMPageCapacity int // maximum pages resident in Tier 1 HBM3e
	C2CPageCapacity int // maximum pages resident in Tier 2 Grace LPDDR5X C2C

	mu          sync.Mutex
	initialized bool
	root        *radixNode
	allNodes    []*radixNode
	reqPages    map[string][]*radixNode
	freeHBMIDs  []int
	freeC2CIDs  []int
	hbmResident int
	c2cResident int
	clock       uint64
}

// NewRadixPageAllocator constructs a two-tier HBM3e + Grace NVLink-C2C prefix
// page allocator.
func NewRadixPageAllocator(pageSize, pageBytes, hbmPageCapacity, c2cPageCapacity int) (*RadixPageAllocator, error) {
	if pageSize <= 0 {
		return nil, fmt.Errorf("nvl72 radix allocator: PageSize %d must be > 0", pageSize)
	}
	if pageBytes <= 0 {
		return nil, fmt.Errorf("nvl72 radix allocator: PageBytes %d must be > 0", pageBytes)
	}
	if hbmPageCapacity <= 0 {
		return nil, fmt.Errorf("nvl72 radix allocator: HBMPageCapacity %d must be > 0", hbmPageCapacity)
	}
	if c2cPageCapacity < 0 {
		return nil, fmt.Errorf("nvl72 radix allocator: C2CPageCapacity %d must be >= 0", c2cPageCapacity)
	}
	a := &RadixPageAllocator{
		PageSize:        pageSize,
		PageBytes:       pageBytes,
		HBMPageCapacity: hbmPageCapacity,
		C2CPageCapacity: c2cPageCapacity,
	}
	if err := a.initLocked(); err != nil {
		return nil, err
	}
	return a, nil
}

func (a *RadixPageAllocator) initLocked() error {
	if a.PageSize <= 0 {
		a.PageSize = 16
	}
	if a.PageBytes <= 0 {
		a.PageBytes = a.PageSize * 640
	}
	if a.HBMPageCapacity <= 0 {
		return fmt.Errorf("nvl72 radix allocator: HBMPageCapacity %d must be > 0", a.HBMPageCapacity)
	}
	if a.C2CPageCapacity < 0 {
		return fmt.Errorf("nvl72 radix allocator: C2CPageCapacity %d must be >= 0", a.C2CPageCapacity)
	}
	if a.initialized {
		return nil
	}
	a.root = &radixNode{
		children:  make(map[uint64][]*radixNode),
		hbmPageID: -1,
		c2cPageID: -1,
	}
	a.reqPages = make(map[string][]*radixNode)
	a.freeHBMIDs = make([]int, a.HBMPageCapacity)
	for i := range a.HBMPageCapacity {
		a.freeHBMIDs[i] = a.HBMPageCapacity - 1 - i
	}
	a.freeC2CIDs = make([]int, a.C2CPageCapacity)
	for i := range a.C2CPageCapacity {
		a.freeC2CIDs[i] = a.C2CPageCapacity - 1 - i
	}
	a.initialized = true
	return nil
}

func hashTokenPage(parentHash uint64, tokens []int32) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], parentHash)
	_, _ = h.Write(buf[:])
	for _, tok := range tokens {
		binary.LittleEndian.PutUint32(buf[:4], uint32(tok))
		_, _ = h.Write(buf[:4])
	}
	return h.Sum64()
}

func (a *RadixPageAllocator) popHBMID() int {
	n := len(a.freeHBMIDs)
	id := a.freeHBMIDs[n-1]
	a.freeHBMIDs = a.freeHBMIDs[:n-1]
	return id
}

func (a *RadixPageAllocator) pushHBMID(id int) {
	if id >= 0 {
		a.freeHBMIDs = append(a.freeHBMIDs, id)
	}
}

func (a *RadixPageAllocator) popC2CID() int {
	n := len(a.freeC2CIDs)
	if n == 0 {
		return -1
	}
	id := a.freeC2CIDs[n-1]
	a.freeC2CIDs = a.freeC2CIDs[:n-1]
	return id
}

func (a *RadixPageAllocator) pushC2CID(id int) {
	if id >= 0 {
		a.freeC2CIDs = append(a.freeC2CIDs, id)
	}
}

// evictOneHBMToC2CLocked selects the least-recently-used zero-refcount HBM3e
// page and demotes it to Tier 2 Grace LPDDR5X C2C (or discards it if
// C2CPageCapacity == 0).
func (a *RadixPageAllocator) evictOneHBMToC2CLocked() (evictedToC2C bool, err error) {
	var victim *radixNode
	for _, n := range a.allNodes {
		if n.tier == tierHBM && n.refCount == 0 {
			if victim == nil || n.lastAccess < victim.lastAccess {
				victim = n
			}
		}
	}
	if victim == nil {
		return false, fmt.Errorf("nvl72 radix allocator: Tier 1 HBM3e full (%d pages) and no zero-refcount page available for C2C demotion", a.HBMPageCapacity)
	}

	a.pushHBMID(victim.hbmPageID)
	victim.hbmPageID = -1
	a.hbmResident--

	if a.C2CPageCapacity == 0 {
		victim.tier = tierNone
		return false, nil
	}

	if a.c2cResident >= a.C2CPageCapacity {
		var c2cVictim *radixNode
		for _, n := range a.allNodes {
			if n.tier == tierC2C {
				if c2cVictim == nil || n.lastAccess < c2cVictim.lastAccess {
					c2cVictim = n
				}
			}
		}
		if c2cVictim != nil {
			a.pushC2CID(c2cVictim.c2cPageID)
			c2cVictim.c2cPageID = -1
			c2cVictim.tier = tierNone
			a.c2cResident--
		}
	}

	victim.tier = tierC2C
	victim.c2cPageID = a.popC2CID()
	a.c2cResident++
	return true, nil
}

// MatchOrAllocatePrefix walks the radix tree for tokenPages, reusing Tier 1
// HBM3e prefix pages, promoting Tier 2 Grace LPDDR5X C2C prefix pages over
// NVLink-C2C (450 GB/s), and allocating new Tier 1 HBM3e pages for uncached
// suffix pages (demoting LRU zero-refcount HBM3e pages to Tier 2 C2C when full).
func (a *RadixPageAllocator) MatchOrAllocatePrefix(reqID string, tokenPages [][]int32) (AllocationReceipt, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if err := a.initLocked(); err != nil {
		return AllocationReceipt{}, err
	}
	if reqID == "" {
		return AllocationReceipt{}, fmt.Errorf("nvl72 radix allocator: empty reqID")
	}
	if _, exists := a.reqPages[reqID]; exists {
		return AllocationReceipt{}, fmt.Errorf("nvl72 radix allocator: request %q already active", reqID)
	}
	if len(tokenPages) == 0 {
		return AllocationReceipt{}, fmt.Errorf("nvl72 radix allocator: request %q has 0 token pages", reqID)
	}
	if len(tokenPages) > a.HBMPageCapacity {
		return AllocationReceipt{}, fmt.Errorf("nvl72 radix allocator: request %q needs %d pages, exceeds HBMPageCapacity %d",
			reqID, len(tokenPages), a.HBMPageCapacity)
	}

	// Count available HBM slots (free slots + evictable zero-refcount HBM pages)
	// before mutating state so a capacity error leaves the allocator clean.
	evictableHBM := 0
	for _, n := range a.allNodes {
		if n.tier == tierHBM && n.refCount == 0 {
			evictableHBM++
		}
	}
	freeHBM := a.HBMPageCapacity - a.hbmResident
	needHBM := 0
	cur := a.root
	for _, page := range tokenPages {
		var matched *radixNode
		if cur != nil {
			h := hashTokenPage(cur.hash, page)
			for _, cand := range cur.children[h] {
				if slices.Equal(cand.tokens, page) {
					matched = cand
					break
				}
			}
		}
		if matched == nil || matched.tier != tierHBM {
			needHBM++
		} else if matched.refCount == 0 {
			// Pinning a zero-refcount HBM hit removes it from the evictable pool.
			evictableHBM--
		}
		cur = matched
	}
	if needHBM > freeHBM+evictableHBM {
		return AllocationReceipt{}, fmt.Errorf("nvl72 radix allocator: request %q needs %d HBM pages, only %d free + %d evictable",
			reqID, needHBM, freeHBM, evictableHBM)
	}

	receipt := AllocationReceipt{
		ReqID:      reqID,
		TotalPages: len(tokenPages),
		PageIDs:    make([]int, len(tokenPages)),
	}
	pinned := make([]*radixNode, 0, len(tokenPages))
	cur = a.root

	for i, page := range tokenPages {
		if len(page) == 0 {
			return AllocationReceipt{}, fmt.Errorf("nvl72 radix allocator: request %q page %d is empty", reqID, i)
		}
		h := hashTokenPage(cur.hash, page)
		var child *radixNode
		for _, cand := range cur.children[h] {
			if slices.Equal(cand.tokens, page) {
				child = cand
				break
			}
		}
		if child == nil {
			tokCopy := make([]int32, len(page))
			copy(tokCopy, page)
			child = &radixNode{
				hash:      h,
				tokens:    tokCopy,
				children:  make(map[uint64][]*radixNode),
				tier:      tierNone,
				hbmPageID: -1,
				c2cPageID: -1,
			}
			cur.children[h] = append(cur.children[h], child)
			a.allNodes = append(a.allNodes, child)
		}

		a.clock++
		switch child.tier {
		case tierHBM:
			// Tier 1 HBM3e cache hit: zero compute, zero C2C transfer.
			child.refCount++
			child.lastAccess = a.clock
			receipt.HBMHitPages++

		case tierC2C:
			// Tier 2 Grace LPDDR5X C2C hit: promote back to Tier 1 HBM3e over NVLink-C2C (450 GB/s).
			a.pushC2CID(child.c2cPageID)
			child.c2cPageID = -1
			child.tier = tierNone
			a.c2cResident--

			if a.hbmResident >= a.HBMPageCapacity {
				demoted, err := a.evictOneHBMToC2CLocked()
				if err != nil {
					return AllocationReceipt{}, err
				}
				if demoted {
					receipt.EvictedToC2CPages++
				}
			}
			child.tier = tierHBM
			child.hbmPageID = a.popHBMID()
			child.refCount++
			child.lastAccess = a.clock
			a.hbmResident++
			receipt.C2CPromotedPages++

		default:
			// Uncached page: allocate fresh Tier 1 HBM3e page (demoting LRU HBM3e page to C2C if full).
			if a.hbmResident >= a.HBMPageCapacity {
				demoted, err := a.evictOneHBMToC2CLocked()
				if err != nil {
					return AllocationReceipt{}, err
				}
				if demoted {
					receipt.EvictedToC2CPages++
				}
			}
			child.tier = tierHBM
			child.hbmPageID = a.popHBMID()
			child.refCount++
			child.lastAccess = a.clock
			a.hbmResident++
			receipt.NewlyAllocatedPages++
		}

		receipt.PageIDs[i] = child.hbmPageID
		pinned = append(pinned, child)
		cur = child
	}

	a.reqPages[reqID] = pinned
	receipt.C2CPromoteBytes = int64(receipt.C2CPromotedPages) * int64(a.PageBytes)
	// NVL72GraceC2CGBps is 450 GB/s = 450,000 bytes/us.
	receipt.C2CPromoteLatencyUs = float64(receipt.C2CPromoteBytes) / (NVL72GraceC2CGBps * 1e3)
	receipt.PrefillTokensSaved = (receipt.HBMHitPages + receipt.C2CPromotedPages) * a.PageSize
	receipt.PrefillTokensToCompute = receipt.NewlyAllocatedPages * a.PageSize
	return receipt, nil
}

// ReleaseRequest decrements reference counts on the pages held by reqID so
// they become eligible for LRU demotion to Tier 2 Grace LPDDR5X C2C while
// remaining in the radix tree for future prefix hits.
func (a *RadixPageAllocator) ReleaseRequest(reqID string) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if err := a.initLocked(); err != nil {
		return err
	}
	pages, ok := a.reqPages[reqID]
	if !ok {
		return fmt.Errorf("nvl72 radix allocator: unknown or already released request %q", reqID)
	}
	for _, n := range pages {
		if n.refCount > 0 {
			n.refCount--
		}
	}
	delete(a.reqPages, reqID)
	return nil
}

// HBMUsedPages returns the number of pages currently resident in Tier 1 HBM3e.
func (a *RadixPageAllocator) HBMUsedPages() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.hbmResident
}

// C2CUsedPages returns the number of pages currently resident in Tier 2 Grace
// LPDDR5X C2C memory.
func (a *RadixPageAllocator) C2CUsedPages() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.c2cResident
}

// ChunkedBatchRequest describes a request participating in continuous batching
// with chunked prefill and prefix-cache token savings.
type ChunkedBatchRequest struct {
	ReqID              string
	TotalPromptTokens  int
	CachedPrefixTokens int
	IsDecode           bool
}

// ChunkedBatchStep describes one iteration step of a continuous batching
// schedule, combining 1-token active decodes with token-budgeted prefill chunks.
type ChunkedBatchStep struct {
	StepIndex              int
	ScheduledDecodes       []string
	ScheduledPrefillChunks map[string]int
	TotalTokensInStep      int
}

// PlanChunkedPrefillBatch schedules active decode requests (1 token each per
// step to protect TPOT SLA) and partitions the remaining maxTokenBudgetPerStep
// across waiting prefill requests' uncached tokens (TotalPromptTokens - CachedPrefixTokens)
// until all prefills complete.
func PlanChunkedPrefillBatch(requests []ChunkedBatchRequest, maxTokenBudgetPerStep int) ([]ChunkedBatchStep, error) {
	if maxTokenBudgetPerStep <= 0 {
		return nil, fmt.Errorf("nvl72 chunked prefill: maxTokenBudgetPerStep %d must be > 0", maxTokenBudgetPerStep)
	}
	if len(requests) == 0 {
		return nil, nil
	}

	var decodeIDs []string
	type prefillState struct {
		reqID     string
		remaining int
	}
	var prefills []prefillState
	totalRemainingPrefill := 0

	for i, r := range requests {
		if r.ReqID == "" {
			return nil, fmt.Errorf("nvl72 chunked prefill: request %d has empty ReqID", i)
		}
		if r.IsDecode {
			decodeIDs = append(decodeIDs, r.ReqID)
			continue
		}
		if r.TotalPromptTokens < 0 || r.CachedPrefixTokens < 0 || r.CachedPrefixTokens > r.TotalPromptTokens {
			return nil, fmt.Errorf("nvl72 chunked prefill: request %q invalid prompt/cached tokens (%d, %d)",
				r.ReqID, r.TotalPromptTokens, r.CachedPrefixTokens)
		}
		uncached := r.TotalPromptTokens - r.CachedPrefixTokens
		prefills = append(prefills, prefillState{
			reqID:     r.ReqID,
			remaining: uncached,
		})
		totalRemainingPrefill += uncached
	}

	if len(decodeIDs) > maxTokenBudgetPerStep {
		return nil, fmt.Errorf("nvl72 chunked prefill: %d active decode requests exceed maxTokenBudgetPerStep %d",
			len(decodeIDs), maxTokenBudgetPerStep)
	}
	if totalRemainingPrefill > 0 && len(decodeIDs) == maxTokenBudgetPerStep {
		return nil, fmt.Errorf("nvl72 chunked prefill: zero token budget remaining for %d uncached prefill tokens (decodes=%d, budget=%d)",
			totalRemainingPrefill, len(decodeIDs), maxTokenBudgetPerStep)
	}

	// If there are no uncached prefill tokens to schedule, emit a single decode step.
	if totalRemainingPrefill == 0 {
		decodesCopy := make([]string, len(decodeIDs))
		copy(decodesCopy, decodeIDs)
		return []ChunkedBatchStep{
			{
				StepIndex:              0,
				ScheduledDecodes:       decodesCopy,
				ScheduledPrefillChunks: make(map[string]int),
				TotalTokensInStep:      len(decodesCopy),
			},
		}, nil
	}

	var steps []ChunkedBatchStep
	for stepIdx := 0; totalRemainingPrefill > 0; stepIdx++ {
		decodesCopy := make([]string, len(decodeIDs))
		copy(decodesCopy, decodeIDs)
		budgetLeft := maxTokenBudgetPerStep - len(decodesCopy)
		chunks := make(map[string]int)
		prefillTokensInStep := 0

		for i := range prefills {
			if budgetLeft == 0 {
				break
			}
			if prefills[i].remaining == 0 {
				continue
			}
			take := min(prefills[i].remaining, budgetLeft)
			chunks[prefills[i].reqID] += take
			prefills[i].remaining -= take
			totalRemainingPrefill -= take
			budgetLeft -= take
			prefillTokensInStep += take
		}

		steps = append(steps, ChunkedBatchStep{
			StepIndex:              stepIdx,
			ScheduledDecodes:       decodesCopy,
			ScheduledPrefillChunks: chunks,
			TotalTokensInStep:      len(decodesCopy) + prefillTokensInStep,
		})
	}

	return steps, nil
}
