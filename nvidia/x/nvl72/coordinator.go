package nvl72

import (
	"errors"
	"fmt"
	"slices"
	"sync"
)

// Errors returned by [Coordinator].
var (
	ErrNoEligibleRank = errors.New("no active rank below the load cap")
	ErrRankNotActive  = errors.New("rank is not active")
	ErrNoSpareRank    = errors.New("no spare rank")
	ErrXIDNotFatal    = errors.New("xid does not take a rank out of service")
)

// fatalXIDs are the XID codes on which [Coordinator.FailoverRank] replaces
// a rank: 48 (double-bit ECC error), 79 (GPU fell off the bus), 119 (GSP
// RPC timeout) and 149 (NVLink fault).
var fatalXIDs = map[int]bool{48: true, 79: true, 119: true, 149: true}

// Coordinator routes prompts to prefill ranks by cached prefix and swaps
// faulted ranks for warm spares. Its methods are safe for concurrent use;
// callers must not modify the fields while methods run.
type Coordinator struct {
	ActiveRanks []int                       // ring / expert-parallel order
	SpareRanks  []int                       // warm spares, used in order
	RadixTrees  map[int]*RadixPageAllocator // per-rank KV prefix cache; nil means empty

	mu sync.Mutex
}

// MatchPrefix returns how many leading pages of tokenPages are cached in
// HBM or C2C memory, without pinning, promoting or touching LRU state.
// Matching stops at the first page whose KV is no longer resident, since
// later pages depend on it.
func (a *RadixPageAllocator) MatchPrefix(tokenPages [][]int32) (hbmPages, c2cPages int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if !a.initialized {
		return 0, 0
	}
	cur := a.root
	for _, page := range tokenPages {
		var next *radixNode
		for _, cand := range cur.children[hashTokenPage(cur.hash, page)] {
			if slices.Equal(cand.tokens, page) {
				next = cand
				break
			}
		}
		if next == nil {
			break
		}
		switch next.tier {
		case tierHBM:
			hbmPages++
		case tierC2C:
			c2cPages++
		default:
			return hbmPages, c2cPages
		}
		cur = next
	}
	return hbmPages, c2cPages
}

// paginate splits tokens into pages of size tokens; the last page may be
// short.
func paginate(tokens []int32, size int) [][]int32 {
	var pages [][]int32
	for len(tokens) > 0 {
		n := min(size, len(tokens))
		pages = append(pages, tokens[:n])
		tokens = tokens[n:]
	}
	return pages
}

// RoutePrompt picks the active rank whose cache holds the longest prefix
// of tokens among ranks with load[rank] < loadCap (loadCap <= 0 means no
// cap). Ties go to more HBM-resident tokens, then lower load, then lower
// rank id. It returns the rank and the number of prompt tokens whose KV it
// already holds.
func (c *Coordinator) RoutePrompt(tokens []int32, load map[int]int, loadCap int) (selectedRank, matchedTokens int, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(tokens) == 0 {
		return -1, 0, errors.New("route prompt: empty prompt")
	}
	best, bestMatch, bestHBM := -1, -1, -1
	for _, r := range c.ActiveRanks {
		if loadCap > 0 && load[r] >= loadCap {
			continue
		}
		match, hbm := 0, 0
		if a := c.RadixTrees[r]; a != nil {
			pages := paginate(tokens, a.PageSize)
			h, c2c := a.MatchPrefix(pages)
			for i := range h + c2c {
				match += len(pages[i])
			}
			for i := range h {
				hbm += len(pages[i])
			}
		}
		better := best < 0 || match > bestMatch ||
			match == bestMatch && (hbm > bestHBM ||
				hbm == bestHBM && (load[r] < load[best] || load[r] == load[best] && r < best))
		if better {
			best, bestMatch, bestHBM = r, match, hbm
		}
	}
	if best < 0 {
		return -1, 0, fmt.Errorf("route prompt: %w", ErrNoEligibleRank)
	}
	return best, bestMatch, nil
}

// FailoverRank replaces faultedRank, which reported xidCode, with the first
// spare rank. The spare takes the faulted rank's position in ActiveRanks,
// so ring and expert-parallel order are kept; the faulted rank's KV cache
// is dropped and no other rank's cache is touched. It returns a copy of the
// new active ranks and the replacement.
//
// Only XIDs 48, 79, 119 and 149 trigger a swap; other codes return
// [ErrXIDNotFatal] and change nothing. Replicating experts onto the
// replacement is left to the caller.
func (c *Coordinator) FailoverRank(faultedRank, xidCode int) (newActiveRanks []int, replacementRank int, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !fatalXIDs[xidCode] {
		return nil, -1, fmt.Errorf("failover rank %d: xid %d: %w", faultedRank, xidCode, ErrXIDNotFatal)
	}
	i := slices.Index(c.ActiveRanks, faultedRank)
	if i < 0 {
		return nil, -1, fmt.Errorf("failover rank %d: %w", faultedRank, ErrRankNotActive)
	}
	if len(c.SpareRanks) == 0 {
		return nil, -1, fmt.Errorf("failover rank %d: %w", faultedRank, ErrNoSpareRank)
	}
	repl := c.SpareRanks[0]
	if slices.Contains(c.ActiveRanks, repl) {
		return nil, -1, fmt.Errorf("failover rank %d: spare %d is already active", faultedRank, repl)
	}
	c.SpareRanks = c.SpareRanks[1:]
	c.ActiveRanks[i] = repl
	delete(c.RadixTrees, faultedRank)
	return slices.Clone(c.ActiveRanks), repl, nil
}
