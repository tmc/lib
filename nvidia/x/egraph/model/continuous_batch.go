package model

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
)

// SequencePhase identifies the execution regime of an in-flight sequence in
// [ContinuousBatchScheduler].
type SequencePhase string

const (
	// PhaseChunkedPrefill indicates a sequence processing a slice of its prompt
	// tokens within the step budget.
	PhaseChunkedPrefill SequencePhase = "chunked_prefill"

	// PhaseDecode indicates a sequence generating a single autoregressive token.
	PhaseDecode SequencePhase = "decode"

	// PhaseSpeculativeVerify indicates a sequence verifying a speculative draft
	// token tree alongside its root/bonus token.
	PhaseSpeculativeVerify SequencePhase = "speculative_verify"
)

// RadixNode is a prefix-tree node in [RadixKVTree] mapping a token slice edge
// to shared physical KV page IDs.
type RadixNode struct {
	Tokens     []int
	PageIDs    []int
	PrefixHash [32]byte
	RefCount   int
	LastAccess uint64
	Parent     *RadixNode
	Children   map[int]*RadixNode
}

// RadixKVTree is a reference-counted prefix-hash radix tree that maps token
// prefixes to shared physical KV cache page IDs and evicts zero-reference leaf
// nodes in LRU order when physical pages are needed.
type RadixKVTree struct {
	mu         sync.Mutex
	pageSize   int
	totalPages int
	freePages  []int
	root       *RadixNode
	clock      uint64
	cacheHits  int
}

// NewRadixKVTree creates a [RadixKVTree] managing totalPages physical KV pages
// with pageSize tokens per page.
func NewRadixKVTree(totalPages, pageSize int) (*RadixKVTree, error) {
	if totalPages <= 0 {
		return nil, fmt.Errorf("radix kv tree: totalPages %d <= 0", totalPages)
	}
	if pageSize <= 0 {
		pageSize = 16
	}
	free := make([]int, totalPages)
	for i := range totalPages {
		free[i] = i
	}
	return &RadixKVTree{
		pageSize:   pageSize,
		totalPages: totalPages,
		freePages:  free,
		root: &RadixNode{
			Children: make(map[int]*RadixNode),
		},
	}, nil
}

// PrefixHash computes a deterministic SHA-256 digest over a token prefix.
func PrefixHash(tokens []int) [32]byte {
	h := sha256.New()
	var buf [8]byte
	for _, tok := range tokens {
		binary.LittleEndian.PutUint64(buf[:], uint64(tok))
		_, _ = h.Write(buf[:])
	}
	var sum [32]byte
	copy(sum[:], h.Sum(nil))
	return sum
}

// PageSize returns the number of tokens stored per physical KV page.
func (t *RadixKVTree) PageSize() int {
	if t == nil {
		return 0
	}
	return t.pageSize
}

// FreePages returns the number of unallocated physical KV pages in the pool.
func (t *RadixKVTree) FreePages() int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.freePages)
}

// UsedPages returns the number of allocated physical KV pages currently held by
// radix tree nodes.
func (t *RadixKVTree) UsedPages() int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.totalPages - len(t.freePages)
}

// CacheHits returns the cumulative count of reused prefix KV pages across all
// [RadixKVTree.InsertPrefix] calls.
func (t *RadixKVTree) CacheHits() int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.cacheHits
}

// MatchPrefix walks the radix tree for tokens, splitting an internal edge if a
// partial match occurs at a page-aligned boundary, and returns the number of
// matched prefix tokens, the shared physical KV page IDs, and the deepest
// matched [RadixNode].
func (t *RadixKVTree) MatchPrefix(tokens []int) (matchedTokens int, pageIDs []int, node *RadixNode) {
	if t == nil || len(tokens) == 0 {
		return 0, nil, nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	aligned := (len(tokens) / t.pageSize) * t.pageSize
	if aligned == 0 {
		aligned = len(tokens)
	}
	return t.matchPrefixLocked(tokens[:aligned])
}

func (t *RadixKVTree) matchPrefixLocked(tokens []int) (int, []int, *RadixNode) {
	cur := t.root
	t.clock++
	cur.LastAccess = t.clock

	var (
		matched int
		pages   []int
		prefix  []int
	)
	rem := tokens
	for len(rem) > 0 {
		child, ok := cur.Children[rem[0]]
		if !ok {
			break
		}
		common := commonPrefixLen(rem, child.Tokens)
		if common > t.pageSize {
			common = (common / t.pageSize) * t.pageSize
		}
		if common == 0 {
			break
		}
		if common < len(child.Tokens) {
			child = t.splitNodeLocked(cur, child, common)
		}
		t.clock++
		child.LastAccess = t.clock
		prefix = append(prefix, child.Tokens...)
		pages = append(pages, child.PageIDs...)
		matched += len(child.Tokens)
		rem = rem[len(child.Tokens):]
		cur = child
	}
	return matched, pages, cur
}

func (t *RadixKVTree) splitNodeLocked(parent, child *RadixNode, splitToks int) *RadixNode {
	splitPages := (splitToks + t.pageSize - 1) / t.pageSize
	if splitPages > len(child.PageIDs) {
		splitPages = len(child.PageIDs)
	}

	midTokens := slices.Clone(child.Tokens[:splitToks])
	midPages := slices.Clone(child.PageIDs[:splitPages])

	var pathToks []int
	for p := parent; p != nil && p != t.root; p = p.Parent {
		pathToks = append(slices.Clone(p.Tokens), pathToks...)
	}
	pathToks = append(pathToks, midTokens...)

	mid := &RadixNode{
		Tokens:     midTokens,
		PageIDs:    midPages,
		PrefixHash: PrefixHash(pathToks),
		RefCount:   child.RefCount,
		LastAccess: child.LastAccess,
		Parent:     parent,
		Children:   make(map[int]*RadixNode, 1),
	}

	child.Tokens = slices.Clone(child.Tokens[splitToks:])
	child.PageIDs = slices.Clone(child.PageIDs[splitPages:])
	child.Parent = mid
	mid.Children[child.Tokens[0]] = child
	parent.Children[mid.Tokens[0]] = mid
	return mid
}

// InsertPrefix matches the longest cached prefix of tokens in t, allocates
// physical KV pages for any uncached suffix (evicting zero-reference leaves in
// LRU order when needed), increments the reference count along the matched/new
// path, and returns all physical KV page IDs, the number of cached pages hit,
// and the terminal [RadixNode].
func (t *RadixKVTree) InsertPrefix(tokens []int) (pageIDs []int, cachedPages int, node *RadixNode, err error) {
	if t == nil {
		return nil, 0, nil, errors.New("radix kv tree: nil tree")
	}
	if len(tokens) == 0 {
		return nil, 0, t.root, nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	matched, hitPages, cur := t.matchPrefixLocked(tokens)
	cachedPages = len(hitPages)
	t.cacheHits += cachedPages

	rem := tokens[matched:]
	allPages := slices.Clone(hitPages)
	if len(rem) > 0 {
		neededPages := (len(rem) + t.pageSize - 1) / t.pageSize
		allocPages, allocErr := t.allocPagesLocked(neededPages)
		if allocErr != nil {
			return nil, 0, nil, allocErr
		}
		leaf := &RadixNode{
			Tokens:     slices.Clone(rem),
			PageIDs:    allocPages,
			PrefixHash: PrefixHash(tokens),
			LastAccess: t.clock,
			Parent:     cur,
			Children:   make(map[int]*RadixNode),
		}
		cur.Children[rem[0]] = leaf
		cur = leaf
		allPages = append(allPages, allocPages...)
	}

	t.retainLocked(cur)
	return allPages, cachedPages, cur, nil
}

// Retain increments the reference count for node and all of its ancestors.
func (t *RadixKVTree) Retain(node *RadixNode) {
	if t == nil || node == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.retainLocked(node)
}

func (t *RadixKVTree) retainLocked(node *RadixNode) {
	t.clock++
	for p := node; p != nil && p != t.root; p = p.Parent {
		p.RefCount++
		p.LastAccess = t.clock
	}
}

// Release decrements the reference count for node and all of its ancestors,
// making zero-reference leaves eligible for future LRU eviction.
func (t *RadixKVTree) Release(node *RadixNode) {
	if t == nil || node == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.clock++
	for p := node; p != nil && p != t.root; p = p.Parent {
		if p.RefCount > 0 {
			p.RefCount--
		}
		p.LastAccess = t.clock
	}
}

// EvictLRU evicts zero-reference leaf nodes in least-recently-used order until
// at least neededPages physical KV pages have been returned to the free pool.
func (t *RadixKVTree) EvictLRU(neededPages int) (int, error) {
	if t == nil {
		return 0, errors.New("radix kv tree: nil tree")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.evictLRULocked(neededPages)
}

func (t *RadixKVTree) allocPagesLocked(needed int) ([]int, error) {
	if needed <= 0 {
		return nil, nil
	}
	if len(t.freePages) < needed {
		deficit := needed - len(t.freePages)
		if _, err := t.evictLRULocked(deficit); err != nil {
			return nil, err
		}
	}
	if len(t.freePages) < needed {
		return nil, fmt.Errorf("radix kv tree: out of physical KV pages: need %d, have %d free", needed, len(t.freePages))
	}
	out := slices.Clone(t.freePages[:needed])
	t.freePages = t.freePages[needed:]
	return out, nil
}

func (t *RadixKVTree) evictLRULocked(neededPages int) (int, error) {
	freed := 0
	for freed < neededPages {
		victim := t.findLRUZeroRefLeafLocked(t.root)
		if victim == nil {
			break
		}
		freed += len(victim.PageIDs)
		t.freePages = append(t.freePages, victim.PageIDs...)
		if victim.Parent != nil && len(victim.Tokens) > 0 {
			delete(victim.Parent.Children, victim.Tokens[0])
		}
		victim.Parent = nil
	}
	if freed < neededPages {
		return freed, fmt.Errorf("radix kv tree: evicted %d pages, still short of %d needed (active pinned refs prevent eviction)", freed, neededPages)
	}
	return freed, nil
}

func (t *RadixKVTree) findLRUZeroRefLeafLocked(cur *RadixNode) *RadixNode {
	if cur == nil {
		return nil
	}
	var best *RadixNode
	for _, child := range cur.Children {
		if len(child.Children) == 0 {
			if child.RefCount == 0 {
				if best == nil || child.LastAccess < best.LastAccess {
					best = child
				}
			}
			continue
		}
		if cand := t.findLRUZeroRefLeafLocked(child); cand != nil {
			if best == nil || cand.LastAccess < best.LastAccess {
				best = cand
			}
		}
	}
	return best
}

func commonPrefixLen(a, b []int) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// InFlightSequence describes a single active request managed by
// [ContinuousBatchScheduler].
type InFlightSequence struct {
	ID             string
	Phase          SequencePhase
	PromptTokens   []int
	PrefillOffset  int
	ChunkBudget    int
	DecodeToken    int
	DraftTokens    []int
	AcceptanceRate float64
	PageIDs        []int
	CachedPages    int
	RadixNode      *RadixNode
}

// PackedRow represents one token slot inside a scheduled megakernel batch step.
// Padded rows (`RowIndex >= ActiveTokens`) have `Valid == false`,
// `WriteKV == false`, and `KVPageID == -1` so padded rows never corrupt
// physical KV cache pages.
type PackedRow struct {
	RowIndex int
	Valid    bool
	SeqID    string
	Phase    SequencePhase
	Token    int
	TokenPos int
	KVPageID int
	WriteKV  bool
}

// ScheduledBatchStep records the packed token rows, active-row bitmask, padded
// bucket selection, and megakernel table entry for one continuous-batching step.
type ScheduledBatchStep struct {
	ActiveTokens    int
	BucketM         int
	PaddedRows      int
	ActiveRowMask   uint64
	Rows            []PackedRow
	BucketEntry     *specdecode.BucketMegakernelEntry
	Transition      specdecode.BucketTransition
	PrefixCacheHits int
}

// RowIsActive reports whether row idx is masked active in s.ActiveRowMask.
func (s *ScheduledBatchStep) RowIsActive(idx int) bool {
	if s == nil || idx < 0 || idx >= 64 {
		return false
	}
	return (s.ActiveRowMask & (uint64(1) << uint(idx))) != 0
}

// ContinuousBatchScheduler packs heterogeneous in-flight sequences (chunked
// prefill, single-token decode, and speculative draft trees) into an active
// token step of size M_active, routes M_active to the minimal padded bucket
// B in {1, 2, 4, 8} (B >= M_active) in a [specdecode.MultiBucketMegakernelTable],
// and masks padded rows so they cannot write to physical KV pages.
type ContinuousBatchScheduler struct {
	mu            sync.Mutex
	tree          *RadixKVTree
	table         *specdecode.MultiBucketMegakernelTable
	maxStepTokens int
	sequences     []*InFlightSequence
}

// NewContinuousBatchScheduler constructs a [ContinuousBatchScheduler] backed by
// tree and table. When maxStepTokens <= 0, it defaults to the largest bucket M
// in table (typically 8).
func NewContinuousBatchScheduler(
	tree *RadixKVTree,
	table *specdecode.MultiBucketMegakernelTable,
	maxStepTokens int,
) (*ContinuousBatchScheduler, error) {
	if tree == nil {
		return nil, errors.New("continuous batch scheduler: nil RadixKVTree")
	}
	if table == nil {
		return nil, errors.New("continuous batch scheduler: nil MultiBucketMegakernelTable")
	}
	if maxStepTokens <= 0 {
		for _, e := range table.Entries() {
			if e.Spec.BucketM > maxStepTokens {
				maxStepTokens = e.Spec.BucketM
			}
		}
		if maxStepTokens <= 0 {
			maxStepTokens = 8
		}
	}
	return &ContinuousBatchScheduler{
		tree:          tree,
		table:         table,
		maxStepTokens: maxStepTokens,
	}, nil
}

// Admit registers seq in the scheduler, resolving its prompt tokens through the
// [RadixKVTree] to reuse shared prefix KV pages and allocate pages for any
// uncached suffix.
func (s *ContinuousBatchScheduler) Admit(seq *InFlightSequence) error {
	if s == nil {
		return errors.New("continuous batch scheduler: nil scheduler")
	}
	if seq == nil || seq.ID == "" {
		return errors.New("continuous batch scheduler: sequence ID required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, existing := range s.sequences {
		if existing.ID == seq.ID {
			return fmt.Errorf("continuous batch scheduler: duplicate sequence ID %q", seq.ID)
		}
	}

	if len(seq.PromptTokens) > 0 {
		pages, hits, node, err := s.tree.InsertPrefix(seq.PromptTokens)
		if err != nil {
			return fmt.Errorf("continuous batch scheduler admit %s: %w", seq.ID, err)
		}
		seq.PageIDs = pages
		seq.CachedPages = hits
		seq.RadixNode = node
		if seq.Phase == PhaseChunkedPrefill && seq.PrefillOffset == 0 && hits > 0 {
			cachedToks := min(hits*s.tree.PageSize(), len(seq.PromptTokens))
			if cachedToks < len(seq.PromptTokens) {
				seq.PrefillOffset = cachedToks
			}
		}
	}
	if seq.Phase == "" {
		if seq.PrefillOffset < len(seq.PromptTokens) {
			seq.Phase = PhaseChunkedPrefill
		} else if len(seq.DraftTokens) > 0 {
			seq.Phase = PhaseSpeculativeVerify
		} else {
			seq.Phase = PhaseDecode
		}
	}
	s.sequences = append(s.sequences, seq)
	return nil
}

// Complete finishes seqID, releasing its pinned [RadixNode] reference count so
// its KV pages can be shared by future prefixes or evicted under memory pressure.
func (s *ContinuousBatchScheduler) Complete(seqID string) error {
	if s == nil {
		return errors.New("continuous batch scheduler: nil scheduler")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	for i, seq := range s.sequences {
		if seq.ID == seqID {
			if seq.RadixNode != nil {
				s.tree.Release(seq.RadixNode)
				seq.RadixNode = nil
			}
			s.sequences = append(s.sequences[:i], s.sequences[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("continuous batch scheduler: sequence %q not found", seqID)
}

// SelectMinimalBucket returns the smallest bucket B in {1, 2, 4, 8} (or from
// the table's configured buckets) satisfying B >= activeM.
func (s *ContinuousBatchScheduler) SelectMinimalBucket(activeM int) (*specdecode.BucketMegakernelEntry, int, error) {
	if s == nil || s.table == nil {
		return nil, 0, errors.New("continuous batch scheduler: nil table")
	}
	if activeM <= 0 {
		return nil, 0, fmt.Errorf("continuous batch scheduler: activeM %d <= 0", activeM)
	}
	entries := s.table.Entries()
	if len(entries) == 0 {
		return nil, 0, errors.New("continuous batch scheduler: multi-bucket table has no entries")
	}
	for _, e := range entries {
		if e.Spec.BucketM >= activeM {
			return e, e.Spec.BucketM, nil
		}
	}
	last := entries[len(entries)-1]
	if activeM > last.Spec.BucketM {
		return nil, 0, fmt.Errorf("continuous batch scheduler: activeM %d exceeds max bucket M=%d", activeM, last.Spec.BucketM)
	}
	return last, last.Spec.BucketM, nil
}

// ScheduleStep packs active sequences into an execution step of size M_active,
// routes M_active through the [specdecode.MultiBucketMegakernelTable] to select
// the minimal padded bucket B >= M_active, and populates the active-row mask
// so padded rows [M_active, B) have WriteKV=false and KVPageID=-1.
func (s *ContinuousBatchScheduler) ScheduleStep(ctx context.Context) (*ScheduledBatchStep, error) {
	if s == nil {
		return nil, errors.New("continuous batch scheduler: nil scheduler")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.sequences) == 0 {
		return nil, errors.New("continuous batch scheduler: no active sequences to schedule")
	}

	var (
		validRows  []PackedRow
		totalHits  int
		acceptSum  float64
		acceptCnt  int
		pageSize   = max(s.tree.PageSize(), 1)
		stepBudget = s.maxStepTokens
	)

	pageForPos := func(seq *InFlightSequence, pos int) int {
		if len(seq.PageIDs) == 0 {
			return 0
		}
		idx := pos / pageSize
		if idx < 0 {
			idx = 0
		}
		if idx >= len(seq.PageIDs) {
			idx = len(seq.PageIDs) - 1
		}
		return seq.PageIDs[idx]
	}

	for _, seq := range s.sequences {
		if len(validRows) >= stepBudget {
			break
		}
		remBudget := stepBudget - len(validRows)
		totalHits += seq.CachedPages

		switch seq.Phase {
		case PhaseDecode:
			pos := len(seq.PromptTokens)
			validRows = append(validRows, PackedRow{
				RowIndex: len(validRows),
				Valid:    true,
				SeqID:    seq.ID,
				Phase:    PhaseDecode,
				Token:    seq.DecodeToken,
				TokenPos: pos,
				KVPageID: pageForPos(seq, pos),
				WriteKV:  true,
			})

		case PhaseSpeculativeVerify:
			toks := make([]int, 0, 1+len(seq.DraftTokens))
			if seq.DecodeToken != 0 || len(seq.DraftTokens) == 0 {
				toks = append(toks, seq.DecodeToken)
			}
			toks = append(toks, seq.DraftTokens...)
			if len(toks) > remBudget {
				toks = toks[:remBudget]
			}
			basePos := len(seq.PromptTokens)
			for i, tok := range toks {
				pos := basePos + i
				validRows = append(validRows, PackedRow{
					RowIndex: len(validRows),
					Valid:    true,
					SeqID:    seq.ID,
					Phase:    PhaseSpeculativeVerify,
					Token:    tok,
					TokenPos: pos,
					KVPageID: pageForPos(seq, pos),
					WriteKV:  true,
				})
			}
			if seq.AcceptanceRate > 0 {
				acceptSum += seq.AcceptanceRate
				acceptCnt++
			}

		case PhaseChunkedPrefill:
			remPrompt := len(seq.PromptTokens) - seq.PrefillOffset
			if remPrompt <= 0 {
				seq.Phase = PhaseDecode
				pos := len(seq.PromptTokens)
				validRows = append(validRows, PackedRow{
					RowIndex: len(validRows),
					Valid:    true,
					SeqID:    seq.ID,
					Phase:    PhaseDecode,
					Token:    seq.DecodeToken,
					TokenPos: pos,
					KVPageID: pageForPos(seq, pos),
					WriteKV:  true,
				})
				continue
			}
			chunk := seq.ChunkBudget
			if chunk <= 0 || chunk > remPrompt {
				chunk = remPrompt
			}
			if chunk > remBudget {
				chunk = remBudget
			}
			for i := range chunk {
				pos := seq.PrefillOffset + i
				tok := seq.PromptTokens[pos]
				validRows = append(validRows, PackedRow{
					RowIndex: len(validRows),
					Valid:    true,
					SeqID:    seq.ID,
					Phase:    PhaseChunkedPrefill,
					Token:    tok,
					TokenPos: pos,
					KVPageID: pageForPos(seq, pos),
					WriteKV:  true,
				})
			}
			seq.PrefillOffset += chunk
			if seq.PrefillOffset >= len(seq.PromptTokens) {
				seq.Phase = PhaseDecode
			}
		}
	}

	activeM := len(validRows)
	if activeM == 0 {
		return nil, errors.New("continuous batch scheduler: step packed 0 active tokens")
	}

	minEntry, bucketM, err := s.SelectMinimalBucket(activeM)
	if err != nil {
		return nil, err
	}

	// Route through the MultiBucketMegakernelTable controller so adaptive
	// bucket telemetry and hot-swap hooks stay synchronized with B >= M_active.
	avgAccept := 0.85
	if acceptCnt > 0 {
		avgAccept = acceptSum / float64(acceptCnt)
	}
	if activeM <= 2 && acceptCnt == 0 {
		avgAccept = 0.25
	}
	_, trans, obsErr := s.table.ObserveAndSelect(ctx, activeM-1, avgAccept)
	if obsErr != nil {
		return nil, fmt.Errorf("continuous batch scheduler bucket route: %w", obsErr)
	}
	if ctrl := s.table.Controller(); ctrl != nil && trans.NewBucket.BucketM != bucketM {
		if switched, sErr := ctrl.SwitchBucket(ctx, minEntry.Spec); sErr == nil {
			trans = switched
		}
	}

	rows := make([]PackedRow, bucketM)
	var mask uint64
	for i := range bucketM {
		if i < activeM {
			rows[i] = validRows[i]
			rows[i].RowIndex = i
			if i < 64 {
				mask |= uint64(1) << uint(i)
			}
		} else {
			rows[i] = PackedRow{
				RowIndex: i,
				Valid:    false,
				SeqID:    "",
				Token:    0,
				TokenPos: -1,
				KVPageID: -1,
				WriteKV:  false,
			}
		}
	}

	return &ScheduledBatchStep{
		ActiveTokens:    activeM,
		BucketM:         bucketM,
		PaddedRows:      bucketM - activeM,
		ActiveRowMask:   mask,
		Rows:            rows,
		BucketEntry:     minEntry,
		Transition:      trans,
		PrefixCacheHits: totalHits,
	}, nil
}
