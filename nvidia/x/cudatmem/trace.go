package cudatmem

import (
	"errors"
	"fmt"
	"sort"
)

// EventKind identifies a tcgen05 Tensor Memory lifecycle or datapath operation.
type EventKind string

const (
	EventAlloc       EventKind = "tcgen05.alloc"
	EventDealloc     EventKind = "tcgen05.dealloc"
	EventCp          EventKind = "tcgen05.cp"
	EventMMA         EventKind = "tcgen05.mma"
	EventCommit      EventKind = "tcgen05.commit"
	EventWait        EventKind = "tcgen05.wait"
	EventLd          EventKind = "tcgen05.ld"
	EventSt          EventKind = "tcgen05.st"
	EventShift       EventKind = "tcgen05.shift"
	EventClusterSync EventKind = "cluster.sync"
)

var (
	ErrSingleThreadIssue = errors.New("cudatmem: mma/cp/commit/shift require ActiveThreads == 1 (elect.sync)")
	ErrWarpRowPartition  = errors.New("cudatmem: ld/st violates 4-warp 32-row datapath partition")
	ErrScaleNotStaged    = errors.New("cudatmem: block-scale mma requires prior tcgen05.cp into scale tmem")
	ErrOperandANotReady  = errors.New("cudatmem: tmem operand A requires prior mma + commit + wait")
	ErrCTA2Protocol      = errors.New("cudatmem: cta_group::2 cooperative lifecycle violation")
)

// TraceEvent records one device-side tcgen05 operation for protocol verification.
type TraceEvent struct {
	Kind            EventKind
	CTA             int
	Warp            int
	ActiveThreads   int    // e.g., 32 for alloc/ld/st, 1 (elect.sync) for mma/cp/commit
	CTAGroup        int    // 1 or 2
	Offset          uint32 // TMEM column offset
	Columns         uint32 // TMEM column width
	RowStart        uint32 // row range [RowStart, RowStart+Rows)
	Rows            uint32
	ScaleOffset     uint32 // for block-scaled MMA: TMEM column holding scales staged by tcgen05.cp
	HasScaleInTMEM  bool
	SourceAFromTMEM bool
	SourceAOffset   uint32
}

// TraceSummary reports aggregate statistics for a verified tcgen05 trace.
type TraceSummary struct {
	Target              Target
	CTAs                int
	CTAGroup            int
	MaxAllocatedColumns uint32
	Allocations         int
	Deallocations       int
	Copies              int
	MMAs                int
	Commits             int
	Waits               int
	Loads               int
	Stores              int
	Shifts              int
	ClusterSyncs        int
}

type allocRecord struct {
	alloc    *Allocation
	ctaGroup int
	offset   uint32
	columns  uint32
	rows     uint32
}

type mmaStage uint8

const (
	mmaNone mmaStage = iota
	mmaWritten
	mmaCommitted
	mmaWaited
)

type ctaState struct {
	alloc         *Allocator
	permit        *Permit
	live          map[uint32]*allocRecord // keyed by base Offset
	cpStaged      map[uint32]uint32       // base Offset -> columns staged by tcgen05.cp
	cpInFlight    map[uint32]bool         // base Offset -> tcgen05.cp in flight awaiting EventWait
	mmaState      map[uint32]mmaStage     // base Offset -> MMA completion stage
	mmaPendingUse map[uint32]uint32       // base Offset -> active BeginUser count awaiting EventWait
	allocatedCols uint32
	maxCols       uint32
}

type pairState struct {
	active              bool
	allocs              map[int]map[uint32]uint32 // cta -> offset -> columns (current live cta_group::2 allocations)
	history             map[int]map[uint32]uint32 // cta -> offset -> columns (all cta_group::2 allocations)
	lastAccessStep      int
	lastClusterSyncStep int
}

// VerifyTrace validates a sequence of device-side tcgen05 events against the
// cudatmem Allocator state machine and the sm_100a/sm_103a/sm_100f/sm_103f
// PTX 9.4 datapath rules:
//  1. Single-thread elect.sync issue for tcgen05.mma, tcgen05.cp, tcgen05.commit.
//  2. 4-warp 32-row epilogue datapath partitioning for tcgen05.ld and tcgen05.st.
//  3. tcgen05.cp scale staging prior to block-scaled tcgen05.mma.
//  4. TMEM-sourced operand A readiness (prior tcgen05.mma + commit + wait).
//  5. 2-CTA (cta_group::2) matched allocation, CTA-0 MMA issue, and cluster.sync before dealloc.
//  6. Underlying cudatmem.Allocator unit, ordering, busy-user, and leak checks.
func VerifyTrace(target Target, events []TraceEvent) (*TraceSummary, error) {
	probe, err := New(target)
	if err != nil {
		return nil, err
	}
	_ = probe.Close()

	summary := &TraceSummary{
		Target:   target,
		CTAGroup: 1,
	}
	if len(events) == 0 {
		return summary, nil
	}

	ctas := make(map[int]*ctaState)
	getCTA := func(id int) (*ctaState, error) {
		if id < 0 {
			return nil, fmt.Errorf("cudatmem: negative CTA index %d", id)
		}
		st, ok := ctas[id]
		if !ok {
			a, err := New(target)
			if err != nil {
				return nil, err
			}
			st = &ctaState{
				alloc:         a,
				live:          make(map[uint32]*allocRecord),
				cpStaged:      make(map[uint32]uint32),
				cpInFlight:    make(map[uint32]bool),
				mmaState:      make(map[uint32]mmaStage),
				mmaPendingUse: make(map[uint32]uint32),
			}
			ctas[id] = st
		}
		return st, nil
	}

	pairs := make(map[int]*pairState)
	getPair := func(cta int) *pairState {
		base := cta &^ 1
		p, ok := pairs[base]
		if !ok {
			p = &pairState{
				allocs:              map[int]map[uint32]uint32{base: {}, base | 1: {}},
				history:             map[int]map[uint32]uint32{base: {}, base | 1: {}},
				lastAccessStep:      -1,
				lastClusterSyncStep: -1,
			}
			pairs[base] = p
		}
		return p
	}

	findAlloc := func(st *ctaState, offset, cols uint32) (*allocRecord, error) {
		for _, rec := range st.live {
			if offset >= rec.offset && cols <= rec.columns && offset-rec.offset <= rec.columns-cols {
				return rec, nil
			}
		}
		return nil, fmt.Errorf("%w: ctas[%d] range [%d, %d) is not allocated", ErrClosed, 0, offset, offset+cols)
	}

	validateCTAGroup := func(g int) error {
		if g != 1 && g != 2 {
			return fmt.Errorf("%w: invalid CTAGroup %d (want 1 or 2)", ErrParticipation, g)
		}
		if g == 2 {
			summary.CTAGroup = 2
		}
		return nil
	}

	for step, ev := range events {
		switch ev.Kind {
		case EventClusterSync:
			if ev.CTAGroup != 0 && ev.CTAGroup != 1 && ev.CTAGroup != 2 {
				return nil, fmt.Errorf("step %d (%s): %w: invalid CTAGroup %d", step, ev.Kind, ErrParticipation, ev.CTAGroup)
			}
			if ev.CTAGroup == 2 {
				summary.CTAGroup = 2
			}
			for _, p := range pairs {
				p.lastClusterSyncStep = step
			}
			summary.ClusterSyncs++

		case EventAlloc:
			if err := validateCTAGroup(ev.CTAGroup); err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			st, err := getCTA(ev.CTA)
			if err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			if st.permit == nil {
				p, err := st.alloc.AcquirePermit(uint32(ev.ActiveThreads))
				if err != nil {
					return nil, fmt.Errorf("step %d (%s): acquire permit on CTA %d: %w", step, ev.Kind, ev.CTA, err)
				}
				st.permit = p
			}
			rows := ev.Rows
			if rows == 0 {
				rows = 128
			}
			x, err := st.alloc.Allocate(st.permit, uint32(ev.ActiveThreads), ev.Columns, rows)
			if err != nil {
				return nil, fmt.Errorf("step %d (%s): CTA %d allocate(%d cols, %d rows): %w", step, ev.Kind, ev.CTA, ev.Columns, rows, err)
			}
			if x.Offset() != ev.Offset {
				return nil, fmt.Errorf("step %d (%s): CTA %d offset mismatch: event has %d, allocator returned %d: %w",
					step, ev.Kind, ev.CTA, ev.Offset, x.Offset(), ErrInvalidUnit)
			}
			st.live[ev.Offset] = &allocRecord{
				alloc:    x,
				ctaGroup: ev.CTAGroup,
				offset:   ev.Offset,
				columns:  ev.Columns,
				rows:     rows,
			}
			st.allocatedCols += ev.Columns
			if st.allocatedCols > st.maxCols {
				st.maxCols = st.allocatedCols
			}
			if st.maxCols > summary.MaxAllocatedColumns {
				summary.MaxAllocatedColumns = st.maxCols
			}
			if ev.CTAGroup == 2 {
				p := getPair(ev.CTA)
				p.active = true
				p.allocs[ev.CTA][ev.Offset] = ev.Columns
				p.history[ev.CTA][ev.Offset] = ev.Columns
			}
			summary.Allocations++

		case EventDealloc:
			if err := validateCTAGroup(ev.CTAGroup); err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			st, err := getCTA(ev.CTA)
			if err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			rec, ok := st.live[ev.Offset]
			if !ok {
				return nil, fmt.Errorf("step %d (%s): CTA %d offset %d not live: %w", step, ev.Kind, ev.CTA, ev.Offset, ErrClosed)
			}
			if rec.columns != ev.Columns {
				return nil, fmt.Errorf("step %d (%s): CTA %d dealloc columns %d != allocated %d: %w",
					step, ev.Kind, ev.CTA, ev.Columns, rec.columns, ErrInvalidUnit)
			}
			if rec.ctaGroup != ev.CTAGroup {
				return nil, fmt.Errorf("step %d (%s): CTA %d CTAGroup %d != allocated CTAGroup %d: %w",
					step, ev.Kind, ev.CTA, ev.CTAGroup, rec.ctaGroup, ErrCTA2Protocol)
			}
			if ev.CTAGroup == 2 {
				p := getPair(ev.CTA)
				peer := ev.CTA ^ 1
				if peerCols, ok := p.history[peer][ev.Offset]; !ok || peerCols != ev.Columns {
					return nil, fmt.Errorf("step %d (%s): %w: CTA %d and peer CTA %d do not have matching cta_group::2 allocation at offset %d",
						step, ev.Kind, ErrCTA2Protocol, ev.CTA, peer, ev.Offset)
				}
				if p.lastAccessStep >= 0 && p.lastClusterSyncStep <= p.lastAccessStep {
					return nil, fmt.Errorf("step %d (%s): %w: CTA %d dealloc before cluster.sync (last access step %d, last cluster.sync step %d)",
						step, ev.Kind, ErrCTA2Protocol, ev.CTA, p.lastAccessStep, p.lastClusterSyncStep)
				}
				delete(p.allocs[ev.CTA], ev.Offset)
			}
			if err := st.alloc.Release(st.permit, rec.alloc, uint32(ev.ActiveThreads)); err != nil {
				return nil, fmt.Errorf("step %d (%s): CTA %d release offset %d: %w", step, ev.Kind, ev.CTA, ev.Offset, err)
			}
			delete(st.live, ev.Offset)
			delete(st.cpStaged, ev.Offset)
			delete(st.cpInFlight, ev.Offset)
			delete(st.mmaState, ev.Offset)
			delete(st.mmaPendingUse, ev.Offset)
			if st.allocatedCols >= ev.Columns {
				st.allocatedCols -= ev.Columns
			}
			summary.Deallocations++

		case EventCp:
			if err := validateCTAGroup(ev.CTAGroup); err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			if ev.ActiveThreads != 1 {
				return nil, fmt.Errorf("step %d (%s): got ActiveThreads=%d: %w", step, ev.Kind, ev.ActiveThreads, ErrSingleThreadIssue)
			}
			if ev.Columns == 0 {
				return nil, fmt.Errorf("step %d (%s): zero columns: %w", step, ev.Kind, ErrInvalidUnit)
			}
			targets := []int{ev.CTA}
			if ev.CTAGroup == 2 {
				if ev.CTA%2 != 0 {
					return nil, fmt.Errorf("step %d (%s): %w: cta_group::2 cp must be issued by even CTA, got CTA %d",
						step, ev.Kind, ErrCTA2Protocol, ev.CTA)
				}
				targets = []int{ev.CTA &^ 1, ev.CTA | 1}
			}
			for _, cid := range targets {
				st, err := getCTA(cid)
				if err != nil {
					return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
				}
				rec, err := findAlloc(st, ev.Offset, ev.Columns)
				if err != nil {
					return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, cid, err)
				}
				if err := rec.alloc.BeginUser(); err != nil {
					return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, cid, err)
				}
				if err := rec.alloc.CompleteUser(); err != nil {
					return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, cid, err)
				}
				st.cpStaged[ev.Offset] = ev.Columns
				st.cpInFlight[rec.offset] = true
			}
			if ev.CTAGroup == 2 {
				getPair(ev.CTA).lastAccessStep = step
			}
			summary.Copies++

		case EventMMA:
			if err := validateCTAGroup(ev.CTAGroup); err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			if ev.ActiveThreads != 1 {
				return nil, fmt.Errorf("step %d (%s): got ActiveThreads=%d: %w", step, ev.Kind, ev.ActiveThreads, ErrSingleThreadIssue)
			}
			if ev.Columns == 0 {
				return nil, fmt.Errorf("step %d (%s): zero columns: %w", step, ev.Kind, ErrInvalidUnit)
			}
			targets := []int{ev.CTA}
			if ev.CTAGroup == 2 {
				if ev.CTA%2 != 0 {
					return nil, fmt.Errorf("step %d (%s): %w: cta_group::2 mma must be issued by even CTA (CTA %% 2 == 0), got CTA %d",
						step, ev.Kind, ErrCTA2Protocol, ev.CTA)
				}
				cta0, cta1 := ev.CTA&^1, ev.CTA|1
				targets = []int{cta0, cta1}
				st0, err := getCTA(cta0)
				if err != nil {
					return nil, err
				}
				st1, err := getCTA(cta1)
				if err != nil {
					return nil, err
				}
				rec0, err0 := findAlloc(st0, ev.Offset, ev.Columns)
				rec1, err1 := findAlloc(st1, ev.Offset, ev.Columns)
				if err0 != nil || err1 != nil || rec0.offset != rec1.offset || rec0.columns != rec1.columns || rec0.ctaGroup != 2 || rec1.ctaGroup != 2 {
					return nil, fmt.Errorf("step %d (%s): %w: paired CTAs (%d, %d) lack matching cta_group::2 TMEM allocations at offset %d (cols %d)",
						step, ev.Kind, ErrCTA2Protocol, cta0, cta1, ev.Offset, ev.Columns)
				}
			}

			for _, cid := range targets {
				st, err := getCTA(cid)
				if err != nil {
					return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
				}
				if ev.HasScaleInTMEM {
					stagedCols, ok := st.cpStaged[ev.ScaleOffset]
					if !ok || stagedCols == 0 {
						return nil, fmt.Errorf("step %d (%s): CTA %d scale offset %d: %w",
							step, ev.Kind, cid, ev.ScaleOffset, ErrScaleNotStaged)
					}
					if _, err := findAlloc(st, ev.ScaleOffset, stagedCols); err != nil {
						return nil, fmt.Errorf("step %d (%s): CTA %d scale offset %d not in live allocation: %w",
							step, ev.Kind, cid, ev.ScaleOffset, ErrScaleNotStaged)
					}
				}
				if ev.SourceAFromTMEM {
					if st.mmaState[ev.SourceAOffset] != mmaWaited {
						return nil, fmt.Errorf("step %d (%s): CTA %d source A offset %d (stage=%d): %w",
							step, ev.Kind, cid, ev.SourceAOffset, st.mmaState[ev.SourceAOffset], ErrOperandANotReady)
					}
					if _, err := findAlloc(st, ev.SourceAOffset, 1); err != nil {
						return nil, fmt.Errorf("step %d (%s): CTA %d source A offset %d not allocated: %w",
							step, ev.Kind, cid, ev.SourceAOffset, ErrOperandANotReady)
					}
				}
				rec, err := findAlloc(st, ev.Offset, ev.Columns)
				if err != nil {
					return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, cid, err)
				}
				if err := rec.alloc.BeginUser(); err != nil {
					return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, cid, err)
				}
				st.mmaPendingUse[rec.offset]++
				st.mmaState[ev.Offset] = mmaWritten
			}
			if ev.CTAGroup == 2 {
				getPair(ev.CTA).lastAccessStep = step
			}
			summary.MMAs++

		case EventCommit:
			if err := validateCTAGroup(ev.CTAGroup); err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			if ev.ActiveThreads != 1 {
				return nil, fmt.Errorf("step %d (%s): got ActiveThreads=%d: %w", step, ev.Kind, ev.ActiveThreads, ErrSingleThreadIssue)
			}
			targets := []int{ev.CTA}
			if ev.CTAGroup == 2 {
				if ev.CTA%2 != 0 {
					return nil, fmt.Errorf("step %d (%s): %w: cta_group::2 commit must be issued by even CTA, got CTA %d",
						step, ev.Kind, ErrCTA2Protocol, ev.CTA)
				}
				targets = []int{ev.CTA &^ 1, ev.CTA | 1}
			}
			for _, cid := range targets {
				st, err := getCTA(cid)
				if err != nil {
					return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
				}
				for off, stage := range st.mmaState {
					if stage == mmaWritten {
						st.mmaState[off] = mmaCommitted
					}
				}
			}
			summary.Commits++

		case EventWait:
			if err := validateCTAGroup(ev.CTAGroup); err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			st, err := getCTA(ev.CTA)
			if err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			for off, stage := range st.mmaState {
				if stage == mmaCommitted {
					st.mmaState[off] = mmaWaited
				}
			}
			for baseOff, count := range st.mmaPendingUse {
				if rec, ok := st.live[baseOff]; ok {
					for range count {
						if err := rec.alloc.CompleteUser(); err != nil {
							return nil, fmt.Errorf("step %d (%s): CTA %d complete user: %w", step, ev.Kind, ev.CTA, err)
						}
					}
				}
				delete(st.mmaPendingUse, baseOff)
			}
			clear(st.cpInFlight)
			if ev.CTAGroup == 2 {
				getPair(ev.CTA).lastAccessStep = step
			}
			summary.Waits++

		case EventShift:
			if err := validateCTAGroup(ev.CTAGroup); err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			if ev.ActiveThreads != 1 || ev.Warp != 0 {
				return nil, fmt.Errorf("step %d (%s): got Warp=%d ActiveThreads=%d: %w",
					step, ev.Kind, ev.Warp, ev.ActiveThreads, ErrSingleThreadIssue)
			}
			if ev.Columns == 0 || (ev.Rows != 0 && ev.Rows != 128) {
				return nil, fmt.Errorf("step %d (%s): invalid columns %d or rows %d (want 128): %w",
					step, ev.Kind, ev.Columns, ev.Rows, ErrInvalidUnit)
			}
			targets := []int{ev.CTA}
			if ev.CTAGroup == 2 {
				if ev.CTA%2 != 0 {
					return nil, fmt.Errorf("step %d (%s): %w: cta_group::2 shift must be issued by even CTA, got CTA %d",
						step, ev.Kind, ErrCTA2Protocol, ev.CTA)
				}
				targets = []int{ev.CTA &^ 1, ev.CTA | 1}
			}
			for _, cid := range targets {
				st, err := getCTA(cid)
				if err != nil {
					return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
				}
				rec, err := findAlloc(st, ev.Offset, ev.Columns)
				if err != nil {
					return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, cid, err)
				}
				if rec.rows != 128 {
					return nil, fmt.Errorf("step %d (%s): CTA %d allocation rows %d != 128: %w",
						step, ev.Kind, cid, rec.rows, ErrInvalidUnit)
				}
				if st.mmaPendingUse[rec.offset] > 0 || st.mmaState[rec.offset] == mmaWritten || st.mmaState[rec.offset] == mmaCommitted || st.cpInFlight[rec.offset] {
					return nil, fmt.Errorf("step %d (%s): CTA %d offset %d has in-flight mma/cp awaiting tcgen05.wait: %w",
						step, ev.Kind, cid, ev.Offset, ErrBusy)
				}
				if err := rec.alloc.BeginUser(); err != nil {
					return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, cid, err)
				}
				if err := rec.alloc.CompleteUser(); err != nil {
					return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, cid, err)
				}
			}
			if ev.CTAGroup == 2 {
				getPair(ev.CTA).lastAccessStep = step
			}
			summary.Shifts++

		case EventLd, EventSt:
			if err := validateCTAGroup(ev.CTAGroup); err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			if ev.ActiveThreads != 32 {
				return nil, fmt.Errorf("step %d (%s): ActiveThreads=%d, want 32: %w",
					step, ev.Kind, ev.ActiveThreads, ErrWarpRowPartition)
			}
			if ev.Warp < 0 || ev.Rows == 0 {
				return nil, fmt.Errorf("step %d (%s): invalid warp %d or rows %d: %w",
					step, ev.Kind, ev.Warp, ev.Rows, ErrWarpRowPartition)
			}
			minRow := uint32(32 * (ev.Warp % 4))
			maxRow := minRow + 32
			if ev.RowStart < minRow || ev.RowStart+ev.Rows > maxRow {
				return nil, fmt.Errorf("step %d (%s): warp %d accessed rows [%d, %d), allowed [%d, %d): %w",
					step, ev.Kind, ev.Warp, ev.RowStart, ev.RowStart+ev.Rows, minRow, maxRow, ErrWarpRowPartition)
			}
			st, err := getCTA(ev.CTA)
			if err != nil {
				return nil, fmt.Errorf("step %d (%s): %w", step, ev.Kind, err)
			}
			cols := ev.Columns
			if cols == 0 {
				cols = 1
			}
			rec, err := findAlloc(st, ev.Offset, cols)
			if err != nil {
				return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, ev.CTA, err)
			}
			if err := rec.alloc.BeginUser(); err != nil {
				return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, ev.CTA, err)
			}
			if err := rec.alloc.CompleteUser(); err != nil {
				return nil, fmt.Errorf("step %d (%s): CTA %d: %w", step, ev.Kind, ev.CTA, err)
			}
			if ev.CTAGroup == 2 || rec.ctaGroup == 2 {
				getPair(ev.CTA).lastAccessStep = step
			}
			if ev.Kind == EventLd {
				summary.Loads++
			} else {
				summary.Stores++
			}

		default:
			return nil, fmt.Errorf("step %d: unknown event kind %q", step, ev.Kind)
		}
	}

	// Verify 2-CTA cooperative pairs had matching allocation histories and clean teardown.
	pairKeys := make([]int, 0, len(pairs))
	for k := range pairs {
		pairKeys = append(pairKeys, k)
	}
	sort.Ints(pairKeys)
	for _, base := range pairKeys {
		p := pairs[base]
		if !p.active {
			continue
		}
		cta0, cta1 := base, base|1
		if len(p.history[cta0]) != len(p.history[cta1]) {
			return nil, fmt.Errorf("%w: CTA %d and CTA %d have mismatched cta_group::2 allocation counts (%d vs %d)",
				ErrCTA2Protocol, cta0, cta1, len(p.history[cta0]), len(p.history[cta1]))
		}
		for off, cols0 := range p.history[cta0] {
			if cols1, ok := p.history[cta1][off]; !ok || cols0 != cols1 {
				return nil, fmt.Errorf("%w: CTA %d and CTA %d have mismatched cta_group::2 allocation at offset %d (%d vs %d)",
					ErrCTA2Protocol, cta0, cta1, off, cols0, cols1)
			}
		}
	}

	// Verify all CTA allocators have no leaked allocations and can relinquish & close cleanly.
	ctaKeys := make([]int, 0, len(ctas))
	for k := range ctas {
		ctaKeys = append(ctaKeys, k)
	}
	sort.Ints(ctaKeys)
	for _, cid := range ctaKeys {
		st := ctas[cid]
		if st.permit != nil {
			if err := st.alloc.Relinquish(st.permit); err != nil {
				return nil, fmt.Errorf("CTA %d relinquish permit: %w", cid, err)
			}
		}
		if err := st.alloc.Close(); err != nil {
			return nil, fmt.Errorf("CTA %d close allocator (unreleased TMEM allocation): %w", cid, err)
		}
	}

	summary.CTAs = len(ctas)
	return summary, nil
}
