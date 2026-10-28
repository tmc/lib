package cudahostemu

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"
)

// This file is a cycle-estimating model of one CTA's warp-specialized
// TMA/tensor-core pipeline. Unlike the rest of the package it does not
// compile or run kernel source: a [TimelineProgram] lists each warp's
// asynchronous-pipeline instructions, and [SimulateTimeline] schedules them
// against a fixed latency table. Its cycle counts are estimates from that
// table, useful for comparing schedules of the same kernel, not for
// predicting a device's absolute time.

// A PipeKind is a timeline track group.
type PipeKind string

const (
	PipeProducerWarp PipeKind = "Warp-Producer-TMA"
	PipeConsumerWarp PipeKind = "Warp-Consumer-MMA"
	PipeTMAEngine    PipeKind = "Engine-TMA-Async"
	PipeTensorCore   PipeKind = "Engine-Tensor-Core"
	PipeTMEMSpace    PipeKind = "Space-TMEM-Columns"
	PipeStallWait    PipeKind = "Stall-Scoreboard-Barrier"
)

// Timeline track IDs for the asynchronous engines, above any warp ID.
const (
	TrackTMAEngine  = 1000
	TrackTensorCore = 1001
)

// A TimelineArch is the latency model [SimulateTimeline] uses. The values
// from [TimelineArchFor] are round figures chosen to be in proportion to one
// another, not measurements of a device.
type TimelineArch struct {
	Name        string
	TMEMColumns int // 0 for an architecture without tensor memory

	TMAIssue      int     // cycles to issue cp.async.bulk.tensor
	TMALatency    int     // minimum cycles from issue to completion
	TMABytesCycle float64 // bytes per cycle the TMA engine moves

	MMAIssue     int     // cycles to issue an MMA
	MMAOpsCycle  float64 // multiply-add operations (2 per MAC) per cycle
	TMEMIssue    int     // cycles to issue tcgen05.ld
	TMEMLdLag    int     // cycles after the last MMA completes until tcgen05.ld data is ready
	BarrierIssue int     // cycles for an arrive, a commit, or a satisfied wait
	AllocIssue   int     // cycles for tcgen05.alloc or dealloc
}

// TimelineArchFor returns the model for arch: sm_90a (wgmma, no tensor
// memory), sm_100a, sm_107a, or sm_121a. GB10 (sm_121a) has no tcgen05
// hardware; its entry models the emulated form only.
func TimelineArchFor(arch string) (TimelineArch, error) {
	a := TimelineArch{
		Name:          arch,
		TMEMColumns:   512,
		TMAIssue:      16,
		TMALatency:    380,
		TMABytesCycle: 128,
		MMAIssue:      8,
		MMAOpsCycle:   8192,
		TMEMIssue:     4,
		TMEMLdLag:     24,
		BarrierIssue:  2,
		AllocIssue:    4,
	}
	switch arch {
	case "sm_100a":
	case "sm_107a":
		a.MMAOpsCycle, a.TMABytesCycle = 16384, 256
	case "sm_121a":
		a.MMAOpsCycle, a.TMABytesCycle = 1024, 32
	case "sm_90a":
		a.TMEMColumns, a.MMAOpsCycle, a.TMABytesCycle = 0, 4096, 64
	default:
		return TimelineArch{}, fmt.Errorf("cudahostemu: no timeline model for %q", arch)
	}
	return a, nil
}

// A TimelineOpKind is an instruction a [TimelineProgram] models.
type TimelineOpKind uint8

const (
	OpTMALoad  TimelineOpKind = iota // cp.async.bulk.tensor completing Bytes on mbarrier Bar
	OpArrive                         // mbarrier.arrive on Bar
	OpWait                           // mbarrier.try_wait.parity until phase Phase of Bar completes
	OpMMA                            // tcgen05.mma (wgmma.mma_async on sm_90a) of shape M×N×K
	OpCommit                         // tcgen05.commit: arrive on Bar when this warp's MMAs complete
	OpAlloc                          // tcgen05.alloc of Cols columns, named Label
	OpDealloc                        // tcgen05.dealloc of the allocation named Label
	OpLoad                           // tcgen05.ld
	OpWaitLoad                       // tcgen05.wait::ld
	OpCompute                        // Cycles of other work
)

// A TimelineOp is one modeled instruction. Phase counts completions of the
// barrier from zero; the PTX parity operand is Phase&1.
type TimelineOp struct {
	Kind    TimelineOpKind
	Bar     int
	Phase   int
	Bytes   int
	M, N, K int
	Cols    int
	Cycles  int
	Label   string
}

// Mnemonic returns the PTX instruction op stands for on arch.
func (op TimelineOp) Mnemonic(arch TimelineArch) string {
	switch op.Kind {
	case OpTMALoad:
		return "cp.async.bulk.tensor"
	case OpArrive:
		return "mbarrier.arrive"
	case OpWait:
		return "mbarrier.try_wait.parity"
	case OpMMA:
		if arch.TMEMColumns == 0 {
			return "wgmma.mma_async"
		}
		return "tcgen05.mma"
	case OpCommit:
		if arch.TMEMColumns == 0 {
			return "wgmma.commit_group"
		}
		return "tcgen05.commit"
	case OpAlloc:
		return "tcgen05.alloc"
	case OpDealloc:
		return "tcgen05.dealloc"
	case OpLoad:
		return "tcgen05.ld"
	case OpWaitLoad:
		return "tcgen05.wait::ld"
	case OpCompute:
		if op.Label != "" {
			return op.Label
		}
		return "compute"
	}
	return fmt.Sprintf("TimelineOpKind(%d)", op.Kind)
}

// A TimelineWarp is one warp's instruction stream.
type TimelineWarp struct {
	ID   int
	Role PipeKind // PipeProducerWarp or PipeConsumerWarp
	Ops  []TimelineOp
}

// A TimelineProgram is the modeled instruction streams of one CTA.
type TimelineProgram struct {
	CTA int
	// Barriers holds each mbarrier's arrival count per phase.
	Barriers []int
	Warps    []TimelineWarp
}

// A TimelineSpan is one interval of the timeline: an instruction, an
// asynchronous engine operation, or a stall.
type TimelineSpan struct {
	Name       string         `json:"name"`
	Category   PipeKind       `json:"cat"`
	Track      int            `json:"tid"`
	CTA        int            `json:"pid"`
	StartCycle uint64         `json:"ts"`
	DurCycles  uint64         `json:"dur"`
	Args       map[string]any `json:"args,omitempty"`
}

// A TimelineFlow links the arrival that completed a barrier phase to the
// wait it released.
type TimelineFlow struct {
	ID        uint64
	FromTrack int
	FromCycle uint64
	ToTrack   int
	ToCycle   uint64
	Barrier   int
	Phase     int
}

// A TMEMColumnSlice is a tensor-memory allocation of NumCols columns from
// StartCol, live from AllocCycle to FreeCycle.
type TMEMColumnSlice struct {
	StartCol   int
	NumCols    int
	AllocCycle uint64
	FreeCycle  uint64
	Label      string
}

// A TimelineFinding is a schedule problem [SimulateTimeline] flagged.
type TimelineFinding struct {
	Kind   string // "consumer-stall" or "wait-ld-after-mma"
	Warp   int
	Op     int // index in the warp's Ops, or -1
	Detail string
}

// A TimelineSummary is the result of [SimulateTimeline].
type TimelineSummary struct {
	Arch                string
	CTA                 int
	TotalCycles         uint64
	TensorCoreActiveCyc uint64
	TMAActiveCyc        uint64
	ConsumerStallCyc    uint64  // cycles consumer warps wait with none of their MMAs in flight
	TensorUtilPct       float64 // 100 * TensorCoreActiveCyc / TotalCycles
	TMAOverlapPct       float64 // percent of TMAActiveCyc during which the tensor core is busy
	PeakTMEMCols        int
	Spans               []TimelineSpan
	Flows               []TimelineFlow
	TMEMMap             []TMEMColumnSlice
	Findings            []TimelineFinding
}

// ConsumerStallLimit is the fraction of total cycles of consumer stall
// above which [SimulateTimeline] reports a "consumer-stall" finding.
const ConsumerStallLimit = 0.20

type arrival struct {
	cycle uint64
	track int
	flow  uint64
}

type barrierState struct {
	count    int
	arrivals []arrival
}

// completion returns when phase k of b completes and the arrival that
// completed it, or ok=false if not all its arrivals are known yet.
func (b *barrierState) completion(k int) (arrival, bool) {
	end := (k + 1) * b.count
	if end > len(b.arrivals) {
		return arrival{}, false
	}
	var last arrival
	for _, a := range b.arrivals[k*b.count : end] {
		if a.cycle >= last.cycle {
			last = a
		}
	}
	return last, true
}

type interval struct{ start, end uint64 }

// SimulateTimeline schedules p on the model for arch. Each warp issues its
// instructions in order; a ready warp with the earliest clock runs next.
// The TMA engine and the tensor core each serve one operation at a time, in
// issue order. A wait for a barrier phase blocks its warp until the phase's
// last arrival; a tcgen05.wait::ld blocks until the last MMA issued before
// the matching tcgen05.ld has completed plus TMEMLdLag. SimulateTimeline
// returns an error if every unfinished warp is blocked on a phase that can
// no longer complete, or if a tensor-memory allocation does not fit.
func SimulateTimeline(p TimelineProgram, arch string) (TimelineSummary, error) {
	a, err := TimelineArchFor(arch)
	if err != nil {
		return TimelineSummary{}, err
	}
	s := TimelineSummary{Arch: arch, CTA: p.CTA}
	bars := make([]barrierState, len(p.Barriers))
	for i, c := range p.Barriers {
		if c <= 0 {
			return TimelineSummary{}, fmt.Errorf("cudahostemu: barrier %d: arrival count %d", i, c)
		}
		bars[i].count = c
	}
	n := len(p.Warps)
	clock := make([]uint64, n)
	pc := make([]int, n)
	lastMMA := make([]uint64, n) // completion of each warp's last MMA
	ldReady := make([]uint64, n) // when this warp's outstanding tcgen05.ld data is ready
	var (
		tmaFree, tcFree uint64
		tmaBusy, tcBusy []interval
		flowID          uint64
		tmemLive        []int // indices into s.TMEMMap
		tmemUsed        int
	)
	span := func(name string, cat PipeKind, track int, start, end uint64, args map[string]any) {
		s.Spans = append(s.Spans, TimelineSpan{Name: name, Category: cat, Track: track, CTA: p.CTA, StartCycle: start, DurCycles: end - start, Args: args})
	}
	// stall returns the part of a consumer warp's stall from t to wake
	// after its last MMA completed, and adds it to ConsumerStallCyc.
	stall := func(w int, t, wake uint64) uint64 {
		if p.Warps[w].Role != PipeConsumerWarp {
			return 0
		}
		b := wake - min(wake, max(t, lastMMA[w]))
		s.ConsumerStallCyc += b
		return b
	}
	checkBar := func(w int, op TimelineOp) error {
		if op.Bar < 0 || op.Bar >= len(bars) {
			return fmt.Errorf("cudahostemu: warp %d op %d: barrier %d of %d", p.Warps[w].ID, pc[w], op.Bar, len(bars))
		}
		return nil
	}

	for {
		// Pick the ready warp that can start earliest.
		next, start := -1, uint64(math.MaxUint64)
		unfinished := false
		for w := range n {
			if pc[w] >= len(p.Warps[w].Ops) {
				continue
			}
			unfinished = true
			op := p.Warps[w].Ops[pc[w]]
			t := clock[w]
			if op.Kind == OpWait {
				if err := checkBar(w, op); err != nil {
					return TimelineSummary{}, err
				}
				c, ok := bars[op.Bar].completion(op.Phase)
				if !ok {
					continue
				}
				t = max(t, c.cycle)
			}
			if t < start {
				next, start = w, t
			}
		}
		if !unfinished {
			break
		}
		if next < 0 {
			var blocked []string
			for w := range n {
				if pc[w] < len(p.Warps[w].Ops) {
					op := p.Warps[w].Ops[pc[w]]
					blocked = append(blocked, fmt.Sprintf("warp %d on barrier %d phase %d", p.Warps[w].ID, op.Bar, op.Phase))
				}
			}
			return TimelineSummary{}, fmt.Errorf("cudahostemu: timeline deadlock: %v", blocked)
		}

		w := next
		warp := p.Warps[w]
		op := warp.Ops[pc[w]]
		t := clock[w]
		name := op.Mnemonic(a)
		switch op.Kind {
		case OpTMALoad:
			if err := checkBar(w, op); err != nil {
				return TimelineSummary{}, err
			}
			issued := t + uint64(a.TMAIssue)
			xfer := max(issued, tmaFree)
			tmaFree = xfer + uint64(math.Ceil(float64(op.Bytes)/a.TMABytesCycle))
			done := max(issued+uint64(a.TMALatency), tmaFree)
			flowID++
			bars[op.Bar].arrivals = append(bars[op.Bar].arrivals, arrival{done, TrackTMAEngine, flowID})
			tmaBusy = append(tmaBusy, interval{issued, done})
			span(name, warp.Role, warp.ID, t, issued, nil)
			span("TMA "+fmt.Sprint(op.Bytes)+"B", PipeTMAEngine, TrackTMAEngine, issued, done, map[string]any{"bytes": op.Bytes, "mbarrier": op.Bar})
			clock[w] = issued
		case OpArrive:
			if err := checkBar(w, op); err != nil {
				return TimelineSummary{}, err
			}
			flowID++
			bars[op.Bar].arrivals = append(bars[op.Bar].arrivals, arrival{t, warp.ID, flowID})
			clock[w] = t + uint64(a.BarrierIssue)
			span(name, warp.Role, warp.ID, t, clock[w], nil)
		case OpWait:
			c, _ := bars[op.Bar].completion(op.Phase)
			wake := t
			if c.cycle > t {
				wake = c.cycle
				bubble := stall(w, t, wake)
				span("STALL_MBARRIER", PipeStallWait, warp.ID, t, wake, map[string]any{"mbarrier": op.Bar, "phase": op.Phase, "bubble": bubble})
			}
			s.Flows = append(s.Flows, TimelineFlow{ID: c.flow, FromTrack: c.track, FromCycle: c.cycle, ToTrack: warp.ID, ToCycle: wake, Barrier: op.Bar, Phase: op.Phase})
			clock[w] = wake + uint64(a.BarrierIssue)
			span(name, warp.Role, warp.ID, wake, clock[w], map[string]any{"mbarrier": op.Bar, "parity": op.Phase & 1})
		case OpMMA:
			issued := t + uint64(a.MMAIssue)
			begin := max(issued, tcFree)
			tcFree = begin + uint64(math.Ceil(2*float64(op.M)*float64(op.N)*float64(op.K)/a.MMAOpsCycle))
			lastMMA[w] = tcFree
			tcBusy = append(tcBusy, interval{begin, tcFree})
			span(name, warp.Role, warp.ID, t, issued, nil)
			span(fmt.Sprintf("MMA %dx%dx%d", op.M, op.N, op.K), PipeTensorCore, TrackTensorCore, begin, tcFree, nil)
			clock[w] = issued
		case OpCommit:
			if err := checkBar(w, op); err != nil {
				return TimelineSummary{}, err
			}
			flowID++
			bars[op.Bar].arrivals = append(bars[op.Bar].arrivals, arrival{max(lastMMA[w], t), TrackTensorCore, flowID})
			clock[w] = t + uint64(a.BarrierIssue)
			span(name, warp.Role, warp.ID, t, clock[w], nil)
		case OpAlloc:
			if a.TMEMColumns == 0 {
				return TimelineSummary{}, fmt.Errorf("cudahostemu: %s has no tensor memory", arch)
			}
			col, err := tmemFit(s.TMEMMap, tmemLive, op.Cols, a.TMEMColumns)
			if err != nil {
				return TimelineSummary{}, fmt.Errorf("cudahostemu: warp %d op %d: %w", warp.ID, pc[w], err)
			}
			tmemLive = append(tmemLive, len(s.TMEMMap))
			s.TMEMMap = append(s.TMEMMap, TMEMColumnSlice{StartCol: col, NumCols: op.Cols, AllocCycle: t, Label: op.Label})
			tmemUsed += op.Cols
			s.PeakTMEMCols = max(s.PeakTMEMCols, tmemUsed)
			clock[w] = t + uint64(a.AllocIssue)
			span(name, warp.Role, warp.ID, t, clock[w], map[string]any{"cols": op.Cols, "start": col})
		case OpDealloc:
			i := slices.IndexFunc(tmemLive, func(i int) bool { return s.TMEMMap[i].Label == op.Label })
			if i < 0 {
				return TimelineSummary{}, fmt.Errorf("cudahostemu: warp %d op %d: dealloc of unallocated %q", warp.ID, pc[w], op.Label)
			}
			s.TMEMMap[tmemLive[i]].FreeCycle = t
			tmemUsed -= s.TMEMMap[tmemLive[i]].NumCols
			tmemLive = slices.Delete(tmemLive, i, i+1)
			clock[w] = t + uint64(a.AllocIssue)
			span(name, warp.Role, warp.ID, t, clock[w], nil)
		case OpLoad:
			clock[w] = t + uint64(a.TMEMIssue)
			ldReady[w] = max(ldReady[w], clock[w], tcFree+uint64(a.TMEMLdLag))
			span(name, warp.Role, warp.ID, t, clock[w], nil)
		case OpWaitLoad:
			if waitFollowsMMA(warp.Ops[:pc[w]]) {
				s.Findings = append(s.Findings, TimelineFinding{Kind: "wait-ld-after-mma", Warp: warp.ID, Op: pc[w], Detail: "tcgen05.wait::ld follows tcgen05.mma with only tcgen05.ld between"})
			}
			wake := max(t, ldReady[w])
			if wake > t {
				bubble := stall(w, t, wake)
				span("STALL_TMEM_LD", PipeStallWait, warp.ID, t, wake, map[string]any{"bubble": bubble})
			}
			clock[w] = wake + 1
			span(name, warp.Role, warp.ID, wake, clock[w], nil)
		case OpCompute:
			clock[w] = t + uint64(op.Cycles)
			span(name, warp.Role, warp.ID, t, clock[w], nil)
		default:
			return TimelineSummary{}, fmt.Errorf("cudahostemu: warp %d op %d: unknown kind %d", warp.ID, pc[w], op.Kind)
		}
		pc[w]++
		s.TotalCycles = max(s.TotalCycles, clock[w])
	}
	s.TotalCycles = max(s.TotalCycles, tmaFree, tcFree)
	for _, iv := range tmaBusy {
		s.TotalCycles = max(s.TotalCycles, iv.end)
	}
	for _, i := range tmemLive {
		s.TMEMMap[i].FreeCycle = s.TotalCycles
	}

	tcU := union(tcBusy)
	tmaU := union(tmaBusy)
	s.TensorCoreActiveCyc = length(tcU)
	s.TMAActiveCyc = length(tmaU)
	if s.TotalCycles > 0 {
		s.TensorUtilPct = 100 * float64(s.TensorCoreActiveCyc) / float64(s.TotalCycles)
		if r := float64(s.ConsumerStallCyc) / float64(s.TotalCycles); r > ConsumerStallLimit {
			s.Findings = append(s.Findings, TimelineFinding{Kind: "consumer-stall", Warp: -1, Op: -1, Detail: fmt.Sprintf("consumer warps stall %.0f%% of %d cycles", 100*r, s.TotalCycles)})
		}
	}
	if s.TMAActiveCyc > 0 {
		s.TMAOverlapPct = 100 * float64(length(intersect(tmaU, tcU))) / float64(s.TMAActiveCyc)
	}
	return s, nil
}

// waitFollowsMMA reports whether the last instruction of ops, ignoring
// tcgen05.ld, is an MMA.
func waitFollowsMMA(ops []TimelineOp) bool {
	for i := len(ops) - 1; i >= 0; i-- {
		switch ops[i].Kind {
		case OpLoad:
			continue
		case OpMMA:
			return true
		}
		return false
	}
	return false
}

// tmemFit returns the first column where cols columns are free, allocating
// as tcgen05.alloc does: cols is a power of two of at least 32, and the
// allocation is aligned to its size.
func tmemFit(m []TMEMColumnSlice, live []int, cols, total int) (int, error) {
	if cols < 32 || cols > total || cols&(cols-1) != 0 {
		return 0, fmt.Errorf("tcgen05.alloc of %d columns: want a power of two in [32, %d]", cols, total)
	}
next:
	for col := 0; col+cols <= total; col += cols {
		for _, i := range live {
			if col < m[i].StartCol+m[i].NumCols && m[i].StartCol < col+cols {
				continue next
			}
		}
		return col, nil
	}
	return 0, fmt.Errorf("tcgen05.alloc of %d columns: tensor memory full", cols)
}

// union returns the sorted, disjoint union of ivs.
func union(ivs []interval) []interval {
	ivs = slices.Clone(ivs)
	slices.SortFunc(ivs, func(a, b interval) int {
		switch {
		case a.start < b.start:
			return -1
		case a.start > b.start:
			return 1
		}
		return 0
	})
	var out []interval
	for _, iv := range ivs {
		if iv.end <= iv.start {
			continue
		}
		if k := len(out) - 1; k >= 0 && iv.start <= out[k].end {
			out[k].end = max(out[k].end, iv.end)
			continue
		}
		out = append(out, iv)
	}
	return out
}

// intersect returns the intersection of two sorted disjoint interval lists.
func intersect(a, b []interval) []interval {
	var out []interval
	for i, j := 0, 0; i < len(a) && j < len(b); {
		lo, hi := max(a[i].start, b[j].start), min(a[i].end, b[j].end)
		if lo < hi {
			out = append(out, interval{lo, hi})
		}
		if a[i].end < b[j].end {
			i++
		} else {
			j++
		}
	}
	return out
}

func length(ivs []interval) uint64 {
	var n uint64
	for _, iv := range ivs {
		n += iv.end - iv.start
	}
	return n
}

// traceEvent is one Chrome trace event.
type traceEvent struct {
	Name string         `json:"name,omitempty"`
	Cat  string         `json:"cat,omitempty"`
	Ph   string         `json:"ph"`
	Pid  int            `json:"pid"`
	Tid  int            `json:"tid"`
	Ts   uint64         `json:"ts"`
	Dur  *uint64        `json:"dur,omitempty"`
	ID   uint64         `json:"id,omitempty"`
	BP   string         `json:"bp,omitempty"`
	Args map[string]any `json:"args,omitempty"`
}

// WritePerfettoJSON writes s as a Chrome trace-event JSON document, which
// ui.perfetto.dev and chrome://tracing open. One cycle is written as one
// microsecond. The document holds process and thread names, an "X" event
// per span, an "s" and "f" event pair per flow, and a "C" counter track of
// allocated tensor-memory columns.
func (s TimelineSummary) WritePerfettoJSON(w io.Writer) error {
	var ev []traceEvent
	meta := func(name string, tid int, value string) {
		ev = append(ev, traceEvent{Name: name, Ph: "M", Pid: s.CTA, Tid: tid, Args: map[string]any{"name": value}})
	}
	meta("process_name", 0, fmt.Sprintf("CTA %d (%s)", s.CTA, s.Arch))
	named := map[int]bool{}
	for _, sp := range s.Spans {
		if named[sp.Track] {
			continue
		}
		named[sp.Track] = true
		switch sp.Track {
		case TrackTMAEngine:
			meta("thread_name", sp.Track, "TMA Engine")
		case TrackTensorCore:
			meta("thread_name", sp.Track, "Tensor Core")
		default:
			if sp.Category == PipeProducerWarp || sp.Category == PipeConsumerWarp {
				role := "Producer"
				if sp.Category == PipeConsumerWarp {
					role = "Consumer"
				}
				meta("thread_name", sp.Track, fmt.Sprintf("%s Warp %d", role, sp.Track))
			} else {
				named[sp.Track] = false
			}
		}
	}
	for _, sp := range s.Spans {
		dur := sp.DurCycles
		ev = append(ev, traceEvent{Name: sp.Name, Cat: string(sp.Category), Ph: "X", Pid: s.CTA, Tid: sp.Track, Ts: sp.StartCycle, Dur: &dur, Args: sp.Args})
	}
	for _, f := range s.Flows {
		args := map[string]any{"mbarrier": f.Barrier, "phase": f.Phase}
		ev = append(ev,
			traceEvent{Name: "mbarrier", Cat: "flow", Ph: "s", Pid: s.CTA, Tid: f.FromTrack, Ts: f.FromCycle, ID: f.ID, Args: args},
			traceEvent{Name: "mbarrier", Cat: "flow", Ph: "f", BP: "e", Pid: s.CTA, Tid: f.ToTrack, Ts: f.ToCycle, ID: f.ID, Args: args})
	}
	type change struct {
		at    uint64
		delta int
	}
	var changes []change
	for _, t := range s.TMEMMap {
		changes = append(changes, change{t.AllocCycle, t.NumCols}, change{t.FreeCycle, -t.NumCols})
	}
	slices.SortStableFunc(changes, func(a, b change) int {
		switch {
		case a.at < b.at:
			return -1
		case a.at > b.at:
			return 1
		}
		return 0
	})
	cols := 0
	for i, c := range changes {
		cols += c.delta
		if i+1 < len(changes) && changes[i+1].at == c.at {
			continue
		}
		ev = append(ev, traceEvent{Name: "TMEM columns", Cat: string(PipeTMEMSpace), Ph: "C", Pid: s.CTA, Ts: c.at, Args: map[string]any{"allocated": cols}})
	}
	doc := struct {
		TraceEvents     []traceEvent   `json:"traceEvents"`
		DisplayTimeUnit string         `json:"displayTimeUnit"`
		OtherData       map[string]any `json:"otherData"`
	}{ev, "ns", map[string]any{
		"arch":            s.Arch,
		"time_unit":       "cycle",
		"total_cycles":    s.TotalCycles,
		"tensor_util_pct": s.TensorUtilPct,
		"tma_overlap_pct": s.TMAOverlapPct,
	}}
	enc := json.NewEncoder(w)
	return enc.Encode(doc)
}

// A GEMMTimeline describes the main loop of one CTA's GEMM tile for
// [GEMMTimelineProgram].
type GEMMTimeline struct {
	M, N, K int // MMA shape per K step
	KSteps  int
	Stages  int // shared-memory ring depth
	// Specialized puts TMA issue in a producer warp and MMA issue in a
	// consumer warp. Otherwise one warp loads, waits and multiplies in turn.
	Specialized bool
	// WaitLoadInLoop reads the accumulator with tcgen05.ld and waits for
	// it in every K step, right after the MMA, instead of once after the
	// loop.
	WaitLoadInLoop bool
	ElemBytes      int // operand element size; 0 means 2 (BF16)
}

// GEMMTimelineProgram returns the modeled instruction streams of g's main
// loop: per K step, a TMA load of the A and B tiles into stage k%Stages, a
// wait on that stage's full barrier, an MMA, and a commit to its empty
// barrier, which the load of step k+Stages waits on. The accumulator lives
// in tensor memory, allocated before the loop and read after it.
func GEMMTimelineProgram(g GEMMTimeline) (TimelineProgram, error) {
	if g.M <= 0 || g.N <= 0 || g.K <= 0 || g.KSteps <= 0 || g.Stages <= 0 {
		return TimelineProgram{}, fmt.Errorf("cudahostemu: bad GEMM timeline %+v", g)
	}
	eb := g.ElemBytes
	if eb == 0 {
		eb = 2
	}
	bytes := (g.M + g.N) * g.K * eb
	cols := 32
	for cols < g.N {
		cols *= 2
	}
	// Barriers: full[s] = s, empty[s] = Stages+s, done = 2*Stages.
	full := func(k int) int { return k % g.Stages }
	empty := func(k int) int { return g.Stages + k%g.Stages }
	done := 2 * g.Stages
	p := TimelineProgram{Barriers: slices.Repeat([]int{1}, 2*g.Stages+1)}

	load := func(k int) []TimelineOp {
		var ops []TimelineOp
		if k >= g.Stages {
			ops = append(ops, TimelineOp{Kind: OpWait, Bar: empty(k), Phase: k/g.Stages - 1})
		}
		return append(ops, TimelineOp{Kind: OpTMALoad, Bar: full(k), Bytes: bytes})
	}
	mma := func(k int) []TimelineOp {
		ops := []TimelineOp{
			{Kind: OpWait, Bar: full(k), Phase: k / g.Stages},
			{Kind: OpMMA, M: g.M, N: g.N, K: g.K},
		}
		if g.WaitLoadInLoop {
			ops = append(ops, TimelineOp{Kind: OpLoad}, TimelineOp{Kind: OpWaitLoad})
		}
		return append(ops, TimelineOp{Kind: OpCommit, Bar: empty(k)})
	}
	epilogue := []TimelineOp{
		{Kind: OpCommit, Bar: done},
		{Kind: OpWait, Bar: done, Phase: 0},
		{Kind: OpLoad},
		{Kind: OpWaitLoad},
		{Kind: OpDealloc, Label: "accum"},
	}
	alloc := TimelineOp{Kind: OpAlloc, Cols: cols, Label: "accum"}

	if g.Specialized {
		producer := TimelineWarp{ID: 0, Role: PipeProducerWarp}
		consumer := TimelineWarp{ID: 4, Role: PipeConsumerWarp, Ops: []TimelineOp{alloc}}
		for k := range g.KSteps {
			producer.Ops = append(producer.Ops, load(k)...)
			consumer.Ops = append(consumer.Ops, mma(k)...)
		}
		consumer.Ops = append(consumer.Ops, epilogue...)
		p.Warps = []TimelineWarp{producer, consumer}
		return p, nil
	}
	single := TimelineWarp{ID: 0, Role: PipeConsumerWarp, Ops: []TimelineOp{alloc}}
	// One warp keeps Stages-1 loads in flight ahead of its MMA.
	ahead := g.Stages - 1
	for k := range min(ahead, g.KSteps) {
		single.Ops = append(single.Ops, load(k)...)
	}
	for k := range g.KSteps {
		if j := k + ahead; j < g.KSteps {
			single.Ops = append(single.Ops, load(j)...)
		}
		single.Ops = append(single.Ops, mma(k)...)
	}
	single.Ops = append(single.Ops, epilogue...)
	p.Warps = []TimelineWarp{single}
	return p, nil
}
