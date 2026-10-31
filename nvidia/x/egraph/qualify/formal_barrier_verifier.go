package qualify

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// BarrierViolationKind classifies a formal symbolic verification failure in a
// persistent [hopper.Megakernel].
type BarrierViolationKind string

const (
	// ViolationRAWDominance reports a Read-After-Write (RAW) dataflow edge
	// (writer stage p < reader stage s on buffer b) where stage s's WaitSlots
	// chain does not transitively dominate p.
	ViolationRAWDominance BarrierViolationKind = "raw_dominance"

	// ViolationArenaAliasing reports a Write-After-Read (WAR) or
	// Write-After-Write (WAW) hazard on a reused physical arena buffer (such as
	// eg_arena_even / eg_arena_odd) where the second writer stage does not
	// transitively dominate the last reader (or first writer) of the prior
	// logical intermediate via WaitSlots.
	ViolationArenaAliasing BarrierViolationKind = "arena_aliasing"

	// ViolationNamedBarrierUniformity reports a sub-CTA stage (0 < Threads < 256)
	// whose thread count is not a multiple of warp size (32), is missing the
	// entry guard "if (threadIdx.x >= <Threads>) return;", or retains an
	// unrewritten 256-thread __syncthreads() instead of "bar.sync 1, <Threads>;".
	ViolationNamedBarrierUniformity BarrierViolationKind = "named_barrier_uniformity"

	// ViolationEpochSlotOverlap reports that a stage's SignalSlot or WaitSlots
	// collides with m.EpochSlot() or exceeds m.ReadyWords().
	ViolationEpochSlotOverlap BarrierViolationKind = "epoch_slot_overlap"
)

// BarrierViolation describes one formal verification counterexample with exact
// stage indices and buffer names.
type BarrierViolation struct {
	Kind          BarrierViolationKind `json:"kind"`
	Stage         int                  `json:"stage"`
	UpstreamStage int                  `json:"upstream_stage"`
	ReaderStage   int                  `json:"reader_stage,omitempty"`
	Buffer        string               `json:"buffer,omitempty"`
	Message       string               `json:"message"`
}

// FormalBarrierReport summarizes static symbolic dataflow, barrier-dominance,
// ping-pong arena liveness, named-barrier warp uniformity, and epoch-slot
// non-overlap verification for a [hopper.Megakernel].
type FormalBarrierReport struct {
	MegakernelName             string             `json:"megakernel_name"`
	NumStages                  int                `json:"num_stages"`
	TotalBlocks                int                `json:"total_blocks"`
	EpochSlot                  int                `json:"epoch_slot"`
	ReadyWords                 int                `json:"ready_words"`
	RAWBarrierDominanceOK      bool               `json:"raw_barrier_dominance_ok"`
	ArenaAliasingSafetyOK      bool               `json:"arena_aliasing_safety_ok"`
	NamedBarrierUniformityOK   bool               `json:"named_barrier_uniformity_ok"`
	EpochSlotNonOverlapOK      bool               `json:"epoch_slot_non_overlap_ok"`
	CheckedRAWEdges            int                `json:"checked_raw_edges"`
	CheckedArenaAliases        int                `json:"checked_arena_aliases"`
	CheckedSubCTABarrierStages int                `json:"checked_sub_cta_barrier_stages"`
	Violations                 []BarrierViolation `json:"violations,omitempty"`
}

// Passed reports whether all four formal safety properties hold with zero
// violations.
func (r *FormalBarrierReport) Passed() bool {
	if r == nil {
		return false
	}
	return r.RAWBarrierDominanceOK &&
		r.ArenaAliasingSafetyOK &&
		r.NamedBarrierUniformityOK &&
		r.EpochSlotNonOverlapOK &&
		len(r.Violations) == 0
}

// VerifyMegakernelBarrierSafety performs static symbolic dataflow and
// barrier-dominance verification over m.Stages and m's input/temp/output
// buffers, proving:
//  1. RAW (Read-After-Write) Barrier Dominance: for every buffer b read by
//     stage s that was written by an earlier stage p < s, stage s's WaitSlots
//     chain transitively dominates p.
//  2. WAR / WAW Arena Aliasing Safety: when two distinct logical intermediates
//     share the same physical ping-pong arena buffer (e.g. eg_arena_even /
//     eg_arena_odd) or any reused temporary buffer, the second writer stage
//     transitively dominates the last reader (and all readers) of the first
//     logical value via WaitSlots.
//  3. Named-Barrier Thread Uniformity: every stage with 0 < Threads < 256 has
//     Threads a multiple of 32 (warp size), guards entry with
//     "if (threadIdx.x >= <Threads>) return;", and synchronizes via
//     "bar.sync 1, <Threads>;".
//  4. Device-Epoch Slot Non-Overlap: m.EpochSlot() >= len(m.Stages)-1 and no
//     stage SignalSlot or WaitSlots entry collides with m.EpochSlot().
func VerifyMegakernelBarrierSafety(m *hopper.Megakernel) (*FormalBarrierReport, error) {
	if m == nil || len(m.Stages) == 0 {
		return nil, errors.New("verify megakernel barrier safety: empty megakernel")
	}

	grid, _ := m.Grid()
	nStages := len(m.Stages)
	epochSlot := m.EpochSlot()
	readyWords := m.ReadyWords()

	rep := &FormalBarrierReport{
		MegakernelName:           m.Name,
		NumStages:                nStages,
		TotalBlocks:              grid[0],
		EpochSlot:                epochSlot,
		ReadyWords:               readyWords,
		RAWBarrierDominanceOK:    true,
		ArenaAliasingSafetyOK:    true,
		NamedBarrierUniformityOK: true,
		EpochSlotNonOverlapOK:    true,
	}

	// -------------------------------------------------------------------------
	// Property 4: Device-Epoch Slot Non-Overlap
	// -------------------------------------------------------------------------
	if epochSlot < nStages-1 || readyWords <= epochSlot {
		rep.EpochSlotNonOverlapOK = false
		rep.Violations = append(rep.Violations, BarrierViolation{
			Kind:          ViolationEpochSlotOverlap,
			Stage:         -1,
			UpstreamStage: -1,
			Message: fmt.Sprintf(
				"epoch slot %d (ready words %d) < len(Stages)-1 (%d)",
				epochSlot, readyWords, nStages-1,
			),
		})
	}

	// Map each signaled ready counter slot to the producer stage index that signals it.
	slotProducer := make(map[int]int, nStages)
	for s, st := range m.Stages {
		if st.SignalSlot >= 0 {
			if st.SignalSlot == epochSlot || st.SignalSlot >= readyWords {
				rep.EpochSlotNonOverlapOK = false
				rep.Violations = append(rep.Violations, BarrierViolation{
					Kind:          ViolationEpochSlotOverlap,
					Stage:         s,
					UpstreamStage: s,
					Message: fmt.Sprintf(
						"stage %d (%s) SignalSlot %d collides with epoch slot %d (ready words %d)",
						s, st.Name, st.SignalSlot, epochSlot, readyWords,
					),
				})
			}
			slotProducer[st.SignalSlot] = s
		}
		for _, ws := range st.WaitSlots {
			if ws == epochSlot || ws < 0 || ws >= readyWords {
				rep.EpochSlotNonOverlapOK = false
				rep.Violations = append(rep.Violations, BarrierViolation{
					Kind:          ViolationEpochSlotOverlap,
					Stage:         s,
					UpstreamStage: ws,
					Message: fmt.Sprintf(
						"stage %d (%s) WaitSlots entry %d is invalid or collides with epoch slot %d",
						s, st.Name, ws, epochSlot,
					),
				})
			}
		}
	}

	// Build transitive barrier-dominance relation: dominates[s][p] is true iff
	// stage s transitively waits on stage p (with 0 <= p < s) through valid
	// signaled WaitSlots.
	dominates := make([][]bool, nStages)
	for s := range nStages {
		dominates[s] = make([]bool, nStages)
		st := m.Stages[s]
		for _, ws := range st.WaitSlots {
			p, ok := slotProducer[ws]
			if !ok && ws >= 0 && ws < s && m.Stages[ws].SignalSlot == ws {
				p = ws
				ok = true
			}
			if !ok || p < 0 || p >= s || ws == epochSlot {
				continue
			}
			dominates[s][p] = true
			for anc := 0; anc < p; anc++ {
				if dominates[p][anc] {
					dominates[s][anc] = true
				}
			}
		}
	}

	// Extract symbolic read/write buffer sets for each stage.
	stageReads := make([][]string, nStages)
	stageWrites := make([][]string, nStages)
	stageLogicalWrites := make([][]string, nStages)
	for s, st := range m.Stages {
		reads, writes, logicalWrites := extractStageBufferAccesses(st)
		stageReads[s] = reads
		stageWrites[s] = writes
		stageLogicalWrites[s] = logicalWrites
	}

	// -------------------------------------------------------------------------
	// Property 1: RAW (Read-After-Write) Barrier Dominance
	// -------------------------------------------------------------------------
	for s := 0; s < nStages; s++ {
		for _, buf := range stageReads[s] {
			if buf == "" {
				continue
			}
			// Find all earlier stages p < s that wrote buf (most recent writer
			// and all prior writers).
			for p := s - 1; p >= 0; p-- {
				if !slices.Contains(stageWrites[p], buf) {
					continue
				}
				rep.CheckedRAWEdges++
				if !dominates[s][p] {
					rep.RAWBarrierDominanceOK = false
					rep.Violations = append(rep.Violations, BarrierViolation{
						Kind:          ViolationRAWDominance,
						Stage:         s,
						UpstreamStage: p,
						Buffer:        buf,
						Message: fmt.Sprintf(
							"RAW hazard on buffer %q: reader stage %d (%s, WaitSlots=%v) does not dominate writer stage %d (%s, SignalSlot=%d)",
							buf, s, m.Stages[s].Name, m.Stages[s].WaitSlots, p, m.Stages[p].Name, m.Stages[p].SignalSlot,
						),
					})
				}
				break // Most recent writer checked; earlier writers are covered by WAW/WAR dominance.
			}
		}
	}

	// -------------------------------------------------------------------------
	// Property 2: WAR / WAW Arena Aliasing Safety
	// -------------------------------------------------------------------------
	type writeEpoch struct {
		writerStage int
		logicalName string
		readers     []int
	}
	historyByBuf := make(map[string]*writeEpoch)

	for s := 0; s < nStages; s++ {
		// Record reads first for any buffers not written in the exact same in-place op,
		// then process writes to detect reuse of a physical buffer.
		for idx, buf := range stageWrites[s] {
			if buf == "" {
				continue
			}
			logical := buf
			if idx < len(stageLogicalWrites[s]) && stageLogicalWrites[s][idx] != "" {
				logical = stageLogicalWrites[s][idx]
			}
			prev, exists := historyByBuf[buf]
			if exists {
				rep.CheckedArenaAliases++
				w1 := prev.writerStage
				// WAW check: second writer s must dominate first writer w1.
				if !dominates[s][w1] {
					rep.ArenaAliasingSafetyOK = false
					rep.Violations = append(rep.Violations, BarrierViolation{
						Kind:          ViolationArenaAliasing,
						Stage:         s,
						UpstreamStage: w1,
						Buffer:        buf,
						Message: fmt.Sprintf(
							"WAW arena aliasing hazard on buffer %q: second writer stage %d (%s, logical %q) does not dominate first writer stage %d (%s, logical %q)",
							buf, s, m.Stages[s].Name, logical, w1, m.Stages[w1].Name, prev.logicalName,
						),
					})
				}
				// WAR check: second writer s must dominate every reader r in (w1, s)
				// (and in particular the last reader) of the first logical value.
				for _, r := range prev.readers {
					if r <= w1 || r >= s {
						continue
					}
					if !dominates[s][r] {
						rep.ArenaAliasingSafetyOK = false
						rep.Violations = append(rep.Violations, BarrierViolation{
							Kind:          ViolationArenaAliasing,
							Stage:         s,
							UpstreamStage: w1,
							ReaderStage:   r,
							Buffer:        buf,
							Message: fmt.Sprintf(
								"WAR arena aliasing hazard on buffer %q: second writer stage %d (%s, logical %q, WaitSlots=%v) does not dominate reader stage %d (%s) of first writer stage %d (%s, logical %q)",
								buf, s, m.Stages[s].Name, logical, m.Stages[s].WaitSlots, r, m.Stages[r].Name, w1, m.Stages[w1].Name, prev.logicalName,
							),
						})
					}
				}
			}
			historyByBuf[buf] = &writeEpoch{
				writerStage: s,
				logicalName: logical,
			}
		}

		for _, buf := range stageReads[s] {
			if ep, ok := historyByBuf[buf]; ok && ep.writerStage < s {
				if !slices.Contains(ep.readers, s) {
					ep.readers = append(ep.readers, s)
				}
			}
		}
	}

	// -------------------------------------------------------------------------
	// Property 3: Named-Barrier Thread Uniformity
	// -------------------------------------------------------------------------
	cudaSrc, cudaErr := m.CUDA()
	for s, st := range m.Stages {
		threads := st.Threads
		if threads == 0 {
			threads = 256
		}
		if threads < 0 || threads > 256 {
			rep.NamedBarrierUniformityOK = false
			rep.Violations = append(rep.Violations, BarrierViolation{
				Kind:          ViolationNamedBarrierUniformity,
				Stage:         s,
				UpstreamStage: s,
				Message: fmt.Sprintf(
					"stage %d (%s) has Threads=%d outside valid CTA range 1..256",
					s, st.Name, st.Threads,
				),
			})
			continue
		}
		if threads%32 != 0 {
			rep.NamedBarrierUniformityOK = false
			rep.Violations = append(rep.Violations, BarrierViolation{
				Kind:          ViolationNamedBarrierUniformity,
				Stage:         s,
				UpstreamStage: s,
				Message: fmt.Sprintf(
					"stage %d (%s) has non-warp-multiple Threads=%d (must be a multiple of 32 for bar.sync uniformity)",
					s, st.Name, threads,
				),
			})
			continue
		}
		if threads < 256 {
			rep.CheckedSubCTABarrierStages++
			if cudaErr != nil {
				rep.NamedBarrierUniformityOK = false
				rep.Violations = append(rep.Violations, BarrierViolation{
					Kind:          ViolationNamedBarrierUniformity,
					Stage:         s,
					UpstreamStage: s,
					Message: fmt.Sprintf(
						"stage %d (%s) with Threads=%d failed CUDA emission: %v",
						s, st.Name, threads, cudaErr,
					),
				})
				continue
			}
			devFn := extractStageDeviceFunction(cudaSrc, st.Name)
			wantGuard := fmt.Sprintf("if (threadIdx.x >= %d) return;", threads)
			wantBarSync := fmt.Sprintf("bar.sync 1, %d;", threads)
			if !strings.Contains(devFn, wantGuard) {
				rep.NamedBarrierUniformityOK = false
				rep.Violations = append(rep.Violations, BarrierViolation{
					Kind:          ViolationNamedBarrierUniformity,
					Stage:         s,
					UpstreamStage: s,
					Message: fmt.Sprintf(
						"stage %d (%s) with Threads=%d missing entry guard %q",
						s, st.Name, threads, wantGuard,
					),
				})
			}
			if strings.Contains(devFn, "__syncthreads()") {
				rep.NamedBarrierUniformityOK = false
				rep.Violations = append(rep.Violations, BarrierViolation{
					Kind:          ViolationNamedBarrierUniformity,
					Stage:         s,
					UpstreamStage: s,
					Message: fmt.Sprintf(
						"stage %d (%s) with Threads=%d contains raw __syncthreads() instead of %q",
						s, st.Name, threads, wantBarSync,
					),
				})
			}
			if strings.Contains(devFn, "bar.sync") && !strings.Contains(devFn, wantBarSync) {
				rep.NamedBarrierUniformityOK = false
				rep.Violations = append(rep.Violations, BarrierViolation{
					Kind:          ViolationNamedBarrierUniformity,
					Stage:         s,
					UpstreamStage: s,
					Message: fmt.Sprintf(
						"stage %d (%s) with Threads=%d uses mismatched bar.sync count, want %q",
						s, st.Name, threads, wantBarSync,
					),
				})
			}
		}
	}

	if !rep.Passed() {
		msgs := make([]string, len(rep.Violations))
		for i, v := range rep.Violations {
			msgs[i] = v.Message
		}
		return rep, fmt.Errorf("megakernel %s barrier verification failed (%d violations): %s",
			m.Name, len(rep.Violations), strings.Join(msgs, "; "))
	}
	return rep, nil
}

func extractStageDeviceFunction(cudaSrc, stageName string) string {
	header := fmt.Sprintf("EG_DEV void %s_dev(", stageName)
	start := strings.Index(cudaSrc, header)
	if start < 0 {
		return ""
	}
	rest := cudaSrc[start:]
	if next := strings.Index(rest[len(header):], "\nEG_DEV void "); next >= 0 {
		return rest[:len(header)+next]
	}
	if globalIdx := strings.Index(rest, "\n__global__ void "); globalIdx >= 0 {
		return rest[:globalIdx]
	}
	return rest
}

func extractStageBufferAccesses(st hopper.MegaStage) (reads, writes, logicalWrites []string) {
	addRead := func(name string) {
		if name != "" && name != "nullptr" && name != "ws" && !slices.Contains(reads, name) {
			reads = append(reads, name)
		}
	}
	addWrite := func(physName, logicalName string) {
		if physName == "" || physName == "nullptr" || physName == "ws" {
			return
		}
		if idx := slices.Index(writes, physName); idx >= 0 {
			if logicalName != "" && logicalWrites[idx] == physName {
				logicalWrites[idx] = logicalName
			}
			return
		}
		if logicalName == "" {
			logicalName = physName
		}
		writes = append(writes, physName)
		logicalWrites = append(logicalWrites, logicalName)
	}

	pk := st.Kernel
	switch {
	case pk.Row != nil && (len(pk.Row.Inputs) > 0 || len(pk.Row.Outputs) > 0):
		for _, in := range pk.Row.Inputs {
			buf := in.Buffer
			if buf == "" {
				buf = in.Name
			}
			addRead(buf)
		}
		for i, out := range pk.Row.Outputs {
			logical := out.Name
			if i < len(st.Outputs) && st.Outputs[i].Name != "" {
				logical = st.Outputs[i].Name
			}
			addWrite(out.Name, logical)
			if out.Index != "" {
				addRead(out.Index)
			}
		}

	case pk.RoPE != nil:
		if len(pk.Args) >= 3 {
			nIn := len(pk.Args) - 3
			if nIn < 3 {
				nIn = min(3, len(pk.Args))
			}
			for _, a := range pk.Args[:nIn] {
				addRead(a)
			}
			for i, a := range pk.Args[nIn:] {
				logical := a
				if i < len(st.Outputs) && st.Outputs[i].Name != "" {
					logical = st.Outputs[i].Name
				}
				addWrite(a, logical)
			}
		}

	case pk.TopK != nil:
		if len(pk.Args) >= 3 {
			for _, a := range pk.Args[:len(pk.Args)-2] {
				addRead(a)
			}
			for i, a := range pk.Args[len(pk.Args)-2:] {
				logical := a
				if i < len(st.Outputs) && st.Outputs[i].Name != "" {
					logical = st.Outputs[i].Name
				}
				addWrite(a, logical)
			}
		}

	case pk.Align != nil:
		if len(pk.Args) >= 5 {
			for _, a := range pk.Args[:2] {
				addRead(a)
			}
			for i, a := range pk.Args[2:] {
				logical := a
				if i < len(st.Outputs) && st.Outputs[i].Name != "" {
					logical = st.Outputs[i].Name
				}
				addWrite(a, logical)
			}
		}

	case pk.Tile != nil:
		end := len(pk.Args)
		if pk.Tile.StreamK > 0 && end > 0 {
			end--
		}
		nOut := 1
		if pk.Tile.SumSq {
			nOut = 2
		}
		if end >= nOut {
			for _, a := range pk.Args[:end-nOut] {
				addRead(a)
			}
			for i, a := range pk.Args[end-nOut : end] {
				logical := a
				if i < len(st.Outputs) && st.Outputs[i].Name != "" {
					logical = st.Outputs[i].Name
				}
				addWrite(a, logical)
			}
		}

	case pk.GEMM != nil:
		end := len(pk.Args)
		if pk.GEMM.SplitK > 1 && end > 0 {
			end-- // trailing split-K workspace
		}
		if end >= 1 {
			for _, a := range pk.Args[:end-1] {
				addRead(a)
			}
			logical := pk.Args[end-1]
			if len(st.Outputs) > 0 && st.Outputs[0].Name != "" {
				logical = st.Outputs[0].Name
			}
			addWrite(pk.Args[end-1], logical)
		}

	case pk.Attention != nil:
		if len(st.Outputs) > 0 && st.Outputs[0].Name != "" {
			outBuf := st.Outputs[0].Name
			for _, a := range pk.Args {
				if a == outBuf {
					addWrite(a, outBuf)
					break
				}
				addRead(a)
			}
		} else if len(pk.Args) >= 1 {
			for _, a := range pk.Args[:len(pk.Args)-1] {
				addRead(a)
			}
			addWrite(pk.Args[len(pk.Args)-1], pk.Args[len(pk.Args)-1])
		}

	default:
		// Generic fallback for MMA, Attention, ExpertSum, or synthetic stages.
		if len(pk.Args) >= 1 {
			nOut := max(len(st.Outputs), 1)
			if nOut > len(pk.Args) {
				nOut = 1
			}
			split := len(pk.Args) - nOut
			for _, a := range pk.Args[:split] {
				addRead(a)
			}
			for i, a := range pk.Args[split:] {
				logical := a
				if i < len(st.Outputs) && st.Outputs[i].Name != "" {
					logical = st.Outputs[i].Name
				}
				addWrite(a, logical)
			}
		}
	}

	// Also include any explicit st.Outputs not already recorded in writes.
	if len(writes) == 0 {
		for _, out := range st.Outputs {
			addWrite(out.Name, out.Name)
		}
	}
	return reads, writes, logicalWrites
}

// RewriteMegakernelPingPongArena aliases eligible single-consumer temporary
// buffers in m onto alternating physical ping-pong arena buffers
// ("eg_arena_even" and "eg_arena_odd"). When addWARWaitEdges is true, it
// inserts the required WAR WaitSlots edges from each prior logical reader to
// the next writer of the same ping-pong buffer; when false, it omits those WAR
// WaitSlots edges so formal verification tests can exercise premature arena
// reuse detection.
func RewriteMegakernelPingPongArena(m *hopper.Megakernel, addWARWaitEdges bool) (*hopper.Megakernel, error) {
	if m == nil || len(m.Stages) == 0 {
		return nil, errors.New("rewrite megakernel ping-pong arena: empty megakernel")
	}

	out := &hopper.Megakernel{
		Name:   m.Name,
		Inputs: slices.Clone(m.Inputs),
		Temps:  slices.Clone(m.Temps),
		Output: m.Output,
		Stages: make([]hopper.MegaStage, len(m.Stages)),
	}
	for i, st := range m.Stages {
		stCopy := st
		stCopy.WaitSlots = slices.Clone(st.WaitSlots)
		stCopy.Outputs = slices.Clone(st.Outputs)
		stCopy.Kernel.Args = slices.Clone(st.Kernel.Args)
		if st.Kernel.Row != nil {
			rk := *st.Kernel.Row
			rk.Inputs = slices.Clone(st.Kernel.Row.Inputs)
			rk.Outputs = slices.Clone(st.Kernel.Row.Outputs)
			rk.Values = slices.Clone(st.Kernel.Row.Values)
			stCopy.Kernel.Row = &rk
		}
		out.Stages[i] = stCopy
	}

	tempSet := make(map[string]hopper.Tensor, len(out.Temps))
	for _, t := range out.Temps {
		if t.Name != out.Output.Name {
			tempSet[t.Name] = t
		}
	}

	arenaNames := [2]string{"eg_arena_even", "eg_arena_odd"}
	lastReaderByArena := [2]int{-1, -1}
	lastWriterByArena := [2]int{-1, -1}
	maxElemsByArena := [2]int{0, 0}
	dtypeByArena := [2]hopper.DType{hopper.BF16, hopper.BF16}

	renameMap := make(map[string]string)
	parity := 0

	for s := range out.Stages {
		reads, writes, _ := extractStageBufferAccesses(out.Stages[s])
		// Update reader tracking for already-mapped temporaries.
		for _, rBuf := range reads {
			if mapped, ok := renameMap[rBuf]; ok {
				for p := range 2 {
					if mapped == arenaNames[p] && s > lastReaderByArena[p] {
						lastReaderByArena[p] = s
					}
				}
			}
		}

		for _, wBuf := range writes {
			t, isTemp := tempSet[wBuf]
			if !isTemp {
				continue
			}
			if _, already := renameMap[wBuf]; already {
				continue
			}
			slot := parity % 2
			parity++
			phys := arenaNames[slot]
			renameMap[wBuf] = phys

			elems := 1
			for _, d := range t.Shape {
				if d > 0 {
					elems *= d
				}
			}
			if elems > maxElemsByArena[slot] {
				maxElemsByArena[slot] = elems
				dtypeByArena[slot] = t.DType
			}

			if addWARWaitEdges {
				if r := lastReaderByArena[slot]; r >= 0 && r < s && out.Stages[r].SignalSlot >= 0 {
					if !slices.Contains(out.Stages[s].WaitSlots, r) {
						out.Stages[s].WaitSlots = append(out.Stages[s].WaitSlots, r)
						slices.Sort(out.Stages[s].WaitSlots)
					}
				} else if w := lastWriterByArena[slot]; w >= 0 && w < s && out.Stages[w].SignalSlot >= 0 {
					if !slices.Contains(out.Stages[s].WaitSlots, w) {
						out.Stages[s].WaitSlots = append(out.Stages[s].WaitSlots, w)
						slices.Sort(out.Stages[s].WaitSlots)
					}
				}
			}
			lastWriterByArena[slot] = s
		}
	}

	// Apply physical buffer renames to Kernel.Args and Row.Inputs / Row.Outputs,
	// while keeping logical names in st.Outputs for clear attribution.
	mapBuf := func(name string) string {
		if phys, ok := renameMap[name]; ok {
			return phys
		}
		return name
	}
	for s := range out.Stages {
		st := &out.Stages[s]
		for i, a := range st.Kernel.Args {
			st.Kernel.Args[i] = mapBuf(a)
		}
		if st.Kernel.Row != nil {
			for i := range st.Kernel.Row.Inputs {
				b := st.Kernel.Row.Inputs[i].Buffer
				if b == "" {
					b = st.Kernel.Row.Inputs[i].Name
				}
				if mapped := mapBuf(b); mapped != b {
					st.Kernel.Row.Inputs[i].Buffer = mapped
				}
			}
			for i := range st.Kernel.Row.Outputs {
				st.Kernel.Row.Outputs[i].Name = mapBuf(st.Kernel.Row.Outputs[i].Name)
			}
		}
	}

	// Rebuild Temps with eg_arena_even and eg_arena_odd plus any unmapped temps.
	var newTemps []hopper.Tensor
	for p := range 2 {
		if maxElemsByArena[p] > 0 {
			newTemps = append(newTemps, hopper.Tensor{
				Name:  arenaNames[p],
				DType: dtypeByArena[p],
				Shape: []int{maxElemsByArena[p]},
			})
		}
	}
	for _, t := range out.Temps {
		if _, mapped := renameMap[t.Name]; !mapped {
			newTemps = append(newTemps, t)
		}
	}
	out.Temps = newTemps
	return out, nil
}
