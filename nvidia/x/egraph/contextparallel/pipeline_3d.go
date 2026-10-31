// Package contextparallel compiles persistent 3D Pipeline + Tensor + Context
// Parallel (PP x TP x CP) schedules, including Zero-Bubble ZB-V (V-shape
// bi-directional stage assignment with split B_X input-gradient and W
// weight-gradient backward steps), fused with intra-step Ring-CP KV exchange,
// TP NVSwitch multimem collectives, and .sys-scope P2P stage-handoff ready
// barriers.
package contextparallel

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// Parallel3DSpec configures a 3D Pipeline + Tensor + Context Parallel
// (`PP x TP x CP`) persistent schedule compiler.
type Parallel3DSpec struct {
	PPStages      int // Pipeline Parallel stages
	TPDegree      int // Tensor Parallel degree (intra-node NVSwitch)
	CPDegree      int // Context Parallel degree (Ring / Ulysses)
	NumMicroBatch int
	ZeroBubbleZBV bool // Split backward into B (input grad) and W (weight grad) for zero bubble
	MicroStepUs   float64
	P2PLatencyUs  float64
}

// Validate checks that s has valid positive dimensions and non-negative latencies.
func (s Parallel3DSpec) Validate() error {
	if s.PPStages < 1 {
		return fmt.Errorf("contextparallel: PPStages %d must be >= 1", s.PPStages)
	}
	if s.TPDegree < 1 {
		return fmt.Errorf("contextparallel: TPDegree %d must be >= 1", s.TPDegree)
	}
	if s.CPDegree < 1 {
		return fmt.Errorf("contextparallel: CPDegree %d must be >= 1", s.CPDegree)
	}
	if s.NumMicroBatch < 1 {
		return fmt.Errorf("contextparallel: NumMicroBatch %d must be >= 1", s.NumMicroBatch)
	}
	if s.MicroStepUs <= 0 {
		return fmt.Errorf("contextparallel: MicroStepUs %v must be > 0", s.MicroStepUs)
	}
	if s.P2PLatencyUs < 0 {
		return fmt.Errorf("contextparallel: P2PLatencyUs %v must be >= 0", s.P2PLatencyUs)
	}
	return nil
}

// StepPhase identifies a micro-step phase in a 3D parallel pipeline schedule.
type StepPhase string

const (
	// PhaseF is the Forward pass micro-step.
	PhaseF StepPhase = "F"
	// PhaseB is the Activation / Input-Gradient Backward micro-step (dX) on the
	// critical inter-stage path (or monolithic B+W when ZeroBubbleZBV is false).
	PhaseB StepPhase = "B"
	// PhaseW is the Weight-Gradient Backward micro-step (dW), decoupled from
	// inter-stage P2P dependencies to fill pipeline bubbles in ZB-V.
	PhaseW StepPhase = "W"
)

// P2PReadyBarrier describes a `.sys`-scope P2P stage-handoff ready barrier
// synchronizing producer and consumer PP stages across NVLink / IBGDA RDMA.
type P2PReadyBarrier struct {
	BarrierID      uint64
	Epoch          uint32
	MicroBatchID   int
	VirtualChunk   int
	Phase          StepPhase
	FromPPStage    int
	ToPPStage      int
	Scope          string // ".sys"
	ReleasePTX     string // e.g. st.release.sys.global.u32 [ready_flag], epoch;
	AcquireWaitPTX string // e.g. ld.acquire.sys.global.u32 %r, [ready_flag];
	ReadyAtUs      float64
}

// MicroStep3D represents a single scheduled `F`, `B`, or `W` micro-step on one
// Pipeline Parallel stage, fused with intra-step Ring-CP KV steps and TP
// `multimem` collectives.
type MicroStep3D struct {
	PPStage              int
	VirtualChunk         int // 0 for first V-leg (0..P-1), 1 for second V-leg (P-1..0) in ZB-V
	LayerChunk           int // logical model chunk index in 0..2*P-1 (ZB-V) or 0..P-1 (1F1B)
	MicroBatchID         int
	Phase                StepPhase
	StartUs              float64
	EndUs                float64
	ComputeDurationUs    float64
	RingCPSteps          int
	RingCPOverlapped     bool
	TPMultimemAllReduces int
	TPMultimemOp         string
	WaitBarrierID        int64 // -1 if no upstream P2P stage handoff needed
	SignalBarrierID      int64 // -1 if no downstream P2P stage handoff emitted
}

// Parallel3DSchedule is the compiled per-PP-stage timeline of `F`, `B`, and `W`
// micro-steps, exact pipeline bubble telemetry, and `.sys`-scope P2P
// stage-handoff ready-barrier metadata.
type Parallel3DSchedule struct {
	Spec                   Parallel3DSpec
	ScheduleKind           string // "ZeroBubble-ZB-V" or "Standard-1F1B"
	WorldSize              int    // PPStages * TPDegree * CPDegree
	Stages                 [][]MicroStep3D
	Barriers               []P2PReadyBarrier
	MakespanUs             float64
	StageComputeUs         float64
	BubbleFraction         float64
	Baseline1F1BBubble     float64
	BubbleReductionRatio   float64
	PeakInflight           []int
	SysBarrierPreludeCUDA  string
}

// Compile3DParallelSchedule compiles a complete per-PP-stage timeline of `F`
// (Forward), `B` (Activation Backward), and `W` (Weight Gradient) micro-steps
// using the Zero-Bubble `ZB-V` schedule when `spec.ZeroBubbleZBV` is true (or
// standard `1F1B` otherwise), fused with intra-step Ring-CP KV exchange and TP
// `multimem` collectives, and emits `.sys`-scope P2P stage-handoff ready
// barrier metadata.
func Compile3DParallelSchedule(spec Parallel3DSpec) (*Parallel3DSchedule, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}

	sched, err := compileScheduleMode(spec, spec.ZeroBubbleZBV)
	if err != nil {
		return nil, err
	}

	if spec.ZeroBubbleZBV {
		baseSpec := spec
		baseSpec.ZeroBubbleZBV = false
		base1F1B, err := compileScheduleMode(baseSpec, false)
		if err != nil {
			return nil, err
		}
		sched.Baseline1F1BBubble = base1F1B.BubbleFraction
		if base1F1B.BubbleFraction > 0 {
			sched.BubbleReductionRatio = (base1F1B.BubbleFraction - sched.BubbleFraction) / base1F1B.BubbleFraction
		}
	} else {
		sched.Baseline1F1BBubble = sched.BubbleFraction
	}

	return sched, nil
}

// vStageOf maps logical chunk position pos to its physical PP stage index.
// For 1F1B (numChunks == P), pos in [0, P) maps to stage pos.
// For ZB-V (numChunks == 2*P), V-shape folding places chunk pos in [0, P) on
// stage pos (forward leg 0 -> P-1) and chunk pos in [P, 2*P) on stage
// 2*P - 1 - pos (return leg P-1 -> 0), so both forward legs and backward legs
// begin and end on stage 0 and balance activation memory across all stages.
func vStageOf(P, numChunks, pos int) int {
	if numChunks == 2*P {
		if pos < P {
			return pos
		}
		return 2*P - 1 - pos
	}
	return pos
}

type opKey struct {
	phase StepPhase
	m     int
	pos   int
}

func compileScheduleMode(spec Parallel3DSpec, zbv bool) (*Parallel3DSchedule, error) {
	P := spec.PPStages
	M := spec.NumMicroBatch

	// In ZB-V (V-shape Zero Bubble), each stage holds 2 virtual chunks (total 2*P
	// chunks along the V-folded pipeline), and each chunk's backward is split into
	// B (input grad) and W (weight grad). Each micro-step on a half-layer chunk
	// takes 0.5 * MicroStepUs for F, 0.5 * MicroStepUs for B, and 0.5 * MicroStepUs
	// for W (total work per stage per microbatch = 2 chunks * 1.5 * MicroStepUs =
	// 3 * MicroStepUs).
	// In standard 1F1B, each stage holds 1 chunk (total P chunks), where F takes
	// MicroStepUs and monolithic backward (B followed immediately by W) takes
	// 2 * MicroStepUs (B = MicroStepUs, W = MicroStepUs), so total work per stage
	// per microbatch is also 3 * MicroStepUs.
	numChunks := P
	stepScale := 1.0
	kind := "Standard-1F1B"
	if zbv {
		numChunks = 2 * P
		stepScale = 0.5
		kind = "ZeroBubble-ZB-V"
	}

	fDur := spec.MicroStepUs * stepScale
	bDur := spec.MicroStepUs * stepScale
	wDur := spec.MicroStepUs * stepScale

	// Intra-step TP multimem all-reduces (2 per transformer block: OProj + FFN)
	// and Ring-CP KV steps (CPDegree steps overlapped with tile MMA).
	tpCollectives := 0
	tpOp := "none"
	if spec.TPDegree > 1 {
		tpCollectives = 2
		tpOp = "multimem.ld_reduce.relaxed.sys.global.add.bf16x2"
	}
	cpSteps := max(spec.CPDegree, 1)

	doneEnd := make(map[opKey]float64, 3*M*numChunks)
	signalID := make(map[opKey]int64, 2*M*numChunks)
	stages := make([][]MicroStep3D, P)
	freeAt := make([]float64, P)
	inflight := make([]int, P)
	chunkInflight := make([]int, numChunks)
	peakInflight := make([]int, P)
	var barriers []P2PReadyBarrier
	var nextBarrierID uint64 = 1

	isDone := func(ph StepPhase, m, pos int) bool {
		_, ok := doneEnd[opKey{ph, m, pos}]
		return ok
	}

	// readyTime returns (earliestStartUs, waitBarrierID, isReady).
	readyTime := func(ph StepPhase, m, pos int) (float64, int64, bool) {
		if isDone(ph, m, pos) {
			return 0, -1, false
		}
		myStage := vStageOf(P, numChunks, pos)
		switch ph {
		case PhaseF:
			if pos == 0 {
				return 0, -1, true
			}
			prevKey := opKey{PhaseF, m, pos - 1}
			endT, ok := doneEnd[prevKey]
			if !ok {
				return 0, -1, false
			}
			prevStage := vStageOf(P, numChunks, pos-1)
			p2p := 0.0
			if prevStage != myStage {
				p2p = spec.P2PLatencyUs
			}
			bid, hasBid := signalID[prevKey]
			if !hasBid {
				bid = -1
			}
			return endT + p2p, bid, true

		case PhaseB:
			if pos == numChunks-1 {
				fKey := opKey{PhaseF, m, pos}
				endT, ok := doneEnd[fKey]
				if !ok {
					return 0, -1, false
				}
				return endT, -1, true
			}
			// Must have completed own F(m, pos) and upstream backward.
			fEnd, okF := doneEnd[opKey{PhaseF, m, pos}]
			if !okF {
				return 0, -1, false
			}
			upPhase := PhaseB
			if !zbv {
				// Standard 1F1B has monolithic backward: stage pos+1 only sends dX
				// after its entire backward (B + W) finishes.
				upPhase = PhaseW
			}
			upKey := opKey{upPhase, m, pos + 1}
			upEnd, okUp := doneEnd[upKey]
			if !okUp {
				return 0, -1, false
			}
			upStage := vStageOf(P, numChunks, pos+1)
			p2p := 0.0
			if upStage != myStage {
				p2p = spec.P2PLatencyUs
			}
			bid, hasBid := signalID[opKey{PhaseB, m, pos + 1}]
			if !hasBid {
				bid = -1
			}
			return math.Max(fEnd, upEnd+p2p), bid, true

		case PhaseW:
			bEnd, ok := doneEnd[opKey{PhaseB, m, pos}]
			if !ok {
				return 0, -1, false
			}
			return bEnd, -1, true
		}
		return 0, -1, false
	}

	emitStep := func(stage int, ph StepPhase, m, pos int, start, dur float64, waitBid int64) {
		end := start + dur
		k := opKey{ph, m, pos}
		doneEnd[k] = end

		switch ph {
		case PhaseF:
			inflight[stage]++
			chunkInflight[pos]++
			if inflight[stage] > peakInflight[stage] {
				peakInflight[stage] = inflight[stage]
			}
		case PhaseW:
			inflight[stage]--
			chunkInflight[pos]--
		}

		vChunk := 0
		if zbv && pos >= P {
			vChunk = 1
		}

		sigBid := int64(-1)
		// Determine if this step hands off to another stage across P2P.
		if ph == PhaseF && pos+1 < numChunks {
			nextStage := vStageOf(P, numChunks, pos+1)
			if nextStage != stage {
				bid := nextBarrierID
				nextBarrierID++
				sigBid = int64(bid)
				signalID[k] = sigBid
				epoch := uint32(m + 1)
				barriers = append(barriers, P2PReadyBarrier{
					BarrierID:      bid,
					Epoch:          epoch,
					MicroBatchID:   m,
					VirtualChunk:   vChunk,
					Phase:          PhaseF,
					FromPPStage:    stage,
					ToPPStage:      nextStage,
					Scope:          ".sys",
					ReleasePTX:     fmt.Sprintf("fence.acq_rel.sys; st.release.sys.global.u32 [p2p_fwd_ready+%d], %du;", bid*4, epoch),
					AcquireWaitPTX: fmt.Sprintf("ld.acquire.sys.global.u32 %%r_epoch, [p2p_fwd_ready+%d];", bid*4),
					ReadyAtUs:      end + spec.P2PLatencyUs,
				})
			}
		} else if ph == PhaseB && pos > 0 {
			prevStage := vStageOf(P, numChunks, pos-1)
			if prevStage != stage {
				bid := nextBarrierID
				nextBarrierID++
				sigBid = int64(bid)
				signalID[k] = sigBid
				readyT := end + spec.P2PLatencyUs
				if !zbv {
					readyT = end + wDur + spec.P2PLatencyUs
				}
				epoch := uint32(m + 1)
				barriers = append(barriers, P2PReadyBarrier{
					BarrierID:      bid,
					Epoch:          epoch,
					MicroBatchID:   m,
					VirtualChunk:   vChunk,
					Phase:          PhaseB,
					FromPPStage:    stage,
					ToPPStage:      prevStage,
					Scope:          ".sys",
					ReleasePTX:     fmt.Sprintf("fence.acq_rel.sys; st.release.sys.global.u32 [p2p_bwd_ready+%d], %du;", bid*4, epoch),
					AcquireWaitPTX: fmt.Sprintf("ld.acquire.sys.global.u32 %%r_epoch, [p2p_bwd_ready+%d];", bid*4),
					ReadyAtUs:      readyT,
				})
			}
		}

		stages[stage] = append(stages[stage], MicroStep3D{
			PPStage:              stage,
			VirtualChunk:         vChunk,
			LayerChunk:           pos,
			MicroBatchID:         m,
			Phase:                ph,
			StartUs:              start,
			EndUs:                end,
			ComputeDurationUs:    dur,
			RingCPSteps:          cpSteps,
			RingCPOverlapped:     spec.CPDegree > 1,
			TPMultimemAllReduces: tpCollectives,
			TPMultimemOp:         tpOp,
			WaitBarrierID:        waitBid,
			SignalBarrierID:      sigBid,
		})
		freeAt[stage] = end
	}

	// Stage positions owned by stage s:
	// - In 1F1B: [s]
	// - In ZB-V: [s, 2*P - 1 - s]
	positionsForStage := func(stage int) []int {
		if !zbv {
			return []int{stage}
		}
		return []int{stage, 2*P - 1 - stage}
	}

	totalOps := 3 * M * numChunks
	const eps = 1e-9

	for len(doneEnd) < totalOps {
		progressed := false
		for st := 0; st < P; st++ {
			t := freeAt[st]
			// Check if any ready op can run at time <= t on stage st.
			// Priority in ZB-V:
			//   1. Critical-path B (prefer return leg chunk 2*P-1-st then chunk st, lowest microbatch m)
			//   2. Forward F (prefer chunk0 to feed downstream stages, then chunk1) when below inflight cap
			//   3. Weight-gradient W (deferred bubble filler: runs whenever no B or F is immediately ready at t)
			positions := positionsForStage(st)

			// 1. Try B ready at <= t.
			bestBM, bestBPos, bestBWait := -1, -1, int64(-1)
			// Order positions descending so return-leg B (higher pos) propagates first.
			for i := len(positions) - 1; i >= 0; i-- {
				pos := positions[i]
				for m := 0; m < M; m++ {
					if rT, wBid, ok := readyTime(PhaseB, m, pos); ok && rT <= t+eps {
						bestBM, bestBPos, bestBWait = m, pos, wBid
						break
					}
				}
				if bestBM >= 0 {
					break
				}
			}
			if bestBM >= 0 {
				start := t
				if rT, _, _ := readyTime(PhaseB, bestBM, bestBPos); rT > start {
					start = rT
				}
				emitStep(st, PhaseB, bestBM, bestBPos, start, bDur, bestBWait)
				if !zbv {
					// Standard 1F1B immediately executes W as part of monolithic backward.
					emitStep(st, PhaseW, bestBM, bestBPos, start+bDur, wDur, -1)
				}
				progressed = true
				continue
			}

			// 2. Try F ready at <= t (within per-chunk inflight limit).
			bestFM, bestFPos, bestFWait := -1, -1, int64(-1)
			for i := len(positions) - 1; i >= 0; i-- {
				pos := positions[i]
				chunkCap := P - st
				if zbv {
					chunkCap = 2 * P
				}
				if chunkInflight[pos] >= chunkCap {
					continue
				}
				for m := 0; m < M; m++ {
					if rT, wBid, ok := readyTime(PhaseF, m, pos); ok && rT <= t+eps {
						bestFM, bestFPos, bestFWait = m, pos, wBid
						break
					}
				}
				if bestFM >= 0 {
					break
				}
			}
			if bestFM >= 0 {
				start := t
				if rT, _, _ := readyTime(PhaseF, bestFM, bestFPos); rT > start {
					start = rT
				}
				emitStep(st, PhaseF, bestFM, bestFPos, start, fDur, bestFWait)
				progressed = true
				continue
			}

			// 3. Try W ready at <= t (ZB-V bubble filler).
			if zbv {
				bestWM, bestWPos := -1, -1
				for _, pos := range positions {
					for m := 0; m < M; m++ {
						if rT, _, ok := readyTime(PhaseW, m, pos); ok && rT <= t+eps {
							bestWM, bestWPos = m, pos
							break
						}
					}
					if bestWM >= 0 {
						break
					}
				}
				if bestWM >= 0 {
					start := t
					if rT, _, _ := readyTime(PhaseW, bestWM, bestWPos); rT > start {
						start = rT
					}
					emitStep(st, PhaseW, bestWM, bestWPos, start, wDur, -1)
					progressed = true
					continue
				}
			}
		}

		if progressed {
			continue
		}

		// Advance idle stages to the earliest next event time across ready work or busy stages.
		nextT := math.Inf(1)
		for st := 0; st < P; st++ {
			positions := positionsForStage(st)
			for _, pos := range positions {
				chunkCap := P - st
				if zbv {
					chunkCap = 2 * P
				}
				for m := 0; m < M; m++ {
					for _, ph := range []StepPhase{PhaseB, PhaseF, PhaseW} {
						if !zbv && ph == PhaseW {
							continue
						}
						if ph == PhaseF && chunkInflight[pos] >= chunkCap {
							continue
						}
						if rT, _, ok := readyTime(ph, m, pos); ok {
							cand := math.Max(freeAt[st], rT)
							if cand < nextT {
								nextT = cand
							}
						}
					}
				}
			}
		}

		if math.IsInf(nextT, 1) {
			return nil, fmt.Errorf("contextparallel: schedule stalled with %d/%d ops completed", len(doneEnd), totalOps)
		}
		for st := 0; st < P; st++ {
			if freeAt[st] < nextT {
				freeAt[st] = nextT
			}
		}
	}

	var makespan float64
	stageCompute := float64(M) * 3.0 * spec.MicroStepUs
	for st := 0; st < P; st++ {
		slices.SortFunc(stages[st], func(a, b MicroStep3D) int {
			switch {
			case a.StartUs < b.StartUs:
				return -1
			case a.StartUs > b.StartUs:
				return 1
			default:
				return 0
			}
		})
		if n := len(stages[st]); n > 0 && stages[st][n-1].EndUs > makespan {
			makespan = stages[st][n-1].EndUs
		}
	}

	bubbleFraction := 0.0
	if makespan > 0 {
		bubbleFraction = math.Max(0.0, (makespan-stageCompute)/makespan)
	}

	return &Parallel3DSchedule{
		Spec:                  spec,
		ScheduleKind:          kind,
		WorldSize:             spec.PPStages * spec.TPDegree * spec.CPDegree,
		Stages:                stages,
		Barriers:              barriers,
		MakespanUs:            makespan,
		StageComputeUs:        stageCompute,
		BubbleFraction:        bubbleFraction,
		PeakInflight:          peakInflight,
		SysBarrierPreludeCUDA: emit3DSysBarrierPrelude(spec, kind, len(barriers)),
	}, nil
}

func emit3DSysBarrierPrelude(spec Parallel3DSpec, kind string, numBarriers int) string {
	var b strings.Builder
	fmt.Fprintf(&b, `// ============================================================================
// 3D Parallelism (%s) Persistent P2P .sys Handoff + Ring-CP + TP Multimem
// PPStages=%d, TPDegree=%d, CPDegree=%d, NumMicroBatch=%d, P2PBarriers=%d
// ============================================================================
EG_DEV void eg_pp3d_p2p_signal_sys(uint32_t* __restrict__ ready_flag, uint32_t epoch) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    __atomic_store_n(ready_flag, epoch, __ATOMIC_RELEASE);
#else
    asm volatile("fence.acq_rel.sys;\n\tst.release.sys.global.u32 [%%0], %%1;" :: "l"(ready_flag), "r"(epoch) : "memory");
#endif
}

EG_DEV void eg_pp3d_p2p_wait_sys(const uint32_t* __restrict__ ready_flag, uint32_t want_epoch) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    while (__atomic_load_n(ready_flag, __ATOMIC_ACQUIRE) < want_epoch) {}
#else
    uint32_t got = 0;
    do {
        asm volatile("ld.acquire.sys.global.u32 %%0, [%%1];" : "=r"(got) : "l"(ready_flag) : "memory");
    } while (got < want_epoch);
#endif
}
`, kind, spec.PPStages, spec.TPDegree, spec.CPDegree, spec.NumMicroBatch, numBarriers)
	return b.String()
}
