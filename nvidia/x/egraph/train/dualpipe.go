package train

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// OpPhase is the kind of work in a [PipelineSlot].
type OpPhase uint8

const (
	PhaseForward          OpPhase = iota // F
	PhaseBwdInput                        // B_X: dX = dY·Wᵀ, on the critical path to the previous stage
	PhaseBwdWeight                       // B_W: dW = Xᵀ·dY, needed only by the optimizer step
	PhaseOverlappedFwdBwd                // F of one microbatch overlapped with B_X of another (DualPipe)
)

func (p OpPhase) String() string {
	switch p {
	case PhaseForward:
		return "F"
	case PhaseBwdInput:
		return "B"
	case PhaseBwdWeight:
		return "W"
	case PhaseOverlappedFwdBwd:
		return "F+B"
	}
	return fmt.Sprintf("OpPhase(%d)", uint8(p))
}

// PipelineSlot is one block of work on one pipeline stage (rank).
//
// Direction 0 visits stages 0..P-1 and direction 1 visits P-1..0 (DualPipe).
// A PhaseOverlappedFwdBwd slot runs the forward of (MicrobatchID, Direction)
// together with the input backward of (PairMicrobatchID, PairDirection).
type PipelineSlot struct {
	Stage            int
	MicrobatchID     int
	Direction        int
	Phase            OpPhase
	PairMicrobatchID int
	PairDirection    int
	StartUs          float64
	EndUs            float64
}

// ScheduleConfig describes a pipeline and the cost of each phase.
//
// CommAll2AllUs is the MoE all-to-all time inside each F and B_X chunk. It is
// exposed (added to the chunk) unless the chunk runs in a
// PhaseOverlappedFwdBwd slot, where each chunk's communication hides behind
// the other's compute.
type ScheduleConfig struct {
	NumStages     int // P pipeline stages (even for DualPipe)
	NumMicrobatch int // M microbatches (even for DualPipe)
	FwdUs         float64
	BwdInputUs    float64
	BwdWeightUs   float64
	CommAll2AllUs float64
	FSDPShardDeg  int // FSDP2 sharding degree; > 1 enables [PipelineSchedule.FSDPOverlap]
	// MaxInflight bounds, per stage, the microbatches whose forward has run
	// but whose weight gradient has not (activation memory). Zero means
	// 2*NumStages. 1F1B ignores it and uses P-s on stage s.
	MaxInflight int
}

// PipelineSchedule is a per-stage timeline.
//
// BubbleFraction is (Makespan·P − work)/(Makespan·P), where work counts only
// compute (F, B_X and B_W time, not exposed communication).
// InteriorBubbleFraction counts only idle time between each stage's first
// and last slot: the warm-up and drain triangles it excludes can overlap the
// neighbouring iterations when stages do not synchronize the optimizer step.
type PipelineSchedule struct {
	Slots                  [][]PipelineSlot // [stage][]slot, in time order
	MakespanUs             float64
	BubbleFraction         float64
	InteriorBubbleFraction float64
	PeakInflight           []int // per stage
}

func (c ScheduleConfig) check(dual bool) error {
	if c.NumStages < 1 {
		return fmt.Errorf("NumStages %d must be >= 1", c.NumStages)
	}
	if c.NumMicrobatch < 1 {
		return fmt.Errorf("NumMicrobatch %d must be >= 1", c.NumMicrobatch)
	}
	if dual && (c.NumStages%2 != 0 || c.NumMicrobatch%2 != 0) {
		return fmt.Errorf("dualpipe needs even NumStages and NumMicrobatch, have %d and %d", c.NumStages, c.NumMicrobatch)
	}
	if c.FwdUs <= 0 || c.BwdInputUs <= 0 || c.BwdWeightUs <= 0 {
		return errors.New("phase times must be > 0")
	}
	if c.CommAll2AllUs < 0 || c.MaxInflight < 0 {
		return errors.New("negative CommAll2AllUs or MaxInflight")
	}
	return nil
}

func (c ScheduleConfig) inflightLimit() int {
	if c.MaxInflight > 0 {
		return c.MaxInflight
	}
	return 2 * c.NumStages
}

// posOf returns the chunk position that stage s runs in direction dir of a
// p-stage pipeline. The mapping is its own inverse: posOf(p, dir, pos) is
// also the stage that runs position pos.
func posOf(p, dir, s int) int {
	if dir == 0 {
		return s
	}
	return p - 1 - s
}

type schedPolicy uint8

const (
	policy1F1B schedPolicy = iota
	policyZeroBubble
	policyDualPipe
)

// pipeSim is a greedy list scheduler: whenever a stage is idle it starts
// the highest-priority ready work.
type pipeSim struct {
	cfg    ScheduleConfig
	policy schedPolicy
	dirOf  []int // microbatch -> direction
	// end[phase][m][pos] is the completion time, or -1 if not run.
	end      [3][][]float64
	inflight []int
	peak     []int
	free     []float64
	slots    [][]PipelineSlot
}

func newPipeSim(cfg ScheduleConfig, policy schedPolicy) *pipeSim {
	P, M := cfg.NumStages, cfg.NumMicrobatch
	s := &pipeSim{
		cfg: cfg, policy: policy,
		dirOf:    make([]int, M),
		inflight: make([]int, P),
		peak:     make([]int, P),
		free:     make([]float64, P),
		slots:    make([][]PipelineSlot, P),
	}
	if policy == policyDualPipe {
		for m := M / 2; m < M; m++ {
			s.dirOf[m] = 1
		}
	}
	for ph := range s.end {
		s.end[ph] = make([][]float64, M)
		for m := range M {
			s.end[ph][m] = make([]float64, P)
			for i := range P {
				s.end[ph][m][i] = -1
			}
		}
	}
	return s
}

// ready reports the earliest time phase ph of microbatch m at position pos
// may start, or false if its inputs have not been scheduled.
func (s *pipeSim) ready(ph OpPhase, m, pos int) (float64, bool) {
	P := s.cfg.NumStages
	if s.end[ph][m][pos] >= 0 {
		return 0, false
	}
	var dep float64
	switch ph {
	case PhaseForward:
		if pos == 0 {
			return 0, true
		}
		dep = s.end[PhaseForward][m][pos-1]
	case PhaseBwdInput:
		switch {
		case pos == P-1:
			dep = s.end[PhaseForward][m][pos]
		case s.policy == policy1F1B:
			// A monolithic backward sends dX only when the whole
			// backward of the next stage, B_W included, is done.
			dep = s.end[PhaseBwdWeight][m][pos+1]
		default:
			dep = s.end[PhaseBwdInput][m][pos+1]
		}
	case PhaseBwdWeight:
		dep = s.end[PhaseBwdInput][m][pos]
	}
	return dep, dep >= 0
}

func (s *pipeSim) limit(stage int) int {
	if s.policy == policy1F1B {
		return s.cfg.NumStages - stage
	}
	return s.cfg.inflightLimit()
}

// pick returns the ready work of phase ph on stage at time t with the lowest
// microbatch id, or -1.
func (s *pipeSim) pick(stage int, ph OpPhase, t float64) int {
	for m := range s.cfg.NumMicrobatch {
		pos := posOf(s.cfg.NumStages, s.dirOf[m], stage)
		if at, ok := s.ready(ph, m, pos); ok && at <= t {
			return m
		}
	}
	return -1
}

func (s *pipeSim) run(ph OpPhase, stage, m int, start, end float64) {
	pos := posOf(s.cfg.NumStages, s.dirOf[m], stage)
	s.end[ph][m][pos] = end
	switch ph {
	case PhaseForward:
		s.inflight[stage]++
		s.peak[stage] = max(s.peak[stage], s.inflight[stage])
	case PhaseBwdWeight:
		s.inflight[stage]--
	}
	s.slots[stage] = append(s.slots[stage], PipelineSlot{
		Stage: stage, MicrobatchID: m, Direction: s.dirOf[m], Phase: ph,
		PairMicrobatchID: -1, PairDirection: -1,
		StartUs: start, EndUs: end,
	})
}

// step tries to start work on an idle stage at time t.
func (s *pipeSim) step(stage int, t float64) bool {
	c := s.cfg
	fDur := c.FwdUs + c.CommAll2AllUs
	bDur := c.BwdInputUs + c.CommAll2AllUs
	canF := s.inflight[stage] < s.limit(stage)
	b := s.pick(stage, PhaseBwdInput, t)
	f := -1
	if canF {
		f = s.pick(stage, PhaseForward, t)
	}
	switch {
	case s.policy == policyDualPipe && b >= 0 && f >= 0:
		end := t + max(c.FwdUs+c.BwdInputUs, 2*c.CommAll2AllUs)
		s.run(PhaseForward, stage, f, t, end)
		s.run(PhaseBwdInput, stage, b, t, end)
		// Fuse the two slots into one overlapped slot.
		n := len(s.slots[stage])
		bs := s.slots[stage][n-1]
		fs := &s.slots[stage][n-2]
		fs.Phase = PhaseOverlappedFwdBwd
		fs.PairMicrobatchID, fs.PairDirection = bs.MicrobatchID, bs.Direction
		s.slots[stage] = s.slots[stage][:n-1]
		s.free[stage] = end
	case b >= 0:
		s.run(PhaseBwdInput, stage, b, t, t+bDur)
		s.free[stage] = t + bDur
		if s.policy == policy1F1B {
			// 1F1B runs the weight gradient as part of the backward.
			s.run(PhaseBwdWeight, stage, b, t+bDur, t+bDur+c.BwdWeightUs)
			s.free[stage] += c.BwdWeightUs
		}
	case f >= 0:
		s.run(PhaseForward, stage, f, t, t+fDur)
		s.free[stage] = t + fDur
	default:
		w := s.pick(stage, PhaseBwdWeight, t)
		if w < 0 {
			return false
		}
		s.run(PhaseBwdWeight, stage, w, t, t+c.BwdWeightUs)
		s.free[stage] = t + c.BwdWeightUs
	}
	return true
}

func (s *pipeSim) schedule() (PipelineSchedule, error) {
	c := s.cfg
	P, M := c.NumStages, c.NumMicrobatch
	total := 3 * P * M
	done := 0
	count := func() int {
		n := 0
		for st := range P {
			for _, sl := range s.slots[st] {
				if sl.Phase == PhaseOverlappedFwdBwd {
					n += 2
				} else {
					n++
				}
			}
		}
		return n
	}
	t := 0.0
	for done < total {
		for st := range P {
			if s.free[st] <= t {
				s.step(st, t)
			}
		}
		done = count()
		// Advance to the next time a stage frees or work completes.
		next := math.Inf(1)
		for st := range P {
			if s.free[st] > t {
				next = min(next, s.free[st])
			}
		}
		if math.IsInf(next, 1) {
			if done < total {
				return PipelineSchedule{}, fmt.Errorf("pipeline schedule: stalled at %.3f us with %d of %d ops", t, done, total)
			}
			break
		}
		t = next
	}
	return s.finish(), nil
}

func (s *pipeSim) finish() PipelineSchedule {
	c := s.cfg
	P := c.NumStages
	sch := PipelineSchedule{Slots: s.slots, PeakInflight: s.peak}
	work := float64(c.NumMicrobatch) * (c.FwdUs + c.BwdInputUs + c.BwdWeightUs)
	var span, idle float64
	for st := range P {
		sl := s.slots[st]
		slices.SortFunc(sl, func(a, b PipelineSlot) int {
			switch {
			case a.StartUs < b.StartUs:
				return -1
			case a.StartUs > b.StartUs:
				return 1
			}
			return 0
		})
		first, last := sl[0].StartUs, sl[len(sl)-1].EndUs
		sch.MakespanUs = max(sch.MakespanUs, last)
		span += last - first
		idle += last - first - work
	}
	sch.BubbleFraction = (sch.MakespanUs*float64(P) - work*float64(P)) / (sch.MakespanUs * float64(P))
	sch.InteriorBubbleFraction = idle / span
	return sch
}

// PlanOneFOneB returns the standard 1F1B schedule with a monolithic backward
// (B_X immediately followed by B_W), as a baseline. Stage s holds at most
// P-s microbatches in flight.
func PlanOneFOneB(cfg ScheduleConfig) (PipelineSchedule, error) {
	if err := cfg.check(false); err != nil {
		return PipelineSchedule{}, fmt.Errorf("plan 1f1b: %w", err)
	}
	return newPipeSim(cfg, policy1F1B).schedule()
}

// PlanZeroBubble returns a zero-bubble (ZB-H2 style) schedule. B_X runs as
// soon as it is ready, F runs while the stage is below its in-flight limit,
// and the deferred B_W fills any time the stage would otherwise idle.
func PlanZeroBubble(cfg ScheduleConfig) (PipelineSchedule, error) {
	if err := cfg.check(false); err != nil {
		return PipelineSchedule{}, fmt.Errorf("plan zero bubble: %w", err)
	}
	return newPipeSim(cfg, policyZeroBubble).schedule()
}

// PlanDualPipe returns a bidirectional DualPipe schedule. Microbatches
// 0..M/2-1 run in direction 0 and M/2..M-1 in direction 1, so stage s holds
// chunk s of one direction and chunk P-1-s of the other. When a stage has
// both a forward and an input backward ready it runs them as one
// PhaseOverlappedFwdBwd slot lasting max(F+B_X, 2·CommAll2AllUs), hiding the
// all-to-all of each behind the other's compute. B_W is deferred as in
// [PlanZeroBubble].
func PlanDualPipe(cfg ScheduleConfig) (PipelineSchedule, error) {
	if err := cfg.check(true); err != nil {
		return PipelineSchedule{}, fmt.Errorf("plan dualpipe: %w", err)
	}
	return newPipeSim(cfg, policyDualPipe).schedule()
}

// VerifyScheduleDAG checks that sched runs every F, B_X and B_W of every
// microbatch exactly once on the stage its direction assigns, that each slot
// lasts at least its compute time, and that
//
//  1. F(s, m) completes before F(s+1, m),
//  2. F(P-1, m) completes before B_X(P-1, m),
//  3. B_X(s, m) completes before B_X(s-1, m),
//  4. B_X(s, m) completes before B_W(s, m), and
//  5. no two slots on one stage overlap,
//
// where s is the position along the microbatch's direction.
func VerifyScheduleDAG(sched PipelineSchedule, cfg ScheduleConfig) error {
	P, M := cfg.NumStages, cfg.NumMicrobatch
	if len(sched.Slots) != P {
		return fmt.Errorf("verify schedule: %d stages, want %d", len(sched.Slots), P)
	}
	const eps = 1e-9
	type key struct {
		ph  OpPhase
		m   int
		pos int
	}
	type span struct{ start, end float64 }
	at := make(map[key]span)
	dirOf := make(map[int]int)
	add := func(ph OpPhase, stage, m, dir int, start, end float64) error {
		if m < 0 || m >= M {
			return fmt.Errorf("verify schedule: stage %d: microbatch %d out of range", stage, m)
		}
		if dir != 0 && dir != 1 {
			return fmt.Errorf("verify schedule: stage %d: microbatch %d has direction %d", stage, m, dir)
		}
		if d, ok := dirOf[m]; ok && d != dir {
			return fmt.Errorf("verify schedule: microbatch %d runs in both directions", m)
		}
		dirOf[m] = dir
		k := key{ph, m, posOf(P, dir, stage)}
		if _, dup := at[k]; dup {
			return fmt.Errorf("verify schedule: %v of microbatch %d at position %d runs twice", ph, m, k.pos)
		}
		at[k] = span{start, end}
		return nil
	}
	for st, slots := range sched.Slots {
		for i, sl := range slots {
			if sl.Stage != st {
				return fmt.Errorf("verify schedule: slot on stage %d says stage %d", st, sl.Stage)
			}
			if sl.StartUs < -eps {
				return fmt.Errorf("verify schedule: stage %d slot %d starts at %v", st, i, sl.StartUs)
			}
			var need float64
			switch sl.Phase {
			case PhaseForward:
				need = cfg.FwdUs
			case PhaseBwdInput:
				need = cfg.BwdInputUs
			case PhaseBwdWeight:
				need = cfg.BwdWeightUs
			case PhaseOverlappedFwdBwd:
				need = cfg.FwdUs + cfg.BwdInputUs
			default:
				return fmt.Errorf("verify schedule: stage %d slot %d has phase %v", st, i, sl.Phase)
			}
			if sl.EndUs-sl.StartUs < need-eps {
				return fmt.Errorf("verify schedule: stage %d %v slot of microbatch %d lasts %v us, needs %v", st, sl.Phase, sl.MicrobatchID, sl.EndUs-sl.StartUs, need)
			}
			if i > 0 && sl.StartUs < slots[i-1].EndUs-eps {
				return fmt.Errorf("verify schedule: stage %d slots %d and %d overlap", st, i-1, i)
			}
			var err error
			switch sl.Phase {
			case PhaseOverlappedFwdBwd:
				if err = add(PhaseForward, st, sl.MicrobatchID, sl.Direction, sl.StartUs, sl.EndUs); err == nil {
					err = add(PhaseBwdInput, st, sl.PairMicrobatchID, sl.PairDirection, sl.StartUs, sl.EndUs)
				}
			default:
				err = add(sl.Phase, st, sl.MicrobatchID, sl.Direction, sl.StartUs, sl.EndUs)
			}
			if err != nil {
				return err
			}
		}
	}
	before := func(a, b key) error {
		x, ok := at[a]
		if !ok {
			return fmt.Errorf("verify schedule: %v of microbatch %d at position %d never runs", a.ph, a.m, a.pos)
		}
		y, ok := at[b]
		if !ok {
			return fmt.Errorf("verify schedule: %v of microbatch %d at position %d never runs", b.ph, b.m, b.pos)
		}
		if x.end > y.start+eps {
			return fmt.Errorf("verify schedule: microbatch %d: %v at position %d ends at %v after %v at position %d starts at %v",
				a.m, a.ph, a.pos, x.end, b.ph, b.pos, y.start)
		}
		return nil
	}
	for m := range M {
		for pos := range P {
			deps := [][2]key{{{PhaseBwdInput, m, pos}, {PhaseBwdWeight, m, pos}}}
			if pos+1 < P {
				deps = append(deps,
					[2]key{{PhaseForward, m, pos}, {PhaseForward, m, pos + 1}},
					[2]key{{PhaseBwdInput, m, pos + 1}, {PhaseBwdInput, m, pos}})
			} else {
				deps = append(deps, [2]key{{PhaseForward, m, pos}, {PhaseBwdInput, m, pos}})
			}
			for _, d := range deps {
				if err := before(d[0], d[1]); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// FSDPOp is an FSDP2 collective tied to a pipeline stage. On NVL72 and
// NVL144 the reduce-scatter maps to an in-switch multimem reduction
// (fabric.try_red.async.tensor...multimem on Rubin); this planner models only
// its timing.
type FSDPOp struct {
	Stage   int
	Kind    string // "all_gather" or "reduce_scatter"
	StartUs float64
	EndUs   float64
}

// FSDPOverlap places, for each stage, the parameter all-gather so that it
// completes before the stage's first slot (prefetched during the previous
// iteration), and the gradient reduce-scatter right after the stage's last
// B_W. It returns the ops and the reduce-scatter time that extends past the
// pipeline makespan, which is exposed unless the next iteration overlaps it.
func (s PipelineSchedule) FSDPOverlap(cfg ScheduleConfig, allGatherUs, reduceScatterUs float64) ([]FSDPOp, float64, error) {
	if cfg.FSDPShardDeg <= 1 {
		return nil, 0, nil
	}
	if allGatherUs < 0 || reduceScatterUs < 0 {
		return nil, 0, errors.New("fsdp overlap: negative collective time")
	}
	var ops []FSDPOp
	var exposed float64
	for st, slots := range s.Slots {
		if len(slots) == 0 {
			return nil, 0, fmt.Errorf("fsdp overlap: stage %d has no slots", st)
		}
		first := slots[0].StartUs
		ops = append(ops, FSDPOp{Stage: st, Kind: "all_gather", StartUs: first - allGatherUs, EndUs: first})
		lastW := -1.0
		for _, sl := range slots {
			if sl.Phase == PhaseBwdWeight {
				lastW = max(lastW, sl.EndUs)
			}
		}
		if lastW < 0 {
			return nil, 0, fmt.Errorf("fsdp overlap: stage %d has no weight-gradient slot", st)
		}
		end := lastW + reduceScatterUs
		ops = append(ops, FSDPOp{Stage: st, Kind: "reduce_scatter", StartUs: lastW, EndUs: end})
		exposed = max(exposed, end-s.MakespanUs)
	}
	return ops, exposed, nil
}
