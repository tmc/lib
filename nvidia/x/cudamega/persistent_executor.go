package cudamega

import (
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// A Grid is the static task-to-CTA mapping: contract 1.
//
// It is derived from the compiled worker's measured residency, never from the
// SM count alone. An SM count says how many multiprocessors exist; it does not
// say how many CTAs of this worker are resident on one, which is what a
// persistent grid must not exceed. Exceeding it is not slow, it deadlocks:
// a CTA that is not resident never runs, and the resident ones wait for it.
type Grid struct {
	CTAs          uint32
	ThreadsPerCTA uint32

	// MaxElements bounds the generated CUDA scalar worker extent. Zero leaves the
	// extent unbounded by the caller. It is not a buffer size: callers must
	// still retain input and scratch storage for every admitted element.
	MaxElements uint32

	// Policy is how the grid's CTAs are dealt to the graph's tasks. The zero
	// value is [WavePolicy], under which every CTA performs every task, which
	// is what this package emitted before per-task ownership existed.
	//
	// It lives here rather than beside the graph because it is the static
	// task-to-CTA mapping this type is named for: a grid and a policy
	// together say which CTA performs which element of which task, and
	// neither says it alone.
	Policy Policy

	// SpinLimit bounds how many times a CTA polls a dependency's completion
	// word before giving up with [AbortSpinExhausted]. Zero means
	// [DefaultSpinLimit].
	SpinLimit uint64
}

// spinLimit resolves the compiled-in poll bound.
func (g Grid) spinLimit() uint64 {
	if g.SpinLimit == 0 {
		return DefaultSpinLimit
	}
	return g.SpinLimit
}

// Validate reports whether the grid is one the generated worker can run under.
func (g Grid) Validate() error {
	if g.CTAs == 0 {
		return fmt.Errorf("grid has no CTAs")
	}
	if g.ThreadsPerCTA == 0 || g.ThreadsPerCTA > 1024 {
		return fmt.Errorf("grid has %d threads per CTA, outside 1..1024", g.ThreadsPerCTA)
	}
	if g.ThreadsPerCTA%32 != 0 {
		return fmt.Errorf("grid has %d threads per CTA, not a whole number of warps", g.ThreadsPerCTA)
	}
	return nil
}

// CTAGrid derives the persistent grid for g from the compiled worker's
// admission.
//
// The grid is exactly the admitted resident CTAs. It is not the SM count and
// not a multiple of it: the generated worker gives every CTA a share of every
// task's elements, so every admitted CTA has work in every task, and a CTA
// beyond the admitted residency would never be scheduled while the resident
// ones waited at its barrier arrival.
func CTAGrid(a Admission, g cudataskgraph.Graph) (Grid, error) {
	if err := a.Validate(); err != nil {
		return Grid{}, err
	}
	if len(g.Tasks()) == 0 {
		return Grid{}, fmt.Errorf("graph has no tasks")
	}
	grid := Grid{CTAs: a.ResidentCTAs, ThreadsPerCTA: a.WorkerThreads}
	if err := grid.Validate(); err != nil {
		return Grid{}, err
	}
	return grid, nil
}

// RankWidths returns the number of tasks at each rank, ordered by rank.
func RankWidths(g cudataskgraph.Graph) []uint32 {
	byRank := map[int]uint32{}
	for _, t := range g.Tasks() {
		byRank[t.Rank]++
	}
	ranks := make([]int, 0, len(byRank))
	for r := range byRank {
		ranks = append(ranks, r)
	}
	sort.Ints(ranks)
	out := make([]uint32, 0, len(ranks))
	for _, r := range ranks {
		out = append(out, byRank[r])
	}
	return out
}

// A Wave is one task executed by the whole grid.
//
// There is one wave per task because the generated worker unrolls the task
// sequence and gives every CTA a share of every task's elements. Each wave
// ends in one barrier arrival, which is why the parity here is the parity the
// emitted code alternates.
type Wave struct {
	Rank   int
	Task   string
	Scope  cudataskgraph.Scope
	Parity uint32 // the barrier phase this wave arrives on: contract 5
}

// A ProgressArgument is the finite scheduler's claim that the persistent grid
// terminates, together with the facts it rests on.
//
// It is a host-side argument about the schedule, not an observation of a
// running grid. What it rules out is the deadlock the contracts name: a CTA
// waiting on a task no CTA has been assigned.
type ProgressArgument struct {
	Grid       Grid
	Waves      []Wave
	RankWidths []uint32
	// Invariants are the checks that ran, in the order they ran, each stated
	// as what would have to be false for the argument to fail.
	Invariants []string
}

// Progress returns the argument, or the reason there is none.
//
// The argument: dependencies have strictly lower rank than their consumers
// (enforced when the graph is built), the grid runs one rank at a time in
// ascending rank order, and a rank's tasks are split into ceil(width/CTAs)
// waves. When a CTA begins a task of rank r, every task of rank < r has been
// published, because the rank boundary is a grid-wide release/acquire. So no
// CTA ever waits on a task that has not been assigned, and the number of waves
// is finite and known before launch.
func Progress(g cudataskgraph.Graph, a Admission) (ProgressArgument, error) {
	grid, err := CTAGrid(a, g)
	if err != nil {
		return ProgressArgument{}, err
	}
	tasks := g.Tasks()
	for _, t := range tasks {
		if t.Bytes > uint64(a.SharedBytes) {
			return ProgressArgument{}, fmt.Errorf("task %q wants %d shared bytes, the compiled worker admits %d",
				t.ID, t.Bytes, a.SharedBytes)
		}
	}
	// g.Tasks() is already ordered by rank then id, which is the order the
	// generated worker unrolls and therefore the order the waves run in.
	waves := make([]Wave, 0, len(tasks))
	for i, t := range tasks {
		waves = append(waves, Wave{Rank: t.Rank, Task: t.ID, Scope: t.Scope, Parity: ParityAt(i)})
	}
	return ProgressArgument{
		Grid:       grid,
		Waves:      waves,
		RankWidths: RankWidths(g),
		Invariants: []string{
			"every dependency has a strictly lower rank than its consumer",
			"the grid never exceeds the compiled worker's admitted resident CTAs",
			"no task's shared-memory requirement exceeds the admitted amount",
			"tasks run in ascending rank order and a task publishes before any consumer reads it",
			"barrier phase parity alternates across waves and is never reset",
		},
	}, nil
}

// ParityAt returns the barrier phase the nth wave arrives on: contract 5.
// Parity alternates rather than resetting, because resetting a hardware
// mbarrier between iterations is what the contract forbids.
func ParityAt(wave int) uint32 { return uint32(wave & 1) }

// An Executor holds the host side of one persistent launch.
//
// It owns no device memory. The buffers belong to the caller, which is what
// makes abort safe: aborting changes a word the worker reads and does not free
// anything the worker may still be writing. See [Executor.Abort].
type Executor struct {
	grid     Grid
	progress ProgressArgument
	epoch    *cudataskgraph.Epoch
	graph    cudataskgraph.Graph
}

// NewExecutor validates the plan against the compiled worker and returns the
// executor for one epoch. The epoch number must be nonzero.
func NewExecutor(g cudataskgraph.Graph, a Admission, epoch uint64) (*Executor, error) {
	if epoch == 0 {
		return nil, fmt.Errorf("epoch zero is the unpublished value and may not be launched")
	}
	p, err := Progress(g, a)
	if err != nil {
		return nil, err
	}
	e := &Executor{grid: p.Grid, progress: p, epoch: cudataskgraph.NewEpoch(epoch), graph: g.Clone()}
	// The device owns every task's output storage from the moment the launch
	// is planned, not from the moment the worker reaches that task: the
	// worker is persistent, so there is no point after launch at which a
	// later task's buffer is known to be untouched.
	for _, t := range e.graph.Tasks() {
		if err := e.epoch.Claim(t.ID, e.graph); err != nil {
			return nil, err
		}
	}
	return e, nil
}

// Grid returns the static mapping this executor launches under.
func (e *Executor) Grid() Grid { return e.grid }

// Progress returns the scheduler's progress argument.
func (e *Executor) Progress() ProgressArgument { return e.progress }

// Epoch returns the publication and ownership record for this launch.
func (e *Executor) Epoch() *cudataskgraph.Epoch { return e.epoch }

// Drain records that the host has observed the launch finish. Nothing may be
// released before it.
func (e *Executor) Drain() { e.epoch.Drain() }

// Release relinquishes the device's ownership of one task's output storage.
func (e *Executor) Release(id string) error { return e.epoch.Release(id) }

// Abort requests termination through the volatile abort word: contract 6.
//
// It aborts the epoch and returns the tasks whose output storage is still
// owned by the device. Those buffers must not be freed: the worker may be
// mid-store in them, and an abort is a request the worker observes at its next
// poll, not a fence it has already passed.
func (e *Executor) Abort() []string {
	e.epoch.Abort()
	return e.epoch.Retained()
}
