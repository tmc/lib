package cudamega

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// A ForwardProgress is a backend's declaration of the scheduling guarantee
// its workers run under.
//
// It is the boundary of this package, and it is a scheduling guarantee rather
// than a syntax. A worker that spins on a predecessor's completion word is
// correct only if that predecessor is guaranteed to be running: otherwise the
// spin is not a wait, it is a deadlock that no amount of correctly spelled
// synchronization repairs. CUDA supplies the guarantee through occupancy plus
// cooperative launch; OS threads supply it through preemptive scheduling.
// Vulkan and Metal supply it for the threads of one workgroup and for nothing
// wider, so a cross-workgroup spin is legally a hang there.
//
// The zero value declares nothing and is refused by [NewRunner]. That refusal
// is the point: a backend that cannot state the guarantee is discovered at
// construction rather than by a hung device.
type ForwardProgress int

const (
	// NoForwardProgress is the zero value: no guarantee is declared.
	NoForwardProgress ForwardProgress = iota

	// CoResident means every worker the residency query admits is executing
	// concurrently for the whole launch, because the launch was admitted
	// against that residency. This is what a CUDA cooperative launch at or
	// below the occupancy limit provides.
	CoResident

	// IndependentThreads means each worker runs on a thread the operating
	// system schedules independently, so a worker that spins cannot prevent
	// another from running. This is what OS threads provide.
	IndependentThreads
)

func (f ForwardProgress) String() string {
	switch f {
	case NoForwardProgress:
		return "none"
	case CoResident:
		return "co-resident"
	case IndependentThreads:
		return "independent-threads"
	}
	return fmt.Sprintf("ForwardProgress(%d)", int(f))
}

// A Fence is which side of a release/acquire pair a fence performs.
type Fence int

const (
	// Acquire orders the reads that follow it after the completion word the
	// worker has just observed.
	Acquire Fence = iota + 1
	// Release orders the stores that precede it before the completion word
	// the worker is about to publish.
	Release
)

func (f Fence) String() string {
	switch f {
	case Acquire:
		return "acquire"
	case Release:
		return "release"
	}
	return fmt.Sprintf("Fence(%d)", int(f))
}

// A Backend performs, or renders to device source, the primitives a
// persistent worker needs.
//
// The set is deliberately the smallest one that expresses the worker and no
// smaller. Six of the methods are the synchronization primitives the model
// rests on:
//
//   - [Backend.Barrier] is the workgroup barrier, over the threads of one
//     worker.
//   - [Backend.Publish] and [Backend.Ready] are the device-scope atomic add
//     and the acquiring load of a completion word. They are one primitive
//     seen from its two ends; neither is usable without the other.
//   - [Backend.Fence] is the release/acquire fence at a declared scope. At
//     CTA scope a backend may serve it with a workgroup-scoped barrier
//     primitive -- an mbarrier on CUDA -- which is why the optional seventh
//     primitive is a quality of this method rather than a method of its own.
//   - [Backend.NextRound] is where a worker with nothing ready waits. It is
//     the only method that may wait at all.
//   - [Backend.Residency] is the residency query.
//   - [Backend.ForwardProgress] is the co-residency guarantee, which is the
//     actual boundary. See [ForwardProgress].
//
// The remaining methods are structure, not synchronization: they bracket the
// run, the selection loop and the individual tasks, and carry the elementwise
// payload. They are here because the driver in [Runner.Run] must be the only
// description of the worker's shape, and a driver that could not open a
// task's guarded region would leave that shape in each backend.
//
// Taking and waiting are separate methods, and that separation is the point
// rather than a decomposition. [Backend.Ready] never waits: it performs one
// acquiring load and reports what it saw. [Backend.Take] never waits either:
// it commits to a task only when every Ready for that task was satisfied.
// The waiting is [Backend.NextRound], which a worker reaches only by having
// declined every task it owns. A backend that took a task and then blocked on
// it would be one where a worker holding a consumer stops scanning, and a
// grid whose every resident worker holds a consumer never reaches its
// producers at all.
//
// A Backend value describes one worker's view of the program. A performing
// backend therefore has one value per thread; a rendering backend has one
// value for the whole program, because the text it writes stands for every
// thread at once.
type Backend interface {
	// ForwardProgress reports the guarantee this backend's workers run
	// under. A backend that cannot declare one returns NoForwardProgress and
	// is refused by NewRunner.
	ForwardProgress() ForwardProgress

	// Residency reports how many workers of this program may run
	// concurrently, and with what geometry. A program whose grid exceeds it
	// is refused by NewRunner.
	Residency() Admission

	// Begin opens a run of p. It is where a backend makes the refusals the
	// generated worker's entry makes: a geometry that is not the one the
	// program was built for, or an empty extent.
	Begin(p *Program) error

	// BeginSelection opens the worker's selection loop and takes stock of the
	// tasks this worker owns. A rendering backend emits the loop; a
	// performing backend records what it has left to do.
	BeginSelection() error

	// BeginTask opens step i's guarded region and reports whether the step's
	// remaining primitives are performed at all. The guard is ownership and
	// completion -- does this worker perform step i, and has it not already
	// done so -- and it never waits.
	//
	// A rendering backend emits the guard and reports true, because the test
	// is made at run time by the code it emits. A performing backend makes
	// the test itself, and must make it uniformly across the threads of one
	// worker: a guard that some threads of a worker take and others do not
	// leaves the barrier below short an arrival.
	BeginTask(i int) (bool, error)

	// Ready performs one acquiring load of step dep's completion word, at
	// scope s, and records whether it has reached the number of workers that
	// publish it. It does not wait, and it does not decide: what it saw is
	// read back by Take, after the Barrier the driver performs.
	//
	// The number it compares against is dep's own [Owner.Count], not the
	// grid. A task performed by a proper subrange of the grid publishes that
	// many completions and never more, so a wait spelled against the grid is
	// not a slow wait but an unsatisfiable one.
	Ready(i, dep int, s cudataskgraph.Scope) error

	// Barrier is the workgroup barrier over the threads of one worker.
	Barrier() error

	// Take commits this worker to step i, and reports whether it did. It
	// commits exactly when every Ready since the last BeginTask was
	// satisfied, and a step with no dependencies is taken unconditionally.
	// It never waits.
	Take(i int) (bool, error)

	// Fence performs side k of step i's release/acquire pair at scope s.
	// For an acquire, dep is the step whose edge is being ordered; for a
	// release, dep is negative. It reports an error for a scope the backend
	// cannot order.
	//
	// There is one acquire per edge rather than one per task, which is what
	// gives the max(producer, consumer) rule somewhere to live: a task with
	// two dependencies at different scopes has two edges ordered at two
	// scopes, and a single per-task fence could only be right for one of
	// them.
	Fence(i, dep int, s cudataskgraph.Scope, k Fence) error

	// Compute applies step i's operation to this worker's share of the
	// step's elements, reading from step src's output or, when src is
	// negative, from the program input. The share is taken over the step's
	// owning CTA range, not over the grid.
	Compute(i, src int, op Operation) error

	// Publish increments step i's completion word with a device-scope atomic
	// add, once per worker.
	Publish(i int) error

	// EndTask closes the regions BeginTask and Take opened.
	EndTask(i int) error

	// NextRound ends one pass over the worker's owned tasks and reports
	// whether another is needed. It is the only method that waits, and a
	// worker reaches it only having declined every task it owns, so what it
	// waits for is another worker's publication and never its own.
	NextRound() (bool, error)

	// EndSelection closes the loop BeginSelection opened.
	EndSelection() error

	// End closes the run. A rendering backend writes whatever must follow
	// the last task; a performing backend has nothing left to do.
	End() error
}

// A Step is one task as the program runs it.
type Step struct {
	Task      cudataskgraph.Task
	Operation Operation
	// Deps are the indices, in program order, of this step's dependencies.
	Deps []int

	// Owner is the CTA range that performs this task. Its [Owner.Count] is
	// how many completions the task publishes, and therefore the number a
	// consumer's wait must reach.
	Owner Owner

	// EdgeScope[k] is the scope the edge from Deps[k] is ordered at: the
	// higher of the producer's declared scope and this task's.
	//
	// The higher of the two, not the consumer's alone. An acquire is only as
	// wide as the domain it names: a consumer acquiring at device scope has
	// not ordered a producer whose stores were released at system scope, and
	// a release at CTA scope publishes nothing a consumer in another CTA can
	// acquire. The maximum is the weakest rule that orders both ends, which
	// is why it is the rule rather than merely a safe choice.
	EdgeScope []cudataskgraph.Scope

	// ReleaseScope is the scope this task's release is performed at: the
	// highest of its own declared scope and every consumer's. A task nothing
	// depends on releases at its own.
	ReleaseScope cudataskgraph.Scope
}

// A Program is a finite task graph bound to one worker geometry and one
// target: the sequence of primitives a persistent worker performs, with every
// refusal the pair implies already made.
//
// Construction is where target and backend capability meet the graph. A scope
// the target cannot order, an operation the program has no definition for, a
// dependency edge the grid cannot carry -- all of them are refused here, once,
// rather than by each backend separately. What reaches a backend is a program
// that backend can run.
type Program struct {
	steps  []Step
	grid   Grid
	target Target

	// What the generated worker's prologue needs, decided once here because
	// it is a property of the program and not of any backend.
	needSpin, needBarrier, needPhase bool
}

// NewProgram binds a graph and its operations to a grid and a target.
//
// The emitted worker refuses at run time to run under any geometry but grid,
// so grid is part of the program's identity rather than an argument to a
// later call. Target is here for the same reason: it decides which scopes can
// be ordered at all, and a program that could be retargeted after the fact
// would be a program whose refusals had not yet been made.
//
// Unsupported, reported rather than approximated:
//
//   - [cudataskgraph.ScopeCluster]. Below sm_90 the target has no
//     barrier.cluster to order one with. At sm_90 and above it needs the
//     distributed shared memory and participation contract lane 09 owns, and
//     an mbarrier in the local CTA's shared window is not a cluster barrier.
//     Both refuse; they refuse for different reasons, and the message says
//     which, because only the second one lane 09 can lift.
//   - [cudataskgraph.ScopeCTA] below sm_80, which has no mbarrier.
//   - Either end of a dependency edge at CTA scope in a multi-CTA grid. An
//     mbarrier in one CTA's shared memory orders that CTA only, so it can
//     neither publish to another CTA nor observe what another CTA published.
//     The producer's end is refused as well as the consumer's: a CTA-scope
//     release publishes a completion word that a CTA in another block will
//     read, and that is the same defect seen from the other side.
func NewProgram(g cudataskgraph.Graph, operations map[string]Operation, grid Grid, target Target) (*Program, error) {
	return newProgram(g, operations, grid, target, false)
}

func newProgram(g cudataskgraph.Graph, operations map[string]Operation, grid Grid, target Target, typed bool) (*Program, error) {
	tasks := g.Tasks()
	if len(tasks) == 0 || len(tasks) > 1024 || len(operations) != len(tasks) {
		return nil, fmt.Errorf("finite program requires 1..1024 operations matching its tasks")
	}
	if err := grid.Validate(); err != nil {
		return nil, err
	}
	if _, _, _, err := target.Capability(); err != nil {
		return nil, err
	}
	dependedOn := make(map[string]bool, len(tasks))
	for _, t := range tasks {
		for _, d := range t.Deps {
			dependedOn[d] = true
		}
	}
	owners, err := grid.Policy.Assign(tasks, grid.CTAs)
	if err != nil {
		return nil, err
	}
	// A task's release scope is the highest scope any edge leaving it is
	// ordered at, which is known only once every consumer has been seen. It
	// is resolved before the per-task loop because the loop emits it.
	byID := make(map[string]cudataskgraph.Task, len(tasks))
	for _, t := range tasks {
		byID[t.ID] = t
	}
	release := make([]cudataskgraph.Scope, len(tasks))
	for i, t := range tasks {
		release[i] = t.Scope
		for _, c := range tasks {
			for _, d := range c.Deps {
				if d == t.ID && c.Scope > release[i] {
					release[i] = c.Scope
				}
			}
		}
	}
	p := &Program{steps: make([]Step, 0, len(tasks)), grid: grid, target: target}
	indices := make(map[string]int, len(tasks))
	for i, t := range tasks {
		op, ok := operations[t.ID]
		if !ok || op.Kind != Add && op.Kind != Multiply && !(typed && op.Kind == MoEGateUp && op.moe != nil) {
			return nil, fmt.Errorf("task %q has no supported operation", t.ID)
		}
		if op.moe != nil {
			if !typed || op.Kind != MoEGateUp || len(tasks) != 1 || len(t.Deps) != 0 || grid.CTAs != 1 || grid.ThreadsPerCTA != 32 || t.Scope != cudataskgraph.ScopeDevice {
				return nil, fmt.Errorf("typed weight payload requires one independent device-scope task and one 32-thread CTA")
			}
			if err := op.moe.validate(); err != nil {
				return nil, err
			}
			value := *op.moe
			op.moe = &value
		}
		if err := supportsScope(target, t.Scope); err != nil {
			return nil, fmt.Errorf("task %q: %w", t.ID, err)
		}
		if t.Scope == cudataskgraph.ScopeCTA && grid.CTAs > 1 && (len(t.Deps) > 0 || dependedOn[t.ID]) {
			side := "depends on another task"
			if len(t.Deps) == 0 {
				side = "is depended on by another task"
			}
			return nil, fmt.Errorf("task %q %s at CTA scope in a %d-CTA grid; an mbarrier orders one CTA", t.ID, side, grid.CTAs)
		}
		deps := make([]int, 0, len(t.Deps))
		edges := make([]cudataskgraph.Scope, 0, len(t.Deps))
		for _, d := range t.Deps {
			n, ok := indices[d]
			if !ok {
				return nil, fmt.Errorf("task %q precedes dependency %q", t.ID, d)
			}
			// The edge is ordered at the higher of the two scopes it
			// joins. It needs no admissibility check of its own: the
			// higher of two scopes is one of them, and each end is
			// refused above if this target cannot order it. The same
			// argument covers release[i], which is a max over this
			// task's scope and its consumers'.
			e := byID[d].Scope
			if t.Scope > e {
				e = t.Scope
			}
			deps = append(deps, n)
			edges = append(edges, e)
		}
		indices[t.ID] = i
		p.steps = append(p.steps, Step{
			Task:         t,
			Operation:    op,
			Deps:         deps,
			Owner:        owners[i],
			EdgeScope:    edges,
			ReleaseScope: release[i],
		})

		if len(deps) > 0 {
			p.needSpin = true
		}
		// The mbarrier and its phase are needed by the fence sites that
		// order at CTA scope, which is not the same set as the tasks that
		// declare it: a CTA-scope task with a device-scope consumer releases
		// at device scope and arrives at no mbarrier.
		if release[i] == cudataskgraph.ScopeCTA {
			p.needBarrier = true
		}
		for _, e := range edges {
			if e == cudataskgraph.ScopeCTA {
				p.needBarrier, p.needPhase = true, true
			}
		}
	}
	return p, nil
}

// supportsScope reports whether target can order a scope, and why not when it
// cannot.
func supportsScope(target Target, s cudataskgraph.Scope) error {
	switch s {
	case cudataskgraph.ScopeCTA:
		ok, err := target.atLeast(8, 0)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("CTA scope needs mbarrier, which target %s predates", target)
		}
		return nil
	case cudataskgraph.ScopeCluster:
		ok, err := target.atLeast(9, 0)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("cluster scope needs barrier.cluster, which target %s does not have", target)
		}
		return fmt.Errorf("cluster scope on target %s needs lane 09's participation contract", target)
	case cudataskgraph.ScopeDevice, cudataskgraph.ScopeSystem:
		return nil
	}
	return fmt.Errorf("no synchronization is defined for scope %s", scopeName(s))
}

// Steps returns the program's tasks in the order the worker performs them.
func (p *Program) Steps() []Step {
	out := make([]Step, len(p.steps))
	for i, s := range p.steps {
		if s.Operation.moe != nil {
			value := *s.Operation.moe
			s.Operation.moe = &value
		}
		s.Task.Deps = append([]string(nil), s.Task.Deps...)
		s.Deps = append([]int(nil), s.Deps...)
		s.EdgeScope = append([]cudataskgraph.Scope(nil), s.EdgeScope...)
		out[i] = s
	}
	return out
}

// Grid returns the geometry the program was built for.
func (p *Program) Grid() Grid { return p.grid }

// Target returns the architecture the program was built for.
func (p *Program) Target() Target { return p.target }

// A Runner is one program bound to one backend.
//
// Binding is where a backend's declarations are checked against the program,
// so that a backend which cannot run it says so before any worker spins.
type Runner struct {
	p *Program
	b Backend
}

// NewRunner binds p to b, or reports why b may not run p.
//
// Two refusals live here and nowhere else. A backend that declares no
// forward-progress guarantee is refused, because the worker's dependency wait
// is a spin and a spin is only a wait where the predecessor is guaranteed to
// run. A backend whose residency admits fewer workers than the program's grid
// is refused, because the grid would contain a worker that never runs while
// the resident ones waited at its publication.
func NewRunner(p *Program, b Backend) (*Runner, error) {
	if p == nil {
		return nil, fmt.Errorf("nil program")
	}
	if b == nil {
		return nil, fmt.Errorf("nil backend")
	}
	switch g := b.ForwardProgress(); g {
	case CoResident, IndependentThreads:
	case NoForwardProgress:
		return nil, fmt.Errorf("backend declares no forward-progress guarantee; a dependency wait would be a deadlock, not a wait")
	default:
		return nil, fmt.Errorf("backend declares forward-progress guarantee %s, which is not one this package defines", g)
	}
	a := b.Residency()
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("backend residency: %w", err)
	}
	if a.ResidentCTAs < p.grid.CTAs {
		return nil, fmt.Errorf("backend admits %d workers, the program's grid is %d", a.ResidentCTAs, p.grid.CTAs)
	}
	if a.WorkerThreads != p.grid.ThreadsPerCTA {
		return nil, fmt.Errorf("backend admits %d threads per worker, the program's grid has %d", a.WorkerThreads, p.grid.ThreadsPerCTA)
	}
	return &Runner{p: p, b: b}, nil
}

// Run walks the program, performing or rendering each primitive in the order
// the persistent worker performs them.
//
// This is the only description of the worker's shape. A backend chooses what
// each primitive means and nothing else; that a worker scans the tasks it
// owns, takes only the ones already ready, acquires once per incoming edge,
// computes, releases and publishes, and waits only between scans, is decided
// here, once, for every backend there will ever be.
//
// The outer loop is the selection loop. The inner pass visits every step in
// program order, which is the order the emitted text lays them down; a worker
// that owns none of them, or that finds none of its own ready, falls through
// to [Backend.NextRound], which is where waiting happens and the only place
// it happens.
func (r *Runner) Run() error {
	p, b := r.p, r.b
	if err := b.Begin(p); err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	if err := b.BeginSelection(); err != nil {
		return fmt.Errorf("selection: %w", err)
	}
	for {
		for i, s := range p.steps {
			mine, err := b.BeginTask(i)
			if err != nil {
				return fmt.Errorf("task %q: %w", s.Task.ID, err)
			}
			if !mine {
				continue
			}
			// Every incoming edge is polled, not just enough of them to
			// decide: a scan that stopped at the first unsatisfied edge
			// would leave the emitted text with a data-dependent number of
			// acquiring loads, and the artifact could no longer be checked
			// against the edge list.
			for k, d := range s.Deps {
				if err := b.Ready(i, d, s.EdgeScope[k]); err != nil {
					return fmt.Errorf("task %q: ready %q: %w", s.Task.ID, p.steps[d].Task.ID, err)
				}
			}
			if len(s.Deps) > 0 {
				if err := b.Barrier(); err != nil {
					return fmt.Errorf("task %q: barrier: %w", s.Task.ID, err)
				}
			}
			taken, err := b.Take(i)
			if err != nil {
				return fmt.Errorf("task %q: take: %w", s.Task.ID, err)
			}
			if !taken {
				if err := b.EndTask(i); err != nil {
					return fmt.Errorf("task %q: %w", s.Task.ID, err)
				}
				continue
			}
			for k, d := range s.Deps {
				if err := b.Fence(i, d, s.EdgeScope[k], Acquire); err != nil {
					return fmt.Errorf("task %q: %w", s.Task.ID, err)
				}
			}
			src := -1
			if len(s.Deps) > 0 {
				src = s.Deps[0]
			}
			if err := b.Compute(i, src, s.Operation); err != nil {
				return fmt.Errorf("task %q: compute: %w", s.Task.ID, err)
			}
			if err := b.Fence(i, -1, s.ReleaseScope, Release); err != nil {
				return fmt.Errorf("task %q: %w", s.Task.ID, err)
			}
			if err := b.Publish(i); err != nil {
				return fmt.Errorf("task %q: publish: %w", s.Task.ID, err)
			}
			if err := b.EndTask(i); err != nil {
				return fmt.Errorf("task %q: %w", s.Task.ID, err)
			}
		}
		more, err := b.NextRound()
		if err != nil {
			return fmt.Errorf("selection: %w", err)
		}
		if !more {
			break
		}
	}
	if err := b.EndSelection(); err != nil {
		return fmt.Errorf("selection: %w", err)
	}
	return b.End()
}
