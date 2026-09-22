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
// smaller. Five of the methods are the synchronization primitives the model
// rests on:
//
//   - [Backend.Barrier] is the workgroup barrier, over the threads of one
//     worker.
//   - [Backend.Publish] and [Backend.Await] are the device-scope atomic add
//     and the acquiring load of a completion word. They are one primitive
//     seen from its two ends; neither is usable without the other.
//   - [Backend.Fence] is the release/acquire fence at a declared scope. At
//     CTA scope a backend may serve it with a workgroup-scoped barrier
//     primitive -- an mbarrier on CUDA -- which is why the optional sixth
//     primitive is a quality of this method rather than a method of its own.
//   - [Backend.Residency] is the residency query.
//   - [Backend.ForwardProgress] is the co-residency guarantee, which is the
//     actual boundary. See [ForwardProgress].
//
// The remaining methods are structure, not synchronization: they bracket the
// run and the individual tasks and carry the elementwise payload. They are
// here because the driver in [Runner.Run] must be the only description of the
// worker's shape, and a driver that could not open a task's guarded region
// would leave that shape in each backend.
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

	// BeginTask opens step i's abort-guarded region and reports whether the
	// step's remaining primitives are performed at all. A rendering backend
	// emits the guard and reports true, because the test is made at run time
	// by the code it emits. A performing backend makes the test itself, and
	// must make it uniformly across the threads of one worker: a guard that
	// some threads of a worker take and others do not leaves the barrier
	// below short an arrival.
	BeginTask(i int) (bool, error)

	// Await waits until every worker has published step dep, reading that
	// completion word with an acquiring load. It returns when the count is
	// reached, when an abort is observed, or when the backend's own poll
	// bound is exhausted -- never never.
	Await(i, dep int) error

	// Barrier is the workgroup barrier over the threads of one worker.
	Barrier() error

	// Fence performs side k of step i's release/acquire pair at scope s. It
	// reports an error for a scope the backend cannot order.
	Fence(i int, s cudataskgraph.Scope, k Fence) error

	// Compute applies step i's operation to this worker's share of the
	// elements, reading from step src's output or, when src is negative,
	// from the program input.
	Compute(i, src int, op Operation) error

	// Publish increments step i's completion word with a device-scope atomic
	// add, once per worker.
	Publish(i int) error

	// EndTask closes the region BeginTask opened.
	EndTask(i int) error

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
	p := &Program{steps: make([]Step, 0, len(tasks)), grid: grid, target: target}
	indices := make(map[string]int, len(tasks))
	for i, t := range tasks {
		op, ok := operations[t.ID]
		if !ok || op.Kind != Add && op.Kind != Multiply {
			return nil, fmt.Errorf("task %q has no supported operation", t.ID)
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
		for _, d := range t.Deps {
			n, ok := indices[d]
			if !ok {
				return nil, fmt.Errorf("task %q precedes dependency %q", t.ID, d)
			}
			deps = append(deps, n)
		}
		indices[t.ID] = i
		p.steps = append(p.steps, Step{Task: t, Operation: op, Deps: deps})

		if len(deps) > 0 {
			p.needSpin = true
		}
		if t.Scope == cudataskgraph.ScopeCTA {
			p.needBarrier = true
			if len(deps) > 0 {
				p.needPhase = true
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
		s.Task.Deps = append([]string(nil), s.Task.Deps...)
		s.Deps = append([]int(nil), s.Deps...)
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

// Run walks the program once, performing or rendering each primitive in the
// order the persistent worker performs them.
//
// This is the only description of the worker's shape. A backend chooses what
// each primitive means and nothing else; the acquire side of every dependency,
// then the computation, then the release side, then the publishing store is
// decided here, once, for every backend there will ever be.
func (r *Runner) Run() error {
	p, b := r.p, r.b
	if err := b.Begin(p); err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	for i, s := range p.steps {
		run, err := b.BeginTask(i)
		if err != nil {
			return fmt.Errorf("task %q: %w", s.Task.ID, err)
		}
		if !run {
			continue
		}
		for _, d := range s.Deps {
			if err := b.Await(i, d); err != nil {
				return fmt.Errorf("task %q: await %q: %w", s.Task.ID, p.steps[d].Task.ID, err)
			}
			if err := b.Barrier(); err != nil {
				return fmt.Errorf("task %q: barrier: %w", s.Task.ID, err)
			}
		}
		src := -1
		if len(s.Deps) > 0 {
			src = s.Deps[0]
			if err := b.Fence(i, s.Task.Scope, Acquire); err != nil {
				return fmt.Errorf("task %q: %w", s.Task.ID, err)
			}
		}
		if err := b.Compute(i, src, s.Operation); err != nil {
			return fmt.Errorf("task %q: compute: %w", s.Task.ID, err)
		}
		if err := b.Fence(i, s.Task.Scope, Release); err != nil {
			return fmt.Errorf("task %q: %w", s.Task.ID, err)
		}
		if err := b.Publish(i); err != nil {
			return fmt.Errorf("task %q: publish: %w", s.Task.ID, err)
		}
		if err := b.EndTask(i); err != nil {
			return fmt.Errorf("task %q: %w", s.Task.ID, err)
		}
	}
	return b.End()
}
