package cudamegacpu

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// An Executor runs programs on one OS thread per worker thread.
//
// The geometry is the admission's: ResidentCTAs workers of WorkerThreads
// threads each, so a program built for a grid this executor does not admit is
// refused by [cudamega.NewRunner] before any thread starts.
type Executor struct {
	admission cudamega.Admission
	// progress is a field rather than a constant so that a test can build an
	// executor that declares nothing and watch the binding refuse it. It is
	// not settable from outside this package: a backend whose guarantee a
	// caller could weaken would be a backend whose guarantee means nothing.
	progress cudamega.ForwardProgress

	// wrap, when set, is applied to each worker before it is bound. It is
	// the seam the load-bearing test needs: that test has to watch the
	// completion words a worker reads and to hold one worker back, and
	// neither is observable from outside the run. It is unexported and set
	// only by this package's own test.
	wrap func(cudamega.Backend) cudamega.Backend
}

// New returns an executor with the given residency.
//
// SharedBytes has no meaning here -- there is no shared-memory window to
// admit from -- and is required to be nonzero only because [cudamega.Admission]
// validates it. An executor reports it back unchanged so that a program
// admitted against a device's shared-memory limit is not silently admitted
// against a different one here.
func New(a cudamega.Admission) (*Executor, error) {
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("cudamegacpu: %w", err)
	}
	return &Executor{admission: a, progress: cudamega.IndependentThreads}, nil
}

// Admission returns the residency this executor runs at.
func (e *Executor) Admission() cudamega.Admission { return e.admission }

// A Result is one run's output: the scratch buffer every task wrote into, and
// the abort code the run ended with.
type Result struct {
	// Scratch holds len(steps)*len(input) words, task-major.
	Scratch []uint32
	// StartOrder is the overlap instrument, laid out as the emitted worker's
	// buffer is: word zero is the ticket counter, and word
	// 1 + cta*len(steps) + task is the one-based ticket that worker held when
	// it took that task, or zero if it never did. See [cudamega.StartOrderWords].
	StartOrder []uint64
	// Abort is zero for a run that finished, and otherwise one of the
	// cudamega.Abort constants.
	Abort uint32
}

// Run executes p once over input and returns what the workers wrote.
//
// Every buffer the emitted CUDA worker takes as a parameter has a counterpart
// here: input, scratch, the completion words and the abort word. They are
// allocated per run because the worker is persistent only within one epoch,
// and because a buffer reused across runs would let a run pass on a
// predecessor's publications.
//
// abort lets the caller start a run already aborted, which is what the
// emitted worker's host-request path does. epoch must be nonzero, for the
// reason [cudamega.ProgramSource] documents.
func (e *Executor) Run(p *cudamega.Program, input []uint32, epoch uint64, abort uint32) (Result, error) {
	if p == nil {
		return Result{}, fmt.Errorf("cudamegacpu: nil program")
	}
	steps := p.Steps()
	if len(input) == 0 {
		return Result{}, fmt.Errorf("cudamegacpu: empty input")
	}
	r := &run{
		p:     p,
		steps: steps,
		n:     len(input),
		// The launch geometry is the executor's, not the program's. That is
		// what the host does on a device -- it launches at the residency it
		// measured -- and it is what leaves the emitted worker's geometry
		// refusal something to refuse. Taking the geometry from the program
		// would make that check unfalsifiable here.
		ctas:    e.admission.ResidentCTAs,
		threads: e.admission.WorkerThreads,
		input:   input,
		scratch: make([]uint32, len(steps)*len(input)),
		epoch:   epoch,
		exec:    e,
	}
	r.completed = make([]atomic.Uint32, len(steps))
	r.startOrder = make([]uint64, 1+len(steps)*int(r.ctas))
	r.abort.Store(abort)
	r.cta = make([]ctaState, r.ctas)
	for i := range r.cta {
		r.cta[i].barrier = newBarrier(r.threads)
	}

	// Bind before starting anything. A backend that may not run this program
	// must be refused at construction, not discovered by a worker that spins.
	runners := make([]*cudamega.Runner, 0, int(r.ctas)*int(r.threads))
	for cta := uint32(0); cta < r.ctas; cta++ {
		for tid := uint32(0); tid < r.threads; tid++ {
			var b cudamega.Backend = &worker{r: r, cta: cta, tid: tid}
			if e.wrap != nil {
				b = e.wrap(b)
			}
			run, err := cudamega.NewRunner(p, b)
			if err != nil {
				return Result{}, fmt.Errorf("cudamegacpu: %w", err)
			}
			runners = append(runners, run)
		}
	}

	errs := make([]error, len(runners))
	var wg sync.WaitGroup
	wg.Add(len(runners))
	for i, run := range runners {
		go func() {
			defer wg.Done()
			// The guarantee this backend declares is that each worker is
			// scheduled independently of the others. Locking the goroutine to
			// its own OS thread is what makes that true rather than a hope
			// about the Go scheduler.
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			errs[i] = run.Run()
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return Result{}, fmt.Errorf("cudamegacpu: %w", err)
		}
	}
	// Word zero is the ticket counter, the way it is in the emitted worker's
	// buffer: there the counter is the word the atomic adds to, so it ends at
	// the number of tickets dealt. Here the counter is a separate atomic, so
	// it is copied in once the workers have joined. The two buffers are read
	// by the same test, and a layout that differed between them would make
	// that test's agreement meaningless.
	r.startOrder[0] = r.ticket.Load()
	return Result{Scratch: r.scratch, StartOrder: r.startOrder, Abort: r.abort.Load()}, nil
}

// spinLimit bounds a dependency wait, for the reason the emitted worker's
// bound exists: a wait that cannot end leaves a process no signal can help,
// and here it would also hold an OS thread.
//
// It is far above any legitimate wait for a program this package runs. It is
// not a deadline and nothing calibrates it against a poll rate.
const spinLimit = 1 << 26

// A run is the state one execution of a program shares across its workers:
// the buffers the emitted worker takes as parameters, plus one barrier per
// worker.
type run struct {
	p       *cudamega.Program
	steps   []cudamega.Step
	n       int
	ctas    uint32
	threads uint32
	epoch   uint64
	exec    *Executor

	input     []uint32
	scratch   []uint32
	completed []atomic.Uint32
	abort     atomic.Uint32
	cta       []ctaState

	// startOrder and ticket are the overlap instrument, not part of the
	// model. ticket is the device-scope atomic the emitted worker deals
	// tickets from; startOrder is the buffer it writes them into. Each slot
	// is written by exactly one thread of one worker and read only after
	// every worker has finished, so the plain slice needs no atomics.
	startOrder []uint64
	ticket     atomic.Uint64

	// fence is the word a device- or system-scope fence performs a
	// read-modify-write on. Go exposes no standalone fence; a
	// sequentially-consistent atomic operation is the strongest ordering it
	// offers, and performing one on a word every worker shares is the
	// nearest thing to fence.sc.gpu the language has. See worker.Fence.
	fence atomic.Uint64
}

// A ctaState is one worker's private state: its barrier, and the two
// decisions its threads make together.
type ctaState struct {
	barrier *barrier
	// ready and aborted are written by thread zero and read by the rest,
	// with the barrier in between. They are the CPU counterparts of the two
	// shared words the emitted worker declares: a readiness word another CTA
	// is concurrently making stale, and the abort word. Both are read once
	// per worker rather than once per thread, because the threads of one
	// worker must reach the same conclusion or the barrier below is short an
	// arrival.
	ready, aborted bool
}

// A worker is one thread's view of a run. It is the CPU backend.
type worker struct {
	r   *run
	cta uint32
	tid uint32

	// The selection state, all of it per thread and all of it uniform across
	// the threads of one worker by construction: done and remaining change
	// only in EndTask, which every thread of a worker reaches together.
	done      []bool
	remaining int
	spins     uint64
	stopped   bool
	// took is the task this worker committed to in Take, or -1.
	took int
}

var _ cudamega.Backend = (*worker)(nil)

// ForwardProgress reports that each worker runs on an independently scheduled
// OS thread. See the package comment.
func (w *worker) ForwardProgress() cudamega.ForwardProgress { return w.r.exec.progress }

// Residency reports the executor's configured admission. Unlike a device's,
// it is not measured: the number of workers that can run concurrently here is
// the number of threads the caller asked for, because the operating system
// will schedule all of them.
func (w *worker) Residency() cudamega.Admission { return w.r.exec.admission }

// Begin makes the refusals the emitted worker's entry makes: a geometry that
// is not the one the program was built for, and an empty extent.
func (w *worker) Begin(p *cudamega.Program) error {
	g := p.Grid()
	if g.CTAs != w.r.ctas || g.ThreadsPerCTA != w.r.threads {
		return fmt.Errorf("launched %dx%d, generated for %dx%d: abort code %d",
			w.r.ctas, w.r.threads, g.CTAs, g.ThreadsPerCTA, cudamega.AbortBadLaunch)
	}
	if w.r.n == 0 || w.r.epoch == 0 {
		return fmt.Errorf("extent %d at epoch %d: abort code %d", w.r.n, w.r.epoch, cudamega.AbortBadExtent)
	}
	return nil
}

// BeginSelection takes stock of what this worker owns and makes the abort
// observation the emitted worker makes at the top of its scan loop.
//
// The observation is here as well as in NextRound because a run may be
// aborted before it starts: the host writes the word and launches anyway, and
// a worker that only checked between rounds would perform its whole first
// scan first.
func (w *worker) BeginSelection() error {
	w.done = make([]bool, len(w.r.steps))
	w.remaining = 0
	for _, s := range w.r.steps {
		if s.Owner.Contains(w.cta) {
			w.remaining++
		}
	}
	w.took = -1
	w.observeAbort()
	return nil
}

// observeAbort reads the abort word once per worker and tells the worker's
// threads what it said. See ctaState.
func (w *worker) observeAbort() {
	c := &w.r.cta[w.cta]
	if w.tid == 0 {
		c.aborted = w.r.abort.Load() != 0
	}
	c.barrier.wait()
	if c.aborted {
		w.stopped = true
	}
	c.barrier.wait()
}

// BeginTask makes the ownership test, and makes it uniformly across the
// threads of one worker.
//
// It never waits and it reads no word another worker is writing: ownership is
// a function of blockIdx alone, and completion is this worker's own record.
// That is what keeps the threads of a worker in step through the barrier
// below without the emitted worker's divergent-read hazard.
func (w *worker) BeginTask(i int) (bool, error) {
	if w.stopped {
		return false, nil
	}
	mine := w.r.steps[i].Owner.Contains(w.cta) && !w.done[i]
	if mine && w.tid == 0 {
		// The verdict starts clean on every visit, the way the emitted
		// worker's "task_ready = 1u" does: a readiness word carried over
		// from a previous round would record what was false then and never
		// become true, and the worker would spin to the bound on a run that
		// was proceeding correctly.
		w.r.cta[w.cta].ready = true
	}
	return mine, nil
}

// Ready is the acquiring load of a completion word. It does not wait.
//
// Only thread zero loads, exactly as in the emitted worker, and the
// happens-before edge it establishes reaches the rest through the barrier the
// driver performs immediately after. Go's atomics are sequentially
// consistent, so the load here is an acquiring load in the sense the model
// needs: a thread that observes the producer's increment also observes every
// store the producer made before it.
//
// The count it compares against is the dependency's own owner width. Against
// the grid it would be a comparison a subrange-owned task never satisfies,
// and the worker would spin to the bound on a run that was proceeding
// correctly.
func (w *worker) Ready(_, dep int, _ cudataskgraph.Scope) error {
	if w.tid != 0 {
		return nil
	}
	if w.r.completed[dep].Load() < w.r.steps[dep].Owner.Count() {
		w.r.cta[w.cta].ready = false
	}
	return nil
}

// Take commits this worker to a task, or declines it. It never waits: what it
// reads is what thread zero already saw, published through the driver's
// barrier.
func (w *worker) Take(i int) (bool, error) {
	taken := true
	if len(w.r.steps[i].Deps) > 0 {
		c := &w.r.cta[w.cta]
		taken = c.ready
		// Every thread has now read the verdict, so thread zero may write it
		// again. Without this barrier thread zero can reach the next task's
		// Ready while a slower thread is still reading this one's, and the
		// two then disagree about a task the whole worker must agree on. The
		// emitted worker carries the same barrier for the same reason.
		c.barrier.wait()
	}
	if !taken {
		return false, nil
	}
	w.took = i
	if w.tid == 0 {
		w.r.startOrder[1+int(w.cta)*len(w.r.steps)+i] = w.r.ticket.Add(1)
	}
	return true, nil
}

// EndTask retires a task this worker took. It is also called for a task the
// worker declined, which is why it asks Take what happened rather than
// assuming.
func (w *worker) EndTask(i int) error {
	if w.took != i {
		return nil
	}
	w.took = -1
	w.done[i] = true
	w.remaining--
	return nil
}

// NextRound is where this backend waits, and the only place it does.
//
// A worker reaches it having declined every task it owns, so what it waits
// for is another worker's publication. The bound exists for the reason the
// emitted worker's does: a wait that cannot end leaves a process no signal
// can help, and here it would also hold an OS thread.
func (w *worker) NextRound() (bool, error) {
	if w.stopped || w.remaining == 0 {
		return false, nil
	}
	w.spins++
	if w.spins > spinLimit && w.tid == 0 {
		w.r.abort.CompareAndSwap(0, cudamega.AbortSpinExhausted)
	}
	// Yielding is a courtesy to a machine with fewer cores than workers, not
	// the guarantee: the guarantee is that the OS would schedule the producer
	// anyway.
	runtime.Gosched()
	w.observeAbort()
	return !w.stopped, nil
}

func (w *worker) EndSelection() error { return nil }

// Barrier is the workgroup barrier over the threads of one worker.
func (w *worker) Barrier() error {
	w.r.cta[w.cta].barrier.wait()
	return nil
}

// Fence performs one side of a release/acquire pair.
//
// At CTA scope it is the workgroup barrier. That is the optional sixth
// primitive degrading as the model allows: the emitted CUDA orders CTA scope
// with an mbarrier, and a backend without one orders it with the barrier it
// has, which carries the same guarantee over the same set of threads.
//
// At device and system scope it is a sequentially-consistent read-modify-write
// on a word every worker shares. Go has no standalone fence, and a
// sequentially-consistent atomic is the strongest ordering the language
// offers; performing one on a shared word places this thread's prior and
// subsequent memory operations on either side of a point in a total order
// every other worker also observes. This orders this thread's operations;
// it does not wait for the other threads' stores. Publish gathers those
// stores before releasing the completion word, and Ready acquires that word.
// System scope is not distinguished from
// device scope, because this backend has no second agent to be ordered
// against: there is one address space and the host is the device.
func (w *worker) Fence(_, _ int, s cudataskgraph.Scope, k cudamega.Fence) error {
	switch k {
	case cudamega.Acquire, cudamega.Release:
	default:
		return fmt.Errorf("unknown fence side %s", k)
	}
	switch s {
	case cudataskgraph.ScopeCTA:
		w.r.cta[w.cta].barrier.wait()
		return nil
	case cudataskgraph.ScopeDevice, cudataskgraph.ScopeSystem:
		w.r.fence.Add(0)
		return nil
	}
	return fmt.Errorf("no %s is defined for scope %d", k, s)
}

// Compute applies the operation to this thread's share of the task's
// elements, striding by the owning CTA range's thread count exactly as the
// emitted worker does.
//
// The stride is the owner's, not the grid's. Under a policy where a consumer
// and its producer have different owners, that is what makes the element a
// consumer reads one another worker wrote -- and therefore what makes the
// dependency wait something the result depends on rather than a coincidence
// of the fixture.
func (w *worker) Compute(i, src int, op cudamega.Operation) error {
	n := w.r.n
	from := w.r.input
	if src >= 0 {
		from = w.r.scratch[src*n : src*n+n]
	}
	to := w.r.scratch[i*n : i*n+n]
	o := w.r.steps[i].Owner
	stride := int(o.Count()) * int(w.r.threads)
	for e := int(w.cta-o.Lo)*int(w.r.threads) + int(w.tid); e < n; e += stride {
		v := from[e]
		switch op.Kind {
		case cudamega.Add:
			to[e] = v + op.Scalar
		case cudamega.Multiply:
			to[e] = v * op.Scalar
		default:
			return fmt.Errorf("unsupported operation %d", op.Kind)
		}
	}
	return nil
}

// Publish increments the task's completion word once per worker after every
// thread of that worker has finished its stores. The workgroup barrier
// carries those stores to thread zero; its atomic increment releases them to
// a consumer's Ready load. The consumer's barrier then carries the acquired
// stores to its remaining threads. Other workers do not join this barrier.
func (w *worker) Publish(i int) error {
	w.r.cta[w.cta].barrier.wait()
	if w.tid == 0 {
		w.r.completed[i].Add(1)
	}
	return nil
}

func (w *worker) End() error { return nil }

// A barrier is a reusable barrier over a fixed number of threads.
//
// The phase counter is what makes it reusable without a reset, which is the
// same property the emitted mbarrier's alternating parity has: a barrier that
// had to be reset between waves would have a window in which an early arriver
// passes the previous wave's barrier.
type barrier struct {
	n     uint32
	mu    sync.Mutex
	cond  *sync.Cond
	count uint32
	phase uint64
}

func newBarrier(n uint32) *barrier {
	b := &barrier{n: n}
	b.cond = sync.NewCond(&b.mu)
	return b
}

func (b *barrier) wait() {
	b.mu.Lock()
	defer b.mu.Unlock()
	phase := b.phase
	b.count++
	if b.count == b.n {
		b.count = 0
		b.phase++
		b.cond.Broadcast()
		return
	}
	for phase == b.phase {
		b.cond.Wait()
	}
}
