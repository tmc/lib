package cudamega

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// A cudaSource is the backend that renders the primitives as CUDA C++.
//
// It performs nothing. Every method appends the text that performs the
// primitive when the emitted program runs, which is why one value stands for
// the whole grid: the text it writes is executed by every thread.
type cudaSource struct {
	p          *Program
	defs, body strings.Builder

	// readyOpen records that a readiness block is open, so that the barrier
	// which follows the last acquiring load closes it. The driver's calls
	// are Ready* then Barrier, and the block spans exactly that run.
	readyOpen bool
}

// ForwardProgress reports the guarantee a cooperative launch at or below the
// measured occupancy provides.
//
// It is declared by the rendering backend rather than discovered by it: the
// occupancy is measured on the compiled artifact by the harness that launches
// it, and [Admission] is where that measurement enters this package. What is
// asserted here is that the emitted worker is written to be launched that
// way, and a launch which is not is the one the emitted geometry refusal
// catches.
func (c *cudaSource) ForwardProgress() ForwardProgress { return CoResident }

// Residency reports the grid the program was built for. A rendering backend
// admits exactly the geometry it renders for and nothing else: the worker it
// writes refuses at run time to run under any other.
func (c *cudaSource) Residency() Admission {
	return Admission{ResidentCTAs: c.p.grid.CTAs, WorkerThreads: c.p.grid.ThreadsPerCTA, SharedBytes: 1}
}

func (c *cudaSource) Begin(p *Program) error {
	if c.p != p {
		return fmt.Errorf("backend was built for another program")
	}
	for _, s := range p.steps {
		if s.Operation.Kind != Add && s.Operation.Kind != Multiply || s.Operation.moe != nil {
			return fmt.Errorf("scalar CUDA backend refuses typed payload")
		}
	}
	fmt.Fprintf(&c.defs, "constexpr unsigned task_count = %d;\n", len(p.steps))
	fmt.Fprintf(&c.defs, "constexpr unsigned grid_ctas = %d;\n", p.grid.CTAs)
	fmt.Fprintf(&c.defs, "constexpr unsigned grid_threads = %d;\n", p.grid.ThreadsPerCTA)
	// Only a dependency wait polls, so only a graph with a dependency edge
	// declares the bound. Declaring it unconditionally leaves an unreferenced
	// constant in every dependency-free worker, which nvcc reports.
	if p.needSpin {
		fmt.Fprintf(&c.defs, "constexpr unsigned long long spin_limit = %dull;\n", p.grid.spinLimit())
	}
	// One producer count per task some other task waits on. It is the
	// task's own owner width, and the reason it is a per-task constant
	// rather than grid_ctas is that a task performed by a subrange of the
	// grid publishes that many completions and never more.
	dependedOn := make([]bool, len(p.steps))
	for _, s := range p.steps {
		for _, d := range s.Deps {
			dependedOn[d] = true
		}
	}
	for i, s := range p.steps {
		if dependedOn[i] {
			fmt.Fprintf(&c.defs, "constexpr unsigned producers_%d = %du; // task %q is performed by CTAs %s\n",
				i, s.Owner.Count(), s.Task.ID, s.Owner)
		}
	}
	c.defs.WriteString("static_assert(task_count >= 1, \"the generated worker has at least one task\");\n")
	for i, s := range p.steps {
		operator := "+"
		if s.Operation.Kind == Multiply {
			operator = "*"
		}
		fmt.Fprintf(&c.defs, "__device__ unsigned op_%d(unsigned x) { return x %s %du; }\n", i, operator, s.Operation.Scalar)
	}
	return nil
}

// BeginSelection writes the worker's selection state and opens the loop that
// scans it.
//
// The loop is what separates taking a task from waiting for one. A worker
// that finds nothing ready falls out of the scan and goes round again; it
// never sits inside a task it has taken. The abort word is read once per
// round by thread zero into shared memory rather than by every thread at
// every guard: the guards below contain a __syncthreads, and a word another
// CTA may be writing is not a thing the threads of one CTA can be relied on
// to read alike.
func (c *cudaSource) BeginSelection() error {
	p := c.p
	b := &c.body
	b.WriteString("\n  __shared__ unsigned selection_abort;\n")
	if p.needSpin {
		b.WriteString("  __shared__ unsigned task_ready;\n")
	}
	for i, s := range p.steps {
		// The lower bound is emitted only where it can be false. Against
		// zero it is a comparison nvcc reports as always true, and a
		// generated warning in every wave-policy worker would train the eye
		// to ignore the ones that mean something.
		test := fmt.Sprintf("blockIdx.x < %du", s.Owner.Hi)
		if s.Owner.Lo > 0 {
			test = fmt.Sprintf("blockIdx.x >= %du && %s", s.Owner.Lo, test)
		}
		fmt.Fprintf(b, "  const bool owns_%d = (%s); // task %q\n", i, test, s.Task.ID)
	}
	for i := range p.steps {
		fmt.Fprintf(b, "  unsigned done_%d = 0u;\n", i)
	}
	b.WriteString("  unsigned remaining = 0u")
	for i := range p.steps {
		fmt.Fprintf(b, " + (unsigned)owns_%d", i)
	}
	b.WriteString(";\n")
	if p.needSpin {
		b.WriteString("  unsigned long long spins = 0ull;\n")
	}
	b.WriteString("  while (remaining) {\n")
	// The shared word is written by one thread, read by all, and then not
	// written again until every reader has finished with it. That is what
	// the second barrier is for: without it thread zero can go round the
	// scan and overwrite the word while a slower thread of the same worker
	// is still reading the previous round's answer, and the two then take
	// different branches at a barrier that needs both.
	b.WriteString("    if (threadIdx.x == 0) selection_abort = *abort_code;\n")
	b.WriteString("    __syncthreads();\n")
	b.WriteString("    const unsigned stop = selection_abort;\n")
	b.WriteString("    __syncthreads();\n")
	b.WriteString("    if (stop) break;\n")
	return nil
}

// BeginTask opens the task's ownership guard. The guard is emitted, never
// taken: whether the region runs is a question the device answers, and it is
// a question with no wait in it.
func (c *cudaSource) BeginTask(i int) (bool, error) {
	s := c.p.steps[i]
	fmt.Fprintf(&c.body, "\n    // task %d %q scope=%s rank=%d deps=%v owner=%s\n",
		i, s.Task.ID, scopeName(s.Task.Scope), s.Task.Rank, s.Deps, s.Owner)
	fmt.Fprintf(&c.body, "    if (owns_%d && !done_%d) {\n", i, i)
	return true, nil
}

// Ready renders one acquiring load of a dependency's completion word.
//
// The load is an acquiring load, not a volatile one. They are not the same
// thing and the difference is the whole reason this primitive exists:
// volatile forbids the compiler from caching the load and orders nothing at
// all against the producer's stores, so the reads that follow may still be
// satisfied from stale lines. ld.acquire orders them. The fence the driver
// emits after the barrier below is still required, because only thread zero
// executes this load and the acquire it performs is its own.
//
// It is compared against producers_<dep>, the number of CTAs that perform
// that task. Against grid_ctas it would be a comparison no subrange-owned
// task can ever satisfy.
//
// There is no loop here. The load happens once per scan; the scan is the
// loop, and it is bounded in EndSelection.
func (c *cudaSource) Ready(i, dep int, s cudataskgraph.Scope) error {
	scope, err := ptxScope(s)
	if err != nil {
		return err
	}
	b := &c.body
	if !c.readyOpen {
		b.WriteString("      if (threadIdx.x == 0) {\n")
		b.WriteString("        unsigned published = 0u;\n")
		b.WriteString("        task_ready = 1u;\n")
		c.readyOpen = true
	}
	addr := "%1"
	if dep > 0 {
		addr = fmt.Sprintf("%%1+%d", 4*dep)
	}
	fmt.Fprintf(b, "        asm volatile(\"ld.acquire.%s.u32 %%0, [%s];\" : \"=r\"(published) : \"l\"(completed) : \"memory\"); // task %d\n", scope, addr, dep)
	fmt.Fprintf(b, "        if (published < producers_%d) task_ready = 0u;\n", dep)
	fmt.Fprintf(b, "        if (published > producers_%d) { task_ready = 0u; atomicCAS((unsigned *)abort_code, 0u, %du); }\n", dep, AbortBadCompletion)
	return nil
}

// Barrier closes the readiness block, if one is open, and emits the workgroup
// barrier. Thread zero's loads reach the rest of the worker through it.
func (c *cudaSource) Barrier() error {
	if c.readyOpen {
		c.body.WriteString("      }\n")
		c.readyOpen = false
	}
	c.body.WriteString("      __syncthreads();\n")
	return nil
}

// Take renders the commitment to a task, and the record of when it happened.
//
// The condition is what thread zero read, never a wait: a worker that arrives
// here with task_ready clear leaves the task alone and goes on scanning. A
// task with no dependencies is taken unconditionally, and the brace is still
// emitted so that EndTask closes the same number of regions either way.
func (c *cudaSource) Take(i int) (bool, error) {
	b := &c.body
	if len(c.p.steps[i].Deps) > 0 {
		// Read once into a register, then release the shared word with a
		// barrier before anyone can write it again. See BeginSelection.
		fmt.Fprintf(b, "      const unsigned take_%d = task_ready;\n", i)
		b.WriteString("      __syncthreads();\n")
		fmt.Fprintf(b, "      if (take_%d) {\n", i)
	} else {
		b.WriteString("      { // no dependency: ready on sight\n")
	}
	fmt.Fprintf(b, "        if (threadIdx.x == 0) start_order[1 + (size_t)blockIdx.x*task_count + %d] = atomicAdd(start_order, 1ull) + 1ull;\n", i)
	return true, nil
}

func (c *cudaSource) Fence(i, dep int, s cudataskgraph.Scope, k Fence) error {
	var instruction string
	var err error
	edge := "release"
	switch k {
	case Acquire:
		instruction, err = acquireFence(s)
		edge = fmt.Sprintf("acquire of task %d", dep)
	case Release:
		instruction, err = releaseFence(s)
	default:
		err = fmt.Errorf("unknown fence side %s", k)
	}
	if err != nil {
		return err
	}
	fmt.Fprintf(&c.body, "        %s // task %d: %s at %s scope\n", instruction, i, edge, scopeName(s))
	return nil
}

// Compute deals the task's elements to the CTAs that own it.
//
// The stride is the owner's width, not the grid's, and the first element is
// taken from the CTA's position within the owner range. That is the one
// change that makes a dependency wait observable at all: under a policy where
// producer and consumer have different owners, the element a consumer reads
// was written by another CTA, and a wait that did not happen is a wrong
// answer rather than a coincidence.
func (c *cudaSource) Compute(i, src int, op Operation) error {
	s := c.p.steps[i]
	expr := "input"
	if src >= 0 {
		expr = fmt.Sprintf("scratch + (size_t)%d*n", src)
	}
	b := &c.body
	fmt.Fprintf(b, "        const unsigned *src_%d = %s;\n", i, expr)
	first := "(size_t)blockIdx.x"
	if s.Owner.Lo > 0 {
		first = fmt.Sprintf("(size_t)(blockIdx.x - %du)", s.Owner.Lo)
	}
	fmt.Fprintf(b, "        for (size_t e = %s*blockDim.x + threadIdx.x; e < n; e += (size_t)%du*blockDim.x)\n",
		first, s.Owner.Count())
	fmt.Fprintf(b, "          scratch[(size_t)%d*n + e] = op_%d(src_%d[e]);\n", i, i, i)
	return nil
}

func (c *cudaSource) Publish(i int) error {
	fmt.Fprintf(&c.body, "        if (threadIdx.x == 0) atomicAdd(completed + %d, 1u); // publish task %d\n", i, i)
	return nil
}

// EndTask closes the region Take opened and retires the task from this
// worker's scan.
func (c *cudaSource) EndTask(i int) error {
	fmt.Fprintf(&c.body, "        done_%d = 1u; remaining--;\n", i)
	c.body.WriteString("      }\n")
	c.body.WriteString("    }\n")
	return nil
}

// NextRound reports that the rendering backend needs no second pass: it has
// written the text of one scan, and the loop EndSelection closes is what runs
// it again.
func (c *cudaSource) NextRound() (bool, error) { return false, nil }

// EndSelection closes the scan loop and bounds it.
//
// The bound is in the emitted loop, not in a harness. An external timeout
// cannot stop a CTA that will not yield: killing the host process leaves the
// grid resident. It is installed on the round counter rather than on a
// per-dependency poll because the round is now the only thing that repeats.
func (c *cudaSource) EndSelection() error {
	b := &c.body
	if c.p.needSpin {
		b.WriteString("\n    ++spins;\n")
		fmt.Fprintf(b, "    if (spins > spin_limit && threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, %du);\n", AbortSpinExhausted)
	}
	b.WriteString("  }\n")
	return nil
}

// End writes the tables the reference kernel reads. They come last because
// they are a summary of the whole task list, and the driver has no step at
// which to ask for one.
func (c *cudaSource) End() error {
	c.defs.WriteString("__device__ int dependency(unsigned task) { switch(task) {\n")
	for i, s := range c.p.steps {
		dep := -1
		if len(s.Deps) > 0 {
			dep = s.Deps[0]
		}
		fmt.Fprintf(&c.defs, "case %d: return %d;\n", i, dep)
	}
	c.defs.WriteString("default: return -2; }}\n__device__ unsigned dispatch(unsigned task, unsigned x) { switch(task) {\n")
	for i := range c.p.steps {
		fmt.Fprintf(&c.defs, "case %d: return op_%d(x);\n", i, i)
	}
	c.defs.WriteString("default: return 0; }}\n")
	return nil
}

// source returns the whole translation unit.
func (c *cudaSource) source() string {
	return programPrologue + c.defs.String() +
		workerPrologue(c.p.needBarrier, c.p.needPhase, c.p.grid.MaxElements) + c.body.String() +
		workerEpilogue + ordinaryKernel
}

// ptxScope maps a synchronization domain to the PTX scope qualifier an
// acquiring load of a completion word in that domain carries.
func ptxScope(s cudataskgraph.Scope) (string, error) {
	switch s {
	case cudataskgraph.ScopeCTA:
		return "cta", nil
	case cudataskgraph.ScopeDevice:
		return "gpu", nil
	case cudataskgraph.ScopeSystem:
		return "sys", nil
	}
	return "", fmt.Errorf("no acquiring load is defined for scope %s", scopeName(s))
}
