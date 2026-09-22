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
	fmt.Fprintf(&c.defs, "constexpr unsigned task_count = %d;\n", len(p.steps))
	fmt.Fprintf(&c.defs, "constexpr unsigned grid_ctas = %d;\n", p.grid.CTAs)
	// Only a dependency wait polls, so only a graph with a dependency edge
	// declares the bound. Declaring it unconditionally leaves an unreferenced
	// constant in every dependency-free worker, which nvcc reports.
	if p.needSpin {
		fmt.Fprintf(&c.defs, "constexpr unsigned long long spin_limit = %dull;\n", p.grid.spinLimit())
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

// BeginTask opens the task's guarded region. The guard is emitted, never
// taken: whether the region runs is a question the device answers.
func (c *cudaSource) BeginTask(i int) (bool, error) {
	s := c.p.steps[i]
	fmt.Fprintf(&c.body, "\n  // task %d %q scope=%s rank=%d deps=%v\n", i, s.Task.ID, scopeName(s.Task.Scope), s.Task.Rank, s.Deps)
	c.body.WriteString("  if (!*abort_code) {\n")
	return true, nil
}

func (c *cudaSource) EndTask(int) error {
	c.body.WriteString("  }\n")
	return nil
}

// Await renders the bounded poll of a dependency's completion word.
//
// The load is an acquiring load, not a volatile one. They are not the same
// thing and the difference is the whole reason this primitive exists:
// volatile forbids the compiler from caching the load and orders nothing at
// all against the producer's stores, so the reads that follow may still be
// satisfied from stale lines. ld.acquire orders them. The fence the driver
// emits after the barrier below is still required, because only thread zero
// executes this load and the acquire it performs is its own.
//
// The bound is in the emitted loop, not in a harness. An external timeout
// cannot stop a CTA that will not yield: killing the host process leaves the
// grid resident.
func (c *cudaSource) Await(i, dep int) error {
	scope, err := ptxScope(c.p.steps[i].Task.Scope)
	if err != nil {
		return err
	}
	addr := "%1"
	if dep > 0 {
		addr = fmt.Sprintf("%%1+%d", 4*dep)
	}
	b := &c.body
	b.WriteString("    if (threadIdx.x == 0) {\n")
	b.WriteString("      unsigned long long spins = 0ull;\n")
	b.WriteString("      unsigned done = 0u;\n")
	fmt.Fprintf(b, "      while (true) { // poll the completion word of task %d\n", dep)
	fmt.Fprintf(b, "        asm volatile(\"ld.acquire.%s.u32 %%0, [%s];\" : \"=r\"(done) : \"l\"(completed) : \"memory\");\n", scope, addr)
	b.WriteString("        if (done >= grid_ctas) break;\n")
	b.WriteString("        if (*abort_code) break;\n")
	fmt.Fprintf(b, "        if (++spins > spin_limit) { atomicCAS((unsigned *)abort_code, 0u, %du); break; }\n", AbortSpinExhausted)
	b.WriteString("      }\n")
	b.WriteString("    }\n")
	return nil
}

func (c *cudaSource) Barrier() error {
	c.body.WriteString("    __syncthreads();\n")
	return nil
}

func (c *cudaSource) Fence(i int, s cudataskgraph.Scope, k Fence) error {
	var instruction string
	var err error
	switch k {
	case Acquire:
		instruction, err = acquireFence(s)
	case Release:
		instruction, err = releaseFence(s)
	default:
		err = fmt.Errorf("unknown fence side %s", k)
	}
	if err != nil {
		return err
	}
	fmt.Fprintf(&c.body, "    %s // %s for task %d\n", instruction, k, i)
	return nil
}

func (c *cudaSource) Compute(i, src int, op Operation) error {
	expr := "input"
	if src >= 0 {
		expr = fmt.Sprintf("scratch + (size_t)%d*n", src)
	}
	b := &c.body
	b.WriteString("    if (!*abort_code) {\n")
	fmt.Fprintf(b, "      const unsigned *src = %s;\n", expr)
	b.WriteString("      for (size_t e = (size_t)blockIdx.x*blockDim.x + threadIdx.x; e < n; e += (size_t)grid_ctas*blockDim.x)\n")
	fmt.Fprintf(b, "        scratch[(size_t)%d*n + e] = op_%d(src[e]);\n", i, i)
	b.WriteString("    }\n")
	return nil
}

func (c *cudaSource) Publish(i int) error {
	fmt.Fprintf(&c.body, "    if (threadIdx.x == 0) atomicAdd(completed + %d, 1u); // publish task %d\n", i, i)
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
		workerPrologue(c.p.needBarrier, c.p.needPhase) + c.body.String() +
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
