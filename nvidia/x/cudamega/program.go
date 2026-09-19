package cudamega

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// Operation is a finite elementwise task. Add and Multiply compute input plus
// or times Scalar using uint32 arithmetic.
type Operation struct {
	Kind   Opcode
	Scalar uint32
}

type Opcode uint8

const (
	Add Opcode = iota + 1
	Multiply
)

// Abort codes the generated worker writes to the volatile abort buffer. Zero
// means no abort; the host distinguishes the causes.
const (
	AbortNone        = 0
	AbortHostRequest = 1 // the host wrote a nonzero code before or during the launch
	AbortBadLaunch   = 2 // the launch geometry is not the one the source was generated for
	AbortBadExtent   = 3 // n or epoch is zero
	// AbortSpinExhausted means a CTA polled a dependency's completion word
	// more times than the compiled-in bound allows and gave up.
	//
	// The bound is in the kernel, not in the harness, because an external
	// timeout cannot stop a CTA that will not yield: killing the host process
	// leaves the grid resident. This code is distinguishable from the others
	// so that a run which hit the bound is never read as a run that
	// completed.
	AbortSpinExhausted = 4
)

// DefaultSpinLimit is the poll-iteration bound a dependency wait uses when
// [Grid.SpinLimit] is zero.
//
// It is a safety bound, not a deadline: it is far above any legitimate wait
// for the fixtures in this package and is here so that a grid which would
// otherwise spin forever exits instead. Nothing calibrates it against a
// measured poll rate, so it is not a timeout in seconds and must not be
// reported as one.
const DefaultSpinLimit = 1 << 24

// ProgramSource emits the persistent worker and the ordinary-launch reference
// for the same tasks.
//
// The task loop is unrolled: one block of code per task, in rank order, each
// carrying the synchronization its own scope declares. That is what makes the
// six contracts visible in the artifact rather than only in this comment.
//
// The caller supplies grid, which comes from [CTAGrid] and therefore from the
// compiled worker's measured residency. The emitted worker refuses at runtime
// to run under any other geometry: a persistent grid whose CTA count differs
// from the one the code was generated for has a different static task-to-CTA
// mapping, and the abort buffer is how it says so.
//
// Buffers: scratch has len(tasks)*n uint32 elements; completed has len(tasks)
// words, zeroed by the host before each epoch, and counts arriving CTAs;
// abort_code is one volatile word. Epoch must be nonzero. The caller retains
// every buffer until stream completion.
//
// Unsupported, reported rather than approximated:
//
//   - [cudataskgraph.ScopeCluster]. A cluster arrival needs distributed shared
//     memory and the participation contract lane 09 owns, and an mbarrier in
//     the local CTA's shared window is not a cluster barrier.
//   - Either end of a dependency edge at CTA scope in a multi-CTA grid. An
//     mbarrier in one CTA's shared memory orders that CTA only, so it can
//     neither publish to another CTA nor observe what another CTA published.
//     The producer's end is refused as well as the consumer's: a CTA-scope
//     release publishes a completion word that a CTA in another block will
//     read, and that is the same defect seen from the other side.
func ProgramSource(g cudataskgraph.Graph, operations map[string]Operation, grid Grid) (string, error) {
	tasks := g.Tasks()
	if len(tasks) == 0 || len(tasks) > 1024 || len(operations) != len(tasks) {
		return "", fmt.Errorf("finite program requires 1..1024 operations matching its tasks")
	}
	if err := grid.Validate(); err != nil {
		return "", err
	}
	dependedOn := make(map[string]bool, len(tasks))
	for _, t := range tasks {
		for _, d := range t.Deps {
			dependedOn[d] = true
		}
	}
	indices := make(map[string]int, len(tasks))
	needBarrier, needPhase, needSpin := false, false, false
	for _, t := range tasks {
		if len(t.Deps) > 0 {
			needSpin = true
		}
		if t.Scope == cudataskgraph.ScopeCTA {
			needBarrier = true
			if len(t.Deps) > 0 {
				needPhase = true
			}
		}
	}
	var defs, body strings.Builder
	fmt.Fprintf(&defs, "constexpr unsigned task_count = %d;\n", len(tasks))
	fmt.Fprintf(&defs, "constexpr unsigned grid_ctas = %d;\n", grid.CTAs)
	// Only a dependency wait polls, so only a graph with a dependency edge
	// declares the bound. Declaring it unconditionally leaves an unreferenced
	// constant in every dependency-free worker, which nvcc reports.
	if needSpin {
		fmt.Fprintf(&defs, "constexpr unsigned long long spin_limit = %dull;\n", grid.spinLimit())
	}
	defs.WriteString("static_assert(task_count >= 1, \"the generated worker has at least one task\");\n")

	for i, t := range tasks {
		op, ok := operations[t.ID]
		if !ok || op.Kind != Add && op.Kind != Multiply {
			return "", fmt.Errorf("task %q has no supported operation", t.ID)
		}
		if t.Scope == cudataskgraph.ScopeCluster {
			return "", fmt.Errorf("task %q declares cluster scope, which needs lane 09's participation contract", t.ID)
		}
		if t.Scope == cudataskgraph.ScopeCTA && grid.CTAs > 1 && (len(t.Deps) > 0 || dependedOn[t.ID]) {
			side := "depends on another task"
			if len(t.Deps) == 0 {
				side = "is depended on by another task"
			}
			return "", fmt.Errorf("task %q %s at CTA scope in a %d-CTA grid; an mbarrier orders one CTA", t.ID, side, grid.CTAs)
		}
		deps := make([]int, 0, len(t.Deps))
		for _, d := range t.Deps {
			n, ok := indices[d]
			if !ok {
				return "", fmt.Errorf("task %q precedes dependency %q", t.ID, d)
			}
			deps = append(deps, n)
		}
		indices[t.ID] = i

		operator := "+"
		if op.Kind == Multiply {
			operator = "*"
		}
		fmt.Fprintf(&defs, "__device__ unsigned op_%d(unsigned x) { return x %s %du; }\n", i, operator, op.Scalar)
		src := "input"
		if len(deps) > 0 {
			src = fmt.Sprintf("scratch + (size_t)%d*n", deps[0])
		}
		if err := emitTask(&body, i, t, deps, src); err != nil {
			return "", err
		}
	}
	defs.WriteString("__device__ int dependency(unsigned task) { switch(task) {\n")
	for i, t := range tasks {
		dep := -1
		if len(t.Deps) > 0 {
			dep = indices[t.Deps[0]]
		}
		fmt.Fprintf(&defs, "case %d: return %d;\n", i, dep)
	}
	defs.WriteString("default: return -2; }}\n__device__ unsigned dispatch(unsigned task, unsigned x) { switch(task) {\n")
	for i := range tasks {
		fmt.Fprintf(&defs, "case %d: return op_%d(x);\n", i, i)
	}
	defs.WriteString("default: return 0; }}\n")

	return programPrologue + defs.String() + workerPrologue(needBarrier, needPhase) + body.String() + workerEpilogue + ordinaryKernel, nil
}

// emitTask writes one task's block: the acquire side of every dependency, the
// computation, the release side, and the publishing store. The order of those
// four is the whole of contract 4, and it is the order the artifact is
// asserted against.
func emitTask(b *strings.Builder, i int, t cudataskgraph.Task, deps []int, src string) error {
	fmt.Fprintf(b, "\n  // task %d %q scope=%s rank=%d deps=%v\n", i, t.ID, scopeName(t.Scope), t.Rank, deps)
	fmt.Fprintf(b, "  if (!*abort_code) {\n")

	// Acquire: wait for every dependency, then fence so the reads below cannot
	// be hoisted above the wait.
	for _, d := range deps {
		fmt.Fprintf(b, "    if (threadIdx.x == 0) {\n")
		fmt.Fprintf(b, "      unsigned long long spins = 0ull;\n")
		fmt.Fprintf(b, "      while (((volatile unsigned*)completed)[%d] < grid_ctas) {\n", d)
		fmt.Fprintf(b, "        if (*abort_code) break;\n")
		fmt.Fprintf(b, "        if (++spins > spin_limit) { atomicCAS((unsigned *)abort_code, 0u, %du); break; }\n", AbortSpinExhausted)
		fmt.Fprintf(b, "      }\n")
		fmt.Fprintf(b, "    }\n")
		fmt.Fprintf(b, "    __syncthreads();\n")
	}
	if len(deps) > 0 {
		acq, err := acquireFence(t.Scope)
		if err != nil {
			return err
		}
		fmt.Fprintf(b, "    %s // acquire for task %d\n", acq, i)
	}

	fmt.Fprintf(b, "    if (!*abort_code) {\n")
	fmt.Fprintf(b, "      const unsigned *src = %s;\n", src)
	fmt.Fprintf(b, "      for (size_t e = (size_t)blockIdx.x*blockDim.x + threadIdx.x; e < n; e += (size_t)grid_ctas*blockDim.x)\n")
	fmt.Fprintf(b, "        scratch[(size_t)%d*n + e] = op_%d(src[e]);\n", i, i)
	fmt.Fprintf(b, "    }\n")

	// Release: every thread of the CTA is past its stores, then the fence the
	// scope declares, then the publishing store.
	rel, err := releaseFence(t.Scope)
	if err != nil {
		return err
	}
	fmt.Fprintf(b, "    %s // release for task %d\n", rel, i)
	fmt.Fprintf(b, "    if (threadIdx.x == 0) atomicAdd(completed + %d, 1u); // publish task %d\n", i, i)
	fmt.Fprintf(b, "  }\n")
	return nil
}

// acquireFence returns the acquire side for a scope. ScopeCTA's is the
// mbarrier wait, which is also what makes the CTA's own prior stores visible
// to the threads that resume after it.
func acquireFence(s cudataskgraph.Scope) (string, error) {
	switch s {
	case cudataskgraph.ScopeCTA:
		return ctaBarrierWait, nil
	case cudataskgraph.ScopeDevice:
		return `asm volatile("fence.sc.gpu;" ::: "memory");`, nil
	case cudataskgraph.ScopeSystem:
		return `asm volatile("fence.sc.sys;" ::: "memory");`, nil
	}
	return "", fmt.Errorf("no acquire is defined for scope %s", scopeName(s))
}

// releaseFence returns the release side for a scope.
func releaseFence(s cudataskgraph.Scope) (string, error) {
	switch s {
	case cudataskgraph.ScopeCTA:
		return ctaBarrierArrive, nil
	case cudataskgraph.ScopeDevice:
		return `asm volatile("fence.sc.gpu;" ::: "memory");`, nil
	case cudataskgraph.ScopeSystem:
		return `asm volatile("fence.sc.sys;" ::: "memory");`, nil
	}
	return "", fmt.Errorf("no release is defined for scope %s", scopeName(s))
}

func scopeName(s cudataskgraph.Scope) string {
	switch s {
	case cudataskgraph.ScopeCTA:
		return "cta"
	case cudataskgraph.ScopeCluster:
		return "cluster"
	case cudataskgraph.ScopeDevice:
		return "device"
	case cudataskgraph.ScopeSystem:
		return "system"
	}
	return "unknown"
}

// ctaBarrierArrive is the CTA-scope release: every thread arrives at the
// shared mbarrier with release semantics, which orders its stores before the
// arrival token other threads observe.
const ctaBarrierArrive = `asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(cta_barrier_addr) : "memory");`

// ctaBarrierWait is the CTA-scope acquire. The phase parity alternates across
// arrivals so the barrier is never reset, which is contract 5.
const ctaBarrierWait = `{ asm volatile("{ .reg .pred p; L_wait_%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1; @!p bra L_wait_%=; }" :: "r"(cta_barrier_addr), "r"(cta_phase) : "memory"); cta_phase ^= 1u; }`

const programPrologue = "#include <cuda_runtime.h>\n"

// workerPrologue is the generated worker's entry: the geometry and extent
// refusals of contract 6, then the shared mbarrier. The barrier and its phase
// are declared only when some task needs them, so that a graph with no CTA
// synchronization does not carry a barrier nothing arrives at.
func workerPrologue(barrier, phase bool) string {
	var b strings.Builder
	b.WriteString(`
extern "C" __global__ void dag_worker(const unsigned *input, unsigned *scratch,
    unsigned *completed, volatile unsigned *abort_code, unsigned n, unsigned epoch) {
  if (gridDim.x != grid_ctas || gridDim.y != 1 || gridDim.z != 1 || blockDim.y != 1 || blockDim.z != 1) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, `)
	fmt.Fprintf(&b, "%du", AbortBadLaunch)
	b.WriteString(`);
    return;
  }
  if (!n || !epoch) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, `)
	fmt.Fprintf(&b, "%du", AbortBadExtent)
	b.WriteString(`);
    return;
  }
`)
	if barrier {
		b.WriteString(`  __shared__ __align__(8) unsigned long long cta_barrier;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
        "r"((unsigned)__cvta_generic_to_shared(&cta_barrier)), "r"(blockDim.x) : "memory");
  }
  __syncthreads();
  const unsigned cta_barrier_addr = (unsigned)__cvta_generic_to_shared(&cta_barrier);
`)
	}
	if phase {
		b.WriteString("  unsigned cta_phase = 0u;\n")
	}
	return b.String()
}

const workerEpilogue = `
}
`

const ordinaryKernel = `
// ordinary_task is the reference implementation: the same tasks, one ordinary
// launch each, in rank order, with the host's stream ordering standing in for
// every fence above. It exists so the persistent result has something to be
// equal to.
extern "C" __global__ void ordinary_task(const unsigned *input, unsigned *scratch,
    unsigned n, unsigned task) {
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  int dep = dependency(task);
  if (i < n) {
    const unsigned *src = dep < 0 ? input : scratch + (size_t)dep*n;
    scratch[(size_t)task*n+i] = dispatch(task, src[i]);
  }
}
`

// ProgramScratchBytes checks the finite worker's scratch extent.
func ProgramScratchBytes(tasks, elements uint64) (uint64, error) {
	if tasks == 0 || tasks > 1024 || elements == 0 || elements > math.MaxUint32 || elements > math.MaxUint64/4/tasks {
		return 0, fmt.Errorf("invalid finite program extent")
	}
	return tasks * elements * 4, nil
}
