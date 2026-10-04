package cudamega

import (
	_ "embed"
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
	moe    *MoEWeightConfig
}

type Opcode uint8

const (
	Add Opcode = iota + 1
	Multiply
	// MoEGateUp is the typed payload emitted only by MoEWeightSource.
	// The scalar ProgramSource and public NewProgram refuse it.
	MoEGateUp
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
// for the same tasks, as CUDA C++ for target.
//
// It is [NewProgram] followed by rendering on the CUDA backend, and exists
// because emitting the source is what most callers want from the pair. The
// task loop is unrolled: one block of code per task, in rank order, each
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
// start_order has [StartOrderWords] uint64 words and records when each CTA
// took each task; abort_code is one volatile word. The caller retains every
// buffer until stream completion.
//
// Epoch must be nonzero, and the parameter is 64 bits wide because the host's
// epoch counter is: a 32-bit parameter takes epoch 2³² to zero, which the
// worker refuses as the unpublished value, and takes 2³²+k to k, which it
// accepts as a different epoch. Neither is caught by anything narrower.
//
// See [NewProgram] for what the pair refuses.
func ProgramSource(g cudataskgraph.Graph, operations map[string]Operation, grid Grid, target Target) (string, error) {
	p, err := NewProgram(g, operations, grid, target)
	if err != nil {
		return "", err
	}
	return p.Source()
}

// Source renders the program as CUDA C++ for its target.
//
// The text is what the CUDA backend writes when the driver walks the program,
// so it cannot drift from what that backend does: there is no second emitter.
func (p *Program) Source() (string, error) {
	c := &cudaSource{p: p}
	r, err := NewRunner(p, c)
	if err != nil {
		return "", err
	}
	if err := r.Run(); err != nil {
		return "", err
	}
	return c.source(), nil
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
// arrival token other threads observe. It is the optional sixth primitive:
// a backend without an mbarrier serves CTA scope with its workgroup barrier.
const ctaBarrierArrive = `asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(cta_barrier_addr) : "memory");`

// ctaBarrierWait is the CTA-scope acquire. The phase parity alternates across
// arrivals so the barrier is never reset, which is contract 5.
const ctaBarrierWait = `{ asm volatile("{ .reg .pred p; L_wait_%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1; @!p bra L_wait_%=; }" :: "r"(cta_barrier_addr), "r"(cta_phase) : "memory"); cta_phase ^= 1u; }`

// The fixed CUDA of the generated program, held as CUDA rather than as Go
// string literals so that it can be read, diffed and syntax-highlighted as
// what it is. Each file is a fragment of one translation unit, not a
// standalone one: the entry refers to constants the emitter writes above it,
// and the reference kernel calls the dispatch table it writes. The only
// substitutions are the abort codes, which are %-verbs so that the Go
// constants stay the single authority for their values.
//
//go:embed cudasrc/prologue.cu
var programPrologue string

//go:embed cudasrc/worker_entry.cu
var workerEntry string

//go:embed cudasrc/cta_barrier.cu
var ctaBarrierInit string

//go:embed cudasrc/ordinary_task.cu
var ordinaryKernel string

// workerPrologue is the generated worker's entry: the geometry and extent
// refusals of contract 6, then the shared mbarrier. The barrier and its phase
// are declared only when some task needs them, so that a graph with no CTA
// synchronization does not carry a barrier nothing arrives at.
func workerPrologue(barrier, phase bool) string {
	var b strings.Builder
	fmt.Fprintf(&b, workerEntry, AbortBadLaunch, AbortBadExtent)
	if barrier {
		b.WriteString(ctaBarrierInit)
	}
	if phase {
		b.WriteString("  unsigned cta_phase = 0u;\n")
	}
	return b.String()
}

const workerEpilogue = `
}
`

// StartOrderWords is the length of the start-order buffer, in uint64 words,
// for a program of the given shape.
//
// The buffer is the overlap instrument. Word zero is the ticket counter; word
// 1 + cta*tasks + task is the ticket that CTA held when it took that task,
// one-based so that zero means the CTA never took it. Tickets come from a
// single device-scope atomic, so they order task starts across CTAs and not
// only within one, which is the whole question the instrument exists to
// answer: whether the workers are in the same task at the same time.
func StartOrderWords(tasks, ctas uint64) (uint64, error) {
	if tasks == 0 || tasks > 1024 || ctas == 0 || ctas > math.MaxUint32 {
		return 0, fmt.Errorf("invalid start-order extent")
	}
	return 1 + tasks*ctas, nil
}

// ProgramScratchBytes checks the finite worker's scratch extent.
func ProgramScratchBytes(tasks, elements uint64) (uint64, error) {
	if tasks == 0 || tasks > 1024 || elements == 0 || elements > math.MaxUint32 || elements > math.MaxUint64/4/tasks {
		return 0, fmt.Errorf("invalid finite program extent")
	}
	return tasks * elements * 4, nil
}
