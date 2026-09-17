package cudamega

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// Operation is a finite elementwise task. Add and Multiply compute input plus
// or times Scalar using uint32 arithmetic. A task has at most one dependency.
type Operation struct {
	Kind   Opcode
	Scalar uint32
}

type Opcode uint8

const (
	Add Opcode = iota + 1
	Multiply
)

// ProgramSource emits a single-CTA finite worker and an ordinary-launch
// kernel for the same operations. Scratch has len(g.Tasks())*n uint32 elements;
// completion has len(g.Tasks()) words initialized to zero. Epoch must be
// nonzero. The caller retains scratch, completion and abort until stream
// completion. Multiple CTAs and cross-CTA dependency scopes are unsupported.
func ProgramSource(g cudataskgraph.Graph, operations map[string]Operation) (string, error) {
	tasks := g.Tasks()
	if len(tasks) == 0 || len(tasks) > 1024 || len(operations) != len(tasks) {
		return "", fmt.Errorf("finite program requires 1..1024 operations matching its tasks")
	}
	indices := make(map[string]int, len(tasks))
	var defs strings.Builder
	fmt.Fprintf(&defs, "constexpr unsigned task_count = %d;\n", len(tasks))
	for i, t := range tasks {
		if t.Scope != cudataskgraph.ScopeCTA || len(t.Deps) > 1 {
			return "", fmt.Errorf("task %q requires unsupported scope or fan-in", t.ID)
		}
		op, ok := operations[t.ID]
		if !ok || op.Kind != Add && op.Kind != Multiply {
			return "", fmt.Errorf("task %q has no supported operation", t.ID)
		}
		dep := -1
		if len(t.Deps) == 1 {
			var ok bool
			dep, ok = indices[t.Deps[0]]
			if !ok {
				return "", fmt.Errorf("task %q precedes dependency", t.ID)
			}
		}
		indices[t.ID] = i
		operator := "+"
		if op.Kind == Multiply {
			operator = "*"
		}
		fmt.Fprintf(&defs, "__device__ unsigned op_%d(unsigned x) { return x %s %du; }\n", i, operator, op.Scalar)
		fmt.Fprintf(&defs, "// task %d dependency %d\n", i, dep)
	}
	defs.WriteString("__device__ int dependency(unsigned task) { switch(task) {\n")
	for i, t := range tasks {
		dep := -1
		if len(t.Deps) == 1 {
			dep = indices[t.Deps[0]]
		}
		fmt.Fprintf(&defs, "case %d: return %d;\n", i, dep)
	}
	defs.WriteString("default: return -2; }}\n__device__ unsigned dispatch(unsigned task, unsigned x) { switch(task) {\n")
	for i := range tasks {
		fmt.Fprintf(&defs, "case %d: return op_%d(x);\n", i, i)
	}
	defs.WriteString("default: return 0; }}\n")
	return "#include <cuda_runtime.h>\n" + defs.String() + finiteProgramBody, nil
}

// ProgramScratchBytes checks the finite worker's scratch extent.
func ProgramScratchBytes(tasks, elements uint64) (uint64, error) {
	if tasks == 0 || tasks > 1024 || elements == 0 || elements > math.MaxUint32 || elements > math.MaxUint64/4/tasks {
		return 0, fmt.Errorf("invalid finite program extent")
	}
	return tasks * elements * 4, nil
}

const finiteProgramBody = `
extern "C" __global__ void dag_worker(const unsigned *input, unsigned *scratch,
    unsigned *completed, unsigned *abort_code, unsigned n, unsigned epoch) {
  if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 || blockDim.y != 1 || blockDim.z != 1) return;
  __shared__ unsigned stop;
  for (unsigned task = 0; task < task_count; ++task) {
    int dep = dependency(task);
    if (threadIdx.x == 0) {
      stop = atomicAdd(abort_code, 0u);
      if (!epoch || !n || (dep >= 0 && atomicAdd(completed + dep, 0u) != epoch)) {
        atomicCAS(abort_code, 0u, 2u);
        stop = 2;
      }
    }
    __syncthreads();
    if (stop) return;
    const unsigned *src = dep < 0 ? input : scratch + (size_t)dep*n;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
      scratch[(size_t)task*n+i] = dispatch(task, src[i]);
    __threadfence_block();
    __syncthreads();
    if (threadIdx.x == 0) atomicExch(completed + task, epoch);
    __syncthreads();
  }
}
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
