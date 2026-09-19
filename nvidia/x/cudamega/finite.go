package cudamega

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// Admission is the measured worker residency contract supplied by the
// compiled artifact and target record.
//
// It is an input, not a derivation: the numbers come from
// cudaOccupancyMaxActiveBlocksPerMultiprocessor on the compiled worker for the
// target device, multiplied by that device's multiprocessor count. Nothing in
// this package can compute them, and a grid built from a guess is the
// deadlock the contracts exist to prevent.
type Admission struct {
	ResidentCTAs  uint32
	WorkerThreads uint32
	SharedBytes   uint32
}

// Validate reports whether the admission is one a persistent grid can be
// derived from.
func (a Admission) Validate() error {
	if a.ResidentCTAs == 0 || a.WorkerThreads == 0 {
		return fmt.Errorf("worker admission is empty")
	}
	if a.WorkerThreads > 1024 {
		return fmt.Errorf("worker exceeds block thread limit")
	}
	if a.SharedBytes == 0 {
		return fmt.Errorf("worker shared memory admission is empty")
	}
	return nil
}

// A Scheduler is the finite task graph bound to one compiled worker's
// admission.
//
// It holds no schedule of its own. The schedule is [ProgressArgument], which
// the scheduler computes once at construction: there is one static
// task-to-CTA mapping per (graph, admission) pair, and a scheduler that could
// return a different one on a later call would be a second mapping the
// generated source does not implement.
type Scheduler struct {
	graph     cudataskgraph.Graph
	admission Admission
	progress  ProgressArgument
}

// NewScheduler validates the graph against the admission and returns the
// scheduler, or the reason the graph does not fit the compiled worker.
func NewScheduler(g cudataskgraph.Graph, a Admission) (*Scheduler, error) {
	p, err := Progress(g, a)
	if err != nil {
		return nil, err
	}
	return &Scheduler{graph: g.Clone(), admission: a, progress: p}, nil
}

// Assign returns the wave order the generated worker runs, which is the
// static task-to-CTA mapping of contract 1: every admitted CTA takes a share
// of every task's elements, and the tasks run in the order returned here.
func (s *Scheduler) Assign() ([]Wave, error) {
	if s == nil {
		return nil, fmt.Errorf("nil scheduler")
	}
	return append([]Wave(nil), s.progress.Waves...), nil
}

// Progress returns the argument that the schedule terminates.
func (s *Scheduler) Progress() ProgressArgument { return s.progress }

// Admission returns the compiled worker's residency the schedule was built
// against.
func (s *Scheduler) Admission() Admission { return s.admission }

// Graph returns an independent snapshot of the scheduled graph.
func (s *Scheduler) Graph() cudataskgraph.Graph { return s.graph.Clone() }

// FiniteWorkerSource emits the legacy ordinary fallback for opcode 1
// (elementwise increment). It is not the persistent taskgraph qualification
// fixture; use ProgramSource for that finite subset.
func FiniteWorkerSource() string {
	return `extern "C" __global__ void finite_worker(const unsigned* in, unsigned* out, volatile unsigned* abort, unsigned n, unsigned epoch) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n && *abort==0) { out[i]=in[i]+1; __threadfence(); }
  __syncthreads();
}`
}

// OrdinaryReference is the CPU oracle for the legacy ordinary fallback.
func OrdinaryReference(in []uint32) []uint32 {
	out := make([]uint32, len(in))
	for i, v := range in {
		out[i] = v + 1
	}
	return out
}
