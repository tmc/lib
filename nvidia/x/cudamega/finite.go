package cudamega

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// Admission is the measured worker residency contract supplied by the
// compiled artifact and target record.
type Admission struct {
	ResidentCTAs  uint32
	WorkerThreads uint32
	SharedBytes   uint32
}

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

type Scheduler struct {
	graph     cudataskgraph.Graph
	admission Admission
}

func NewScheduler(g cudataskgraph.Graph, a Admission) (*Scheduler, error) {
	if e := a.Validate(); e != nil {
		return nil, e
	}
	if len(g.Tasks()) > int(a.ResidentCTAs)*1024 {
		return nil, fmt.Errorf("finite graph exceeds admitted worker capacity")
	}
	for _, t := range g.Tasks() {
		if t.Bytes > uint64(a.SharedBytes) {
			return nil, fmt.Errorf("task %q exceeds admitted shared memory", t.ID)
		}
	}
	return &Scheduler{graph: g.Clone(), admission: a}, nil
}
func (s *Scheduler) Assign() ([]cudataskgraph.Task, error) {
	if s == nil {
		return nil, fmt.Errorf("nil scheduler")
	}
	ts := s.graph.Tasks()
	for i, t := range ts {
		if uint32(i) >= s.admission.ResidentCTAs && t.Rank == 0 {
			return nil, fmt.Errorf("producer cannot be assigned to resident CTA")
		}
	}
	return ts, nil
}
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
