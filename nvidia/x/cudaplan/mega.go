package cudaplan

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// ErrNotLowerable reports a step a region cannot be lowered through, because
// what [cudataskgraph.New] requires is not determined by what [StepPlan]
// carries. It is a refusal, not a failure: the caller may lower a smaller
// region, or run the step on another backend.
var ErrNotLowerable = errors.New("cudaplan: step cannot be lowered to a task graph")

// A Region names a contiguous run of steps: the half-open index range
// [Start, End) into the slice [ExecutablePlan.Steps] returns.
type Region struct {
	Start int
	End   int
}

// A Binding is the side table entry for one lowered step.
//
// [cudataskgraph.Task] carries scheduling and nothing else — an identity, its
// dependencies, a rank, a shared-memory claim and a scope. Everything a step
// needs in order to be *executed* has no counterpart there and would be lost
// by the lowering, so it is kept here, keyed by task ID. The split is
// deliberate: the task graph is validated and hashed by
// [cudataskgraph.Graph.Identity], and folding artifact pointers or workspace
// offsets into it would make two topologically identical schedules differ.
type Binding struct {
	// StepIndex is the step's index in the slice that was lowered, not its
	// index within the region.
	StepIndex int

	NodeID          int
	NodeName        string
	Backend         TargetBackend
	Artifact        *cudakernel.Artifact
	InputNames      []string
	OutputName      string
	IsOutput        bool
	WorkspaceOffset int
	WorkspaceBytes  int

	// ExternalDeps holds the indices of steps this step reads that lie before
	// the region. They are not task-graph edges: the producer is not a task,
	// so nothing in the graph can order against it. Ordering across the
	// region boundary is the caller's, and this field is what tells the
	// caller which values it owes.
	ExternalDeps []int
}

// A Lowering is a validated task graph plus the side table that carries what
// the task graph cannot. It is the whole result of lowering a region: a graph
// without its bindings cannot be executed, and bindings without their graph
// have no order.
type Lowering struct {
	graph    cudataskgraph.Graph
	ids      []string
	bindings map[string]Binding
}

// Graph returns the validated task graph.
func (l Lowering) Graph() cudataskgraph.Graph { return l.graph }

// TaskIDs returns the lowered task IDs in region order.
func (l Lowering) TaskIDs() []string { return append([]string(nil), l.ids...) }

// Len returns the number of steps that were lowered.
func (l Lowering) Len() int { return len(l.ids) }

// Binding returns the side table entry for a task ID.
func (l Lowering) Binding(id string) (Binding, bool) {
	b, ok := l.bindings[id]
	if !ok {
		return Binding{}, false
	}
	b.InputNames = append([]string(nil), b.InputNames...)
	b.Artifact = captureArtifact(b.Artifact)
	b.ExternalDeps = append([]int(nil), b.ExternalDeps...)
	return b, true
}

// LowerRegion lowers a contiguous run of steps into a task graph and its side
// table.
//
// The lowering is not a field copy. [cudataskgraph.New] demands two things a
// [StepPlan] does not carry, and both are derived here:
//
// Rank. New rejects any edge whose dependency does not have a strictly smaller
// rank. The rank derived here is the longest path from a step with no
// in-region dependency — topological depth, not step index. Step index would
// satisfy New and be wrong: two independent steps share a depth, and giving
// them different ranks asserts an order the plan does not have.
//
// Scope. New rejects the zero value, so a scope must be derived, and the
// derivation is a policy:
//
//   - A step whose output another step in the region reads is at least
//     [cudataskgraph.ScopeDevice]. A CTA-scope edge only orders one CTA, and
//     the emitter refuses either end of such an edge in a multi-CTA grid.
//   - A step that is a plan output is [cudataskgraph.ScopeSystem]: the host
//     reads the value after the launch.
//   - A step that is neither is refused. Nothing in the plan says who reads
//     its output, so nothing in the plan determines how far the write must be
//     visible, and the permissive answer is a guess that always succeeds.
//
// [cudataskgraph.ScopeCTA] is therefore never derived. A step's output lives
// in the plan workspace, which is global memory, and no field of a StepPlan
// distinguishes a value that stays inside one CTA. Deriving CTA scope would
// require information the plan does not have.
//
// Task.Bytes is left zero. It is a shared-memory claim the emitter budgets
// against, and StepPlan.WorkspaceBytes is the size of an output buffer in
// global memory. They are different quantities, so WorkspaceBytes stays in
// the side table and the claim is the caller's to supply.
func LowerRegion(steps []StepPlan, r Region) (Lowering, error) {
	if r.Start < 0 || r.End > len(steps) || r.Start >= r.End {
		return Lowering{}, fmt.Errorf("cudaplan: lower region [%d,%d) of %d steps: empty or out of range", r.Start, r.End, len(steps))
	}
	n := r.End - r.Start

	ids := make([]string, n)
	index := make(map[string]int, n)
	for i := range n {
		s := steps[r.Start+i]
		if s.NodeName == "" {
			return Lowering{}, fmt.Errorf("cudaplan: lower step %d: %w: no node name to key a task by", r.Start+i, ErrNotLowerable)
		}
		if j, dup := index[s.NodeName]; dup {
			return Lowering{}, fmt.Errorf("cudaplan: lower step %d: %w: node name %q is already used by step %d", r.Start+i, ErrNotLowerable, s.NodeName, r.Start+j)
		}
		ids[i] = s.NodeName
		index[s.NodeName] = i
	}

	deps := make([][]int, n)     // region-relative dependency indices
	external := make([][]int, n) // step indices before the region
	consumed := make([]bool, n)
	for i := range n {
		s := steps[r.Start+i]
		for _, d := range s.Dependencies {
			switch {
			case d < 0 || d >= len(steps):
				return Lowering{}, fmt.Errorf("cudaplan: lower step %d: dependency %d is not a step", r.Start+i, d)
			case d >= r.End:
				return Lowering{}, fmt.Errorf("cudaplan: lower step %d: %w: depends on step %d, which is outside the region", r.Start+i, ErrNotLowerable, d)
			case d < r.Start:
				external[i] = append(external[i], d)
			case d-r.Start >= i:
				return Lowering{}, fmt.Errorf("cudaplan: lower step %d: dependency %d does not precede it", r.Start+i, d)
			default:
				deps[i] = append(deps[i], d-r.Start)
				consumed[d-r.Start] = true
			}
		}
	}

	// Rank is the longest path to a step with no in-region dependency. Every
	// dependency precedes its step, so one forward pass settles it.
	rank := make([]int, n)
	for i := range n {
		for _, j := range deps[i] {
			if rank[j]+1 > rank[i] {
				rank[i] = rank[j] + 1
			}
		}
	}

	tasks := make([]cudataskgraph.Task, n)
	bindings := make(map[string]Binding, n)
	for i := range n {
		s := steps[r.Start+i]
		scope, err := deriveScope(s, consumed[i])
		if err != nil {
			return Lowering{}, fmt.Errorf("cudaplan: lower step %d (%s): %w", r.Start+i, s.NodeName, err)
		}
		depIDs := make([]string, len(deps[i]))
		for k, j := range deps[i] {
			depIDs[k] = ids[j]
		}
		tasks[i] = cudataskgraph.Task{
			ID:    ids[i],
			Deps:  depIDs,
			Rank:  rank[i],
			Scope: scope,
		}
		bindings[ids[i]] = Binding{
			StepIndex:       r.Start + i,
			NodeID:          s.NodeID,
			NodeName:        s.NodeName,
			Backend:         s.Backend,
			Artifact:        captureArtifact(s.Artifact),
			InputNames:      append([]string(nil), s.InputNames...),
			OutputName:      s.OutputName,
			IsOutput:        s.IsOutput,
			WorkspaceOffset: s.WorkspaceOffset,
			WorkspaceBytes:  s.WorkspaceBytes,
			ExternalDeps:    external[i],
		}
	}

	g, err := cudataskgraph.New(tasks)
	if err != nil {
		return Lowering{}, fmt.Errorf("cudaplan: lower region [%d,%d): %w", r.Start, r.End, err)
	}
	return Lowering{graph: g, ids: ids, bindings: bindings}, nil
}

// deriveScope states how far a step's output must be visible, or refuses.
func deriveScope(s StepPlan, consumed bool) (cudataskgraph.Scope, error) {
	switch s.Backend {
	case BackendLaunch, BackendGraph, BackendMega:
	case BackendCuBLASLt:
		return 0, fmt.Errorf("%w: %s dispatches through a vendor library, which the worker cannot inline as a task", ErrNotLowerable, s.Backend)
	default:
		return 0, fmt.Errorf("%w: unknown backend %s", ErrNotLowerable, s.Backend)
	}
	if s.Artifact == nil {
		return 0, fmt.Errorf("%w: no kernel artifact, so the task has no body", ErrNotLowerable)
	}
	switch {
	case s.IsOutput:
		return cudataskgraph.ScopeSystem, nil
	case consumed:
		return cudataskgraph.ScopeDevice, nil
	default:
		return 0, fmt.Errorf("%w: no step in the region reads its output and it is not a plan output, so nothing determines its visibility scope", ErrNotLowerable)
	}
}
