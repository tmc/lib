package cudaplan_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// elementwisePTX is a one-input elementwise kernel. Only its schema and bytes
// matter here: no test in this file launches anything.
const elementwisePTX = `//
.version 8.0
.target sm_121a
.address_size 64

.visible .entry elementwise_kernel(
	.param .u64 param_out,
	.param .u64 param_in
)
{
	ret;
}
`

func buildElementwiseArtifact(t *testing.T) *cudakernel.Artifact {
	t.Helper()
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "param_out", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "param_in", Type: "ptr", IsPointer: true},
	)
	if err != nil {
		t.Fatalf("NewSchema: %v", err)
	}
	art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(elementwisePTX), "elementwise_kernel", schema, "sm_121a")
	if err != nil {
		t.Fatalf("NewArtifact: %v", err)
	}
	return art
}

// compileRegionFixture compiles a real elementwise region with no device.
//
// The region is a diamond, not a chain: relu and neg have no dependency on
// each other, so a lowering that used step index as rank would order them and
// a lowering that computes depth will not.
//
//	relu(x)  neg(w)      rank 0
//	     \    /
//	    add(.,.)         rank 1
//	        |
//	     silu(.)         rank 2, plan output
func compileRegionFixture(t *testing.T, withArtifacts bool) *cudaplan.ExecutablePlan {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{4, 8}, cudair.Float32)
	w := g.AddInput("w", []int{4, 8}, cudair.Float32)
	p := g.AddUnary("relu", x)
	q := g.AddUnary("neg", w)
	r := g.AddBinary("add", p, q)
	s := g.AddUnary("silu", r)
	if err := g.SetOutputs(s); err != nil {
		t.Fatalf("SetOutputs: %v", err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	opts := []cudaplan.PlanOption{}
	if withArtifacts {
		reg := cudaplan.NewKernelRegistry()
		art := buildElementwiseArtifact(t)
		for _, op := range []string{"relu", "neg", "add", "silu"} {
			if err := reg.Register(op, art); err != nil {
				t.Fatalf("Register %q: %v", op, err)
			}
		}
		opts = append(opts, cudaplan.WithKernelRegistry(reg))
	}
	plan, err := cudaplan.Compile(snap, opts...)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	t.Cleanup(func() { plan.Close() })
	return plan
}

// TestLowerRegionRoundTripsAPlanRegion is the denominator gate: a real
// compiled region is lowered, every step is accounted for, and the resulting
// graph is one cudamega's emitter accepts for a multi-CTA grid.
func TestLowerRegionRoundTripsAPlanRegion(t *testing.T) {
	plan := compileRegionFixture(t, true)
	steps := plan.Steps()
	if len(steps) == 0 {
		t.Fatal("fixture compiled no steps")
	}

	low, err := cudaplan.LowerRegion(steps, cudaplan.Region{Start: 0, End: len(steps)})
	if err != nil {
		t.Fatalf("LowerRegion: %v", err)
	}
	if low.Len() == 0 {
		t.Fatal("lowered no steps: an empty population passes every assertion below")
	}
	if low.Len() != len(steps) {
		t.Fatalf("lowered %d of %d steps", low.Len(), len(steps))
	}
	t.Logf("denominator: %d steps compiled, %d lowered, %d refused", len(steps), low.Len(), len(steps)-low.Len())

	// Every task ID resolves to a side table entry naming its step.
	for i, id := range low.TaskIDs() {
		b, ok := low.Binding(id)
		if !ok {
			t.Fatalf("task %q has no side table entry", id)
		}
		if b.StepIndex != i || b.NodeName != steps[i].NodeName {
			t.Fatalf("task %q maps to step %d (%s), want step %d (%s)", id, b.StepIndex, b.NodeName, i, steps[i].NodeName)
		}
	}

	// The emitter accepts the derived graph for a 2-CTA grid. The operations
	// are synthetic: cudamega.Operation is Add or Multiply only, so this
	// round trip validates rank, scope and topology, not the step's real
	// arithmetic.
	ops := make(map[string]cudamega.Operation, low.Len())
	for _, id := range low.TaskIDs() {
		ops[id] = cudamega.Operation{Kind: cudamega.Add, Scalar: 1}
	}
	grid := cudamega.Grid{CTAs: 2, ThreadsPerCTA: 128, MaxElements: 32} // fixture shape is 4×8
	src, err := cudamega.ProgramSource(low.Graph(), ops, grid, cudamega.SM80)
	if err != nil {
		t.Fatalf("ProgramSource on the lowered graph: %v", err)
	}
	if !strings.Contains(src, "task_count = 4") {
		t.Fatalf("emitted source does not declare 4 tasks:\n%s", src)
	}

	// Control: the same topology at CTA scope is refused by the same call, so
	// the derived ScopeDevice above is load-bearing and not decoration.
	var ctaTasks []cudataskgraph.Task
	for _, task := range low.Graph().Tasks() {
		task.Scope = cudataskgraph.ScopeCTA
		ctaTasks = append(ctaTasks, task)
	}
	ctaGraph, err := cudataskgraph.New(ctaTasks)
	if err != nil {
		t.Fatalf("control graph: %v", err)
	}
	if _, err := cudamega.ProgramSource(ctaGraph, ops, grid, cudamega.SM80); err == nil {
		t.Fatal("control did not fire: the emitter accepted a CTA-scope edge in a 2-CTA grid, so the scope derivation proves nothing")
	} else {
		t.Logf("control fired: %v", err)
	}
}

// TestLowerRegionDerivesRankFromDepth perturbs the dependency structure and
// shows the derived ranks move with it. The chain and the fan share a step
// count and a step order; only the edges differ.
func TestLowerRegionDerivesRankFromDepth(t *testing.T) {
	art := buildElementwiseArtifact(t)
	step := func(name string, deps []int, isOutput bool) cudaplan.StepPlan {
		return cudaplan.StepPlan{
			NodeName:     name,
			Backend:      cudaplan.BackendLaunch,
			Artifact:     art,
			OutputName:   name,
			IsOutput:     isOutput,
			Dependencies: deps,
		}
	}
	tests := []struct {
		name  string
		steps []cudaplan.StepPlan
		want  map[string]int
	}{{
		name: "chain",
		steps: []cudaplan.StepPlan{
			step("a", nil, false),
			step("b", []int{0}, false),
			step("c", []int{1}, true),
		},
		want: map[string]int{"a": 0, "b": 1, "c": 2},
	}, {
		name: "fan-in: two independent producers share a rank",
		steps: []cudaplan.StepPlan{
			step("a", nil, false),
			step("b", nil, false),
			step("c", []int{0, 1}, true),
		},
		want: map[string]int{"a": 0, "b": 0, "c": 1},
	}, {
		name: "diamond: rank is the longest path, not the shortest",
		steps: []cudaplan.StepPlan{
			step("a", nil, false),
			step("b", []int{0}, false),
			step("c", []int{1}, false),
			step("d", []int{0, 2}, true),
		},
		want: map[string]int{"a": 0, "b": 1, "c": 2, "d": 3},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			low, err := cudaplan.LowerRegion(tt.steps, cudaplan.Region{Start: 0, End: len(tt.steps)})
			if err != nil {
				t.Fatalf("LowerRegion: %v", err)
			}
			got := map[string]int{}
			for _, task := range low.Graph().Tasks() {
				got[task.ID] = task.Rank
			}
			if len(got) != len(tt.want) {
				t.Fatalf("lowered %d tasks, want %d", len(got), len(tt.want))
			}
			for id, want := range tt.want {
				if got[id] != want {
					t.Errorf("task %q rank = %d, want %d", id, got[id], want)
				}
			}
		})
	}
}

// TestRankIsNotStepIndex is the control for the rank derivation. A lowering
// that returned the step index would satisfy cudataskgraph.New and would
// serialize two steps the plan leaves independent; this test fails against
// that implementation and against a constant rank.
func TestRankIsNotStepIndex(t *testing.T) {
	art := buildElementwiseArtifact(t)
	steps := []cudaplan.StepPlan{
		{NodeName: "a", Backend: cudaplan.BackendLaunch, Artifact: art},
		{NodeName: "b", Backend: cudaplan.BackendLaunch, Artifact: art},
		{NodeName: "sink", Backend: cudaplan.BackendLaunch, Artifact: art, IsOutput: true, Dependencies: []int{0, 1}},
	}
	low, err := cudaplan.LowerRegion(steps, cudaplan.Region{Start: 0, End: 3})
	if err != nil {
		t.Fatalf("LowerRegion: %v", err)
	}
	rank := map[string]int{}
	for _, task := range low.Graph().Tasks() {
		rank[task.ID] = task.Rank
	}
	if rank["a"] != rank["b"] {
		t.Fatalf("independent steps got ranks %d and %d: the lowering serialized them", rank["a"], rank["b"])
	}
	if rank["sink"] <= rank["a"] {
		t.Fatalf("sink rank %d does not exceed its dependencies' rank %d", rank["sink"], rank["a"])
	}
	// Proof the control can fire: the index-as-rank answer is a different
	// one, and the assertions above reject it.
	index := map[string]int{"a": 0, "b": 1, "sink": 2}
	if index["a"] == index["b"] {
		t.Fatal("the control fixture cannot distinguish depth from index")
	}
	if rank["b"] == index["b"] {
		t.Fatalf("task b has rank %d, which is its step index: the lowering may not be deriving depth", rank["b"])
	}
}

// TestLowerRegionRefusals names every case the lowering refuses, and each
// fixture differs from a lowerable one in exactly the refused respect.
func TestLowerRegionRefusals(t *testing.T) {
	art := buildElementwiseArtifact(t)
	ok := func(name string, deps []int, isOutput bool) cudaplan.StepPlan {
		return cudaplan.StepPlan{NodeName: name, Backend: cudaplan.BackendLaunch, Artifact: art, OutputName: name, IsOutput: isOutput, Dependencies: deps}
	}
	tests := []struct {
		name       string
		steps      []cudaplan.StepPlan
		region     cudaplan.Region
		want       string
		notLowerab bool
	}{{
		name:   "empty region",
		steps:  []cudaplan.StepPlan{ok("a", nil, true)},
		region: cudaplan.Region{Start: 0, End: 0},
		want:   "empty or out of range",
	}, {
		name:   "region out of range",
		steps:  []cudaplan.StepPlan{ok("a", nil, true)},
		region: cudaplan.Region{Start: 0, End: 2},
		want:   "empty or out of range",
	}, {
		name: "unnamed step",
		steps: []cudaplan.StepPlan{
			{Backend: cudaplan.BackendLaunch, Artifact: art, IsOutput: true},
		},
		region:     cudaplan.Region{Start: 0, End: 1},
		want:       "no node name",
		notLowerab: true,
	}, {
		name:       "duplicate node name",
		steps:      []cudaplan.StepPlan{ok("a", nil, false), ok("a", []int{0}, true)},
		region:     cudaplan.Region{Start: 0, End: 2},
		want:       "already used by step",
		notLowerab: true,
	}, {
		name:       "output nobody reads",
		steps:      []cudaplan.StepPlan{ok("a", nil, false), ok("b", nil, true)},
		region:     cudaplan.Region{Start: 0, End: 2},
		want:       "nothing determines its visibility scope",
		notLowerab: true,
	}, {
		name:       "vendor library backend",
		steps:      []cudaplan.StepPlan{{NodeName: "mm", Backend: cudaplan.BackendCuBLASLt, Artifact: art, IsOutput: true}},
		region:     cudaplan.Region{Start: 0, End: 1},
		want:       "vendor library",
		notLowerab: true,
	}, {
		name:       "unknown backend",
		steps:      []cudaplan.StepPlan{{NodeName: "z", Backend: cudaplan.TargetBackend(99), Artifact: art, IsOutput: true}},
		region:     cudaplan.Region{Start: 0, End: 1},
		want:       "unknown backend",
		notLowerab: true,
	}, {
		name:       "no artifact",
		steps:      []cudaplan.StepPlan{{NodeName: "a", Backend: cudaplan.BackendLaunch, IsOutput: true}},
		region:     cudaplan.Region{Start: 0, End: 1},
		want:       "no kernel artifact",
		notLowerab: true,
	}, {
		name:       "dependency after the region",
		steps:      []cudaplan.StepPlan{ok("a", nil, false), ok("b", []int{2}, true), ok("c", nil, true)},
		region:     cudaplan.Region{Start: 0, End: 2},
		want:       "outside the region",
		notLowerab: true,
	}, {
		name:   "dependency is not a step",
		steps:  []cudaplan.StepPlan{ok("a", []int{7}, true)},
		region: cudaplan.Region{Start: 0, End: 1},
		want:   "is not a step",
	}}

	refused := 0
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := cudaplan.LowerRegion(tt.steps, tt.region)
			if err == nil {
				t.Fatal("lowered a region it cannot derive a schedule for")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %q does not name %q", err, tt.want)
			}
			if got := errors.Is(err, cudaplan.ErrNotLowerable); got != tt.notLowerab {
				t.Fatalf("errors.Is(err, ErrNotLowerable) = %v, want %v (%v)", got, tt.notLowerab, err)
			}
			refused++
		})
	}
	t.Logf("denominator: %d refusal fixtures, %d refused", len(tests), refused)

	// The control for the whole table: the shape these fixtures deviate from
	// does lower. Without it every refusal above could come from one
	// unconditional error.
	base := []cudaplan.StepPlan{ok("a", nil, false), ok("b", []int{0}, true)}
	if _, err := cudaplan.LowerRegion(base, cudaplan.Region{Start: 0, End: 2}); err != nil {
		t.Fatalf("control: the lowerable baseline was refused: %v", err)
	}
}

// TestSideTableRetainsDroppedStepFields asserts every StepPlan field that
// cudataskgraph.Task has no room for is recoverable from the side table.
func TestSideTableRetainsDroppedStepFields(t *testing.T) {
	art := buildElementwiseArtifact(t)
	steps := []cudaplan.StepPlan{
		{NodeID: 11, NodeName: "producer", Backend: cudaplan.BackendGraph, Artifact: art,
			InputNames: []string{"x"}, OutputName: "producer", WorkspaceOffset: 256, WorkspaceBytes: 1024},
		{NodeID: 12, NodeName: "consumer", Backend: cudaplan.BackendLaunch, Artifact: art,
			InputNames: []string{"producer", "w"}, OutputName: "consumer", IsOutput: true,
			WorkspaceOffset: 1280, WorkspaceBytes: 2048, Dependencies: []int{0}},
	}
	low, err := cudaplan.LowerRegion(steps, cudaplan.Region{Start: 0, End: 2})
	if err != nil {
		t.Fatalf("LowerRegion: %v", err)
	}
	for i, s := range steps {
		b, ok := low.Binding(s.NodeName)
		if !ok {
			t.Fatalf("step %d has no side table entry", i)
		}
		switch {
		case b.NodeID != s.NodeID:
			t.Errorf("%s: NodeID = %d, want %d", s.NodeName, b.NodeID, s.NodeID)
		case b.Backend != s.Backend:
			t.Errorf("%s: Backend = %v, want %v", s.NodeName, b.Backend, s.Backend)
		case *b.Artifact != *s.Artifact:
			t.Errorf("%s: Artifact pointer not retained", s.NodeName)
		case strings.Join(b.InputNames, ",") != strings.Join(s.InputNames, ","):
			t.Errorf("%s: InputNames = %v, want %v", s.NodeName, b.InputNames, s.InputNames)
		case b.OutputName != s.OutputName:
			t.Errorf("%s: OutputName = %q, want %q", s.NodeName, b.OutputName, s.OutputName)
		case b.IsOutput != s.IsOutput:
			t.Errorf("%s: IsOutput = %v, want %v", s.NodeName, b.IsOutput, s.IsOutput)
		case b.WorkspaceOffset != s.WorkspaceOffset:
			t.Errorf("%s: WorkspaceOffset = %d, want %d", s.NodeName, b.WorkspaceOffset, s.WorkspaceOffset)
		case b.WorkspaceBytes != s.WorkspaceBytes:
			t.Errorf("%s: WorkspaceBytes = %d, want %d", s.NodeName, b.WorkspaceBytes, s.WorkspaceBytes)
		}
	}

	// Control: the side table is the only copy. Task.Bytes must not be taking
	// WorkspaceBytes, which is an output buffer size and not the shared-memory
	// claim Task.Bytes means.
	for _, task := range low.Graph().Tasks() {
		if task.Bytes != 0 {
			t.Errorf("task %q carries Bytes = %d; Bytes is a shared-memory claim the plan does not state", task.ID, task.Bytes)
		}
	}

	// Control: a mutation of the returned binding does not reach the table.
	b, _ := low.Binding("consumer")
	b.InputNames[0] = "clobbered"
	if again, _ := low.Binding("consumer"); again.InputNames[0] != "producer" {
		t.Fatal("side table aliases its caller's slice")
	}
}

// TestLowerRegionRecordsExternalDependencies asserts an edge that leaves the
// region is recorded rather than dropped: the task graph cannot order against
// a step that is not a task, so the caller must be told what it owes.
func TestLowerRegionRecordsExternalDependencies(t *testing.T) {
	art := buildElementwiseArtifact(t)
	steps := []cudaplan.StepPlan{
		{NodeName: "before", Backend: cudaplan.BackendLaunch, Artifact: art, IsOutput: true},
		{NodeName: "head", Backend: cudaplan.BackendLaunch, Artifact: art, Dependencies: []int{0}},
		{NodeName: "tail", Backend: cudaplan.BackendLaunch, Artifact: art, IsOutput: true, Dependencies: []int{1}},
	}
	low, err := cudaplan.LowerRegion(steps, cudaplan.Region{Start: 1, End: 3})
	if err != nil {
		t.Fatalf("LowerRegion: %v", err)
	}
	head, ok := low.Binding("head")
	if !ok {
		t.Fatal("head is not in the side table")
	}
	if len(head.ExternalDeps) != 1 || head.ExternalDeps[0] != 0 {
		t.Fatalf("head.ExternalDeps = %v, want [0]", head.ExternalDeps)
	}
	for _, task := range low.Graph().Tasks() {
		for _, d := range task.Deps {
			if d == "before" {
				t.Fatal("a step outside the region became a task dependency")
			}
		}
	}
	// Control: the in-region edge is still an edge, so the assertion above is
	// not passing because no edge survives.
	tail, _ := low.Binding("tail")
	if len(tail.ExternalDeps) != 0 {
		t.Fatalf("tail.ExternalDeps = %v, want none", tail.ExternalDeps)
	}
	var tailDeps []string
	for _, task := range low.Graph().Tasks() {
		if task.ID == "tail" {
			tailDeps = task.Deps
		}
	}
	if len(tailDeps) != 1 || tailDeps[0] != "head" {
		t.Fatalf("tail deps = %v, want [head]", tailDeps)
	}
}

// TestLowerRegionRefusesAPlanWithoutArtifacts is the denominator's other half:
// the same fixture compiled without a kernel registry lowers zero steps, so
// the passing case above is not passing for a reason unrelated to the region.
func TestLowerRegionRefusesAPlanWithoutArtifacts(t *testing.T) {
	plan := compileRegionFixture(t, false)
	steps := plan.Steps()
	if len(steps) == 0 {
		t.Fatal("fixture compiled no steps")
	}
	_, err := cudaplan.LowerRegion(steps, cudaplan.Region{Start: 0, End: len(steps)})
	if !errors.Is(err, cudaplan.ErrNotLowerable) {
		t.Fatalf("LowerRegion = %v, want ErrNotLowerable", err)
	}
	t.Logf("denominator: %d steps compiled, 0 lowered, %d refused: %v", len(steps), len(steps), err)
}
