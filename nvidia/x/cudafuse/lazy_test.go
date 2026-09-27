package cudafuse_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// TestLazyFusionBoundary_ElementwiseChaining verifies that chaining builds an
// unevaluated graph without recording output-allocation bookkeeping.
func TestLazyFusionBoundary_ElementwiseChaining(t *testing.T) {
	ctx := cudafuse.NewLazyContext()
	shape := []int{256, 256}

	x := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
	y := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)

	// Chain multiple lazy elementwise operations
	a1 := x.Add(y)
	a2 := a1.Mul(x)
	a3 := a2.Map("relu")
	a4 := a3.Map("exp")
	a5 := a4.Reshape([]int{65536})
	a6 := a5.Transpose([]int{0})
	a7 := a6.Cycle(2)
	a8 := a7.Replicate(1)

	// Ensure zero intermediate allocations were made during chaining
	if ctx.Tracker.Snapshot().Allocations != 0 {
		t.Fatalf("expected 0 allocations during chaining, got %d", ctx.Tracker.Snapshot().Allocations)
	}

	// Verify the graph is un-evaluated
	if a8.Node.Evaluated {
		t.Fatalf("expected node to remain un-evaluated prior to boundary trigger")
	}

	// Verify node operation sequence
	if a8.Node.Op != "replicate" {
		t.Errorf("expected terminal op replicate, got %s", a8.Node.Op)
	}
}

// TestLazyFusionBoundary_MaterializingReductionTrigger asserts that calling a
// materializing reduction (e.g. sum or reduce) correctly triggers graph compilation
// and flushes the fused kernel boundary.
func TestLazyFusionBoundary_MaterializingReductionTrigger(t *testing.T) {
	ctx := cudafuse.NewLazyContext()
	shape := []int{512}

	a := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
	b := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
	c := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)

	chain := a.Add(b).Mul(c).Map("relu")

	// Pre-boundary: zero allocations, un-evaluated
	if ctx.Tracker.Snapshot().Allocations != 0 {
		t.Fatalf("pre-reduction allocations=%d, want 0", ctx.Tracker.Snapshot().Allocations)
	}
	if chain.Node.Evaluated {
		t.Fatalf("chain was evaluated prematurely")
	}

	// Materializing reduction trigger
	plan, err := chain.Sum()
	if err != nil {
		t.Fatalf("Sum() materialization failed: %v", err)
	}

	// Assert plan and compilation effects
	if len(plan.Regions) == 0 {
		t.Fatalf("expected compiled fusion regions, got 0")
	}
	if plan.OutputBuffers != 1 {
		t.Errorf("output buffers = %d, want 1", plan.OutputBuffers)
	}
	if ctx.Tracker.Snapshot().Allocations != 1 {
		t.Errorf("total allocations after Sum() = %d, want 1", ctx.Tracker.Snapshot().Allocations)
	}
	if !chain.Node.Evaluated {
		t.Fatal("materialized chain root was not marked evaluated")
	}
	for _, region := range plan.Regions {
		for _, node := range region.Nodes {
			if !node.Evaluated {
				t.Errorf("region %d node %s was not marked evaluated", region.ID, node.Name())
			}
		}
	}

	// Upstream fused kernel must have been synthesized
	var foundKernel bool
	for _, reg := range plan.Regions {
		if reg.Kernel != nil && strings.Contains(reg.Kernel.Source, "v_node_") {
			foundKernel = true
			break
		}
	}
	if !foundKernel {
		t.Errorf("expected synthesized CUDA fused kernel in plan regions")
	}
}

// TestLazyFusionBoundary_ConditionallyFusedFilterBoundary asserts that predicate
// compactions (where/filter) insert an explicit materialization boundary and correctly
// isolate upstream elementwise fusion.
func TestLazyFusionBoundary_ConditionallyFusedFilterBoundary(t *testing.T) {
	for _, boundary := range []string{"filter", "where", "keep"} {
		t.Run(boundary, func(t *testing.T) {
			ctx := cudafuse.NewLazyContext()
			shape := []int{1024}
			x := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
			y := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
			mask := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
			upstream := x.Add(y).Mul(x)
			var compact *cudafuse.LazyArray
			switch boundary {
			case "filter":
				compact = upstream.Filter(mask)
			case "where":
				compact = upstream.Where(mask, y)
			case "keep":
				compact = upstream.Keep(mask)
			}
			plan, err := compact.Add(y).Materialize()
			if err != nil {
				t.Fatalf("Materialize: %v", err)
			}
			var upstreamOps, downstreamOps []string
			foundBoundary := false
			for _, region := range plan.Regions {
				var ops []string
				for _, node := range region.Nodes {
					if node.Op != "input" {
						ops = append(ops, node.Op)
					}
				}
				if region.Compaction {
					if region.Boundary != boundary || len(ops) != 1 || ops[0] != boundary {
						t.Fatalf("compaction region = boundary %q ops %v", region.Boundary, ops)
					}
					foundBoundary = true
					continue
				}
				if sameOps(ops, []string{"add", "mul"}) {
					upstreamOps = ops
				}
				if len(ops) == 1 && ops[0] == "add" {
					downstreamOps = ops
				}
			}
			if !foundBoundary || !sameOps(upstreamOps, []string{"add", "mul"}) || !sameOps(downstreamOps, []string{"add"}) {
				t.Fatalf("regions around %s: upstream=%v downstream=%v boundary=%t", boundary, upstreamOps, downstreamOps, foundBoundary)
			}
		})
	}
}

func sameOps(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	counts := make(map[string]int)
	for _, op := range got {
		counts[op]++
	}
	for _, op := range want {
		counts[op]--
	}
	for _, count := range counts {
		if count != 0 {
			return false
		}
	}
	return true
}

// TestLazyFusionBoundary_MemoryAllocationReduction checks host bookkeeping for
// one output allocation after a lazy chain; it does not compare a built eager baseline.
func TestLazyFusionBoundary_MemoryAllocationReduction(t *testing.T) {
	ctx := cudafuse.NewLazyContext()
	shape := []int{1024}

	a := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
	b := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)

	// 5-op chain:
	// Op 1: a + b
	// Op 2: (a + b) * b
	// Op 3: relu(...)
	// Op 4: exp(...)
	// Op 5: sum(...) [materialization boundary]
	op1 := a.Add(b)
	op2 := op1.Mul(b)
	op3 := op2.Map("relu")
	op4 := op3.Map("exp")

	// Ensure intermediate ops incurred 0 allocations
	if ctx.Tracker.Snapshot().Allocations != 0 {
		t.Fatalf("intermediate ops allocated %d buffers, want 0", ctx.Tracker.Snapshot().Allocations)
	}

	plan, err := op4.Sum()
	if err != nil {
		t.Fatalf("Sum failed: %v", err)
	}

	// One output allocation is recorded in the host tracker
	if ctx.Tracker.Snapshot().Allocations != 1 {
		t.Errorf("total allocations = %d, want exactly 1 (intermediate buffer elimination failed)", ctx.Tracker.Snapshot().Allocations)
	}
	if plan.OutputBuffers != 1 {
		t.Errorf("plan output buffers = %d, want 1", plan.OutputBuffers)
	}
}

// TestLazyFusionBoundary_HardwareExecutionPendingReport confirms that execution passes
// return ErrHardwarePending (via errors.Is) for every known target until a native measurement exists,
// and ErrUnsupported for an unknown target, rather than returning faked data. The answer does not
// depend on whether the host has a GPU.
func TestLazyFusionBoundary_HardwareExecutionPendingReport(t *testing.T) {
	ctx := cudafuse.NewLazyContext()
	a := cudafuse.NewLazyArray(ctx, []int{128}, cudair.Float32)
	plan, err := a.Add(a).Sum()
	if err != nil {
		t.Fatalf("plan failed: %v", err)
	}

	// Probed targets from cudadevicetable plus standard sm architectures
	knownTargets := cudadevicetable.Targets()
	if len(knownTargets) == 0 {
		knownTargets = []string{"sm_90", "sm_90a", "sm_100", "sm_120", "sm_121", "sm_121a"}
	}

	for _, target := range knownTargets {
		err := plan.Execute(target)
		if err == nil {
			t.Fatalf("target %s returned nil error (prohibited fake GPU execution)", target)
		}
		if !errors.Is(err, cudaerrors.ErrHardwarePending) {
			t.Errorf("target %s error = %v, want matching ErrHardwarePending", target, err)
		}
	}

	unknownTargets := []string{
		"sm_unknown",
		"sm_10",
		"cpu",
		"invalid_arch",
		"unknown", "sm_99", "sm_42", "sm_121zz", "sm_36",
	}

	for _, target := range unknownTargets {
		err := plan.Execute(target)
		if err == nil {
			t.Fatalf("unknown target %s returned nil error, want ErrUnsupported", target)
		}
		if !errors.Is(err, cudaerrors.ErrUnsupported) {
			t.Errorf("unknown target %s error = %v, want matching ErrUnsupported", target, err)
		}
	}
}

func TestLazyFusionBoundary_EveryFusibleOpMaterializes(t *testing.T) {
	tests := []struct {
		name  string
		build func(*cudafuse.LazyArray, *cudafuse.LazyArray) *cudafuse.LazyArray
	}{
		{name: "add", build: func(x, y *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Add(y) }},
		{name: "sub", build: func(x, y *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Sub(y) }},
		{name: "mul", build: func(x, y *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Mul(y) }},
		{name: "div", build: func(x, y *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Div(y) }},
		{name: "relu", build: func(x, _ *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Map("relu") }},
		{name: "silu", build: func(x, _ *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Map("silu") }},
		{name: "exp", build: func(x, _ *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Map("exp") }},
		{name: "sqrt", build: func(x, _ *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Map("sqrt") }},
		{name: "neg", build: func(x, _ *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Map("neg") }},
		{name: "reshape", build: func(x, _ *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Reshape([]int{4}) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := cudafuse.NewLazyContext()
			shape := []int{4}
			if tt.name == "reshape" {
				shape = []int{2, 2}
			}
			x := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
			y := cudafuse.NewLazyArray(ctx, []int{4}, cudair.Float32)
			chain := tt.build(x, y).Add(y)
			plan, err := chain.Materialize()
			if err != nil {
				t.Fatalf("Materialize: %v", err)
			}
			foundKernel := false
			for _, region := range plan.Regions {
				if region.Compaction || cudafuse.IsMaterializingBoundary(region.Boundary) {
					continue
				}
				if region.Kernel == nil {
					t.Fatalf("non-boundary region %d has no kernel", region.ID)
				}
				foundKernel = true
			}
			if !foundKernel {
				t.Fatal("no non-boundary kernel region")
			}
		})
	}

	for _, op := range []string{"transpose", "gather", "cycle", "replicate"} {
		t.Run(op+"_explicit_boundary", func(t *testing.T) {
			ctx := cudafuse.NewLazyContext()
			x := cudafuse.NewLazyArray(ctx, []int{4}, cudair.Float32)
			y := cudafuse.NewLazyArray(ctx, []int{4}, cudair.Float32)
			var boundary *cudafuse.LazyArray
			switch op {
			case "transpose":
				boundary = x.Transpose([]int{0})
			case "gather":
				boundary = x.Gather(y)
			case "cycle":
				boundary = x.Cycle(2)
			case "replicate":
				boundary = x.Replicate(2)
			}
			plan, err := boundary.Add(y).Materialize()
			if err != nil {
				t.Fatalf("Materialize: %v", err)
			}
			foundBoundary, foundDownstreamKernel := false, false
			for _, region := range plan.Regions {
				if region.Boundary == op {
					foundBoundary = region.Reason != ""
				}
				if !cudafuse.IsMaterializingBoundary(region.Boundary) && region.Kernel != nil {
					foundDownstreamKernel = true
				}
			}
			if !foundBoundary || !foundDownstreamKernel {
				t.Fatalf("regions do not show %s boundary and downstream kernel: %+v", op, plan.Regions)
			}
		})
	}

	ctx := cudafuse.NewLazyContext()
	x := cudafuse.NewLazyArray(ctx, []int{4}, cudair.Float32)
	unknown := x.Map("bogus")
	if unknown.Node.Err == nil || !strings.Contains(unknown.Node.Err.Error(), "bogus") {
		t.Fatalf("Map(bogus) construction error = %v, want recorded error naming bogus", unknown.Node.Err)
	}
	_, err := unknown.Materialize()
	if err == nil || !strings.Contains(err.Error(), "bogus") {
		t.Fatalf("Map(bogus).Materialize() error = %v, want an error naming bogus", err)
	}
}

func TestLazyArray_NilOperandErrors(t *testing.T) {
	ctx := cudafuse.NewLazyContext()
	a := cudafuse.NewLazyArray(ctx, []int{4}, cudair.Float32)
	for _, tt := range []struct {
		name  string
		array *cudafuse.LazyArray
	}{
		{name: "Add", array: a.Add(nil)},
		{name: "Mul", array: a.Mul(nil)},
		{name: "Map2", array: a.Map2("add", nil)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.array.Materialize(); err == nil || !strings.Contains(err.Error(), "non-nil operand") {
				t.Fatalf("Materialize error = %v, want explicit nil-operand error", err)
			}
		})
	}
}

func TestLazyArray_ReshapeRejectsChangedElementCount(t *testing.T) {
	x := cudafuse.NewLazyArray(cudafuse.NewLazyContext(), []int{4}, cudair.Float32)
	reshaped := x.Reshape([]int{3})
	if reshaped.Node.Err == nil {
		t.Fatal("Reshape([3]) recorded no error for a four-element input")
	}
	if _, err := reshaped.Materialize(); err == nil {
		t.Fatal("Materialize accepted reshape from four elements to three")
	}
}

func TestLazyFusionBoundary_RegionCompileErrorPropagates(t *testing.T) {
	ctx := cudafuse.NewLazyContext()
	x := cudafuse.NewLazyArray(ctx, []int{4}, cudair.Float16)
	plan, err := x.Add(x).Materialize()
	if err == nil {
		t.Fatal("Materialize accepted a region cudafuse cannot lower")
	}
	if plan != nil {
		t.Fatalf("Materialize returned a partial plan with a region compilation error: %#v", plan)
	}
}

func TestLazyFusionBoundary_OutputBytes(t *testing.T) {
	for _, tt := range []struct {
		name  string
		dtype cudair.DataType
		want  int64
	}{
		{"float32", cudair.Float32, 16},
		{"float16", cudair.Float16, 8},
		{"bfloat16", cudair.BFloat16, 8},
		{"int64", cudair.Int64, 32},
		{"int8", cudair.Int8, 4},
	} {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := cudafuse.NewLazyArray(nil, []int{4}, tt.dtype).Materialize()
			if err != nil {
				t.Fatal(err)
			}
			if plan.AllocatedBytes != tt.want {
				t.Fatalf("AllocatedBytes = %d, want %d", plan.AllocatedBytes, tt.want)
			}
		})
	}
	empty, err := cudafuse.NewLazyArray(nil, []int{0}, cudair.Float32).Materialize()
	if err != nil || empty.AllocatedBytes != 0 {
		t.Fatalf("zero-shape allocation = %v, %v; want 0 bytes", empty, err)
	}
	if _, err := cudafuse.NewLazyArray(nil, []int{-1}, cudair.Float32).Materialize(); err == nil || !strings.Contains(err.Error(), "negative") {
		t.Fatalf("negative-shape error = %v, want negative-dimension error", err)
	}
}

func TestLazyFusionBoundary_TrackerSnapshot(t *testing.T) {
	tracker := &cudafuse.MemoryTracker{}
	tracker.RecordAlloc(12)
	tracker.RecordFree(5)
	stats := tracker.Snapshot()
	if stats.Allocations != 1 || stats.AllocatedBytes != 12 || stats.FreedBytes != 5 {
		t.Fatalf("Snapshot() = %+v", stats)
	}
}
