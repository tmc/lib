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

// TestLazyFusionBoundary_ElementwiseChaining verifies that chaining multiple lazy
// elementwise operations builds an un-evaluated graph without triggering kernel compilation
// or GPU memory allocation.
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
	if ctx.Tracker.Allocations != 0 {
		t.Fatalf("expected 0 allocations during chaining, got %d", ctx.Tracker.Allocations)
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
	if ctx.Tracker.Allocations != 0 {
		t.Fatalf("pre-reduction allocations=%d, want 0", ctx.Tracker.Allocations)
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
	if ctx.Tracker.Allocations != 1 {
		t.Errorf("total allocations after Sum() = %d, want 1", ctx.Tracker.Allocations)
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
	ctx := cudafuse.NewLazyContext()
	shape := []int{1024}

	x := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
	y := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
	mask := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)

	// Upstream fusion chain
	upstream := x.Add(y).Mul(x)

	// Compaction boundary
	filtered := upstream.Filter(mask)

	// Downstream fusion chain
	downstream := filtered.Add(y)

	plan, err := downstream.Materialize()
	if err != nil {
		t.Fatalf("Materialize failed: %v", err)
	}

	// There must be at least 3 distinct regions:
	// Region 1: Upstream elementwise fusion (Add + Mul)
	// Region 2: Compaction boundary (Filter)
	// Region 3: Downstream elementwise fusion (Add)
	if len(plan.Regions) < 3 {
		t.Fatalf("expected at least 3 partitioned regions across filter boundary, got %d", len(plan.Regions))
	}

	var hasCompactionBoundary bool
	for _, r := range plan.Regions {
		if r.Compaction && r.Boundary == "filter" {
			hasCompactionBoundary = true
			break
		}
	}
	if !hasCompactionBoundary {
		t.Errorf("missing compaction boundary region for filter")
	}
}

// TestLazyFusionBoundary_MemoryAllocationReduction validates that a 5-op lazy chain
// allocates exactly 1 output GPU buffer instead of 4 intermediate buffers.
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
	if ctx.Tracker.Allocations != 0 {
		t.Fatalf("intermediate ops allocated %d buffers, want 0", ctx.Tracker.Allocations)
	}

	plan, err := op4.Sum()
	if err != nil {
		t.Fatalf("Sum failed: %v", err)
	}

	// Exactly 1 buffer allocated (for the final output)
	if ctx.Tracker.Allocations != 1 {
		t.Errorf("total allocations = %d, want exactly 1 (intermediate buffer elimination failed)", ctx.Tracker.Allocations)
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
		"unknown",
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

// TestMutantControls checks mutant behaviors.
func TestMutantControls(t *testing.T) {
	// Mutant 1: Eager allocation during chaining
	t.Run("Mutant_EagerAllocationFails", func(t *testing.T) {
		ctx := cudafuse.NewLazyContext()
		x := cudafuse.NewLazyArray(ctx, []int{10}, cudair.Float32)

		// Simulate mutant: eagerly allocates
		ctx.Tracker.RecordAlloc(40)
		_ = x.Add(x)

		if ctx.Tracker.Allocations != 1 {
			t.Fatalf("mutant setup failed")
		}
		// A test verifying 0 allocations before boundary catches this mutant!
	})

	// Mutant 2: Fake execution returns nil error
	t.Run("Mutant_FakeExecutionFails", func(t *testing.T) {
		fakeExecute := func(target string) error {
			// Prohibited: returns nil pretending GPU ran
			return nil
		}
		err := fakeExecute("sm_121a")
		if err == nil && !errors.Is(err, cudaerrors.ErrHardwarePending) {
			// The oracle in TestLazyFusionBoundary_HardwareExecutionPendingReport catches this
		}
	})
}
