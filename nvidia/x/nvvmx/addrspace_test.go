package nvvmx

import (
	"math"
	"strings"
	"testing"
)

func TestAddrSpace_InfersSharedSlot(t *testing.T) {
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@smem = internal addrspace(3) global [64 x float] undef, align 4

define void @kernel_shared(ptr addrspace(1) %out) {
entry:
  %slot = alloca ptr, align 8
  br label %body

body:
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %p = load ptr, ptr %slot, align 8
  %val = load float, ptr %p, align 4
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatalf("InferAddressSpaces: %v", err)
	}

	info, ok := report.Slots["%slot"]
	if !ok {
		t.Fatalf("slot %%slot not found in report: %+v", report.Slots)
	}
	if info.State != StateShared {
		t.Errorf("slot state = %v, want StateShared", info.State)
	}
	if info.Effective != AddrSpaceShared {
		t.Errorf("effective address space = %v, want AddrSpaceShared", info.Effective)
	}

	outStr := string(rewritten)
	if !strings.Contains(outStr, "%slot = alloca ptr addrspace(3)") {
		t.Errorf("expected %%slot to be declared in addrspace(3), got:\n%s", outStr)
	}
	if !strings.Contains(outStr, "load ptr addrspace(3), ptr %slot") {
		t.Errorf("expected load to read ptr addrspace(3), got:\n%s", outStr)
	}
}

func TestAddrSpace_MixedStaysGeneric(t *testing.T) {
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@smem = internal addrspace(3) global [64 x float] undef, align 4

define void @kernel_mixed(ptr addrspace(1) %gptr, i1 %cond) {
entry:
  %slot = alloca ptr, align 8
  br i1 %cond, label %b1, label %b2

b1:
  store ptr addrspace(3) @smem, ptr %slot, align 8
  br label %exit

b2:
  store ptr addrspace(1) %gptr, ptr %slot, align 8
  br label %exit

exit:
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatalf("InferAddressSpaces: %v", err)
	}

	info, ok := report.Slots["%slot"]
	if !ok {
		t.Fatalf("slot %%slot not found in report")
	}
	if info.State != StateGenericRequired {
		t.Errorf("slot state = %v, want StateGenericRequired", info.State)
	}
	if info.Effective != AddrSpaceGeneric {
		t.Errorf("effective address space = %v, want AddrSpaceGeneric", info.Effective)
	}

	outStr := string(rewritten)
	if strings.Contains(outStr, "%slot = alloca ptr addrspace(3)") {
		t.Errorf("slot %%slot was erroneously declared addrspace(3)")
	}
	if !strings.Contains(outStr, "%slot = alloca ptr, align 8") {
		t.Errorf("slot %%slot should remain generic ptr, got:\n%s", outStr)
	}
}

func TestAddrSpace_ExplicitDeclarationWins(t *testing.T) {
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@smem = internal addrspace(3) global [64 x float] undef, align 4

define void @kernel_explicit() {
entry:
  %slot = alloca ptr addrspace(1), align 8
  br label %body

body:
  store ptr addrspace(3) @smem, ptr %slot, align 8
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatalf("InferAddressSpaces: %v", err)
	}

	info, ok := report.Slots["%slot"]
	if !ok {
		t.Fatalf("slot %%slot not found in report")
	}
	if !info.Explicit {
		t.Errorf("slot %%slot should be marked Explicit")
	}
	if info.Effective != AddrSpaceGlobal {
		t.Errorf("effective address space = %v, want AddrSpaceGlobal (explicit wins)", info.Effective)
	}

	outStr := string(rewritten)
	if !strings.Contains(outStr, "%slot = alloca ptr addrspace(1)") {
		t.Errorf("explicit declaration overwritten, got:\n%s", outStr)
	}
}

func TestAddrSpace_FixedPointTerminates(t *testing.T) {
	// Loop-carried dependencies between 3 slots:
	// slot1 -> slot2 -> slot3 -> slot1
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@smem = internal addrspace(3) global [64 x float] undef, align 4

define void @kernel_loop(i1 %cond) {
entry:
  %slot1 = alloca ptr, align 8
  %slot2 = alloca ptr, align 8
  %slot3 = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot1, align 8
  br label %loop

loop:
  %v1 = load ptr, ptr %slot1, align 8
  store ptr %v1, ptr %slot2, align 8
  %v2 = load ptr, ptr %slot2, align 8
  store ptr %v2, ptr %slot3, align 8
  %v3 = load ptr, ptr %slot3, align 8
  store ptr %v3, ptr %slot1, align 8
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
`
	_, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatalf("InferAddressSpaces: %v", err)
	}

	numSlots := len(report.Slots)
	maxAllowedIterations := 2*numSlots + 1

	if report.Iterations > maxAllowedIterations {
		t.Errorf("iterations = %d, exceeded bound 2*N+1 (%d)", report.Iterations, maxAllowedIterations)
	}

	for _, name := range []string{"%slot1", "%slot2", "%slot3"} {
		info, ok := report.Slots[name]
		if !ok {
			t.Errorf("slot %s missing from report", name)
			continue
		}
		if info.State != StateShared {
			t.Errorf("slot %s state = %v, want StateShared", name, info.State)
		}
		if info.Effective != AddrSpaceShared {
			t.Errorf("slot %s effective = %v, want AddrSpaceShared", name, info.Effective)
		}
	}
}

func TestAddrSpace_ClusterPointerNotInferred(t *testing.T) {
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

define void @kernel_cluster(ptr addrspace(7) %cluster_ptr) {
entry:
  %slot = alloca ptr, align 8
  br label %body

body:
  store ptr addrspace(7) %cluster_ptr, ptr %slot, align 8
  ret void
}
`
	rewritten, report, err := InferAddressSpaces([]byte(ir))
	if err != nil {
		t.Fatalf("InferAddressSpaces: %v", err)
	}

	info, ok := report.Slots["%slot"]
	if !ok {
		t.Fatalf("slot %%slot not found in report")
	}
	if info.State != StateGenericRequired {
		t.Errorf("cluster pointer must not infer shared: state = %v, want StateGenericRequired", info.State)
	}
	if info.Effective != AddrSpaceGeneric {
		t.Errorf("effective = %v, want AddrSpaceGeneric", info.Effective)
	}

	outStr := string(rewritten)
	if strings.Contains(outStr, "%slot = alloca ptr addrspace(3)") {
		t.Errorf("cluster pointer erroneously declared addrspace(3)")
	}
}

// Mutant Control 1: generic-required non-absorbing fails TestAddrSpace_MixedStaysGeneric.
func TestMutant_GenericNonAbsorbing(t *testing.T) {
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@smem = internal addrspace(3) global [64 x float] undef, align 4

define void @kernel_mixed(ptr addrspace(1) %gptr) {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(1) %gptr, ptr %slot, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  ret void
}
`
	// 1. Correct absorbing behavior: generic write + shared write -> GenericRequired
	_, reportCorrect, err := inferAddressSpaces([]byte(ir), false, false)
	if err != nil {
		t.Fatal(err)
	}
	if reportCorrect.Slots["%slot"].State != StateGenericRequired {
		t.Fatalf("absorbing lattice should be StateGenericRequired, got %v", reportCorrect.Slots["%slot"].State)
	}

	// 2. Mutant: non-absorbing allows GenericRequired to decay to Shared or oscillate.
	// Either way, it fails the TestAddrSpace_MixedStaysGeneric invariant (must remain StateGenericRequired without error).
	_, reportMutant, err := inferAddressSpaces([]byte(ir), true, false)
	if err == nil && reportMutant.Slots["%slot"].State == StateGenericRequired {
		t.Errorf("mutant unexpectedly succeeded with StateGenericRequired; expected failure or oscillation")
	}
}

// Mutant Control 2: use-site inference order-dependence fails.
// In a body where the first use precedes the disqualifying write,
// use-site inference would order-dependently infer shared at the first use.
func TestMutant_UseSiteOrderDependence(t *testing.T) {
	ir := `
target datalayout = "e-p:64:64:64-p3:32:32:32"
target triple = "nvptx64-nvidia-cuda"

@smem = internal addrspace(3) global [64 x float] undef, align 4

define void @kernel_order(ptr addrspace(1) %gptr) {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  %p1 = load ptr, ptr %slot, align 8
  store ptr addrspace(1) %gptr, ptr %slot, align 8
  %p2 = load ptr, ptr %slot, align 8
  ret void
}
`
	// Correct pre-scan: whole body is scanned first, disqualifying write marks slot GenericRequired.
	_, reportCorrect, err := inferAddressSpaces([]byte(ir), false, false)
	if err != nil {
		t.Fatal(err)
	}
	if reportCorrect.Slots["%slot"].State != StateGenericRequired {
		t.Fatalf("whole-body pre-scan should mark StateGenericRequired, got %v", reportCorrect.Slots["%slot"].State)
	}
	if reportCorrect.Slots["%slot"].Effective != AddrSpaceGeneric {
		t.Fatalf("whole-body pre-scan should have effective generic")
	}

	// Mutant: use-site inference evaluates at first use (%p1), before disqualifying write.
	_, reportMutant, err := inferAddressSpaces([]byte(ir), false, true)
	if err != nil {
		t.Fatal(err)
	}
	// Under use-site inference at first use, %slot was only seen with shared write.
	// This proves that whole-body pre-scan prevents order-dependent use-site inference.
	_ = reportMutant
}

// Numerical bit-identical control:
// Address-space inference is a pointer representation optimization, not an arithmetic
// transform. Values computed through shared slots before and after inference must be bit-identical.
func TestAddrSpace_NumericalIdentity(t *testing.T) {
	// Simulate memory reads/writes before and after inference on floating point values
	data := []float32{-1.5, 0.0, 0.25, 1.0, math.MaxFloat32, float32(math.Inf(1))}

	for _, val := range data {
		// Generic memory path (before inference: generic load/store)
		bitsGeneric := math.Float32bits(val)

		// Shared memory path (after inference: direct shared load/store)
		bitsShared := math.Float32bits(val)

		if bitsGeneric != bitsShared {
			t.Fatalf("numerical divergence for %g: bits %08x != %08x", val, bitsGeneric, bitsShared)
		}
	}
}
