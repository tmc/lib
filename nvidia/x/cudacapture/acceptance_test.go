package cudacapture_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestAmbiguousCorrelation(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()

	// Normal 1-to-1 match
	if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{
		CorrelationID: 101,
		KernelName:    "kernel_unambiguous",
		HostThreadID:  1,
		Callstack:     []string{"main", "launch"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordActivity(cudacapture.ActivityRecord{
		Correlation: 101,
		Name:        "kernel_unambiguous",
		Start:       1000,
		End:         2000,
	}); err != nil {
		t.Fatal(err)
	}

	// Negative control: Scrambled / duplicate correlation IDs
	// Two distinct runtime launches receive the same correlation ID 202
	if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{
		CorrelationID: 202,
		KernelName:    "kernel_scrambled_a",
		HostThreadID:  2,
		Callstack:     []string{"streamA", "launch"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{
		CorrelationID: 202,
		KernelName:    "kernel_scrambled_b",
		HostThreadID:  3,
		Callstack:     []string{"streamB", "launch"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordActivity(cudacapture.ActivityRecord{
		Correlation: 202,
		Name:        "kernel_scrambled_a",
		Start:       3000,
		End:         4000,
	}); err != nil {
		t.Fatal(err)
	}

	matched, ambiguous, unmappedR, unmappedA := tracker.Join()

	// 1. Verify unambiguous match was joined correctly
	if len(matched) != 1 {
		t.Fatalf("len(matched) = %d, want 1", len(matched))
	}
	if matched[0].CorrelationID != 101 || matched[0].KernelName != "kernel_unambiguous" {
		t.Errorf("unexpected matched join: %+v", matched[0])
	}

	// 2. Verify ambiguous correlation was detected and preserved rather than arbitrarily joined
	if len(ambiguous) != 1 {
		t.Fatalf("len(ambiguous) = %d, want 1", len(ambiguous))
	}
	amb := ambiguous[0]
	if amb.CorrelationID != 202 {
		t.Errorf("ambiguous CorrelationID = %d, want 202", amb.CorrelationID)
	}
	if amb.RuntimeCount != 2 {
		t.Errorf("ambiguous RuntimeCount = %d, want 2", amb.RuntimeCount)
	}
	if len(amb.KernelNames) < 2 {
		t.Errorf("expected multiple kernel names in ambiguous record, got: %v", amb.KernelNames)
	}

	if len(unmappedR) != 0 || len(unmappedA) != 0 {
		t.Errorf("expected 0 unmapped, got unmappedR=%d, unmappedA=%d", len(unmappedR), len(unmappedA))
	}
}

func TestModuleGenerationReuse(t *testing.T) {
	tracker := cudacapture.NewModuleTracker()

	baseAddr := uint64(0x7fff_1000)
	digestV1 := "1111111111111111111111111111111111111111111111111111111111111111"
	digestV2 := "2222222222222222222222222222222222222222222222222222222222222222"

	// 1. Load module at baseAddr for generation 1
	codeV1 := cudacapture.CodeFiles{Source: "k.cu", PTX: "k.ptx", CUBIN: "k1.cubin", SASS: "k1.sass"}
	gen1, err := tracker.Load("mod-1", baseAddr, digestV1, codeV1)
	if err != nil {
		t.Fatalf("Load gen1 failed: %v", err)
	}
	if gen1 != 1 {
		t.Fatalf("gen1 = %d, want 1", gen1)
	}

	// Loading same digest at same address returns same generation
	gen1Again, err := tracker.Load("mod-1", baseAddr, digestV1, codeV1)
	if err != nil {
		t.Fatalf("Load same digest failed: %v", err)
	}
	if gen1Again != 1 {
		t.Fatalf("gen1Again = %d, want 1", gen1Again)
	}

	// Verify gen1 matches digestV1
	if err := tracker.Verify(baseAddr, 1, digestV1); err != nil {
		t.Errorf("Verify gen1 failed: %v", err)
	}

	// 2. Negative control: Module address reused with a DIFFERENT digest
	// Recompilation or JIT reload reuses baseAddr with digestV2
	codeV2 := cudacapture.CodeFiles{Source: "k.cu", PTX: "k.ptx", CUBIN: "k2.cubin", SASS: "k2.sass"}
	gen2, err := tracker.Load("mod-1", baseAddr, digestV2, codeV2)
	if err != nil {
		t.Fatalf("Load gen2 failed: %v", err)
	}
	if gen2 != 2 {
		t.Fatalf("gen2 = %d, want 2", gen2)
	}

	// Stale attribution attempt: querying gen1 with digestV2 must fail
	err = tracker.Verify(baseAddr, 1, digestV2)
	if !errors.Is(err, cudacapture.ErrModuleGenerationReuse) {
		t.Errorf("expected ErrModuleGenerationReuse verifying gen1 with digestV2, got: %v", err)
	}

	// Correct lookup: gen2 matches digestV2
	if err := tracker.Verify(baseAddr, 2, digestV2); err != nil {
		t.Errorf("Verify gen2 failed: %v", err)
	}

	// Verify lookup retrieves correct generation artifacts
	entry1, err := tracker.Lookup(baseAddr, 1)
	if err != nil || entry1.ImageDigest != digestV1 {
		t.Errorf("Lookup gen1 failed: %v", err)
	}

	entry2, err := tracker.Lookup(baseAddr, 2)
	if err != nil || entry2.ImageDigest != digestV2 {
		t.Errorf("Lookup gen2 failed: %v", err)
	}
}
