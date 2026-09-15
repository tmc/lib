package cudamemtrack_test

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamemtrack"
)

func TestTrackerLifecycle(t *testing.T) {
	tr := cudamemtrack.New()

	mustAlloc(t, tr, 0x1000, 1024, "tensor_a")
	mustAlloc(t, tr, 0x2000, 2048, "tensor_b")

	s := tr.Summary()
	if s.ActiveAllocations != 2 {
		t.Errorf("ActiveAllocations = %d, want 2", s.ActiveAllocations)
	}
	if s.ActiveBytes != 3072 {
		t.Errorf("ActiveBytes = %d, want 3072", s.ActiveBytes)
	}
	if s.PeakBytes != 3072 {
		t.Errorf("PeakBytes = %d, want 3072", s.PeakBytes)
	}

	if err := tr.Free(0x1000); err != nil {
		t.Fatalf("Free failed: %v", err)
	}

	s = tr.Summary()
	if s.ActiveAllocations != 1 {
		t.Errorf("ActiveAllocations = %d, want 1", s.ActiveAllocations)
	}
	if s.ActiveBytes != 2048 {
		t.Errorf("ActiveBytes = %d, want 2048", s.ActiveBytes)
	}
	if s.TotalFreedBytes != 1024 {
		t.Errorf("TotalFreedBytes = %d, want 1024", s.TotalFreedBytes)
	}

	// Double free should error
	if err := tr.Free(0x1000); err == nil {
		t.Error("expected error on double free, got nil")
	}

	// Free remaining
	if err := tr.Free(0x2000); err != nil {
		t.Fatalf("Free failed: %v", err)
	}

	if err := tr.CheckLeaks(); err != nil {
		t.Errorf("unexpected leaks: %v", err)
	}
}

func TestLeakDetection(t *testing.T) {
	tr := cudamemtrack.New()
	mustAlloc(t, tr, 0x5000, 4096, "leaked_kv_cache")

	err := tr.CheckLeaks()
	if err == nil {
		t.Fatal("expected leak error, got nil")
	}
	if !strings.Contains(err.Error(), "1 memory leaks detected") || !strings.Contains(err.Error(), "leaked_kv_cache") {
		t.Errorf("unexpected error string: %v", err)
	}
}

func TestFragmentation(t *testing.T) {
	tests := []struct {
		name        string
		totalFree   uint64
		largestFree uint64
		wantFrag    float64
	}{
		{"zero_free", 0, 0, 0.0},
		{"unfragmented_single_block", 1000, 1000, 0.0},
		{"equal_halves", 1000, 500, 0.5},
		{"ten_percent_largest", 1000, 100, 0.9},
		// An inconsistent pair is not a healthy heap: it must not read as 0.0.
		{"largest_exceeds_total_is_nan", 1000, 1200, math.NaN()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cudamemtrack.Fragmentation(tt.totalFree, tt.largestFree)
			if math.IsNaN(tt.wantFrag) {
				if !math.IsNaN(got) {
					t.Errorf("Fragmentation(%d, %d) = %f, want NaN", tt.totalFree, tt.largestFree, got)
				}
				return
			}
			if diff := got - tt.wantFrag; diff < -0.001 || diff > 0.001 {
				t.Errorf("Fragmentation(%d, %d) = %f, want %f", tt.totalFree, tt.largestFree, got, tt.wantFrag)
			}
		})
	}
}

func TestComputeFragmentationBlocks(t *testing.T) {
	blocks := []cudamemtrack.Block{
		{Address: 0x1000, Size: 200, Free: true},
		{Address: 0x1200, Size: 100, Free: false}, // allocated
		{Address: 0x1300, Size: 800, Free: true},  // largest free
	}

	// Total free: 200 + 800 = 1000. Largest free = 800.
	// Fragmentation = 1.0 - (800 / 1000) = 0.20
	got := cudamemtrack.ComputeFragmentation(blocks)
	want := 0.20
	if diff := got - want; diff < -0.001 || diff > 0.001 {
		t.Errorf("ComputeFragmentation = %f, want %f", got, want)
	}
}

func TestWriteReport(t *testing.T) {
	tr := cudamemtrack.New()
	mustAlloc(t, tr, 0x8000, 512, "weights")

	var buf bytes.Buffer
	if err := tr.WriteReport(&buf); err != nil {
		t.Fatalf("WriteReport failed: %v", err)
	}

	out := buf.String()
	for _, expected := range []string{
		"GPU Memory Allocation Summary",
		"Active Allocations: 1 (512 bytes)",
		"Peak Usage:         512 bytes",
		"0x8000 - 512 bytes (weights)",
	} {
		if !strings.Contains(out, expected) {
			t.Errorf("report missing expected substring: %q", expected)
		}
	}
}

func mustAlloc(t *testing.T, tr *cudamemtrack.Tracker, addr uintptr, size uint64, tag string) {
	t.Helper()
	if err := tr.Alloc(addr, size, tag); err != nil {
		t.Fatalf("Alloc(0x%x, %d, %q): %v", addr, size, tag, err)
	}
}

// TestAllocRejectsLiveAddress pins that re-allocating a live address is refused.
//
// An earlier version silently retired the first block: it left Live and CheckLeaks,
// and its bytes were added to TotalFreedBytes, so a missed Free was reported as
// freed memory and the leak disappeared.
func TestAllocRejectsLiveAddress(t *testing.T) {
	tr := cudamemtrack.New()
	mustAlloc(t, tr, 0x1000, 1024, "first")

	err := tr.Alloc(0x1000, 2048, "second")
	if err == nil {
		t.Fatal("Alloc on a live address returned nil, want error")
	}
	if !strings.Contains(err.Error(), "already tracked") {
		t.Errorf("unexpected error string: %v", err)
	}

	s := tr.Summary()
	if s.ActiveAllocations != 1 {
		t.Errorf("ActiveAllocations = %d, want 1", s.ActiveAllocations)
	}
	if s.ActiveBytes != 1024 {
		t.Errorf("ActiveBytes = %d, want 1024", s.ActiveBytes)
	}
	if s.TotalFreedBytes != 0 {
		t.Errorf("TotalFreedBytes = %d, want 0: nothing was freed", s.TotalFreedBytes)
	}
	if s.TotalAllocations != 1 {
		t.Errorf("TotalAllocations = %d, want 1: the refused call recorded nothing", s.TotalAllocations)
	}

	// The first block is still a leak; it must not have been erased.
	if err := tr.CheckLeaks(); err == nil {
		t.Error("CheckLeaks returned nil, want the unfreed first block reported")
	}
}

func TestAllocRejectsDegenerateArguments(t *testing.T) {
	tests := []struct {
		name string
		addr uintptr
		size uint64
	}{
		{"zero_address", 0, 1024},
		{"zero_size", 0x1000, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := cudamemtrack.New()
			if err := tr.Alloc(tt.addr, tt.size, "x"); err == nil {
				t.Fatalf("Alloc(0x%x, %d) = nil, want error", tt.addr, tt.size)
			}
			if s := tr.Summary(); s.TotalAllocations != 0 {
				t.Errorf("TotalAllocations = %d, want 0", s.TotalAllocations)
			}
		})
	}
}
