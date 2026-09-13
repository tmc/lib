package cudamemtrack_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamemtrack"
)

func TestTrackerLifecycle(t *testing.T) {
	tr := cudamemtrack.New()

	tr.Alloc(0x1000, 1024, "tensor_a")
	tr.Alloc(0x2000, 2048, "tensor_b")

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
	tr.Alloc(0x5000, 4096, "leaked_kv_cache")

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
		{"largest_exceeds_total_sanity", 1000, 1200, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cudamemtrack.Fragmentation(tt.totalFree, tt.largestFree)
			diff := got - tt.wantFrag
			if diff < -0.001 || diff > 0.001 {
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
	tr.Alloc(0x8000, 512, "weights")

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
