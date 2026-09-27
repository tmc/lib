package cudashared

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

func TestByteRangeOverlap(t *testing.T) {
	tests := []struct {
		name string
		aOff uint64
		aSz  uint32
		bOff uint64
		bSz  uint32
		want bool
	}{
		{"identical single byte", 0x1000, 1, 0x1000, 1, true},
		{"adjacent bytes in same word", 0x1000, 1, 0x1001, 1, false},
		{"adjacent 2-byte halfwords", 0x1000, 2, 0x1002, 2, false},
		{"partial overlap at end", 0x1000, 4, 0x1002, 4, true},
		{"contained interval", 0x1000, 8, 0x1002, 2, true},
		{"zero size A", 0x1000, 0, 0x1000, 4, false},
		{"zero size B", 0x1000, 4, 0x1000, 0, false},
		{"completely disjoint far apart", 0x1000, 4, 0x2000, 4, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ByteRangeOverlap(tc.aOff, tc.aSz, tc.bOff, tc.bSz)
			if got != tc.want {
				t.Errorf("ByteRangeOverlap(0x%x, %d, 0x%x, %d) = %v, want %v",
					tc.aOff, tc.aSz, tc.bOff, tc.bSz, got, tc.want)
			}
		})
	}
}

func TestWordOverlap(t *testing.T) {
	tests := []struct {
		name string
		aOff uint64
		aSz  uint32
		bOff uint64
		bSz  uint32
		want bool
	}{
		{"adjacent bytes same word", 0x1000, 1, 0x1001, 1, true},
		{"byte 0 and byte 3 same word", 0x1000, 1, 0x1003, 1, true},
		{"byte 3 of word 0 and byte 0 of word 1", 0x1003, 1, 0x1004, 1, false},
		{"spanning word boundary", 0x1002, 4, 0x1004, 1, true},
		{"zero size", 0x1000, 0, 0x1000, 4, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := WordOverlap(tc.aOff, tc.aSz, tc.bOff, tc.bSz)
			if got != tc.want {
				t.Errorf("WordOverlap(0x%x, %d, 0x%x, %d) = %v, want %v",
					tc.aOff, tc.aSz, tc.bOff, tc.bSz, got, tc.want)
			}
		})
	}
}

func TestHazardMatrix_ByteIndependenceFalsePositive(t *testing.T) {
	// Models cuHadron false_positives/shared_bytewise_readwrite:
	// Thread 0 reads byte 0 of shared_data (offset 0x100)
	// Thread 32 writes byte 1 of shared_data (offset 0x101)
	m := NewHazardMatrix("sm_90")
	if err := m.AddAccess(MemoryAccess{
		Scope:     ScopeCTA,
		ThreadID:  0,
		BlockID:   0,
		Space:     MemoryShared,
		Offset:    0x100,
		SizeBytes: 1,
		IsWrite:   false,
		Stage:     0,
	}); err != nil {
		t.Fatalf("AddAccess read: %v", err)
	}
	if err := m.AddAccess(MemoryAccess{
		Scope:     ScopeCTA,
		ThreadID:  32,
		BlockID:   0,
		Space:     MemoryShared,
		Offset:    0x101,
		SizeBytes: 1,
		IsWrite:   true,
		Stage:     0,
	}); err != nil {
		t.Fatalf("AddAccess write: %v", err)
	}

	records, err := m.Evaluate()
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}

	r := records[0]
	if !r.WordConflict {
		t.Errorf("expected WordConflict == true on same 32-bit word")
	}
	if r.ByteConflict {
		t.Errorf("expected ByteConflict == false on distinct bytes")
	}
	if r.IsRace {
		t.Errorf("false positive: IsRace reported true for byte-independent accesses")
	}

	hasRace, err := m.HasRace()
	if err != nil {
		t.Fatalf("HasRace: %v", err)
	}
	if hasRace {
		t.Errorf("HasRace reported true, want false (byte-independent accesses are race-free)")
	}
}

func TestHazardMatrix_ClusterDSMEMRace(t *testing.T) {
	// Models cuHadron dsmem/shared_readwrite_race without cluster.sync()
	m := NewHazardMatrix("sm_90")
	// CTA 0 writes to shared_data[0]
	if err := m.AddAccess(MemoryAccess{
		Scope:     ScopeCluster,
		ThreadID:  0,
		BlockID:   0,
		ClusterID: 0,
		Space:     MemoryDSMEM,
		Offset:    0x200,
		SizeBytes: 4,
		IsWrite:   true,
		Stage:     0,
	}); err != nil {
		t.Fatalf("AddAccess writer: %v", err)
	}
	// CTA 1 reads from CTA 0's shared_data[0] via DSMEM concurrently (Stage 0)
	if err := m.AddAccess(MemoryAccess{
		Scope:     ScopeCluster,
		ThreadID:  0,
		BlockID:   1,
		ClusterID: 0,
		Space:     MemoryDSMEM,
		Offset:    0x200,
		SizeBytes: 4,
		IsWrite:   false,
		Stage:     0,
	}); err != nil {
		t.Fatalf("AddAccess reader: %v", err)
	}

	records, err := m.Evaluate()
	if err != nil {
		t.Fatalf("Evaluate on sm_90: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	if !records[0].IsRace {
		t.Errorf("expected cluster DSMEM read-write conflict to be flagged as a race")
	}
}

func TestHazardMatrix_ClusterHardwarePendingOnSM80(t *testing.T) {
	// Target architecture sm_80 lacks thread block cluster and DSMEM hardware.
	m := NewHazardMatrix("sm_80")
	if err := m.AddAccess(MemoryAccess{
		Scope:     ScopeCluster,
		ThreadID:  0,
		BlockID:   0,
		ClusterID: 0,
		Space:     MemoryDSMEM,
		Offset:    0x200,
		SizeBytes: 4,
		IsWrite:   true,
		Stage:     0,
	}); err != nil {
		t.Fatalf("AddAccess: %v", err)
	}

	_, err := m.Evaluate()
	if err == nil {
		t.Fatalf("Evaluate on sm_80 with DSMEM should report error")
	}
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Errorf("expected ErrHardwarePending on sm_80, got %v", err)
	}
}
