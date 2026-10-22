//go:build !windows

package moonep

import (
	"reflect"
	"testing"
)

func TestSymmBufferManager(t *testing.T) {
	mgr := NewSymmBufferManager()

	specs := []SymmBufferKey{
		{GroupName: "ep_small", MaxBytes: 4096, AlignBytes: 256},
		{GroupName: "ep_large", MaxBytes: 65536, AlignBytes: 512},
		{GroupName: "ep_small", MaxBytes: 8192, AlignBytes: 256}, // high-water update for ep_small
		{GroupName: "cp_group", MaxBytes: 32768, AlignBytes: 256},
	}

	planned, err := mgr.PlanAllocations(specs)
	if err != nil {
		t.Fatalf("PlanAllocations unexpected error: %v", err)
	}
	if len(planned) != 3 {
		t.Fatalf("len(planned) = %d, want 3", len(planned))
	}

	// Verify largest-first ordering: ep_large (65536), cp_group (32768), ep_small (8192).
	wantOrder := []SymmBufferKey{
		{GroupName: "ep_large", MaxBytes: 65536, AlignBytes: 512},
		{GroupName: "cp_group", MaxBytes: 32768, AlignBytes: 256},
		{GroupName: "ep_small", MaxBytes: 8192, AlignBytes: 256},
	}
	if !reflect.DeepEqual(planned, wantOrder) {
		t.Fatalf("PlanAllocations = %+v, want %+v", planned, wantOrder)
	}

	// Valid zero-allocation sub-view and cache reuse.
	view1, err := mgr.GetSymmView("ep_small", []int{64, 64}, 2)
	if err != nil {
		t.Fatalf("GetSymmView unexpected error: %v", err)
	}
	if view1.ByteOffset != 0 || view1.ElemBytes != 2 || !reflect.DeepEqual(view1.Shape, []int{64, 64}) {
		t.Fatalf("unexpected view1: %+v", view1)
	}
	// Mutating returned slice must not corrupt cached view.
	view1.Shape[0] = 9999
	view2, err := mgr.GetSymmView("ep_small", []int{64, 64}, 2)
	if err != nil {
		t.Fatalf("second GetSymmView error: %v", err)
	}
	if !reflect.DeepEqual(view2.Shape, []int{64, 64}) {
		t.Fatalf("cached view shape corrupted: %v", view2.Shape)
	}

	// Exceeding MaxBytes fails closed.
	if _, err := mgr.GetSymmView("ep_small", []int{128, 64}, 2); err == nil {
		t.Fatalf("expected error when view (16384 B) exceeds MaxBytes (8192 B)")
	}
	// Unknown group fails closed.
	if _, err := mgr.GetSymmView("nonexistent", []int{16}, 2); err == nil {
		t.Fatalf("expected error for unknown group")
	}
}

func TestPartitionSMBudget(t *testing.T) {
	tests := []struct {
		name     string
		totalSMs int
		commSMs  int
		wantComm int
		wantComp int
		wantErr  bool
	}{
		{name: "b200_default_24", totalSMs: 148, commSMs: 0, wantComm: 24, wantComp: 124},
		{name: "h800_explicit_24", totalSMs: 132, commSMs: 24, wantComm: 24, wantComp: 108},
		{name: "custom_even_16", totalSMs: 80, commSMs: 16, wantComm: 16, wantComp: 64},
		{name: "odd_comm_sms_rejected", totalSMs: 148, commSMs: 25, wantErr: true},
		{name: "comm_exceeds_total_rejected", totalSMs: 24, commSMs: 24, wantErr: true},
		{name: "too_few_total_sms", totalSMs: 2, commSMs: 2, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			comm, comp, err := PartitionSMBudget(tc.totalSMs, tc.commSMs)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got (%d, %d)", comm, comp)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if comm != tc.wantComm || comp != tc.wantComp {
				t.Fatalf("PartitionSMBudget(%d, %d) = (%d, %d), want (%d, %d)",
					tc.totalSMs, tc.commSMs, comm, comp, tc.wantComm, tc.wantComp)
			}
		})
	}
}
