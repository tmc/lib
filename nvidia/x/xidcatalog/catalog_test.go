package xidcatalog

import "testing"

func TestEveryCatalogEntry(t *testing.T) {
	if len(catalog) != 172 {
		t.Fatalf("catalog rows = %d, want 172", len(catalog))
	}
	for id, r := range catalog {
		if id != r.entry.Code || r.entry.Name == "" || r.entry.RecoveryActionEvent != (id == 154) {
			t.Fatalf("invalid row %d: %+v", id, r)
		}
		for j, arch := range []Architecture{A100, H100, B100, GB200} {
			got, ok := Lookup(id, arch)
			want := r.products&(1<<j) != 0
			if ok != want || ok && got != r.entry {
				t.Fatalf("Lookup(%d, %s) = %+v, %v", id, arch, got, ok)
			}
		}
	}
}

func TestCatalogActionsAndUnknowns(t *testing.T) {
	for _, tt := range []struct {
		id                uint64
		arch              Architecture
		action            string
		recovery, present bool
	}{
		{13, H100, "RESTART_APP", false, true},
		{46, A100, "RESET_GPU", false, true},
		{150, B100, "WORKFLOW_NVLINK5_ERR", false, true},
		{150, H100, "", false, false},
		{154, GB200, "XID_154", true, true},
		{172, H100, "", false, true},
		{0, H100, "", false, false},
		{999, H100, "", false, false},
		{13, "GB10", "", false, false},
		{13, "Hopper", "", false, false},
	} {
		got, ok := Lookup(tt.id, tt.arch)
		if ok != tt.present || got.ImmediateAction != tt.action || got.RecoveryActionEvent != tt.recovery {
			t.Fatalf("Lookup(%d,%s) = %+v, %v", tt.id, tt.arch, got, ok)
		}
	}
}
