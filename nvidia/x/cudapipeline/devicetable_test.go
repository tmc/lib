package cudapipeline

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// TestSM121LimitsAgreeWithDeviceTable checks the SM121 constants against the
// GB10 profile, whose values were probed on a GB10. They stay constants
// because callers use them where a constant is required (an untyped
// operand of int64 fields); this test is what keeps them from being a
// second, unchecked copy of the table.
func TestSM121LimitsAgreeWithDeviceTable(t *testing.T) {
	p, err := cudadevicetable.LookupProfile("GB10")
	if err != nil {
		t.Fatal(err)
	}
	a, ok := cudadevicetable.Lookup("sm_121")
	if !ok {
		t.Fatal("sm_121 not in the device table")
	}
	regs, err := a.ProbedAttr("maxRegistersPerMultiprocessor")
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name  string
		got   int64
		table cudadevicetable.Value
	}{
		{"SM121MaxOptinSharedMemBytes", SM121MaxOptinSharedMemBytes, p.SharedPerBlockOptin},
		{"SM121TotalSharedMemBytes", SM121TotalSharedMemBytes, p.SharedPerSM},
		{"SM121MaxWarpsPerSM", SM121MaxWarpsPerSM, cudadevicetable.Value{Value: p.ThreadsPerSM.Value / ThreadsPerWarp, Source: p.ThreadsPerSM.Source}},
		{"SM121MaxBlocksPerSM", SM121MaxBlocksPerSM, p.BlocksPerSM},
		{"SM121TotalRegsPerSM", SM121TotalRegsPerSM, cudadevicetable.Value{Value: regs, Source: cudadevicetable.Probed}},
	}
	for _, tt := range tests {
		if tt.table.Source != cudadevicetable.Probed {
			t.Errorf("%s: GB10 table value is %s, want probed", tt.name, tt.table.Source)
		}
		if tt.got != tt.table.Value {
			t.Errorf("%s = %d, cudadevicetable has %d", tt.name, tt.got, tt.table.Value)
		}
	}
}
