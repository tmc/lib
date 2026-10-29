package tcgen05

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// TestMaxDynamicSharedAgreesWithProfiles pins maxDynamicShared and checks
// that every tcgen05 target the device table profiles shares sm_100a's
// limit, which is the one the schedules here are checked against.
func TestMaxDynamicSharedAgreesWithProfiles(t *testing.T) {
	if maxDynamicShared != 227*1024 {
		t.Errorf("maxDynamicShared = %d, want %d", maxDynamicShared, 227*1024)
	}
	for _, target := range []string{"sm_100a", "sm_103a"} {
		for _, p := range cudadevicetable.TargetProfiles(target) {
			if got := int(p.SharedPerBlockOptin.Value); got != maxDynamicShared {
				t.Errorf("%s (%s) opt-in shared memory = %d, want %d", target, p.Product, got, maxDynamicShared)
			}
		}
	}
}
