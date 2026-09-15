package cudaoccupancy

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// TestEveryAliasNamesItsOwnArchitecture pins the one rule that makes
// knownArchitectures safe to extend: a key may only reach a struct describing
// the same architecture. A feature suffix is fine, because sm_90a and sm_90
// are the same SM geometry with a different instruction set; a different chip
// is not, because nothing in the returned value records the substitution.
//
// Without this rule the map is a place to record a guess. LookupDevice
// reports no error for a key that is present, so a key pointing at a
// neighbouring architecture turns "we have no data for this part" into a
// confident wrong answer: Calculate divides by MaxRegistersPerSM,
// MaxSharedMemoryPerSM and MaxWarpsPerSM and emits both an occupancy
// percentage and a LimitingFactor attributing it. A missing key is a returned
// error the caller must handle; a wrong key is a plausible number.
func TestEveryAliasNamesItsOwnArchitecture(t *testing.T) {
	for key, props := range knownArchitectures {
		want := cudadevicetable.Normalize(key)
		switch key {
		case "gb10":
			want = "sm_121"
		case "gb200":
			want = "sm_100"
		}
		got := cudadevicetable.Normalize(props.Architecture)
		if got != want {
			t.Errorf("knownArchitectures[%q] holds properties for %q: a lookup of "+
				"%s returns another architecture's geometry and reports no error",
				key, props.Architecture, want)
		}
	}
}

// TestUnknownArchitectureIsAnError pins the fail-closed half of the same rule:
// an architecture this package has no measurement for must return an error
// rather than the nearest entry. See x/cuptirange/layout.go for the same
// stance on CUPTI struct layouts, where a guess is a native memory error.
func TestUnknownArchitectureIsAnError(t *testing.T) {
	for _, arch := range []string{"sm_120", "sm120"} {
		if _, err := LookupDevice(arch); err == nil {
			t.Errorf("LookupDevice(%q) succeeded; this package has no measurement "+
				"for that part and must say so rather than substitute a neighbour", arch)
		}
	}
}
