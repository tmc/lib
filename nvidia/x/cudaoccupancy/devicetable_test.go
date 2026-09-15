package cudaoccupancy_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

// The DeviceProperties in arch.go are hand-entered from datasheets. Where a
// real device has been probed, the two must agree -- and until this test
// existed, nothing checked that they did. Two tables of the same hardware
// facts maintained in different files can disagree indefinitely, because a
// wrong capacity produces a plausible occupancy number rather than an error.
//
// This is a cross-check, not a source of truth: cudadevicetable holds only
// what somebody measured, so the overlap is small and grows as devices are
// probed. The test fails if the overlap is empty, because a comparison with
// nothing in it passes for the wrong reason.
var probedFields = []struct {
	attr string                                     // key in cudadevicetable
	get  func(cudaoccupancy.DeviceProperties) int64 // the hand-entered counterpart
}{
	{"warpSize", func(d cudaoccupancy.DeviceProperties) int64 { return int64(d.WarpSize) }},
	{"maxThreadsPerBlock", func(d cudaoccupancy.DeviceProperties) int64 { return int64(d.MaxThreadsPerBlock) }},
	{"maxBlocksPerMultiprocessor", func(d cudaoccupancy.DeviceProperties) int64 { return int64(d.MaxBlocksPerSM) }},
	{"maxRegistersPerMultiprocessor", func(d cudaoccupancy.DeviceProperties) int64 { return int64(d.MaxRegistersPerSM) }},
	{"maxSharedMemoryPerMultiprocessor", func(d cudaoccupancy.DeviceProperties) int64 { return int64(d.MaxSharedMemoryPerSM) }},
	{"maxSharedMemoryPerBlock", func(d cudaoccupancy.DeviceProperties) int64 { return int64(d.MaxSharedMemoryPerBlock) }},
	{"maxSharedMemoryPerBlockOptin", func(d cudaoccupancy.DeviceProperties) int64 { return int64(d.MaxSharedMemoryOptin) }},
}

func TestHandTableAgreesWithProbedDevices(t *testing.T) {
	compared := 0
	for _, arch := range cudadevicetable.Architectures() {
		rec, ok := cudadevicetable.Lookup(arch)
		if !ok {
			t.Fatalf("%s listed but not found", arch)
		}
		hand, err := cudaoccupancy.LookupDevice(arch)
		if err != nil {
			t.Errorf("%s has been probed but cudaoccupancy does not know it: %v", arch, err)
			continue
		}
		for _, f := range probedFields {
			want, err := rec.ProbedAttr(f.attr)
			if err != nil {
				continue // not probed on this device; nothing to check against
			}
			compared++
			if got := f.get(hand); got != want {
				t.Errorf("%s %s: cudaoccupancy has %d, the device reports %d",
					arch, f.attr, got, want)
			}
		}
	}
	if compared == 0 {
		t.Fatal("no hand-entered value was checked against a probed one: either " +
			"devices.json has no probed attributes that probedFields names, or the " +
			"names have drifted apart. An empty comparison is not a passing one.")
	}
	t.Logf("checked %d hand-entered values against probed readings", compared)
}

// TestDerivedWarpsPerSMAgreesWithProbe checks the one hand-entered field that
// has no attribute of its own. MaxWarpsPerSM is not a device attribute; it is
// maxThreadsPerMultiprocessor divided by warpSize, and the division is the
// only thing that can confirm the hand-entered number.
func TestDerivedWarpsPerSMAgreesWithProbe(t *testing.T) {
	checked := 0
	for _, arch := range cudadevicetable.Architectures() {
		rec, ok := cudadevicetable.Lookup(arch)
		if !ok {
			continue
		}
		threads, err := rec.ProbedAttr("maxThreadsPerMultiprocessor")
		if err != nil {
			continue
		}
		warp, err := rec.ProbedAttr("warpSize")
		if err != nil || warp == 0 {
			continue
		}
		hand, err := cudaoccupancy.LookupDevice(arch)
		if err != nil {
			continue
		}
		checked++
		if want := threads / warp; int64(hand.MaxWarpsPerSM) != want {
			t.Errorf("%s MaxWarpsPerSM: cudaoccupancy has %d, the device implies %d (%d threads/SM / %d per warp)",
				arch, hand.MaxWarpsPerSM, want, threads, warp)
		}
	}
	if checked == 0 {
		t.Skip("no probed device reports both maxThreadsPerMultiprocessor and warpSize")
	}
}

// TestProbedArchitectureIsNotAliasedToAnother guards a defect found by
// inspection: knownArchitectures maps "sm_120" to the SM121 struct, whose
// Architecture field reads "sm_121". A caller asking about one architecture
// gets a report stamped with another, and nothing in the value says so.
func TestProbedArchitectureIsNotAliasedToAnother(t *testing.T) {
	for _, arch := range cudadevicetable.Architectures() {
		hand, err := cudaoccupancy.LookupDevice(arch)
		if err != nil {
			continue
		}
		if got := cudadevicetable.Normalize(hand.Architecture); got != arch {
			t.Errorf("LookupDevice(%q) returned properties stamped %q; a record that "+
				"misreports its own architecture cannot be attributed by a reader",
				arch, hand.Architecture)
		}
	}
}
