package cudakernel_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

// This package holds a third copy of the per-architecture geometry that
// cudaoccupancy and cudadevicetable also hold, and until this file existed
// nothing compared them. Duplicated tables do not diverge loudly: a wrong
// shared-memory capacity yields a plausible launch configuration, so the
// copies can disagree for as long as nobody happens to read both.
//
// The feature flags are no longer duplicated at all -- they are derived from
// the probed table at startup -- so what remains to check is the geometry.

var geometryFields = []struct {
	attr string
	hand func(cudakernel.TargetInfo) int64
	occ  func(cudaoccupancy.DeviceProperties) int64
}{
	{"maxSharedMemoryPerMultiprocessor",
		func(t cudakernel.TargetInfo) int64 { return int64(t.MaxSharedMemoryBytes) },
		func(d cudaoccupancy.DeviceProperties) int64 { return int64(d.MaxSharedMemoryPerSM) }},
	{"maxSharedMemoryPerBlockOptin",
		func(t cudakernel.TargetInfo) int64 { return int64(t.MaxOptInSharedBytes) },
		func(d cudaoccupancy.DeviceProperties) int64 { return int64(d.MaxSharedMemoryOptin) }},
	{"maxBlocksPerMultiprocessor",
		func(t cudakernel.TargetInfo) int64 { return int64(t.MaxBlocksPerSM) },
		func(d cudaoccupancy.DeviceProperties) int64 { return int64(d.MaxBlocksPerSM) }},
}

// TestGeometryAgreesWithProbedDevices compares this package's numbers with the
// device that was actually measured.
func TestGeometryAgreesWithProbedDevices(t *testing.T) {
	compared := 0
	for _, arch := range cudadevicetable.Architectures() {
		rec, _ := cudadevicetable.Lookup(arch)
		for _, name := range []string{arch, arch + "a"} {
			info, err := cudakernel.TargetSM(name)
			if err != nil {
				continue // this package does not claim to know the target
			}
			for _, f := range geometryFields {
				want, err := rec.ProbedAttr(f.attr)
				if err != nil {
					continue
				}
				compared++
				if got := f.hand(info); got != want {
					t.Errorf("%s %s: cudakernel has %d, the device reports %d", name, f.attr, got, want)
				}
			}
		}
	}
	if compared == 0 {
		t.Fatal("no value was checked against a probed one; an empty comparison is not a passing one")
	}
}

// TestGeometryAgreesWithCudaoccupancy compares the two hand-maintained copies
// with each other, which covers the architectures no device here has probed --
// H100 and GB200 among them. It cannot show either is right. It can show they
// are not both right, which is the failure that actually occurred: this
// package and cudapipeline disagreed about wgmma on sm_100a for as long as
// both existed.
func TestGeometryAgreesWithCudaoccupancy(t *testing.T) {
	compared := 0
	for _, name := range []string{"sm_90", "sm_90a", "sm_100", "sm_100a", "sm_121", "sm_121a"} {
		info, err := cudakernel.TargetSM(name)
		if err != nil {
			t.Fatalf("TargetSM(%q): %v", name, err)
		}
		dev, err := cudaoccupancy.LookupDevice(name)
		if err != nil {
			t.Errorf("cudakernel knows %s but cudaoccupancy does not: %v", name, err)
			continue
		}
		for _, f := range geometryFields {
			compared++
			if got, want := f.hand(info), f.occ(dev); got != want {
				t.Errorf("%s %s: cudakernel has %d, cudaoccupancy has %d", name, f.attr, got, want)
			}
		}
		if got, want := int64(info.WarpsPerSM), int64(dev.MaxWarpsPerSM); got != want {
			t.Errorf("%s warps per SM: cudakernel has %d, cudaoccupancy has %d", name, got, want)
		}
		compared++
	}
	if compared == 0 {
		t.Fatal("no field was compared")
	}
}

// TestFeatureFlagsCarryTheirSource catches the derivation silently failing.
// If the probed table stopped resolving, every Supports field would read false
// -- a plausible answer that happens to be wrong for every target -- and
// nothing else would notice.
func TestFeatureFlagsCarryTheirSource(t *testing.T) {
	for _, name := range []string{"sm_90", "sm_90a", "sm_100", "sm_100a", "sm_121", "sm_121a"} {
		info, err := cudakernel.TargetSM(name)
		if err != nil {
			t.Fatalf("TargetSM(%q): %v", name, err)
		}
		if info.FeatureSource == "" {
			t.Errorf("%s: capability flags carry no source; the derivation did not run", name)
		}
	}
	// The derived answers must still discriminate: wgmma on the Hopper
	// suffixed target and nowhere else among these.
	wgmma := map[string]bool{}
	for _, name := range []string{"sm_90", "sm_90a", "sm_100a", "sm_121a"} {
		info, _ := cudakernel.TargetSM(name)
		wgmma[name] = info.SupportsWGMMA
	}
	if !wgmma["sm_90a"] {
		t.Error("sm_90a should support wgmma")
	}
	for _, name := range []string{"sm_90", "sm_100a", "sm_121a"} {
		if wgmma[name] {
			t.Errorf("%s should not support wgmma: ptxas rejects it there", name)
		}
	}
}
