package cudatest_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

// TestRequireFilters checks that Require applies its requirements and hands
// back a device that satisfies them.
//
// It used to wrap the body in a zero-tolerance BeginDeviceMemoryCheck, which
// could not test what it appeared to. The baseline is captured after Require
// has already returned, and nothing between the baseline and the deferred
// Check allocates or frees anything, so the assertion reduced to "process-wide
// free memory did not fall between two adjacent samples". No leak in the code
// it names could reach it. What could reach it was any other process touching
// the GPU: under `go test ./x/...`, which runs packages in parallel, it failed
// about two runs in three, reporting several megabytes of another package's
// allocations as growth. The assertion is gone rather than relocated, because
// there was nothing in it to keep -- BeginDeviceMemoryCheck is covered by
// TestDeviceMemoryCheckCUDA, which is gated for exactly this reason.
func TestRequireFilters(t *testing.T) {
	const wantMajor, wantMinor = 7, 0
	dev := cudatest.Require(t, cudatest.MinCompute(wantMajor, wantMinor))
	if dev == nil {
		t.Fatal("Require returned a nil device without skipping")
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major < wantMajor || (major == wantMajor && minor < wantMinor) {
		t.Errorf("Require returned device %d with sm_%d%d, below the requested sm_%d%d",
			dev.Ordinal(), major, minor, wantMajor, wantMinor)
	}
	t.Logf("Acquired device %d: sm_%d%d", dev.Ordinal(), major, minor)
}
