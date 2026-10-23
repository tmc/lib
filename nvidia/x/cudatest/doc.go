// Package cudatest provides owned CUDA devices, hardware requirements, and
// scoped device-memory checks for tests.
//
// This package is a test-support mechanism, not a production CUDA runtime.
// Several NVIDIA packages use it from their hardware tests. A skipped
// RequireOwned call establishes no physical evidence; callers that require
// CUDA execution must check retained test output for an actual run.
//
// # Owned Devices
//
// [RequireOwned] loads the driver, takes the canonical GPU lease, and only
// then initializes the driver and selects a device satisfying the
// requirements. It skips the test when this machine has no driver, no
// devices, or no suitable device:
//
//	func TestMyKernel(t *testing.T) {
//		// admission bounds the lease wait and the checks after it,
//		// not driver calls or cleanup.
//		admission, cancel := context.WithTimeout(t.Context(), 30*time.Second)
//		defer cancel()
//		dev, owner := cudatest.RequireOwned(admission, t,
//			cudatest.MinCompute(8, 0),
//			cudatest.MinVRAM(4*1024*1024*1024),
//		)
//		stream, err := dev.NewStream()
//		owner.DeferStream(stream)
//		if err != nil {
//			t.Fatal(err)
//		}
//		g, err := cudagraph.New(dev)
//		if g != nil {
//			owner.Defer(g.Destroy)
//		}
//		if err != nil {
//			t.Fatal(err)
//		}
//		// ... test implementation ...
//	}
//
// The device is borrowed from its [DeviceOwner], which holds the device and
// the lease. Do not close the device or release the lease directly. Register
// every resource created on the device with the owner, including a non-nil
// result returned with an error, before checking that error. The owner is
// closed by a single cleanup registered with the test: streams first, then
// other resources in reverse registration order, then the device, then the
// lease. A release that fails stops the cleanup and fails the test, and
// everything not yet released is kept rather than released out of order.
//
// [Require] and [RequireDevice] remain for existing callers. They do not own
// what they open; see their documentation.
//
// The canonical lease has a known gap: when [cudalock.Acquire] fails to close
// its namespace guard, it loses the handle, and the returned error carries no
// [io.Closer] from which an owner could release the lease.
//
// # Memory Checks
//
// [BeginDeviceMemoryCheck] starts a check that compares process-global free
// device memory around a serialized scope. It does not prove that contexts,
// modules, streams, events, host memory, or allocations outside the CUDA
// context were released.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudatest
