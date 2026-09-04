// Package cudatest provides CUDA hardware requirements and scoped device-memory
// checks for tests.
//
// # Hardware Capability Filters
//
// Tests using GPU hardware can use [Require] to cleanly skip tests when running
// in CI or on machines without the required compute capability or VRAM:
//
//	func TestMyKernel(t *testing.T) {
//		dev := cudatest.Require(t,
//			cudatest.MinCompute(8, 0),
//			cudatest.MinVRAM(4 * 1024 * 1024 * 1024),
//		)
//		check, err := cudatest.BeginDeviceMemoryCheck(dev, 0)
//		if err != nil {
//			t.Fatal(err)
//		}
//		defer func() {
//			if err := check.Check(); err != nil {
//				t.Error(err)
//			}
//		}()
//		// ... test implementation ...
//	}
//
// A memory check compares process-global free device memory around a serialized
// scope. It does not prove that contexts, modules, streams, events, host memory,
// or allocations outside the CUDA context were released.
package cudatest
