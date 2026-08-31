// Package cudatest provides test harness utilities, hardware capability assertions,
// and GPU resource leak detection for CUDA Go testing.
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
//		defer cudatest.VerifyNoLeaks(t)
//		// ... test implementation ...
//	}
package cudatest
