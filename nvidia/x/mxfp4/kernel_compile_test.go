package mxfp4

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

// TestKernelsCompile runs the embedded CUDA source through nvrtc. The package's
// other value-level test is a quantize/dequantize round trip, which exercises
// no CUDA at all, so before this gate the kernels were only compiled when a
// device happened to be present.
//
// The source is unexported, so this test lives in the package.
func TestKernelsCompile(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	if _, err := cudatest.CompileSource(mxfp4KernelsCU, major, minor); err != nil {
		t.Fatalf("mxfp4 kernels do not compile: %v", err)
	}
}
