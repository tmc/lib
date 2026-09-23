package cudasample_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// TestTemplateCompiles runs the embedded kernel through nvrtc, so that a
// syntax or type error is reported without a device. The package's value-level
// coverage, which compares GPU argmax against SampleArgmaxCPU, needs one.
func TestTemplateCompiles(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	if _, err := cudatest.CompileSource(cudasample.ArgmaxKernelTemplate, major, minor); err != nil {
		t.Fatalf("ArgmaxKernelTemplate does not compile: %v", err)
	}
}
