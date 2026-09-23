package cudalayer_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayer"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// TestTemplateCompiles runs the embedded kernel through nvrtc. The package's
// only other test covers SharedMemoryBytes configuration validation, so before
// this gate the kernel source had no test of any kind.
//
// It needs the toolkit but no GPU.
func TestTemplateCompiles(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	if _, err := cudatest.CompileSource(cudalayer.ProjectionResidualRMSNormTemplate, major, minor); err != nil {
		t.Fatalf("ProjectionResidualRMSNormTemplate does not compile: %v", err)
	}
}
