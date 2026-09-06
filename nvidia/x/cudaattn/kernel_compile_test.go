package cudaattn_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// TestTemplatesCompile runs every embedded kernel through nvrtc. The templates
// are CUDA programs, and until this gate existed no test compiled them: the
// only assertions were that certain names, and in one case a comment, appeared
// in the source. A kernel that does not compile is caught here rather than at
// the first launch on a device.
//
// It needs the toolkit but no GPU.
func TestTemplatesCompile(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	for _, tt := range []struct {
		name string
		src  string
	}{
		{"CausalFlashAttention", cudaattn.CausalFlashAttentionTemplate},
		{"BatchedPagedAttention", cudaattn.BatchedPagedAttentionTemplate},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := cudatest.CompileSource(tt.src, major, minor); err != nil {
				t.Fatalf("%s does not compile: %v", tt.name, err)
			}
		})
	}
}
