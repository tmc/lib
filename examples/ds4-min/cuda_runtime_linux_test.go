//go:build linux

package main

import (
	"os"
	"testing"
)

func TestCUDAFoundation(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if len(r.model) != pinnedModelBytes {
		t.Fatalf("mapped bytes = %d, want %d", len(r.model), pinnedModelBytes)
	}
	if r.name == "" || r.major == 0 {
		t.Fatalf("invalid CUDA identity %q sm_%d%d", r.name, r.major, r.minor)
	}
	if r.mapped.host == 0 || r.mapped.device == 0 || r.mapped.bytes == 0 {
		t.Fatalf("model range was not registered: %+v", r.mapped)
	}
	if r.mapped.bytes > cudaMaxRegisteredModelRange+uint(os.Getpagesize()) {
		t.Fatalf("registered bytes = %d, want a bounded range", r.mapped.bytes)
	}
	if r.probeTensor == "" || r.probeDevice != r.probeHost {
		t.Fatalf("CUDA read of %q = %#04x, host = %#04x", r.probeTensor, r.probeDevice, r.probeHost)
	}
	t.Logf("CUDA read of %q = host %#04x, device %#04x; registered %d bytes", r.probeTensor, r.probeHost, r.probeDevice, r.mapped.bytes)
}
