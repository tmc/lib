//go:build linux || darwin

package nvtelemetry

import (
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/internal/abistub"
	"github.com/tmc/lib/nvidia/nvml"
)

// The device handle selects the architecture returned by this CPU fixture.
// Optional metadata symbols are absent, just as on an older NVML library.
const architectureStub = `
#include <stdint.h>
int nvmlDeviceGetArchitecture(void *device, uint32_t *arch) {
  *arch = (uint32_t)(uintptr_t)device;
  return 0;
}
`

func TestNVMLArchitectureMetadata(t *testing.T) {
	path := os.Getenv("NVTELEMETRY_ARCH_STUB")
	if path == "" {
		abistub.Child(t, t.Name(), "NVTELEMETRY_ARCH_STUB", abistub.Build(t, architectureStub))
		return
	}
	if err := nvml.Load(path); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := nvml.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, tt := range []struct {
		name string
		arch uint32
		want string
	}{
		{"Hopper", 9, "0x9"},
		{"Blackwell", 10, "0xa"},
		{"unrecognized", 42, "0x2a"},
		{"unknown", 0xffffffff, "0xffffffff"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			sampler, err := NewNVMLSampler(nil, nvml.Device(tt.arch))
			if err != nil {
				t.Fatal(err)
			}
			if got := sampler.Devices()[0].Architecture; got != tt.want {
				t.Fatalf("architecture = %q, want %q", got, tt.want)
			}
		})
	}
}
