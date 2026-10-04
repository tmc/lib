package nvrtcx

import (
	"testing"

	"github.com/tmc/lib/nvidia/nvrtc"
)

func TestHasNVRTCCache(t *testing.T) {
	tests := []struct {
		major, minor int32
		want         bool
	}{
		{11, 8, false},
		{12, 2, false},
		{12, 8, false},
		{12, 9, true},
		{13, 0, true},
	}
	for _, tt := range tests {
		if got := hasNVRTCCache(tt.major, tt.minor); got != tt.want {
			t.Errorf("hasNVRTCCache(%d, %d) = %v, want %v", tt.major, tt.minor, got, tt.want)
		}
	}
}

// TestNoCacheCompiles compiles with WithNoCache on whatever NVRTC is
// installed. Before 12.9 NVRTC rejects --no-cache, so this fails there unless
// Compile withholds the flag from the program it compiles.
func TestNoCacheCompiles(t *testing.T) {
	if err := nvrtc.Load(""); err != nil {
		t.Skipf("no NVRTC: %v", err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("NVRTC %d.%d %s", major, minor, nvrtc.Path())
	if _, err := Compile(`extern "C" __global__ void k(float* p) { p[0] = 1; }`,
		WithFlag("--gpu-architecture=compute_80"), WithNoCache()); err != nil {
		t.Fatal(err)
	}
}
