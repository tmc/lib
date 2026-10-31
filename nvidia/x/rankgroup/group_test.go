package rankgroup

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestParseMode(t *testing.T) {
	for _, tt := range []struct {
		in   string
		want Mode
		ok   bool
	}{
		{"serial", Serial, true},
		{"streams", Streams, true},
		{"green", Green, true},
		{"mps", 0, false},
	} {
		got, err := ParseMode(tt.in)
		if (err == nil) != tt.ok || got != tt.want {
			t.Errorf("ParseMode(%q) = %v, %v", tt.in, got, err)
		}
	}
}

func TestOnDeviceRanks(t *testing.T) {
	for _, n := range []int{0, MaxRanks + 1} {
		if _, err := OnDevice(nil, n, Serial); err == nil {
			t.Errorf("OnDevice with %d ranks succeeded", n)
		}
	}
	if _, err := OnDevices(make([]*cuda.Device, MaxRanks+1)); err == nil {
		t.Error("OnDevices with too many ranks succeeded")
	}
}

// TestPointersLayout pins the kernel's view of Pointers: MaxRanks 8-byte
// device addresses, as struct rankgroup_ptrs declares.
func TestPointersLayout(t *testing.T) {
	if got := unsafe.Sizeof(Pointers{}); got != 8*MaxRanks {
		t.Errorf("Pointers is %d bytes, want %d", got, 8*MaxRanks)
	}
	if !strings.Contains(deviceAllReduceSource, "const float* p[8];") || MaxRanks != 8 {
		t.Error("kernel pointer struct disagrees with MaxRanks")
	}
}
