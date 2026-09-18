package cudamem_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// TestChunkedArena checks the three properties a caller of NewChunkedArena
// relies on: sub-allocations are 256-byte aligned, they do not overlap, and they
// come out of one device allocation per block rather than one per call. The last
// is what the arena exists for, and pointer arithmetic is the only way to see it
// from outside the package: consecutive sub-allocations sit at the aligned stride
// their sizes imply, which separate driver allocations would not.
func TestChunkedArena(t *testing.T) {
	if err := driver.Load(""); err != nil {
		t.Skipf("CUDA driver not available: %v", err)
	}
	if err := cuda.Init(); err != nil {
		t.Fatalf("cuInit failed with libcuda loaded from %q: %v", driver.Path(), err)
	}
	count, err := cuda.DeviceCount()
	if err != nil || count == 0 {
		t.Skip("no CUDA devices available")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("NewDevice: %v", err)
	}

	const chunk = 1 << 20
	tests := []struct {
		name   string
		length int // float32 elements
		stride int // bytes to the next sub-allocation
	}{
		{"exact multiple", 64, 256},
		{"rounds up", 1, 256},
		{"several blocks worth", 300, 1280},
		{"rounds up again", 65, 512},
	}

	a := cudamem.NewChunkedArena(dev, chunk)
	defer func() {
		if err := a.Free(); err != nil {
			t.Errorf("Free: %v", err)
		}
	}()

	var got []cudamem.Slice[float32]
	for _, tt := range tests {
		got = append(got, cudamem.AllocIn[float32](a, tt.name, tt.length))
	}
	if err := a.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	for i, tt := range tests {
		s := got[i]
		if s.Len() != tt.length {
			t.Errorf("%s: Len = %d, want %d", tt.name, s.Len(), tt.length)
		}
		if uintptr(s.Ptr())%256 != 0 {
			t.Errorf("%s: pointer %v is not 256-byte aligned", tt.name, s.Ptr())
		}
		if i+1 < len(tests) {
			if d := uintptr(got[i+1].Ptr()) - uintptr(s.Ptr()); d != uintptr(tt.stride) {
				t.Errorf("%s: next sub-allocation is %d bytes on, want %d", tt.name, d, tt.stride)
			}
		}
	}

	// Write every slice before reading any, so an overlap shows up as a
	// neighbor's pattern rather than as one slice's own.
	for i, tt := range tests {
		host := make([]float32, tt.length)
		for j := range host {
			host[j] = float32(i+1)*1e6 + float32(j)
		}
		if err := got[i].CopyFromHost(host); err != nil {
			t.Fatalf("%s: CopyFromHost: %v", tt.name, err)
		}
	}
	for i, tt := range tests {
		host := make([]float32, tt.length)
		if err := got[i].CopyToHost(host); err != nil {
			t.Fatalf("%s: CopyToHost: %v", tt.name, err)
		}
		for j := range host {
			if want := float32(i+1)*1e6 + float32(j); host[j] != want {
				t.Fatalf("%s: element %d is %g, want %g: sub-allocations overlap", tt.name, j, host[j], want)
			}
		}
	}

	// A request larger than the chunk gets a block of its own, so it cannot be
	// at the stride the previous allocation would imply.
	big := cudamem.AllocIn[float32](a, "larger than chunk", chunk)
	if err := a.Err(); err != nil {
		t.Fatalf("Err after oversized request: %v", err)
	}
	if big.Len() != chunk {
		t.Errorf("oversized Len = %d, want %d", big.Len(), chunk)
	}
	if uintptr(big.Ptr())%256 != 0 {
		t.Errorf("oversized pointer %v is not 256-byte aligned", big.Ptr())
	}
}
