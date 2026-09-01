package cudalaunch_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

type mockSlice struct {
	ptr driver.DevicePtr
}

func (m mockSlice) Ptr() driver.DevicePtr {
	return m.ptr
}

func TestDims(t *testing.T) {
	d1 := cudalaunch.Dims(64)
	if d1.X != 64 || d1.Y != 1 || d1.Z != 1 {
		t.Errorf("unexpected Dims(64): %+v", d1)
	}

	d2 := cudalaunch.Dims(64, 4)
	if d2.X != 64 || d2.Y != 4 || d2.Z != 1 {
		t.Errorf("unexpected Dims(64, 4): %+v", d2)
	}

	d3 := cudalaunch.Dims(64, 4, 2)
	if d3.X != 64 || d3.Y != 4 || d3.Z != 2 {
		t.Errorf("unexpected Dims(64, 4, 2): %+v", d3)
	}
}

func TestLauncherBuilder(t *testing.T) {
	var mockFn driver.Function = 1234
	l := cudalaunch.Kernel(mockFn).
		Grid(128, 2).
		Block(256).
		SharedMem(2048)

	if l == nil {
		t.Fatal("expected non-nil launcher")
	}
}

type Float4 struct {
	X, Y, Z, W float32
}

func TestPackArgs16ByteAlignment(t *testing.T) {
	// Call PackArgs directly via Exported/Internal helper
	f4 := Float4{X: 1.0, Y: 2.0, Z: 3.0, W: 4.0}
	params, cleanup, err := cudalaunch.PackArgs(f4, int32(42), uint64(100))
	if err != nil {
		t.Fatalf("PackArgs failed: %v", err)
	}
	defer cleanup()

	if len(params) != 3 {
		t.Fatalf("expected 3 params, got %d", len(params))
	}

	// Verify struct param pointer is strictly 16-byte aligned
	ptrVal := uintptr(params[0])
	if ptrVal%16 != 0 {
		t.Errorf("expected 16-byte aligned struct parameter, got ptr=0x%x (ptr %% 16 = %d)", ptrVal, ptrVal%16)
	}

	// Verify content correctness
	recovered := *(*Float4)(params[0])
	if recovered != f4 {
		t.Errorf("recovered struct %+v != expected %+v", recovered, f4)
	}
}
