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

type mockDeviceSlice struct {
	ptr driver.DevicePtr
	n   int
}

func (m mockDeviceSlice) Ptr() driver.DevicePtr { return m.ptr }
func (m mockDeviceSlice) Len() int              { return m.n }

func TestPackArgsNullSlice(t *testing.T) {
	tests := []struct {
		name string
		arg  any
		want driver.DevicePtr
	}{
		{"zero slice", mockDeviceSlice{}, 0},
		{"empty subslice of live buffer", mockDeviceSlice{ptr: 0x1000, n: 0}, 0},
		{"populated slice", mockDeviceSlice{ptr: 0x1000, n: 8}, 0x1000},
		{"plain provider", mockSlice{ptr: 0x2000}, 0x2000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params, cleanup, err := cudalaunch.PackArgs(tt.arg)
			if err != nil {
				t.Fatalf("PackArgs(%v) failed: %v", tt.arg, err)
			}
			defer cleanup()
			if got := *(*driver.DevicePtr)(params[0]); got != tt.want {
				t.Errorf("PackArgs(%v) packed 0x%x, want 0x%x", tt.arg, got, tt.want)
			}
		})
	}
}

func TestLaunchAttributes(t *testing.T) {
	pdlAttr := cudalaunch.ProgrammaticStreamSerializationAttr(true)
	if pdlAttr.Id != driver.LaunchAttributeProgrammaticStreamSerialization {
		t.Errorf("Id = %v, want %v", pdlAttr.Id, driver.LaunchAttributeProgrammaticStreamSerialization)
	}
	if pdlAttr.Value[0] != 1 {
		t.Errorf("Value[0] = %d, want 1", pdlAttr.Value[0])
	}

	pdlOff := cudalaunch.ProgrammaticStreamSerializationAttr(false)
	if pdlOff.Value[0] != 0 {
		t.Errorf("Value[0] = %d, want 0", pdlOff.Value[0])
	}

	coopAttr := cudalaunch.CooperativeAttr(true)
	if coopAttr.Id != driver.LaunchAttributeCooperative {
		t.Errorf("Id = %v, want %v", coopAttr.Id, driver.LaunchAttributeCooperative)
	}
	if coopAttr.Value[0] != 1 {
		t.Errorf("Value[0] = %d, want 1", coopAttr.Value[0])
	}

	var mockFn driver.Function = 42
	l := cudalaunch.Kernel(mockFn).
		Grid(64, 2).
		Block(128).
		WithProgrammaticStreamSerialization(true).
		WithCooperative(true)

	attrs := l.Attrs()
	if len(attrs) != 2 {
		t.Fatalf("Attrs len = %d, want 2", len(attrs))
	}
	if attrs[0].Id != driver.LaunchAttributeProgrammaticStreamSerialization {
		t.Errorf("attrs[0].Id = %v, want %v", attrs[0].Id, driver.LaunchAttributeProgrammaticStreamSerialization)
	}
	if attrs[1].Id != driver.LaunchAttributeCooperative {
		t.Errorf("attrs[1].Id = %v, want %v", attrs[1].Id, driver.LaunchAttributeCooperative)
	}
}
