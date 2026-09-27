package cudalaunch_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

type mockSlice struct {
	ptr driver.DevicePtr
}

func (m mockSlice) Ptr() driver.DevicePtr {
	return m.ptr
}

func TestLauncherBuilder(t *testing.T) {
	var mockFn driver.Function = 1234
	l := cudalaunch.Kernel(mockFn).
		Grid(128, 2, 1).
		Block(256, 1, 1).
		SharedMem(2048)

	if l == nil {
		t.Fatal("expected non-nil launcher")
	}
}

func TestResolveArgsAndConfigZero(t *testing.T) {
	shape, _ := cudashape.New(16, 32, 64)
	tile, _ := cudashape.New(2, 4, 8)
	p, err := cudalayout.NewPartition(1, shape, tile, 4, cudalayout.WithPartitionStrides([]int{2048, 64, 1}))
	if err != nil {
		t.Fatal(err)
	}
	l := cudalaunch.Kernel(123).SharedMem(4096)
	if _, err := l.ResolveArgs(p); err != nil {
		t.Fatal(err)
	}
	d, err := l.Describe()
	if err != nil {
		t.Fatal(err)
	}
	if d.Grid != driver.Dims(8, 8, 8) {
		t.Fatalf("grid = %v, want (8,8,8)", d.Grid)
	}
	l.Config(cudalaunch.Config{})
	d, err = l.Describe()
	if err != nil {
		t.Fatal(err)
	}
	if d.SharedMem != 4096 {
		t.Fatalf("shared memory = %d, want 4096", d.SharedMem)
	}
	if _, err := cudalaunch.Kernel(123).Grid(1, 1, 1).ResolveArgs(p); !errors.Is(err, cudalaunch.ErrGridMismatch) {
		t.Fatalf("ResolveArgs grid error = %v, want ErrGridMismatch", err)
	}
	if _, err := cudalaunch.Kernel(123).WithClusterDim(driver.Dims(3, 1, 1)).ResolveArgs(p); !errors.Is(err, cudalaunch.ErrClusterGridMismatch) {
		t.Fatalf("ResolveArgs cluster error = %v, want ErrClusterGridMismatch", err)
	}
}

type Float4 struct {
	X, Y, Z, W float32
}

func TestPackArgs16ByteAlignment(t *testing.T) {
	// Call PackArgs directly via Exported/Internal helper
	f4 := Float4{X: 1.0, Y: 2.0, Z: 3.0, W: 4.0}
	params, err := cudalaunch.PackArgs(f4, int32(42), uint64(100))
	if err != nil {
		t.Fatalf("PackArgs failed: %v", err)
	}

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
			params, err := cudalaunch.PackArgs(tt.arg)
			if err != nil {
				t.Fatalf("PackArgs(%v) failed: %v", tt.arg, err)
			}
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
	if v := *pdlAttr.Value.ProgrammaticStreamSerializationAllowed(); v != 1 {
		t.Errorf("ProgrammaticStreamSerializationAllowed = %d, want 1", v)
	}

	pdlOff := cudalaunch.ProgrammaticStreamSerializationAttr(false)
	if v := *pdlOff.Value.ProgrammaticStreamSerializationAllowed(); v != 0 {
		t.Errorf("ProgrammaticStreamSerializationAllowed = %d, want 0", v)
	}

	coopAttr := cudalaunch.CooperativeAttr(true)
	if coopAttr.Id != driver.LaunchAttributeCooperative {
		t.Errorf("Id = %v, want %v", coopAttr.Id, driver.LaunchAttributeCooperative)
	}
	if v := *coopAttr.Value.Cooperative(); v != 1 {
		t.Errorf("Cooperative = %d, want 1", v)
	}

	var mockFn driver.Function = 42
	l := cudalaunch.Kernel(mockFn).
		Grid(64, 2, 1).
		Block(128, 1, 1).
		WithProgrammaticStreamSerialization(true).
		WithCooperative(true).
		WithClusterDim(driver.Dims(2, 2, 1))

	attrs := l.Attrs()
	if len(attrs) != 3 {
		t.Fatalf("Attrs len = %d, want 3", len(attrs))
	}
	if attrs[0].Id != driver.LaunchAttributeProgrammaticStreamSerialization {
		t.Errorf("attrs[0].Id = %v, want %v", attrs[0].Id, driver.LaunchAttributeProgrammaticStreamSerialization)
	}
	if attrs[1].Id != driver.LaunchAttributeCooperative {
		t.Errorf("attrs[1].Id = %v, want %v", attrs[1].Id, driver.LaunchAttributeCooperative)
	}
	if attrs[2].Id != driver.LaunchAttributeClusterDimension {
		t.Errorf("attrs[2].Id = %v, want %v", attrs[2].Id, driver.LaunchAttributeClusterDimension)
	}
}

func TestClusterDimAttr(t *testing.T) {
	attr := cudalaunch.ClusterDimAttr(driver.Dims(2, 4, 1))
	if attr.Id != driver.LaunchAttributeClusterDimension {
		t.Fatalf("attr.Id = %v, want %v", attr.Id, driver.LaunchAttributeClusterDimension)
	}
	if got, want := *attr.Value.ClusterDim(), driver.Dims(2, 4, 1); got != want {
		t.Errorf("ClusterDim = %v, want %v", got, want)
	}
}
