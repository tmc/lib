package openxla

import (
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

func TestDenseRowMajor(t *testing.T) {
	for _, tt := range []struct {
		name                                  string
		rank                                  int
		axes                                  []int64
		tiles                                 uintptr
		badHeader, missingAxes, want, wantErr bool
	}{
		{name: "scalar", want: true},
		{name: "matrix", rank: 2, axes: []int64{1, 0}, want: true},
		{name: "column_major", rank: 2, axes: []int64{0, 1}},
		{name: "tiled", rank: 2, axes: []int64{1, 0}, tiles: 1},
		{name: "rank_mismatch", rank: 2, axes: []int64{0}, wantErr: true},
		{name: "missing_axes", rank: 2, axes: []int64{1, 0}, missingAxes: true, wantErr: true},
		{name: "truncated", badHeader: true, wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			api := new(Api)
			raw := &Buffer{}
			dims := make([]int64, tt.rank)
			api.Buffer_ElementType = purego.NewCallback(func(a *Buffer_ElementType_Args) *ErrorHandle { a.Type = Buffer_Type_F32; return nil })
			api.Buffer_Dimensions = purego.NewCallback(func(a *Buffer_Dimensions_Args) *ErrorHandle {
				a.Dims = unsafe.SliceData(dims)
				a.Num_dims = uintptr(len(dims))
				return nil
			})
			api.Buffer_GetMemoryLayout = purego.NewCallback(func(a *Buffer_GetMemoryLayout_Args) *ErrorHandle {
				if a.Buffer != raw || a.Struct_size != PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE {
					t.Error("invalid layout ABI arguments")
				}
				a.Layout = Buffer_MemoryLayout{Struct_size: PJRT_Buffer_MemoryLayout_STRUCT_SIZE, Type: Buffer_MemoryLayout_Type_Tiled}
				a.Layout.Union_pad = Buffer_MemoryLayout_Tiled{Struct_size: PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE, Minor_to_major: unsafe.SliceData(tt.axes), Minor_to_major_size: uintptr(len(tt.axes)), Num_tiles: tt.tiles}
				if tt.badHeader {
					a.Layout.Struct_size = 0
				}
				if tt.missingAxes {
					a.Layout.Union_pad.Minor_to_major = nil
				}
				return nil
			})
			b := &BufferWrapper{api: api, buffer: raw}
			got, err := b.IsDenseRowMajor()
			if got != tt.want || (err != nil) != tt.wantErr {
				t.Fatalf("got=%v err=%v", got, err)
			}
		})
	}
}

func TestDeviceStorageSize(t *testing.T) {
	api := new(Api)
	raw := &Buffer{}
	api.Buffer_OnDeviceSizeInBytes = purego.NewCallback(func(a *Buffer_OnDeviceSizeInBytes_Args) *ErrorHandle {
		if a.Buffer != raw || a.Struct_size != PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE {
			t.Error("invalid size ABI arguments")
		}
		a.On_device_size_in_bytes = 24
		return nil
	})
	b := &BufferWrapper{api: api, buffer: raw}
	if got, err := b.OnDeviceSizeInBytes(); got != 24 || err != nil {
		t.Fatalf("size=%d err=%v", got, err)
	}
	api.Buffer_OnDeviceSizeInBytes = 0
	if _, err := b.OnDeviceSizeInBytes(); err == nil {
		t.Fatal("missing size slot succeeded")
	}
	if _, err := b.IsDenseRowMajor(); err == nil {
		t.Fatal("missing layout slot succeeded")
	}
	b = nil
	if _, err := b.OnDeviceSizeInBytes(); err == nil {
		t.Fatal("nil size succeeded")
	}
	if _, err := b.IsDenseRowMajor(); err == nil {
		t.Fatal("nil layout succeeded")
	}
}
