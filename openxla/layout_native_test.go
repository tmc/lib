package openxla

import (
	"os"
	"testing"
	"unsafe"
)

func TestNativeLayoutMetadata(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN")
	}
	plugin, err := OpenPlugin(path)
	if err != nil {
		t.Fatal(err)
	}
	client, err := plugin.NewClient()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := client.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	devices, err := client.AddressableDevices()
	if err != nil || len(devices) == 0 {
		t.Fatalf("devices=%v error=%v", devices, err)
	}
	data := make([]float32, 6)
	b, err := client.BufferFromHost(unsafe.Pointer(&data[0]), Buffer_Type_F32, []int64{2, 3}, devices[0])
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	for _, initialized := range []bool{false, true} {
		args := Buffer_GetMemoryLayout_Args{Struct_size: PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE, Buffer: b.buffer}
		if initialized {
			args.Layout.Struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE
			args.Layout.Union_pad.Struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE
		}
		if err := decodeError(b.api, callPJRT(b.api.Buffer_GetMemoryLayout, uintptr(unsafe.Pointer(&args)))); err != nil {
			t.Fatal(err)
		}
		layout := args.Layout
		t.Logf("initialized=%v outer_size=%d inner_size=%d kind=%d axes_count=%d tiles=%d", initialized, layout.Struct_size, layout.Union_pad.Struct_size, layout.Type, layout.Union_pad.Minor_to_major_size, layout.Union_pad.Num_tiles)
		if layout.Type != Buffer_MemoryLayout_Type_Tiled || layout.Union_pad.Minor_to_major_size != 2 || layout.Union_pad.Minor_to_major == nil {
			t.Fatal("unexpected layout")
		}
		t.Logf("axes=%v", unsafe.Slice(layout.Union_pad.Minor_to_major, 2))
	}
	if dense, err := b.IsDenseRowMajor(); err != nil || !dense {
		t.Fatalf("native dense layout=%v error=%v", dense, err)
	}

}
