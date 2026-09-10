package openxla

import (
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

func TestClientWrapperCreateViewOfDeviceBuffer(t *testing.T) {
	api := new(Api)
	client, device, output := &Client{}, &Device{}, &Buffer{}
	c := &ClientWrapper{api: api, client: client}
	data := []byte{1, 2, 3, 4}
	dims := []int64{4}
	layout := &Buffer_MemoryLayout{Struct_size: PJRT_Buffer_MemoryLayout_STRUCT_SIZE}
	api.Buffer_Destroy = purego.NewCallback(func(*Buffer_Destroy_Args) *ErrorHandle { return nil })
	api.Client_CreateViewOfDeviceBuffer = purego.NewCallback(func(a *Client_CreateViewOfDeviceBuffer_Args) *ErrorHandle {
		if a.Struct_size != PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE || a.Client != client || a.Device_buffer_ptr != unsafe.Pointer(&data[0]) || a.Element_type != Buffer_Type_U8 || a.Dims != &dims[0] || a.Num_dims != 1 || a.Device != device || a.Layout != layout || a.On_delete_callback != 7 || a.On_delete_callback_arg != unsafe.Pointer(&data[1]) {
			t.Fatal("bad create view args")
		}
		a.Buffer = output
		return nil
	})
	b, err := c.CreateViewOfDeviceBuffer(unsafe.Pointer(&data[0]), Buffer_Type_U8, dims, layout, device, 7, unsafe.Pointer(&data[1]))
	if err != nil || b == nil || b.buffer != output {
		t.Fatalf("CreateView = %v, %v", b, err)
	}
	api.Client_CreateViewOfDeviceBuffer = 0
	if _, err := c.CreateViewOfDeviceBuffer(unsafe.Pointer(&data[0]), Buffer_Type_U8, dims, nil, device, 0, nil); err == nil {
		t.Fatal("missing slot succeeded")
	}
	if _, err := (&ClientWrapper{}).CreateViewOfDeviceBuffer(unsafe.Pointer(&data[0]), Buffer_Type_U8, dims, nil, device, 0, nil); err == nil {
		t.Fatal("closed client succeeded")
	}
}

func TestCreateViewPreflight(t *testing.T) {
	for _, name := range []string{"create", "destroy", "data", "device", "nil_client"} {
		t.Run(name, func(t *testing.T) {
			api := new(Api)
			api.Client_CreateViewOfDeviceBuffer = purego.NewCallback(func(*Client_CreateViewOfDeviceBuffer_Args) *ErrorHandle {
				t.Error("unexpected view dispatch")
				return nil
			})
			api.Buffer_Destroy = purego.NewCallback(func(*Buffer_Destroy_Args) *ErrorHandle { t.Error("unexpected destroy"); return nil })
			c := &ClientWrapper{api: api, client: &Client{}}
			var data byte
			ptr, dev := unsafe.Pointer(&data), &Device{}
			switch name {
			case "create":
				api.Client_CreateViewOfDeviceBuffer = 0
			case "destroy":
				api.Buffer_Destroy = 0
			case "data":
				ptr = nil
			case "device":
				dev = nil
			case "nil_client":
				c = nil
			}
			if b, err := c.CreateViewOfDeviceBuffer(ptr, Buffer_Type_U8, nil, nil, dev, 0, nil); err == nil || b != nil {
				t.Fatalf("buffer=%v error=%v", b, err)
			}
		})
	}
}

func TestCreateViewNilOutput(t *testing.T) {
	api := new(Api)
	called := 0
	api.Client_CreateViewOfDeviceBuffer = purego.NewCallback(func(*Client_CreateViewOfDeviceBuffer_Args) *ErrorHandle { called++; return nil })
	api.Buffer_Destroy = purego.NewCallback(func(*Buffer_Destroy_Args) *ErrorHandle { t.Error("unexpected destroy"); return nil })
	c := &ClientWrapper{api: api, client: &Client{}}
	var data byte
	b, err := c.CreateViewOfDeviceBuffer(unsafe.Pointer(&data), Buffer_Type_U8, nil, nil, &Device{}, 0, nil)
	if err == nil || b != nil || called != 1 {
		t.Fatalf("buffer=%v error=%v calls=%d", b, err, called)
	}
}
