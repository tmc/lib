package openxla

import (
	"github.com/ebitengine/purego"
	"testing"
)

func TestBufferDevice(t *testing.T) {
	raw := &Buffer{}
	device := &Device{}
	api := &Api{}
	b := &BufferWrapper{api: api, buffer: raw}
	if _, err := b.Device(); err == nil {
		t.Fatal("accepted missing device query")
	}
	api.Buffer_Device = purego.NewCallback(func(args *Buffer_Device_Args) *ErrorHandle {
		if args.Struct_size != PJRT_Buffer_Device_Args_STRUCT_SIZE || args.Buffer != raw {
			t.Error("wrong device query arguments")
		}
		args.Device = device
		return nil
	})
	got, err := b.Device()
	if err != nil || got != device {
		t.Fatalf("got %p, %v", got, err)
	}
	api.Buffer_Device = purego.NewCallback(func(*Buffer_Device_Args) *ErrorHandle { return nil })
	if _, err := b.Device(); err == nil {
		t.Fatal("accepted null device")
	}
	b.buffer = nil
	if _, err := b.Device(); err == nil {
		t.Fatal("queried closed buffer")
	}
}
