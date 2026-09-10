package openxla

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

func TestExternalReferenceArguments(t *testing.T) {
	api := new(Api)
	raw := &Buffer{}
	b := &BufferWrapper{api: api, buffer: raw}
	var storage byte
	count := 0
	api.Buffer_IncreaseExternalReferenceCount = purego.NewCallback(func(a *Buffer_IncreaseExternalReferenceCount_Args) *ErrorHandle {
		if a.Struct_size != PJRT_Buffer_IncreaseExternalReferenceCount_Args_STRUCT_SIZE || a.Buffer != raw {
			t.Error("invalid increase arguments")
		}
		count++
		return nil
	})
	api.Buffer_DecreaseExternalReferenceCount = purego.NewCallback(func(a *Buffer_DecreaseExternalReferenceCount_Args) *ErrorHandle {
		if a.Struct_size != PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE || a.Buffer != raw {
			t.Error("invalid decrease arguments")
		}
		count--
		return nil
	})
	api.Buffer_OpaqueDeviceMemoryDataPointer = purego.NewCallback(func(a *Buffer_OpaqueDeviceMemoryDataPointer_Args) *ErrorHandle {
		if a.Struct_size != PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE || a.Buffer != raw || count != 1 {
			t.Error("invalid pointer arguments or missing reference")
		}
		a.Device_memory_ptr = unsafe.Pointer(&storage)
		return nil
	})
	if err := b.IncreaseExternalReferenceCount(); err != nil {
		t.Fatal(err)
	}
	ptr, err := b.OpaqueDeviceMemoryDataPointer()
	if err != nil || ptr != unsafe.Pointer(&storage) {
		t.Fatalf("pointer=%v error=%v", ptr, err)
	}
	if err := b.DecreaseExternalReferenceCount(); err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("remaining reference count=%d", count)
	}
	api.Buffer_DecreaseExternalReferenceCount = 0
	if err := b.IncreaseExternalReferenceCount(); err == nil || count != 0 {
		t.Fatal("acquired reference without release slot")
	}
}

func TestExternalReferenceErrors(t *testing.T) {
	for _, name := range []string{"increase", "decrease", "pointer"} {
		t.Run(name, func(t *testing.T) {
			api := new(Api)
			message := []byte("reference failed")
			freed := 0
			api.Error_Message = purego.NewCallback(func(a *Error_Message_Args) { a.Message = &message[0]; a.Message_size = uintptr(len(message)) })
			api.Error_Destroy = purego.NewCallback(func(*Error_Destroy_Args) { freed++ })
			api.Buffer_IncreaseExternalReferenceCount = purego.NewCallback(func(*Buffer_IncreaseExternalReferenceCount_Args) *ErrorHandle { return &ErrorHandle{} })
			api.Buffer_DecreaseExternalReferenceCount = purego.NewCallback(func(*Buffer_DecreaseExternalReferenceCount_Args) *ErrorHandle { return &ErrorHandle{} })
			api.Buffer_OpaqueDeviceMemoryDataPointer = purego.NewCallback(func(*Buffer_OpaqueDeviceMemoryDataPointer_Args) *ErrorHandle { return &ErrorHandle{} })
			b := &BufferWrapper{api: api, buffer: &Buffer{}}
			invoke := func() error {
				switch name {
				case "increase":
					return b.IncreaseExternalReferenceCount()
				case "decrease":
					return b.DecreaseExternalReferenceCount()
				default:
					ptr, err := b.OpaqueDeviceMemoryDataPointer()
					if ptr != nil {
						t.Error("failed query returned pointer")
					}
					return err
				}
			}
			if err := invoke(); err == nil || !strings.Contains(err.Error(), "reference failed") || freed != 1 {
				t.Fatalf("error=%v freed=%d", err, freed)
			}
			api.Buffer_IncreaseExternalReferenceCount = 0
			api.Buffer_DecreaseExternalReferenceCount = 0
			api.Buffer_OpaqueDeviceMemoryDataPointer = 0
			if err := invoke(); err == nil || freed != 1 {
				t.Fatal("missing slot succeeded or dispatched")
			}
			b = nil
			if err := invoke(); err == nil {
				t.Fatal("nil buffer succeeded")
			}
		})
	}
}
