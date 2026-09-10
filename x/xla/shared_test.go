package xla

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/openxla"
)

func ExampleEngine_NewSharedBuffer() {
	var engine *Engine
	_, _, err := engine.NewSharedBuffer(F32, []int64{2}, nil)
	fmt.Println(err)
	// Output: xla: engine is closed
}

func ExampleBuffer_SharedData() {
	var buffer Buffer
	_, err := buffer.SharedData()
	fmt.Println(buffer.IsShared(), err)
	// Output: false xla: buffer is closed
}

func TestSharedLifetime(t *testing.T) {
	for _, mode := range []string{"delayed", "create failure", "view ready failure", "shape mismatch", "non CPU", "destroy failure", "joined failure"} {
		t.Run(mode, func(t *testing.T) {
			api := &openxla.Api{StructSize: unsafe.Sizeof(openxla.Api{})}
			api.Error_Destroy = purego.NewCallback(func(*openxla.Error_Destroy_Args) {})
			api.Error_Message = purego.NewCallback(func(*openxla.Error_Message_Args) {})
			api.Client_Create = purego.NewCallback(func(a *openxla.Client_Create_Args) *openxla.ErrorHandle { a.Client = &openxla.Client{}; return nil })
			api.Client_Destroy = purego.NewCallback(func(*openxla.Client_Destroy_Args) *openxla.ErrorHandle { return nil })
			devices := []*openxla.Device{{}}
			api.Client_AddressableDevices = purego.NewCallback(func(a *openxla.Client_AddressableDevices_Args) *openxla.ErrorHandle {
				a.Addressable_devices = (*openxla.Device)(unsafe.Pointer(&devices[0]))
				a.Num_addressable_devices = 1
				return nil
			})
			api.Client_BufferFromHostBuffer = purego.NewCallback(func(a *openxla.Client_BufferFromHostBuffer_Args) *openxla.ErrorHandle {
				a.Buffer = &openxla.Buffer{}
				return nil
			})
			api.Buffer_IsOnCpu = purego.NewCallback(func(a *openxla.Buffer_IsOnCpu_Args) *openxla.ErrorHandle { a.Is_on_cpu = mode != "non CPU"; return nil })
			var data unsafe.Pointer
			var deleter func(unsafe.Pointer, unsafe.Pointer)
			creates, destroys, ready := 0, 0, 0
			api.Client_CreateViewOfDeviceBuffer = purego.NewCallback(func(a *openxla.Client_CreateViewOfDeviceBuffer_Args) *openxla.ErrorHandle {
				creates++
				data = a.Device_buffer_ptr
				purego.RegisterFunc(&deleter, a.On_delete_callback)
				if _, ok := sharedAllocations.Load(data); !ok {
					t.Error("unrooted allocation at dispatch")
				}
				if a.On_delete_callback_arg != nil || uintptr(data)%64 != 0 {
					t.Error("invalid owner or alignment")
				}
				minor := unsafe.Slice(a.Layout.Union_pad.Minor_to_major, a.Layout.Union_pad.Minor_to_major_size)
				if len(minor) != 2 || minor[0] != 1 || minor[1] != 0 {
					t.Errorf("layout %v", minor)
				}
				if mode == "create failure" {
					return &openxla.ErrorHandle{}
				}
				a.Buffer = &openxla.Buffer{}
				return nil
			})
			api.Buffer_ReadyEvent = purego.NewCallback(func(a *openxla.Buffer_ReadyEvent_Args) *openxla.ErrorHandle { a.Event = &openxla.Event{}; return nil })
			api.Event_Await = purego.NewCallback(func(*openxla.Event_Await_Args) *openxla.ErrorHandle {
				ready++
				if (mode == "view ready failure" || mode == "joined failure") && ready == 2 {
					return &openxla.ErrorHandle{}
				}
				return nil
			})
			api.Event_Destroy = purego.NewCallback(func(*openxla.Event_Destroy_Args) *openxla.ErrorHandle { return nil })
			api.Buffer_Destroy = purego.NewCallback(func(*openxla.Buffer_Destroy_Args) *openxla.ErrorHandle {
				destroys++
				if (mode == "destroy failure" || mode == "joined failure") && destroys == 2 {
					return &openxla.ErrorHandle{}
				}
				return nil
			})
			api.Buffer_ElementType = purego.NewCallback(func(a *openxla.Buffer_ElementType_Args) *openxla.ErrorHandle { a.Type = U8; return nil })
			dims := []int64{2, 3}
			if mode == "shape mismatch" {
				dims[1] = 4
			}
			api.Buffer_Dimensions = purego.NewCallback(func(a *openxla.Buffer_Dimensions_Args) *openxla.ErrorHandle {
				a.Dims = &dims[0]
				a.Num_dims = 2
				return nil
			})
			e, err := NewEngineFromAPI(api)
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			b, flat, err := e.NewSharedBuffer(U8, []int64{2, 3}, devices[0])
			if mode == "non CPU" {
				if err == nil || creates != 0 || destroys != 1 {
					t.Fatalf("non CPU: %v %d %d", err, creates, destroys)
				}
				return
			}
			if mode == "create failure" {
				if err == nil {
					t.Fatal("creation succeeded")
				}
				if _, ok := sharedAllocations.Load(data); ok {
					t.Fatal("failed creation retained storage")
				}
				return
			}
			if mode == "view ready failure" || mode == "shape mismatch" || mode == "joined failure" {
				if err == nil || b != nil {
					t.Fatalf("validation: %v", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				flat[0] = 42
				closeErr := b.Close()
				if b.shared != nil {
					t.Fatal("closed buffer retains shared owner")
				}
				if (closeErr != nil) != (mode == "destroy failure") {
					t.Fatalf("close: %v", closeErr)
				}
				if err := b.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "joined failure" && (err == nil || !strings.Contains(err.Error(), "close buffer") || !strings.Contains(err.Error(), "\n")) {
				t.Fatalf("lost joined cleanup error: %v", err)
			}
			if destroys != 2 {
				t.Fatalf("destroyed %d buffers", destroys)
			}
			runtime.GC()
			if _, ok := sharedAllocations.Load(data); !ok {
				t.Fatal("Close released storage before deleter")
			}
			if mode == "delayed" && *(*byte)(data) != 42 {
				t.Fatal("storage changed before callback")
			}
			deleter(data, nil)
			if _, ok := sharedAllocations.Load(data); ok {
				t.Fatal("deleter retained allocation")
			}
			deleter(data, nil) // Duplicate notification must not unpin twice.
		})
	}
}

func TestSharedValidation(t *testing.T) {
	var e *Engine
	for _, dims := range [][]int64{{-1}, {1 << 62}} {
		if _, _, err := e.NewSharedBuffer(F32, dims, nil); err == nil || strings.Contains(err.Error(), "closed") {
			t.Fatalf("validation: %v", err)
		}
	}
}
