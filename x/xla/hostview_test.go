package xla

import (
	"errors"
	"fmt"
	"runtime"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/openxla"
)

func ExampleBuffer_WithReadOnlyFloats() {
	var b Buffer
	err := b.WithReadOnlyFloats(func(values []float32) error { fmt.Println(values); return nil })
	fmt.Println(err)
	// Output: xla: buffer is closed
}

func ExampleEngine_EnableCPUHostAccess() {
	var e Engine
	fmt.Println(e.EnableCPUHostAccess())
	// Output: xla: engine is closed
}

func TestHostViewLifetime(t *testing.T) {
	for _, mode := range []string{"ok", "not enabled", "non CPU", "column major", "extent", "acquire", "ready", "pointer", "nil pointer", "unaligned", "callback", "release", "joined", "close", "concurrent close", "panic"} {
		t.Run(mode, func(t *testing.T) {
			api := &openxla.Api{StructSize: unsafe.Sizeof(openxla.Api{})}
			api.Error_Destroy = purego.NewCallback(func(*openxla.Error_Destroy_Args) {})
			api.Error_Message = purego.NewCallback(func(*openxla.Error_Message_Args) {})
			api.Client_Create = purego.NewCallback(func(a *openxla.Client_Create_Args) *openxla.ErrorHandle { a.Client = &openxla.Client{}; return nil })
			api.Client_Destroy = purego.NewCallback(func(*openxla.Client_Destroy_Args) *openxla.ErrorHandle { return nil })
			api.Client_BufferFromHostBuffer = purego.NewCallback(func(a *openxla.Client_BufferFromHostBuffer_Args) *openxla.ErrorHandle {
				a.Buffer = &openxla.Buffer{}
				return nil
			})
			destroyed := 0
			api.Buffer_Destroy = purego.NewCallback(func(*openxla.Buffer_Destroy_Args) *openxla.ErrorHandle { destroyed++; return nil })
			api.Buffer_IsOnCpu = purego.NewCallback(func(a *openxla.Buffer_IsOnCpu_Args) *openxla.ErrorHandle { a.Is_on_cpu = mode != "non CPU"; return nil })
			dims, minor := []int64{2, 3}, []int64{1, 0}
			if mode == "column major" {
				minor = []int64{0, 1}
			}
			api.Buffer_ElementType = purego.NewCallback(func(a *openxla.Buffer_ElementType_Args) *openxla.ErrorHandle { a.Type = F32; return nil })
			api.Buffer_Dimensions = purego.NewCallback(func(a *openxla.Buffer_Dimensions_Args) *openxla.ErrorHandle {
				a.Dims = &dims[0]
				a.Num_dims = 2
				return nil
			})
			api.Buffer_GetMemoryLayout = purego.NewCallback(func(a *openxla.Buffer_GetMemoryLayout_Args) *openxla.ErrorHandle {
				a.Layout = openxla.Buffer_MemoryLayout{Struct_size: openxla.PJRT_Buffer_MemoryLayout_STRUCT_SIZE, Type: openxla.Buffer_MemoryLayout_Type_Tiled, Union_pad: openxla.Buffer_MemoryLayout_Tiled{Struct_size: openxla.PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE, Minor_to_major: &minor[0], Minor_to_major_size: 2}}
				return nil
			})
			api.Buffer_OnDeviceSizeInBytes = purego.NewCallback(func(a *openxla.Buffer_OnDeviceSizeInBytes_Args) *openxla.ErrorHandle {
				a.On_device_size_in_bytes = 24
				if mode == "extent" {
					a.On_device_size_in_bytes = 28
				}
				return nil
			})
			acquired, released, refs, ready, pointers := 0, 0, 0, 0, 0
			api.Buffer_IncreaseExternalReferenceCount = purego.NewCallback(func(*openxla.Buffer_IncreaseExternalReferenceCount_Args) *openxla.ErrorHandle {
				acquired++
				if mode == "acquire" {
					return &openxla.ErrorHandle{}
				}
				refs++
				return nil
			})
			api.Buffer_DecreaseExternalReferenceCount = purego.NewCallback(func(*openxla.Buffer_DecreaseExternalReferenceCount_Args) *openxla.ErrorHandle {
				released++
				refs--
				if mode == "release" || mode == "joined" {
					return &openxla.ErrorHandle{}
				}
				return nil
			})
			api.Buffer_ReadyEvent = purego.NewCallback(func(a *openxla.Buffer_ReadyEvent_Args) *openxla.ErrorHandle { a.Event = &openxla.Event{}; return nil })
			api.Event_Await = purego.NewCallback(func(*openxla.Event_Await_Args) *openxla.ErrorHandle {
				ready++
				if refs != 1 {
					t.Error("producer wait without reference")
				}
				if mode == "ready" {
					return &openxla.ErrorHandle{}
				}
				return nil
			})
			api.Event_Destroy = purego.NewCallback(func(*openxla.Event_Destroy_Args) *openxla.ErrorHandle { return nil })
			values := []float32{1, 2, 3, 4, 5, 6}
			var pin runtime.Pinner
			pin.Pin(&values[0])
			defer pin.Unpin()
			api.Buffer_OpaqueDeviceMemoryDataPointer = purego.NewCallback(func(a *openxla.Buffer_OpaqueDeviceMemoryDataPointer_Args) *openxla.ErrorHandle {
				pointers++
				if refs != 1 || ready != 1 {
					t.Error("pointer before reference/readiness")
				}
				if mode == "pointer" {
					return &openxla.ErrorHandle{}
				}
				if mode != "nil pointer" {
					a.Device_memory_ptr = unsafe.Pointer(&values[0])
				}
				if mode == "unaligned" {
					a.Device_memory_ptr = unsafe.Add(a.Device_memory_ptr, 1)
				}
				return nil
			})
			e, err := NewEngineFromAPI(api)
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			if mode != "not enabled" {
				if err := e.EnableCPUHostAccess(); err != nil {
					t.Fatal(err)
				}
			}
			b, err := e.NewBufferFromHost(make([]byte, 24), F32, dims, &openxla.Device{})
			if err != nil {
				t.Fatal(err)
			}
			callbacks := 0
			callbackErr := errors.New("callback sentinel")
			invoke := func() (err error) {
				defer func() {
					if value := recover(); value != nil {
						if mode != "panic" {
							panic(value)
						}
						err = callbackErr
					}
				}()
				return b.WithReadOnlyFloats(func(got []float32) error {
					callbacks++
					runtime.GC()
					if refs != 1 || !slices.Equal(got, values) {
						t.Errorf("callback refs=%d values=%v", refs, got)
					}
					if err := e.Close(); err == nil {
						t.Error("closed engine during view")
					}
					if mode == "close" {
						if err := b.Close(); err == nil {
							t.Error("close during view succeeded")
						}
					}
					if mode == "concurrent close" {
						done := make(chan error, 1)
						go func() { done <- b.Close() }()
						if err := <-done; err == nil {
							t.Error("concurrent close succeeded")
						}
					}
					if mode == "panic" {
						panic("callback panic")
					}
					if mode == "callback" || mode == "joined" {
						return callbackErr
					}
					return nil
				})
			}
			err = invoke()
			wantCallback := mode == "ok" || mode == "callback" || mode == "release" || mode == "joined" || mode == "close" || mode == "concurrent close" || mode == "panic"
			if (callbacks == 1) != wantCallback {
				t.Fatalf("callbacks=%d error=%v", callbacks, err)
			}
			wantError := mode != "ok" && mode != "close" && mode != "concurrent close"
			if (err != nil) != wantError {
				t.Fatalf("error=%v", err)
			}
			if mode == "joined" && (!errors.Is(err, callbackErr) || !strings.Contains(err.Error(), "\n")) {
				t.Fatalf("lost joined error: %v", err)
			}
			if refs != 0 || released > 1 || (acquired == 1 && mode != "acquire" && released != 1) {
				t.Fatalf("references acquired=%d released=%d refs=%d", acquired, released, refs)
			}
			if mode == "not enabled" && pointers != 0 {
				t.Fatal("unqualified pointer queried")
			}
			if destroyed != 0 {
				t.Fatal("buffer destroyed during scope")
			}
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
			if destroyed != 1 {
				t.Fatal("buffer not closed after scope")
			}
		})
	}
}
