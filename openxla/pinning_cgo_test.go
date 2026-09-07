//go:build cgo

package openxla

import (
	"runtime"
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/openxla/internal/pincheck"
)

func requireUnpinned(t *testing.T, p unsafe.Pointer) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Error("cgo checker accepted unpinned Go memory; run with cgo checks enabled")
		} else if !strings.Contains(r.(error).Error(), "unpinned Go pointer") {
			t.Errorf("unexpected panic: %v", r)
		}
	}()
	pincheck.Check(p)
}

func TestTransferPinLifetime(t *testing.T) {
	for _, direction := range []string{"upload", "download"} {
		t.Run(direction, func(t *testing.T) {
			storage := make([]byte, 64)
			data := storage[7:39] // Pin must also accept an interior allocation pointer.
			dims := []int64{32}
			dst := unsafe.Pointer(&data[0])
			requireUnpinned(t, dst)
			var borrowed unsafe.Pointer
			awaited, destroyed := false, false
			api := &Api{}
			api.Event_Await = purego.NewCallback(func(args *Event_Await_Args) *Error {
				runtime.GC()
				pincheck.Check(borrowed)
				unsafe.Slice((*byte)(borrowed), 32)[31] = 73
				awaited = true
				return nil
			})
			api.Event_Destroy = purego.NewCallback(func(args *Event_Destroy_Args) *Error {
				pincheck.Check(borrowed)
				destroyed = true
				return nil
			})
			api.Buffer_Destroy = purego.NewCallback(func(args *Buffer_Destroy_Args) *Error { return nil })
			if direction == "upload" {
				api.Client_BufferFromHostBuffer = purego.NewCallback(func(args *Client_BufferFromHostBuffer_Args) *Error {
					pincheck.Check(args.Data)
					pincheck.Check(unsafe.Pointer(args.Dims))
					borrowed = args.Data
					args.Done_with_host_buffer = &Event{}
					args.Buffer = &Buffer{}
					return nil
				})
				c := &ClientWrapper{api: api, client: &Client{}}
				b, err := c.BufferFromHost(dst, Buffer_Type_U8, dims, &Device{})
				if err != nil {
					t.Fatal(err)
				}
				if err := b.Destroy(); err != nil {
					t.Fatal(err)
				}
			} else {
				api.Buffer_ToHostBuffer = purego.NewCallback(func(args *Buffer_ToHostBuffer_Args) *Error {
					pincheck.Check(args.Dst)
					borrowed = args.Dst
					args.Event = &Event{}
					return nil
				})
				b := &BufferWrapper{api: api, buffer: &Buffer{}}
				if err := b.ToHost(dst, uintptr(len(data))); err != nil {
					t.Fatal(err)
				}
			}
			if !awaited || !destroyed || data[31] != 73 {
				t.Fatalf("awaited=%v destroyed=%v last byte=%d", awaited, destroyed, data[31])
			}
			requireUnpinned(t, dst)
		})
	}
}

func TestCompilePinLifetime(t *testing.T) {
	for _, tt := range []struct {
		name    string
		options []byte
		fail    bool
	}{
		{"default", nil, false},
		{"explicit", []byte{26, 4, 32, 2, 40, 3}, false},
		{"error", nil, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			code := make([]byte, 64)[7:39]
			copy(code, "module")
			requireUnpinned(t, unsafe.Pointer(&code[0]))
			var pointers []unsafe.Pointer
			called, errorDestroyed := false, false
			api := &Api{}
			api.Error_Destroy = purego.NewCallback(func(*Error_Destroy_Args) { errorDestroyed = true })
			api.LoadedExecutable_Destroy = purego.NewCallback(func(*LoadedExecutable_Destroy_Args) *Error { return nil })
			api.Client_Compile = purego.NewCallback(func(args *Client_Compile_Args) *Error {
				called = true
				pointers = []unsafe.Pointer{unsafe.Pointer(args.Program.Code), unsafe.Pointer(args.Program.Format), unsafe.Pointer(args.Compile_options), unsafe.Pointer(args.Program)}
				runtime.GC()
				for i, p := range pointers {
					func() {
						defer func() {
							if r := recover(); r != nil {
								t.Errorf("compile pointer %d is not pinned: %v", i, r)
							}
						}()
						pincheck.Check(p)
					}()
				}
				if string(unsafe.Slice(args.Program.Format, args.Program.Format_size)) != "mlir" || args.Program.Code_size != uintptr(len(code)) {
					t.Error("compile program changed during call")
				}
				if tt.fail {
					return &Error{}
				}
				args.Executable = &LoadedExecutable{}
				return nil
			})
			client := &ClientWrapper{api: api, client: &Client{}}
			executable, err := client.Compile(code, "mlir", tt.options)
			if tt.fail {
				if err == nil || executable != nil || !errorDestroyed {
					t.Fatalf("failed compile = %v, %v; error destroyed=%v", executable, err, errorDestroyed)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				if err := executable.Destroy(); err != nil {
					t.Fatal(err)
				}
			}
			if !called {
				t.Fatal("compile did not dispatch")
			}
			for _, p := range pointers {
				requireUnpinned(t, p)
			}
		})
	}
}

func TestDefaultDeviceAssignmentPinLifetime(t *testing.T) {
	var data unsafe.Pointer
	api := &Api{}
	api.Client_DefaultDeviceAssignment = purego.NewCallback(func(a *Client_DefaultDeviceAssignment_Args) *Error {
		data = unsafe.Pointer(a.Default_assignment)
		runtime.GC()
		pincheck.Check(data)
		unsafe.Slice(a.Default_assignment, a.Default_assignment_size)[0] = 7
		return nil
	})
	client := &ClientWrapper{api: api, client: &Client{}}
	ids, err := client.DefaultDeviceAssignment(2, 3)
	if err != nil || len(ids) != 6 || ids[0] != 7 {
		t.Fatalf("assignment %v, %v", ids, err)
	}
	requireUnpinned(t, data)
}
