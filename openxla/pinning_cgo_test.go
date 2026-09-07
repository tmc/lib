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
