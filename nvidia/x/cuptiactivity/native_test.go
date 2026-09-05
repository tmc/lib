package cuptiactivity

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/cupti"
)

func TestCallbackABI(t *testing.T) {
	pool, err := newBufferPool(64, 1)
	if err != nil {
		t.Fatal(err)
	}
	c := &collection{pool: pool, dropped: func(context cupti.Context, stream uint32) (uint, error) {
		if context != 17 || stream != 23 {
			t.Errorf("callback context=%d stream=%d", context, stream)
		}
		return 2, nil
	}}
	nativeSessions.Lock()
	if nativeSessions.active != nil {
		nativeSessions.Unlock()
		t.Fatal("another collection is active")
	}
	nativeSessions.active = c
	nativeSessions.Unlock()
	t.Cleanup(func() { nativeSessions.Lock(); nativeSessions.active = nil; nativeSessions.Unlock() })
	var request func(*uintptr, *uint, *uint)
	var complete func(cupti.Context, uint32, uintptr, uint, uint)
	purego.RegisterFunc(&request, purego.NewCallback(requestBuffer))
	purego.RegisterFunc(&complete, purego.NewCallback(completeBuffer))
	var pointer uintptr
	var size, max uint
	request(&pointer, &size, &max)
	if pointer == 0 || size != 64 || max != 0 {
		t.Fatalf("request=%#x,%d,%d", pointer, size, max)
	}
	runtime.GC()
	var refused uintptr
	request(&refused, &size, &max)
	if refused != 0 || size != 0 {
		t.Fatal("exhausted pool returned a buffer")
	}
	complete(17, 23, pointer, 64, 0)
	if pending, refused := pool.status(); pending != 0 || refused != 1 {
		t.Fatalf("pending=%d refused=%d", pending, refused)
	}
	if c.nativeDropped != 2 || c.failures != 0 {
		t.Fatalf("dropped=%d failures=%d", c.nativeDropped, c.failures)
	}
	nativeSessions.Lock()
	callbacks := nativeSessions.callbacks
	nativeSessions.Unlock()
	if callbacks != 0 {
		t.Fatalf("callbacks=%d", callbacks)
	}
}

func TestNativeNameCopy(t *testing.T) {
	data := []byte("kernel\x00")
	var pin runtime.Pinner
	pin.Pin(&data[0])
	defer pin.Unpin()
	name, err := nativeName(uintptr(unsafe.Pointer(&data[0])))
	if err != nil || name != "kernel" {
		t.Fatalf("name=%q error=%v", name, err)
	}
	data[0] = 'X'
	if name != "kernel" {
		t.Fatal("name aliases native storage")
	}
	if name, err := nativeName(0); err != nil || name != "" {
		t.Fatalf("nil name=%q error=%v", name, err)
	}
}
