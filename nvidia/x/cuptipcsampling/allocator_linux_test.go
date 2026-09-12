//go:build linux

package cuptipcsampling

import (
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

// This native allocator test requires no CUDA device and enables no collector.
func TestSamplingNativeAllocator(t *testing.T) {
	free, err := samplingNameAllocator()
	if err != nil {
		t.Fatal(err)
	}
	sym, err := purego.Dlsym(purego.RTLD_DEFAULT, "malloc")
	if err != nil {
		t.Fatal(err)
	}
	var malloc func(uintptr) uintptr
	purego.RegisterFunc(&malloc, sym)
	ptr := malloc(4)
	if ptr == 0 {
		t.Fatal("native allocation failed")
	}
	defer free(ptr)
	copy(unsafe.Slice((*byte)(nativePointer(ptr)), 4), []byte{'p', 'c', 's', 0})
	got, err := readSamplingName(ptr, 4)
	if err != nil || got != "pcs" {
		t.Fatalf("got=%q err=%v", got, err)
	}
}
