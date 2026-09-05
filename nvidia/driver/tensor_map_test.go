package driver_test

import (
	"fmt"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

func ExampleNewTensorMap() {
	descriptor := driver.NewTensorMap()
	fmt.Println(unsafe.Sizeof(*descriptor))
	fmt.Println(uintptr(unsafe.Pointer(descriptor)) % 128)
	// Output:
	// 128
	// 0
}

func TestTensorMapStorage(t *testing.T) {
	maps := make([]*driver.TensorMap, 256)
	for i := range maps {
		maps[i] = driver.NewTensorMap()
		if uintptr(unsafe.Pointer(maps[i]))%128 != 0 {
			t.Fatal("unaligned descriptor")
		}
		if *maps[i] != (driver.TensorMap{}) {
			t.Fatal("nonzero descriptor")
		}
		maps[i].Opaque[15] = uint64(i + 1)
	}
	runtime.GC()
	for i, p := range maps {
		if p.Opaque[15] != uint64(i+1) {
			t.Fatalf("descriptor %d changed after collection", i)
		}
	}
}
