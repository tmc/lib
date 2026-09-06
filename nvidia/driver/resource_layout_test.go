package driver

import (
	"encoding/binary"
	"testing"
	"unsafe"
)

func TestDevResourceArrayLayout(t *testing.T) {
	// The selected cuda.h C oracle gives size 144, alignment 8, union offset
	// 96 and nextResource offset 136 on the supported 64-bit targets. Check
	// two entries so oversized storage cannot masquerade as the native type.
	var resources [2]DevResource
	if got := unsafe.Sizeof(resources); got != 288 {
		t.Fatalf("resource array size = %d, want 288", got)
	}
	if got := unsafe.Alignof(resources); got != 8 {
		t.Fatalf("resource array alignment = %d, want 8", got)
	}
	if got := unsafe.Offsetof(resources[0].AnonymousUnion); got != 96 {
		t.Fatalf("resource union offset = %d, want 96", got)
	}
	if got := unsafe.Offsetof(resources[0].NextResource); got != 136 {
		t.Fatalf("resource link offset = %d, want 136", got)
	}
	// Simulate writes at independently measured C offsets, then read through
	// Go indexing. This is storage evidence, not a driver execution test.
	storage := unsafe.Slice((*byte)(unsafe.Pointer(&resources[0])), 288)
	for i := range resources {
		binary.NativeEndian.PutUint32(storage[i*144:], uint32(i+1))
		binary.NativeEndian.PutUint32(storage[i*144+96:], uint32(17+i))
	}
	for i, resource := range resources {
		if resource.Type != DevResourceType(i+1) || resource.AnonymousUnion[0] != uint32(17+i) {
			t.Errorf("resource %d = type %d, first union word %d", i, resource.Type, resource.AnonymousUnion[0])
		}
	}
}
