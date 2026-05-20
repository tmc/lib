package abi

import (
	"testing"
	"unsafe"
)

func TestApplicationInfoLayout64(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("layout assertions are for 64-bit targets")
	}
	if got := unsafe.Sizeof(ApplicationInfo{}); got != 48 {
		t.Fatalf("ApplicationInfo size = %d", got)
	}
	if got := unsafe.Offsetof(ApplicationInfo{}.PNext); got != 8 {
		t.Fatalf("ApplicationInfo.PNext offset = %d", got)
	}
	if got := unsafe.Offsetof(ApplicationInfo{}.APIVersion); got != 44 {
		t.Fatalf("ApplicationInfo.APIVersion offset = %d", got)
	}
}

func TestInstanceCreateInfoLayout64(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("layout assertions are for 64-bit targets")
	}
	if got := unsafe.Sizeof(InstanceCreateInfo{}); got != 64 {
		t.Fatalf("InstanceCreateInfo size = %d", got)
	}
	if got := unsafe.Offsetof(InstanceCreateInfo{}.PNext); got != 8 {
		t.Fatalf("InstanceCreateInfo.PNext offset = %d", got)
	}
	if got := unsafe.Offsetof(InstanceCreateInfo{}.PpEnabledExtensionNames); got != 56 {
		t.Fatalf("InstanceCreateInfo.PpEnabledExtensionNames offset = %d", got)
	}
}
