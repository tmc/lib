//go:build linux

package cuptisass

import (
	"github.com/ebitengine/purego"
)

var freeFn func(uintptr)

func init() {
	ptr, err := purego.Dlsym(purego.RTLD_DEFAULT, "free")
	if err == nil && ptr != 0 {
		purego.RegisterFunc(&freeFn, ptr)
	}
}

func nativeFree(ptr uintptr) {
	if ptr != 0 && freeFn != nil {
		freeFn(ptr)
	}
}
