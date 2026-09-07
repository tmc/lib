//go:build cgo

// Package pincheck checks retained Go pointers using the cgo runtime checker.
// It is used only by tests of the purego bindings.
package pincheck

/*
static void borrow(void *p) { (void)p; }
*/
import "C"

import "unsafe"

// Check panics if p points to unpinned Go memory and cgo checks are enabled.
// The pointer container is implicitly pinned by cgo; its pointee must already
// be explicitly pinned by the caller.
func Check(p unsafe.Pointer) {
	holder := [1]unsafe.Pointer{p}
	C.borrow(unsafe.Pointer(&holder[0]))
}
