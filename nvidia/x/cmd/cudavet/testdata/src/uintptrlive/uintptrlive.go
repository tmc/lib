package uintptrlive

import (
	"fmt"
	"runtime"
	"unsafe"
)

// params mirrors driver.KernelNodeParams, whose KernelParams is a uintptr.
type params struct {
	KernelParams uintptr
	Extra        uintptr
}

func setParams(p *params) error { return nil }

func syscall(fn uintptr, args ...uintptr) uintptr { return 0 }

// update is x/cudagraph.Exec.UpdateKernelNode before 2026-09-13.
func update(kernelParams []unsafe.Pointer) error {
	var p params
	p.KernelParams = uintptr(unsafe.Pointer(&kernelParams[0])) // want `kernelParams stored as a uintptr is not traced by the garbage collector; defer runtime.KeepAlive\(kernelParams\) or pin it`
	p.Extra = 0
	return setParams(&p)
}

// updateKept is the fix.
func updateKept(kernelParams []unsafe.Pointer) error {
	var p params
	defer runtime.KeepAlive(kernelParams)
	p.KernelParams = uintptr(unsafe.Pointer(&kernelParams[0]))
	return setParams(&p)
}

// keptAfter keeps kernelParams alive after the last use of p.
func keptAfter(kernelParams []unsafe.Pointer) error {
	var p params
	p.KernelParams = uintptr(unsafe.Pointer(&kernelParams[0]))
	err := setParams(&p)
	runtime.KeepAlive(kernelParams)
	return err
}

// keptTooSoon keeps kernelParams alive only until before the driver reads it.
func keptTooSoon(kernelParams []unsafe.Pointer) error {
	var p params
	p.KernelParams = uintptr(unsafe.Pointer(&kernelParams[0])) // want `kernelParams stored as a uintptr`
	runtime.KeepAlive(kernelParams)
	return setParams(&p)
}

func updatePinned(kernelParams []unsafe.Pointer) error {
	var pin runtime.Pinner
	defer pin.Unpin()
	pin.Pin(&kernelParams[0])
	p := params{KernelParams: uintptr(unsafe.Pointer(&kernelParams[0]))}
	return setParams(&p)
}

// literal stores the address in a composite literal.
func literal(buf []byte) error {
	return setParams(&params{KernelParams: uintptr(unsafe.Pointer(unsafe.SliceData(buf)))}) // want `buf stored as a uintptr`
}

// direct passes the conversion in the argument list, where the compiler keeps
// the referent alive through the call.
func direct(fn uintptr, buf []byte) uintptr {
	return syscall(fn, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
}

// local passes the uintptr on through a variable.
func local(fn uintptr, buf []byte) uintptr {
	p := uintptr(unsafe.Pointer(&buf[0])) // want `buf stored as a uintptr`
	return syscall(fn, p, uintptr(len(buf)))
}

// arithmetic only computes and prints with the address.
func arithmetic(buf []byte) bool {
	p := uintptr(unsafe.Pointer(&buf[0]))
	fmt.Println(p)
	return p%16 == 0
}

var global [64]byte

func globalRef() error {
	return setParams(&params{KernelParams: uintptr(unsafe.Pointer(&global[0]))})
}

// keptAfterCall keeps kernelParams alive past the call; reading a result
// field of p afterwards cannot hand the address on.
func keptAfterCall(kernelParams []unsafe.Pointer) ([]byte, error) {
	var p params
	p.KernelParams = uintptr(unsafe.Pointer(&kernelParams[0]))
	err := setParams(&p)
	runtime.KeepAlive(kernelParams)
	return make([]byte, p.Extra), err
}

type launchArgs struct{ params []unsafe.Pointer }

// keptOwner keeps alive a Go object that holds kernelParams.
func keptOwner(kernelParams []unsafe.Pointer) error {
	owner := &launchArgs{params: kernelParams}
	p := params{KernelParams: uintptr(unsafe.Pointer(&kernelParams[0]))}
	err := setParams(&p)
	runtime.KeepAlive(owner)
	return err
}

// keptHolder keeps alive only the struct holding the uintptr, which keeps
// nothing alive.
func keptHolder(kernelParams []unsafe.Pointer) error {
	p := params{KernelParams: uintptr(unsafe.Pointer(&kernelParams[0]))} // want `kernelParams stored as a uintptr`
	defer runtime.KeepAlive(p)
	return setParams(&p)
}

// fill stores into memory its caller owns; keeping table alive is the
// caller's job.
func fill(out *params, table []uintptr) {
	out.KernelParams = uintptr(unsafe.Pointer(&table[0]))
}

type session struct{ image []byte }

// decode reads memory owned by s, whose lifetime the method cannot see.
func (s *session) decode() error {
	p := params{KernelParams: uintptr(unsafe.Pointer(&s.image[0]))}
	return setParams(&p)
}

// aligned converts back to unsafe.Pointer, which the collector traces.
func aligned(buf []byte) unsafe.Pointer {
	off := (16 - uintptr(unsafe.Pointer(&buf[0]))%16) % 16
	p := unsafe.Pointer(uintptr(unsafe.Pointer(&buf[0])) + off)
	return p
}

// closure pins in the enclosing function.
func closure(buf []byte) error {
	var pin runtime.Pinner
	pin.Pin(&buf[0])
	defer pin.Unpin()
	f := func() error {
		p := params{KernelParams: uintptr(unsafe.Pointer(&buf[0]))}
		return setParams(&p)
	}
	return f()
}
