//go:build linux

package vpi_test

import (
	"errors"
	"fmt"
	"log"
	"syscall"
	"unsafe"

	"github.com/tmc/lib/nvidia/vpi"
)

// Example rescales an image on a backend the machine actually reports.
//
// It has no output comment and so is compiled but not run: executing it
// requires a VPI runtime, which the machine running "go test" is not assumed to
// have. TestPhysicalRescale in physical_test.go is the same sequence written as
// a test that fails loudly when a runtime is present.
func Example() {
	backends, err := vpi.Backends()
	if err != nil {
		log.Fatal(err) // no VPI runtime: *vpi.LibraryError says what was tried
	}
	if !backends.Has(vpi.BackendCPU) {
		log.Fatalf("VPI reports backends %v, without a CPU backend", backends)
	}
	flags := uint64(vpi.BackendCPU) | vpi.RequireBackends

	stream, err := vpi.StreamCreate(flags)
	if err != nil {
		log.Fatal(err)
	}
	defer vpi.StreamDestroy(stream)

	const w, h = 64, 32
	input, err := vpi.ImageCreate(w, h, vpi.ImageFormatU8, flags)
	if err != nil {
		log.Fatal(err)
	}
	defer vpi.ImageDestroy(input)

	output, err := vpi.ImageCreate(w/2, h/2, vpi.ImageFormatU8, flags)
	if err != nil {
		log.Fatal(err)
	}
	defer vpi.ImageDestroy(output)

	// Fill the input. The pointers in the locked ImageData belong to VPI and
	// die at ImageUnlock.
	in, err := vpi.ImageLockData(input, vpi.LockWrite, vpi.ImageBufferHostPitchLinear)
	if err != nil {
		log.Fatal(err)
	}
	plane, err := in.PlaneBytes(0)
	if err != nil {
		log.Fatal(err)
	}
	for i := range plane {
		plane[i] = byte(i)
	}
	if err := vpi.ImageUnlock(input); err != nil {
		log.Fatal(err)
	}

	// Submitting queues the work; it does not run it.
	if err := vpi.SubmitRescale(stream, vpi.BackendCPU, input, output, vpi.InterpLinear, vpi.BorderClamp, 0); err != nil {
		log.Fatal(err)
	}
	if err := vpi.StreamSync(stream); err != nil {
		log.Fatal(err)
	}

	out, err := vpi.ImageLockData(output, vpi.LockRead, vpi.ImageBufferHostPitchLinear)
	if err != nil {
		log.Fatal(err)
	}
	result, err := out.PlaneBytes(0)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(result))
	if err := vpi.ImageUnlock(output); err != nil {
		log.Fatal(err)
	}
}

// ExampleImageCreateWrapper wraps memory the caller owns. The memory comes from
// mmap, not from the Go heap: VPI keeps the address for the lifetime of the
// wrapper, and the Go collector is free to move a heap object.
//
// Compiled, not run; see [Example].
func ExampleImageCreateWrapper() {
	const w, h = 64, 32
	mem, err := syscall.Mmap(-1, 0, w*h, syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		log.Fatal(err)
	}
	// The unmap must happen after ImageDestroy, never before: VPI does not
	// copy the memory and does not free it.
	defer syscall.Munmap(mem)

	var buf vpi.ImageBufferPitchLinear
	buf.Format = vpi.ImageFormatU8
	buf.NumPlanes = 1
	buf.Planes[0] = vpi.ImagePlanePitchLinear{
		PixelType:  vpi.PixelTypeInvalid, // take the pixel type from Format
		Width:      w,
		Height:     h,
		PitchBytes: w,
		Data:       uintptr(unsafe.Pointer(&mem[0])),
	}
	var data vpi.ImageData
	if err := data.SetPitchLinear(vpi.ImageBufferHostPitchLinear, buf); err != nil {
		log.Fatal(err)
	}

	params, err := vpi.InitImageWrapperParams()
	if err != nil {
		log.Fatal(err)
	}
	img, err := vpi.ImageCreateWrapper(&data, &params, uint64(vpi.BackendCPU)|vpi.RequireBackends)
	if err != nil {
		log.Fatal(err)
	}
	defer vpi.ImageDestroy(img)

	width, height, err := vpi.ImageGetSize(img)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(width, height)
}

// ExampleLoad shows the failure path a machine without VPI takes. Nothing in
// this package panics when the library is missing.
//
// Compiled, not run; the message names the paths tried, which differ per
// machine.
func ExampleLoad() {
	err := vpi.Load("")
	var libErr *vpi.LibraryError
	if errors.As(err, &libErr) {
		for _, attempt := range libErr.Attempts {
			fmt.Printf("tried %s: %v\n", attempt.Name, attempt.Err)
		}
		return
	}
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("loaded", vpi.Path())
}

// ExampleImageData_PitchLinear reads the union through its discriminant. It
// needs no library, so it runs.
func ExampleImageData_PitchLinear() {
	var buf vpi.ImageBufferPitchLinear
	buf.Format = vpi.ImageFormatRGBA8
	buf.NumPlanes = 1
	buf.Planes[0] = vpi.ImagePlanePitchLinear{Width: 16, Height: 8, PitchBytes: 64}

	var data vpi.ImageData
	if err := data.SetPitchLinear(vpi.ImageBufferHostPitchLinear, buf); err != nil {
		log.Fatal(err)
	}

	pl, err := data.PitchLinear()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(pl.Format, pl.Planes[0].Width, pl.Planes[0].Height)

	// Asking for a member the discriminant does not select is an error, not a
	// garbage read.
	if _, err := data.CUDAArray(); err != nil {
		fmt.Println(err)
	}
	// Output:
	// RGBA8 16 8
	// vpi: image buffer is ImageBufferHostPitchLinear, not cudaArray_t
}

// ExampleBackend_String formats a backend set. It needs no library.
func ExampleBackend_String() {
	fmt.Println(vpi.BackendCPU | vpi.BackendCUDA)
	fmt.Println(vpi.BackendTegra)
	// Output:
	// CPU|CUDA
	// VIC|NVENC|OFA
}
