//go:build linux
// +build linux

package gdrcopy_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/gdrcopy"
)

// Example shows the whole lifecycle: open, pin, map, copy, and then release in
// the reverse order, which is what gdrapi.h requires.
//
// devicePtr is a CUDA device address, from cuMemAlloc or an equivalent. The
// kernel module pins whole GPU pages, so the mapping generally starts below
// that address and the data is at an offset inside it; Info reports where.
func Example() {
	const (
		devicePtr = uintptr(0) // a CUDA device address
		size      = uintptr(64 * 1024)
	)

	g, err := gdrcopy.Open()
	if err != nil {
		log.Fatal(err)
	}
	defer g.Close()

	mh, err := g.PinBuffer(devicePtr, size, 0, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer g.UnpinBuffer(mh)

	va, err := g.Map(mh, size)
	if err != nil {
		log.Fatal(err)
	}
	defer g.Unmap(mh, va, size)

	// The mapping begins at the GPU-page-aligned base of the pinning, which is
	// at or below devicePtr. Info says where, and Offset does the arithmetic.
	info, err := g.GetInfoV2(mh)
	if err != nil {
		log.Fatal(err)
	}
	off, ok := info.Offset(devicePtr)
	if !ok {
		log.Fatalf("%#x is not inside the pinning at %#x", devicePtr, info.Va)
	}

	// Write through the copy helpers, never by storing into the mapping: the
	// address is an mmap of a device aperture, and these run the copy paths
	// libgdrapi implements for it.
	payload := []byte("hello from the host")
	if err := mh.CopyToMapping(va+off, payload); err != nil {
		log.Fatal(err)
	}

	readback := make([]byte, len(payload))
	if err := mh.CopyFromMapping(readback, va+off); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("mapping type %v, %d bytes at offset %#x\n", info.MappingType, info.MappedSize, off)
}

// ExampleHandle_GetAttribute shows how to ask whether the running gdrdrv and
// GPU driver support forcing a PCIe BAR1 mapping before asking for one.
func ExampleHandle_GetAttribute() {
	g, err := gdrcopy.Open()
	if err != nil {
		log.Fatal(err)
	}
	defer g.Close()

	supported, err := g.GetAttribute(gdrcopy.AttrSupportPinFlagForcePcie)
	if err != nil {
		// Older libgdrapi has no gdr_get_attribute at all.
		log.Fatal(err)
	}
	flags := gdrcopy.PinFlagDefault
	if supported != 0 {
		flags = gdrcopy.PinFlagForcePcie
	}
	fmt.Println(flags)
}
