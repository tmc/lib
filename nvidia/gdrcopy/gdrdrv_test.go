//go:build linux
// +build linux

package gdrcopy_test

import (
	"bytes"
	"errors"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/gdrcopy"
)

// This file is the physical consumer test: the one that proves the binding
// moves real bytes into real GPU memory. It was written without ever being
// run, because the machine it was written on has no GPU.
//
// It needs all of:
//
//   - an NVIDIA GPU visible to the process, with GPUDirect RDMA support
//     (Kepler or later; a display GPU works, a vGPU generally does not);
//   - the NVIDIA GPU driver, so that libcuda.so.1 can allocate device memory;
//   - either the gdrdrv kernel module loaded, which NVIDIA/gdrcopy builds and
//     installs and which appears as /dev/gdrdrv, or a GPU driver and device
//     supporting dma-buf mmap, which libgdrapi uses instead when gdrdrv is
//     absent;
//   - libgdrapi.so.2 on the loader path, or GDRCOPY_LIB_PATH pointing at it.
//
// Where any of those is missing the test skips and says which one. A skip is
// not a pass and proves nothing about this binding. Set
// GDRCOPY_REQUIRE_HARDWARE=1 on a machine that is supposed to have all of it,
// and every one of those skips becomes a failure.

// needs turns a missing prerequisite into a skip that names it, or into a
// failure when the caller has declared the prerequisite must be there.
func needs(t *testing.T, err error, what string) {
	t.Helper()
	if err == nil {
		return
	}
	if os.Getenv("GDRCOPY_REQUIRE_HARDWARE") != "" {
		t.Fatalf("GDRCOPY_REQUIRE_HARDWARE is set but %s: %v", what, err)
	}
	t.Skipf("nothing in this test was checked, because %s: %v", what, err)
}

// TestPinMapCopy is the end-to-end check: allocate device memory with the CUDA
// driver, pin it, map it, write through the mapping with gdrcopy, and read the
// bytes back with cuMemcpyDtoH. The read-back is the part that matters. It is
// the only evidence that the mapping refers to the buffer that was pinned and
// that the copy path put the bytes where the GPU can see them.
func TestPinMapCopy(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	needs(t, driver.Load(""), "libcuda.so.1 is not installed, so no device memory can be allocated")
	needs(t, driver.Init(0), "the CUDA driver would not initialize")
	count, err := driver.DeviceGetCount()
	needs(t, err, "the CUDA driver could not enumerate devices")
	if count == 0 {
		needs(t, errors.New("no devices"), "no NVIDIA GPU is visible to this process")
	}
	dev, err := driver.NewDeviceGet(0)
	needs(t, err, "device 0 could not be opened")
	name, err := dev.GetName()
	if err != nil {
		t.Fatalf("cuDeviceGetName: %v", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("cuDevicePrimaryCtxRetain: %v", err)
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("cuCtxSetCurrent: %v", err)
	}

	// Allocate three GPU pages and pin a region starting a little way into the
	// second, so that the pinning is aligned down to a page boundary below the
	// address handed to PinBuffer and the whole aligned region still lies
	// inside the allocation. That is the case the offset arithmetic exists for.
	const pinSize = gdrcopy.GPUPageSize
	const dataOffset = 0x100
	alloc, err := driver.MemAlloc(uint(3 * gdrcopy.GPUPageSize))
	if err != nil {
		t.Fatalf("cuMemAlloc: %v", err)
	}
	defer alloc.MemFree()
	base := (uintptr(alloc) + gdrcopy.GPUPageSize - 1) & gdrcopy.GPUPageMask
	data := base + dataOffset
	if err := alloc.MemsetD8(0, uint(3*gdrcopy.GPUPageSize)); err != nil {
		t.Fatalf("cuMemsetD8: %v", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatalf("cuCtxSynchronize: %v", err)
	}

	needs(t, gdrcopy.Load(""), "libgdrapi.so.2 is not installed")
	g, err := gdrcopy.Open()
	needs(t, err, "gdr_open failed, so neither gdrdrv nor a dma-buf capable device is available")
	defer func() {
		if err := g.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()
	t.Logf("device %q, libgdrapi at %s", name, gdrcopy.Path())

	mh, err := g.PinBuffer(data, pinSize, 0, 0)
	if err != nil {
		t.Fatalf("PinBuffer(%#x, %d): %v", data, pinSize, err)
	}
	defer func() {
		if err := g.UnpinBuffer(mh); err != nil {
			t.Errorf("UnpinBuffer: %v", err)
		}
	}()

	info, err := g.GetInfoV2(mh)
	if err != nil {
		t.Fatalf("GetInfoV2: %v", err)
	}
	if info.Va&uint64(gdrcopy.GPUPageSize-1) != 0 {
		t.Errorf("Info.Va = %#x, which is not aligned to a %d-byte GPU page", info.Va, gdrcopy.GPUPageSize)
	}
	if uintptr(info.Va) > data {
		t.Errorf("Info.Va = %#x is above the pinned address %#x", info.Va, data)
	}
	if info.Mapped() {
		t.Error("Info.Mapped is set before Map was called")
	}
	off, ok := info.Offset(data)
	if !ok {
		t.Fatalf("the pinned address %#x is reported outside its own pinning at %#x+%d", data, info.Va, info.MappedSize)
	}
	if want := data - uintptr(info.Va); off != want {
		t.Errorf("Offset = %#x, want %#x", off, want)
	}

	mapSize := uintptr(info.MappedSize)
	va, err := g.Map(mh, mapSize)
	if err != nil {
		t.Fatalf("Map(%d): %v", mapSize, err)
	}
	unmapped := false
	defer func() {
		if unmapped {
			return
		}
		if err := g.Unmap(mh, va, mapSize); err != nil {
			t.Errorf("Unmap: %v", err)
		}
	}()

	mapped, err := g.GetInfoV2(mh)
	if err != nil {
		t.Fatalf("GetInfoV2 after Map: %v", err)
	}
	if !mapped.Mapped() {
		t.Error("Info.Mapped is clear after Map succeeded")
	}
	t.Logf("mapping type %v, write-combining %v, %d bytes at %#x", mapped.MappingType, mapped.WCMapping(), mapped.MappedSize, va)

	// Write through the mapping, then read the same bytes back through the
	// CUDA driver. If the mapping pointed somewhere else, or the copy path
	// wrote the wrong width, this is where it shows.
	payload := make([]byte, 4096)
	for i := range payload {
		payload[i] = byte(i*7 + 1)
	}
	if err := mh.CopyToMapping(va+off, payload); err != nil {
		t.Fatalf("CopyToMapping: %v", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatalf("cuCtxSynchronize after CopyToMapping: %v", err)
	}
	fromDevice := make([]byte, len(payload))
	if err := driver.DevicePtr(data).MemcpyDtoH(unsafe.Pointer(&fromDevice[0]), uint(len(fromDevice))); err != nil {
		t.Fatalf("cuMemcpyDtoH: %v", err)
	}
	if !bytes.Equal(fromDevice, payload) {
		t.Fatalf("the GPU sees different bytes than CopyToMapping wrote: first difference at %d", firstDiff(fromDevice, payload))
	}

	// And the other direction: change the buffer from the GPU side and read it
	// back through the mapping.
	if err := driver.DevicePtr(data).MemsetD8(0xa5, uint(len(payload))); err != nil {
		t.Fatalf("cuMemsetD8: %v", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatalf("cuCtxSynchronize after cuMemsetD8: %v", err)
	}
	fromMapping := make([]byte, len(payload))
	if err := mh.CopyFromMapping(fromMapping, va+off); err != nil {
		t.Fatalf("CopyFromMapping: %v", err)
	}
	want := bytes.Repeat([]byte{0xa5}, len(payload))
	if !bytes.Equal(fromMapping, want) {
		t.Fatalf("CopyFromMapping read different bytes than the GPU wrote: first difference at %d", firstDiff(fromMapping, want))
	}

	// The release order is a contract, so check that violating it is refused
	// while the mapping really is live rather than only in the bookkeeping.
	if err := g.UnpinBuffer(mh); !errors.Is(err, gdrcopy.ErrMapped) {
		t.Errorf("UnpinBuffer while mapped = %v, want ErrMapped", err)
	}
	if err := g.Close(); !errors.Is(err, gdrcopy.ErrMapped) {
		t.Errorf("Close while mapped = %v, want ErrMapped", err)
	}
	if err := g.Unmap(mh, va, mapSize); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	unmapped = true

	// The callback flag is set when the GPU driver invalidates the pinning,
	// which has not happened: the allocation is still alive.
	flag, err := g.GetCallbackFlag(mh)
	if err != nil {
		t.Fatalf("GetCallbackFlag: %v", err)
	}
	if flag {
		t.Error("the invalidation callback fired while the device allocation was still live")
	}
}

// TestDriverVersion reports the gdrdrv version against the minimum the pinned
// header supports. It needs the kernel module, not a GPU allocation.
func TestDriverVersion(t *testing.T) {
	needs(t, gdrcopy.Load(""), "libgdrapi.so.2 is not installed")
	g, err := gdrcopy.Open()
	needs(t, err, "gdr_open failed, so neither gdrdrv nor a dma-buf capable device is available")
	defer g.Close()

	major, minor, err := g.DriverGetVersion()
	if err != nil {
		// The dma-buf backend has no kernel module to ask.
		using, attrErr := g.GetAttribute(gdrcopy.AttrUsingDmaBufMmap)
		if attrErr == nil && using != 0 {
			t.Skip("libgdrapi is using the dma-buf backend, which has no gdrdrv to report a version")
		}
		t.Fatalf("DriverGetVersion: %v", err)
	}
	t.Logf("gdrdrv %d.%d", major, minor)
	if major < gdrcopy.MinimumDriverVersionMajor ||
		(major == gdrcopy.MinimumDriverVersionMajor && minor < gdrcopy.MinimumDriverVersionMinor) {
		t.Errorf("gdrdrv %d.%d is older than the %d.%d the pinned header requires",
			major, minor, gdrcopy.MinimumDriverVersionMajor, gdrcopy.MinimumDriverVersionMinor)
	}
}

func firstDiff(a, b []byte) int {
	for i := range a {
		if i >= len(b) || a[i] != b[i] {
			return i
		}
	}
	return len(a)
}
