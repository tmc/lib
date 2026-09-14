//go:build !windows

package nvlshm

import (
	"fmt"
	"runtime"
	"unsafe"

	"golang.org/x/sys/unix"

	"github.com/tmc/lib/nvidia/driver"
)

// allocChunk creates one granularity-aligned physical allocation of size
// bytes on dev and exports it as a POSIX fd. The returned handle is owned by
// the caller (release with driver.MemRelease); the fd is owned by the caller
// (close after the exchange).
func allocChunk(dev driver.Device, size uintptr) (handle uint64, fd int, err error) {
	prop := allocProp(dev)
	handle, err = driver.MemCreate(uint(size), &prop, 0)
	if err != nil {
		return 0, -1, fmt.Errorf("cuMemCreate(%d bytes): %w", size, err)
	}
	var fd32 int32 = -1
	var pin runtime.Pinner
	pin.Pin(&fd32)
	if err := driver.MemExportToShareableHandle(
		uintptr(unsafe.Pointer(&fd32)), handle, driver.MemHandleTypePosixFileDescriptor, 0,
	); err != nil {
		pin.Unpin()
		driver.MemRelease(handle)
		return 0, -1, fmt.Errorf("cuMemExportToShareableHandle: %w", err)
	}
	pin.Unpin()
	if fd32 < 0 {
		driver.MemRelease(handle)
		return 0, -1, fmt.Errorf("cuMemExportToShareableHandle returned invalid fd %d", fd32)
	}
	return handle, int(fd32), nil
}

// mapChunks reserves a contiguous VA of len(fds) chunks and maps fds[i] at
// chunk offset i, granting dev read-write access to every chunk. The imported
// handles are released immediately (each mapping holds its own reference).
// On success the caller owns the reservation; unmap each chunk then free the
// reservation to release it. mapChunks does not close the fds.
func mapChunks(dev driver.Device, chunkBytes uintptr, fds []int) (driver.DevicePtr, error) {
	if chunkBytes == 0 || len(fds) == 0 || chunkBytes > ^uintptr(0)/uintptr(len(fds)) {
		return 0, fmt.Errorf("nvlshm: mapped size overflows uintptr")
	}
	total := chunkBytes * uintptr(len(fds))
	base, err := driver.DevicePtr(0).MemAddressReserve(uint(total), 0, 0)
	if err != nil {
		return 0, fmt.Errorf("cuMemAddressReserve(%d bytes): %w", total, err)
	}
	unwind := func(mapped int) {
		for i := 0; i < mapped; i++ {
			va := base + driver.DevicePtr(uintptr(i)*chunkBytes)
			va.MemUnmap(uint(chunkBytes))
		}
		base.MemAddressFree(uint(total))
	}

	access := driver.MemAccessDesc{
		Location: driver.MemLocation{Type: driver.MemLocationTypeDevice, Id: int32(dev)},
		Flags:    driver.MemAccessFlagsProtReadwrite,
	}
	for i, fd := range fds {
		handle, err := driver.MemImportFromShareableHandle(
			uintptr(fd), driver.MemHandleTypePosixFileDescriptor,
		)
		if err != nil {
			unwind(i)
			return 0, fmt.Errorf("cuMemImportFromShareableHandle(chunk %d): %w", i, err)
		}
		va := base + driver.DevicePtr(uintptr(i)*chunkBytes)
		if err := va.MemMap(uint(chunkBytes), 0, handle, 0); err != nil {
			driver.MemRelease(handle)
			unwind(i)
			return 0, fmt.Errorf("cuMemMap(chunk %d): %w", i, err)
		}
		// All chunks get RW access: writers push into remote chunks.
		if err := va.MemSetAccess(uint(chunkBytes), &access, 1); err != nil {
			driver.MemRelease(handle)
			unwind(i + 1)
			return 0, fmt.Errorf("cuMemSetAccess(chunk %d): %w", i, err)
		}
		if err := driver.MemRelease(handle); err != nil {
			unwind(i + 1)
			return 0, fmt.Errorf("cuMemRelease(imported chunk %d): %w", i, err)
		}
	}
	return base, nil
}

// checkAligned verifies chunkBytes against the granularities the buffer
// needs. multicastDevices is 0 for unicast-only buffers.
func checkAligned(dev driver.Device, chunkBytes uintptr, multicastDevices int) error {
	if chunkBytes == 0 {
		return fmt.Errorf("nvlshm: chunk size must be positive")
	}
	g, err := Granularity(dev)
	if err != nil {
		return err
	}
	need := g
	if multicastDevices > 0 {
		mg, err := MulticastGranularity(multicastDevices)
		if err != nil {
			return err
		}
		if mg > need {
			need = mg
		}
	}
	if chunkBytes%need != 0 {
		return fmt.Errorf("nvlshm: chunk size %d is not a multiple of the required granularity %d; pad with PadTo",
			chunkBytes, need)
	}
	return nil
}

// DistAlloc collectively builds a distributed buffer: every rank contributes
// one chunkBytes chunk on dev, and every rank maps all chunks into one
// contiguous read-write VA range (chunk i on GPU of rank i). chunkBytes must
// be a multiple of [Granularity] and identical on all ranks. All ranks must
// call DistAlloc together.
func DistAlloc(b Bootstrap, dev driver.Device, chunkBytes uintptr) (*DistBuffer, error) {
	return distAlloc(b, dev, chunkBytes, false)
}

// DistAllocMulticast is DistAlloc plus an NVSwitch multicast mapping over the
// same physical chunks (see [DistBuffer.McPtr]). chunkBytes must be a
// multiple of both [Granularity] and [MulticastGranularity]. Fails if the
// device does not support multicast (check [MulticastSupported] to fall back).
func DistAllocMulticast(b Bootstrap, dev driver.Device, chunkBytes uintptr) (*DistBuffer, error) {
	return distAlloc(b, dev, chunkBytes, true)
}

func distAlloc(b Bootstrap, dev driver.Device, chunkBytes uintptr, multicast bool) (*DistBuffer, error) {
	if b == nil {
		return nil, fmt.Errorf("nvlshm: nil bootstrap")
	}
	world := b.WorldSize()
	if world <= 0 || b.Rank() < 0 || b.Rank() >= world {
		return nil, fmt.Errorf("nvlshm: invalid bootstrap rank %d world %d", b.Rank(), world)
	}
	if chunkBytes != 0 && uintptr(world) > ^uintptr(0)/chunkBytes {
		return nil, fmt.Errorf("nvlshm: total size overflows uintptr")
	}
	mcDevices := 0
	if multicast {
		mcDevices = world
		ok, err := MulticastSupported(dev)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("nvlshm: device %d does not support multicast", dev)
		}
	}
	if err := checkAligned(dev, chunkBytes, mcDevices); err != nil {
		return nil, err
	}

	handle, fd, err := allocChunk(dev, chunkBytes)
	if err != nil {
		return nil, err
	}
	// The owned handle is kept until after the map (and, for multicast,
	// after the bind): cuMulticastBindMem requires the owned handle, and
	// the mappings keep the pages alive after it is released.
	defer driver.MemRelease(handle)

	senders := make([]int, world)
	for i := range senders {
		senders[i] = i
	}
	fds, err := exchangeFDs(b, fd, senders)
	unix.Close(fd) // peers have their copies (post-receive barrier)
	if err != nil {
		return nil, err
	}
	closeFDs := func() {
		for _, f := range fds {
			unix.Close(f)
		}
	}

	ordered := make([]int, world)
	for r := 0; r < world; r++ {
		f, ok := fds[r]
		if !ok {
			closeFDs()
			return nil, fmt.Errorf("nvlshm: missing fd from rank %d", r)
		}
		ordered[r] = f
	}
	base, err := mapChunks(dev, chunkBytes, ordered)
	closeFDs()
	if err != nil {
		return nil, err
	}

	buf := &DistBuffer{
		base:       base,
		chunkBytes: chunkBytes,
		world:      world,
		rank:       b.Rank(),
		dev:        dev,
	}
	if multicast {
		if err := buf.attachMulticast(b, handle); err != nil {
			buf.Close()
			return nil, err
		}
	}
	return buf, nil
}

// SingleOwnerAlloc collectively maps one buffer physically resident on the
// owner rank's GPU into every rank's address space (read-write everywhere).
// bytes must be a multiple of [Granularity] and identical on all ranks. All
// ranks must call SingleOwnerAlloc together.
func SingleOwnerAlloc(b Bootstrap, dev driver.Device, bytes uintptr, owner int) (*OwnedBuffer, error) {
	if b == nil {
		return nil, fmt.Errorf("nvlshm: nil bootstrap")
	}
	world := b.WorldSize()
	if world <= 0 || b.Rank() < 0 || b.Rank() >= world {
		return nil, fmt.Errorf("nvlshm: invalid bootstrap rank %d world %d", b.Rank(), world)
	}
	if owner < 0 || owner >= world {
		return nil, fmt.Errorf("nvlshm: owner rank %d out of range [0,%d)", owner, world)
	}
	if err := checkAligned(dev, bytes, 0); err != nil {
		return nil, err
	}

	localFD := -1
	if b.Rank() == owner {
		handle, fd, err := allocChunk(dev, bytes)
		if err != nil {
			return nil, err
		}
		// No multicast on single-owner buffers: the mapping made below
		// keeps the pages alive, so the owned handle can go now.
		defer driver.MemRelease(handle)
		localFD = fd
	}

	fds, err := exchangeFDs(b, localFD, []int{owner})
	if localFD >= 0 {
		unix.Close(localFD)
	}
	if err != nil {
		return nil, err
	}
	ownerFD, ok := fds[owner]
	if !ok {
		return nil, fmt.Errorf("nvlshm: missing fd from owner rank %d", owner)
	}
	base, err := mapChunks(dev, bytes, []int{ownerFD})
	unix.Close(ownerFD)
	if err != nil {
		return nil, err
	}
	return &OwnedBuffer{base: base, bytes: bytes, owner: owner, dev: dev}, nil
}
