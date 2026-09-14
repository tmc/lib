//go:build !windows

package nvlshm

import (
	"fmt"
	"runtime"
	"unsafe"

	"golang.org/x/sys/unix"

	"github.com/tmc/lib/nvidia/driver"
)

// attachMulticast overlays an NVSwitch multicast mapping on buf's chunks.
// ownedHandle is this rank's owned physical allocation handle from
// cuMemCreate — cuMulticastBindMem requires the owned handle; binding an
// imported allocation (or BindAddr on an imported VA) is rejected.
//
// The registration order is a driver contract; each barrier gates a state
// transition and skipping one fails with CUDA_ERROR_ILLEGAL_STATE or hangs:
//
//	rank 0: cuMulticastCreate + export fd ─┐
//	all:    import fd, cuMulticastAddDevice│
//	        ── barrier ──                  │ all devices added
//	all:    cuMulticastBindMem + map VA    │
//	        ── barrier ──                  ┘ all bound
func (buf *DistBuffer) attachMulticast(b Bootstrap, ownedHandle uint64) error {
	rank, world := b.Rank(), b.WorldSize()
	size := buf.chunkBytes

	// Create on rank 0, exchange the fd, import elsewhere.
	var mcHandle uint64
	localFD := -1
	if rank == 0 {
		prop := driver.MulticastObjectProp{
			NumDevices:  uint32(world),
			Size:        uint(size),
			HandleTypes: uint64(driver.MemHandleTypePosixFileDescriptor),
		}
		h, err := driver.MulticastCreate(&prop)
		if err != nil {
			return fmt.Errorf("cuMulticastCreate: %w", err)
		}
		mcHandle = h
		var fd32 int32 = -1
		var pin runtime.Pinner
		pin.Pin(&fd32)
		if err := driver.MemExportToShareableHandle(
			uintptr(unsafe.Pointer(&fd32)), mcHandle, driver.MemHandleTypePosixFileDescriptor, 0,
		); err != nil {
			pin.Unpin()
			driver.MemRelease(mcHandle)
			return fmt.Errorf("cuMemExportToShareableHandle(multicast): %w", err)
		}
		pin.Unpin()
		if fd32 < 0 {
			driver.MemRelease(mcHandle)
			return fmt.Errorf("cuMemExportToShareableHandle(multicast) returned invalid fd %d", fd32)
		}
		localFD = int(fd32)
	}

	fds, err := exchangeFDs(b, localFD, []int{0})
	if localFD >= 0 {
		unix.Close(localFD)
	}
	if err != nil {
		if rank == 0 {
			driver.MemRelease(mcHandle)
		}
		return err
	}
	rootFD := fds[0]
	if rank != 0 {
		h, err := driver.MemImportFromShareableHandle(
			uintptr(rootFD), driver.MemHandleTypePosixFileDescriptor,
		)
		if err != nil {
			unix.Close(rootFD)
			return fmt.Errorf("cuMemImportFromShareableHandle(multicast): %w", err)
		}
		mcHandle = h
	}
	unix.Close(rootFD)

	// From here mcHandle is owned on every rank; release it on error.
	fail := func(err error) error {
		driver.MemRelease(mcHandle)
		return err
	}

	if err := buf.dev.MulticastAdd(mcHandle); err != nil {
		return fail(fmt.Errorf("cuMulticastAddDevice: %w", err))
	}
	// Bind only after every device has been added.
	if err := b.Barrier(); err != nil {
		return fail(fmt.Errorf("nvlshm: post-add barrier: %w", err))
	}

	if err := driver.MulticastBindMem(mcHandle, 0, ownedHandle, 0, uint(size), 0); err != nil {
		return fail(fmt.Errorf("cuMulticastBindMem: %w", err))
	}

	// Map the multicast object at a fresh VA. Alignment must be 0: the
	// driver aligns to granularity itself and rejects an explicit value.
	mcVA, err := driver.DevicePtr(0).MemAddressReserve(uint(size), 0, 0)
	if err != nil {
		return fail(fmt.Errorf("cuMemAddressReserve(multicast): %w", err))
	}
	if err := mcVA.MemMap(uint(size), 0, mcHandle, 0); err != nil {
		mcVA.MemAddressFree(uint(size))
		return fail(fmt.Errorf("cuMemMap(multicast): %w", err))
	}
	access := driver.MemAccessDesc{
		Location: driver.MemLocation{Type: driver.MemLocationTypeDevice, Id: int32(buf.dev)},
		Flags:    driver.MemAccessFlagsProtReadwrite,
	}
	if err := mcVA.MemSetAccess(uint(size), &access, 1); err != nil {
		mcVA.MemUnmap(uint(size))
		mcVA.MemAddressFree(uint(size))
		return fail(fmt.Errorf("cuMemSetAccess(multicast): %w", err))
	}

	// The multicast VA is usable only after every rank has bound.
	if err := b.Barrier(); err != nil {
		mcVA.MemUnmap(uint(size))
		mcVA.MemAddressFree(uint(size))
		return fail(fmt.Errorf("nvlshm: post-bind barrier: %w", err))
	}

	buf.mcVA = mcVA
	buf.mcHandle = mcHandle
	return nil
}
