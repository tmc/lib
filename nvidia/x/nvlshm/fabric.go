//go:build !windows

package nvlshm

import (
	"encoding/hex"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// FabricHandleSize is the byte size of CUmemFabricHandle (64 bytes in CUDA driver API).
const FabricHandleSize = 64

// FabricHandle is a 64-byte opaque CUDA IMEX fabric shareable memory handle
// (CUmemFabricHandle) that can be exchanged across nodes over TCP/MPI Bootstrap.
type FabricHandle [FabricHandleSize]byte

// String returns the lowercase hexadecimal encoding of the 64-byte fabric handle.
func (h FabricHandle) String() string {
	return hex.EncodeToString(h[:])
}

// ParseFabricHandle decodes a 128-character hexadecimal string into a FabricHandle.
func ParseFabricHandle(s string) (FabricHandle, error) {
	var fh FabricHandle
	if len(s) != FabricHandleSize*2 {
		return fh, fmt.Errorf("nvlshm: fabric handle hex length %d, want %d", len(s), FabricHandleSize*2)
	}
	n, err := hex.Decode(fh[:], []byte(s))
	if err != nil {
		return FabricHandle{}, fmt.Errorf("nvlshm: decode fabric handle: %w", err)
	}
	if n != FabricHandleSize {
		return FabricHandle{}, fmt.Errorf("nvlshm: decoded %d bytes, want %d", n, FabricHandleSize)
	}
	return fh, nil
}

// fabricAllocProp returns the VMM allocation properties for multi-node IMEX
// fabric-exportable device memory on dev.
func fabricAllocProp(dev driver.Device) driver.MemAllocationProp {
	return driver.MemAllocationProp{
		Type:                 driver.MemAllocationTypePinned,
		RequestedHandleTypes: driver.MemHandleTypeFabric,
		Location: driver.MemLocation{
			Type: driver.MemLocationTypeDevice,
			Id:   int32(dev),
		},
	}
}

// FabricSupported reports whether dev supports exporting VMM allocations as
// CU_MEM_HANDLE_TYPE_FABRIC handles over the IMEX NVLink fabric.
func FabricSupported(dev driver.Device) (bool, error) {
	v, err := dev.Attribute(driver.DevAttrHandleTypeFabricSupported)
	if err != nil {
		return false, fmt.Errorf("cuDeviceGetAttribute(HANDLE_TYPE_FABRIC_SUPPORTED): %w", err)
	}
	return v != 0, nil
}

// FabricGranularity returns the recommended VMM allocation granularity in bytes
// for CU_MEM_HANDLE_TYPE_FABRIC chunks on dev.
func FabricGranularity(dev driver.Device) (uintptr, error) {
	prop := fabricAllocProp(dev)
	g, err := driver.MemGetAllocationGranularity(&prop, driver.MemAllocGranularityRecommended)
	if err != nil {
		return 0, fmt.Errorf("cuMemGetAllocationGranularity(fabric): %w", err)
	}
	return uintptr(g), nil
}

// FabricMulticastGranularity returns the recommended multicast granularity in
// bytes for a fabric-backed multicast group of numDevices devices.
func FabricMulticastGranularity(numDevices int) (uintptr, error) {
	prop := driver.MulticastObjectProp{
		NumDevices:  uint32(numDevices),
		HandleTypes: uint64(driver.MemHandleTypeFabric),
	}
	g, err := driver.MulticastGetGranularity(&prop, driver.MulticastGranularityRecommended)
	if err != nil {
		return 0, fmt.Errorf("cuMulticastGetGranularity(fabric): %w", err)
	}
	return uintptr(g), nil
}

// ExchangeFabricHandles exchanges 64-byte CUmemFabricHandle values across all
// ranks in [0, worldSize) using boot.BroadcastString and boot.Barrier. Every
// rank receives the full slice of worldSize handles in rank order without
// requiring local AF_UNIX sockets.
func ExchangeFabricHandles(boot Bootstrap, rank, worldSize int, local FabricHandle) ([]FabricHandle, error) {
	if boot == nil {
		return nil, fmt.Errorf("nvlshm: nil bootstrap")
	}
	if worldSize <= 0 || rank < 0 || rank >= worldSize {
		return nil, fmt.Errorf("nvlshm: invalid rank %d world %d", rank, worldSize)
	}
	if boot.Rank() != rank || boot.WorldSize() != worldSize {
		return nil, fmt.Errorf("nvlshm: bootstrap rank/world (%d/%d) mismatch with caller (%d/%d)",
			boot.Rank(), boot.WorldSize(), rank, worldSize)
	}

	localHex := local.String()
	handles := make([]FabricHandle, worldSize)
	for srcRank := 0; srcRank < worldSize; srcRank++ {
		var send string
		if rank == srcRank {
			send = localHex
		}
		got, err := boot.BroadcastString(send, srcRank)
		if err != nil {
			return nil, fmt.Errorf("nvlshm: broadcast fabric handle from rank %d: %w", srcRank, err)
		}
		fh, err := ParseFabricHandle(got)
		if err != nil {
			return nil, fmt.Errorf("nvlshm: invalid fabric handle from rank %d: %w", srcRank, err)
		}
		handles[srcRank] = fh
	}
	if err := boot.Barrier(); err != nil {
		return nil, fmt.Errorf("nvlshm: post-fabric-exchange barrier: %w", err)
	}
	return handles, nil
}

func allocChunkFabric(dev driver.Device, size uintptr) (handle uint64, fh FabricHandle, err error) {
	prop := fabricAllocProp(dev)
	handle, err = driver.MemCreate(uint(size), &prop, 0)
	if err != nil {
		return 0, FabricHandle{}, fmt.Errorf("cuMemCreate(fabric, %d bytes): %w", size, err)
	}
	var pin runtime.Pinner
	pin.Pin(&fh[0])
	err = driver.MemExportToShareableHandle(
		unsafe.Pointer(&fh[0]), handle, driver.MemHandleTypeFabric, 0,
	)
	pin.Unpin()
	if err != nil {
		driver.MemRelease(handle)
		return 0, FabricHandle{}, fmt.Errorf("cuMemExportToShareableHandle(fabric): %w", err)
	}
	return handle, fh, nil
}

func mapChunksFabric(dev driver.Device, chunkBytes uintptr, handles []FabricHandle) (driver.DevicePtr, error) {
	if chunkBytes == 0 || len(handles) == 0 || chunkBytes > ^uintptr(0)/uintptr(len(handles)) {
		return 0, fmt.Errorf("nvlshm: mapped fabric size overflows uintptr")
	}
	total := chunkBytes * uintptr(len(handles))
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
	for i := range handles {
		fh := handles[i]
		var pin runtime.Pinner
		pin.Pin(&fh[0])
		handle, err := driver.MemImportFromShareableHandle(
			uintptr(unsafe.Pointer(&fh[0])), driver.MemHandleTypeFabric,
		)
		pin.Unpin()
		if err != nil {
			unwind(i)
			return 0, fmt.Errorf("cuMemImportFromShareableHandle(fabric chunk %d): %w", i, err)
		}
		va := base + driver.DevicePtr(uintptr(i)*chunkBytes)
		if err := va.MemMap(uint(chunkBytes), 0, handle, 0); err != nil {
			driver.MemRelease(handle)
			unwind(i)
			return 0, fmt.Errorf("cuMemMap(fabric chunk %d): %w", i, err)
		}
		if err := va.MemSetAccess(uint(chunkBytes), &access, 1); err != nil {
			driver.MemRelease(handle)
			unwind(i + 1)
			return 0, fmt.Errorf("cuMemSetAccess(fabric chunk %d): %w", i, err)
		}
		if err := driver.MemRelease(handle); err != nil {
			unwind(i + 1)
			return 0, fmt.Errorf("cuMemRelease(imported fabric chunk %d): %w", i, err)
		}
	}
	return base, nil
}

func checkAlignedFabric(dev driver.Device, chunkBytes uintptr, multicastDevices int) error {
	if chunkBytes == 0 {
		return fmt.Errorf("nvlshm: chunk size must be positive")
	}
	g, err := FabricGranularity(dev)
	if err != nil {
		return err
	}
	need := g
	if multicastDevices > 0 {
		mg, err := FabricMulticastGranularity(multicastDevices)
		if err != nil {
			return err
		}
		if mg > need {
			need = mg
		}
	}
	if chunkBytes%need != 0 {
		return fmt.Errorf("nvlshm: fabric chunk size %d is not a multiple of the required granularity %d; pad with PadTo",
			chunkBytes, need)
	}
	return nil
}

// DistAllocFabric collectively builds a multi-node distributed buffer backed by
// CU_MEM_HANDLE_TYPE_FABRIC handles over the IMEX NVLink fabric. Each rank in
// [0, worldSize) allocates chunkBytes on dev, exports a 64-byte FabricHandle,
// exchanges handles over boot, and maps all worldSize chunks into a contiguous
// read-write VA range.
func DistAllocFabric(boot Bootstrap, rank, worldSize int, dev driver.Device, chunkBytes uintptr) (*DistBuffer, error) {
	return distAllocFabric(boot, rank, worldSize, dev, chunkBytes, false)
}

// DistAllocFabricMulticast is DistAllocFabric plus an NVSwitch multicast
// mapping backed by CU_MEM_HANDLE_TYPE_FABRIC over the same physical chunks.
func DistAllocFabricMulticast(boot Bootstrap, rank, worldSize int, dev driver.Device, chunkBytes uintptr) (*DistBuffer, error) {
	return distAllocFabric(boot, rank, worldSize, dev, chunkBytes, true)
}

func distAllocFabric(boot Bootstrap, rank, worldSize int, dev driver.Device, chunkBytes uintptr, multicast bool) (*DistBuffer, error) {
	if boot == nil {
		return nil, fmt.Errorf("nvlshm: nil bootstrap")
	}
	if worldSize <= 0 || rank < 0 || rank >= worldSize {
		return nil, fmt.Errorf("nvlshm: invalid rank %d world %d", rank, worldSize)
	}
	if boot.Rank() != rank || boot.WorldSize() != worldSize {
		return nil, fmt.Errorf("nvlshm: bootstrap rank/world (%d/%d) mismatch with caller (%d/%d)",
			boot.Rank(), boot.WorldSize(), rank, worldSize)
	}
	if chunkBytes != 0 && uintptr(worldSize) > ^uintptr(0)/chunkBytes {
		return nil, fmt.Errorf("nvlshm: total size overflows uintptr")
	}
	if ok, err := FabricSupported(dev); err != nil {
		return nil, err
	} else if !ok {
		return nil, fmt.Errorf("nvlshm: device %d does not support CU_MEM_HANDLE_TYPE_FABRIC", dev)
	}

	mcDevices := 0
	if multicast {
		mcDevices = worldSize
		ok, err := MulticastSupported(dev)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("nvlshm: device %d does not support multicast", dev)
		}
	}
	if err := checkAlignedFabric(dev, chunkBytes, mcDevices); err != nil {
		return nil, err
	}

	handle, localFH, err := allocChunkFabric(dev, chunkBytes)
	if err != nil {
		return nil, err
	}
	defer driver.MemRelease(handle)

	handles, err := ExchangeFabricHandles(boot, rank, worldSize, localFH)
	if err != nil {
		return nil, err
	}

	base, err := mapChunksFabric(dev, chunkBytes, handles)
	if err != nil {
		return nil, err
	}

	buf := &DistBuffer{
		base:       base,
		chunkBytes: chunkBytes,
		world:      worldSize,
		rank:       rank,
		dev:        dev,
	}
	if multicast {
		if err := buf.attachMulticastFabric(boot, rank, worldSize, handle); err != nil {
			buf.Close()
			return nil, err
		}
	}
	return buf, nil
}

func (buf *DistBuffer) attachMulticastFabric(boot Bootstrap, rank, worldSize int, ownedHandle uint64) error {
	size := buf.chunkBytes

	var mcHandle uint64
	var rootFH FabricHandle
	if rank == 0 {
		prop := driver.MulticastObjectProp{
			NumDevices:  uint32(worldSize),
			Size:        uint(size),
			HandleTypes: uint64(driver.MemHandleTypeFabric),
		}
		h, err := driver.MulticastCreate(&prop)
		if err != nil {
			return fmt.Errorf("cuMulticastCreate(fabric): %w", err)
		}
		mcHandle = h
		var pin runtime.Pinner
		pin.Pin(&rootFH[0])
		err = driver.MemExportToShareableHandle(
			unsafe.Pointer(&rootFH[0]), mcHandle, driver.MemHandleTypeFabric, 0,
		)
		pin.Unpin()
		if err != nil {
			driver.MemRelease(mcHandle)
			return fmt.Errorf("cuMemExportToShareableHandle(fabric multicast): %w", err)
		}
	}

	var send string
	if rank == 0 {
		send = rootFH.String()
	}
	got, err := boot.BroadcastString(send, 0)
	if err != nil {
		if rank == 0 {
			driver.MemRelease(mcHandle)
		}
		return fmt.Errorf("nvlshm: broadcast multicast fabric handle: %w", err)
	}
	rootFH, err = ParseFabricHandle(got)
	if err != nil {
		if rank == 0 {
			driver.MemRelease(mcHandle)
		}
		return fmt.Errorf("nvlshm: parse multicast fabric handle: %w", err)
	}

	if rank != 0 {
		var pin runtime.Pinner
		pin.Pin(&rootFH[0])
		h, err := driver.MemImportFromShareableHandle(
			uintptr(unsafe.Pointer(&rootFH[0])), driver.MemHandleTypeFabric,
		)
		pin.Unpin()
		if err != nil {
			return fmt.Errorf("cuMemImportFromShareableHandle(fabric multicast): %w", err)
		}
		mcHandle = h
	}

	fail := func(err error) error {
		driver.MemRelease(mcHandle)
		return err
	}

	if err := buf.dev.MulticastAdd(mcHandle); err != nil {
		return fail(fmt.Errorf("cuMulticastAddDevice(fabric): %w", err))
	}
	if err := boot.Barrier(); err != nil {
		return fail(fmt.Errorf("nvlshm: post-add fabric barrier: %w", err))
	}

	if err := driver.MulticastBindMem(mcHandle, 0, ownedHandle, 0, uint(size), 0); err != nil {
		return fail(fmt.Errorf("cuMulticastBindMem(fabric): %w", err))
	}

	mcVA, err := driver.DevicePtr(0).MemAddressReserve(uint(size), 0, 0)
	if err != nil {
		return fail(fmt.Errorf("cuMemAddressReserve(fabric multicast): %w", err))
	}
	if err := mcVA.MemMap(uint(size), 0, mcHandle, 0); err != nil {
		mcVA.MemAddressFree(uint(size))
		return fail(fmt.Errorf("cuMemMap(fabric multicast): %w", err))
	}
	access := driver.MemAccessDesc{
		Location: driver.MemLocation{Type: driver.MemLocationTypeDevice, Id: int32(buf.dev)},
		Flags:    driver.MemAccessFlagsProtReadwrite,
	}
	if err := mcVA.MemSetAccess(uint(size), &access, 1); err != nil {
		mcVA.MemUnmap(uint(size))
		mcVA.MemAddressFree(uint(size))
		return fail(fmt.Errorf("cuMemSetAccess(fabric multicast): %w", err))
	}

	if err := boot.Barrier(); err != nil {
		mcVA.MemUnmap(uint(size))
		mcVA.MemAddressFree(uint(size))
		return fail(fmt.Errorf("nvlshm: post-bind fabric barrier: %w", err))
	}

	buf.mcVA = mcVA
	buf.mcHandle = mcHandle
	return nil
}
