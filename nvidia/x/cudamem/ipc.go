package cudamem

import (
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
)

// PageAlignBytes is the 4096-byte page granularity used for direct device
// allocations exported across processes via CUDA IPC.
const PageAlignBytes int64 = 4096

// RoundUpPageAligned rounds a positive byte count n up to the next multiple of
// [PageAlignBytes] (4096 bytes). It returns an error if n <= 0 or if rounding
// would overflow int64.
func RoundUpPageAligned(n int64) (int64, error) {
	if n <= 0 {
		return 0, fmt.Errorf("cudamem: page-aligned size must be > 0, got %d", n)
	}
	if n > math.MaxInt64-(PageAlignBytes-1) {
		return 0, fmt.Errorf("cudamem: page-aligned size overflows int64 for %d", n)
	}
	return ((n + (PageAlignBytes - 1)) / PageAlignBytes) * PageAlignBytes, nil
}

// IPCHandle is a 64-byte interprocess CUDA memory handle matching
// [driver.IpcMemHandle] (CUipcMemHandle).
type IPCHandle [64]byte

// Bytes returns a 64-byte copy of the IPC memory handle.
func (h IPCHandle) Bytes() []byte {
	out := make([]byte, len(h))
	copy(out, h[:])
	return out
}

// Hex returns the lowercase 128-character hexadecimal encoding of h.
func (h IPCHandle) Hex() string {
	return hex.EncodeToString(h[:])
}

// IsZero reports whether every byte in h is zero.
func (h IPCHandle) IsZero() bool {
	return h == (IPCHandle{})
}

// ParseIPCHandleHex decodes a 128-character hexadecimal string (with optional
// leading "0x" or "0X" prefix) into an [IPCHandle].
func ParseIPCHandleHex(s string) (IPCHandle, error) {
	trimmed := strings.TrimSpace(s)
	trimmed = strings.TrimPrefix(trimmed, "0x")
	trimmed = strings.TrimPrefix(trimmed, "0X")
	if len(trimmed) != 2*len(IPCHandle{}) {
		return IPCHandle{}, fmt.Errorf("cudamem: invalid IPC handle hex length %d, want %d", len(trimmed), 2*len(IPCHandle{}))
	}
	raw, err := hex.DecodeString(trimmed)
	if err != nil {
		return IPCHandle{}, fmt.Errorf("cudamem: parse IPC handle hex: %w", err)
	}
	var h IPCHandle
	copy(h[:], raw)
	return h, nil
}

// PageAlignedAlloc describes a page-aligned device memory allocation created via
// [AllocPageAligned].
type PageAlignedAlloc struct {
	Ptr            driver.DevicePtr
	RequestedBytes int64
	AlignedBytes   int64
}

var (
	memAllocPageAlignedFn = func(alignedBytes int64) (driver.DevicePtr, error) {
		if uint64(alignedBytes) > uint64(^uint(0)) {
			return 0, fmt.Errorf("allocation size %d overflows uint", alignedBytes)
		}
		return driver.MemAlloc[byte](int(alignedBytes))
	}
	ipcGetMemHandleFn = func(ptr driver.DevicePtr) (driver.IpcMemHandle, error) {
		return ptr.IpcGetMemHandle()
	}
	ipcOpenMemHandleFn = func(raw driver.IpcMemHandle, flags driver.IpcmemFlags) (driver.DevicePtr, error) {
		return driver.IpcOpenMemHandle(raw, flags)
	}
	ipcCloseMemHandleFn = func(ptr driver.DevicePtr) error {
		return ptr.IpcCloseMemHandle()
	}
)

// AllocPageAligned rounds sizeBytes up to a 4096-byte page boundary via
// [RoundUpPageAligned] and allocates linear device memory via [driver.MemAlloc].
func AllocPageAligned(sizeBytes int64) (PageAlignedAlloc, error) {
	aligned, err := RoundUpPageAligned(sizeBytes)
	if err != nil {
		return PageAlignedAlloc{}, err
	}
	ptr, err := memAllocPageAlignedFn(aligned)
	if err != nil {
		return PageAlignedAlloc{}, fmt.Errorf("cudamem: alloc page-aligned (%d -> %d bytes): %w", sizeBytes, aligned, err)
	}
	return PageAlignedAlloc{
		Ptr:            ptr,
		RequestedBytes: sizeBytes,
		AlignedBytes:   aligned,
	}, nil
}

// ExportIPC exports a 64-byte interprocess memory handle for an existing base
// device allocation via [driver.DevicePtr.IpcGetMemHandle].
func ExportIPC(ptr driver.DevicePtr) (IPCHandle, error) {
	if ptr == 0 {
		return IPCHandle{}, errors.New("cudamem: cannot export IPC handle for zero device pointer")
	}
	raw, err := ipcGetMemHandleFn(ptr)
	if err != nil {
		return IPCHandle{}, fmt.Errorf("cudamem: export IPC handle: %w", err)
	}
	var h IPCHandle
	copy(h[:], raw.Reserved[:])
	return h, nil
}

// OpenIPC maps an exported [IPCHandle] into the current CUDA context with
// [driver.IpcMemLazyEnablePeerAccess]. The returned pointer must be released
// with [CloseIPC].
func OpenIPC(h IPCHandle) (driver.DevicePtr, error) {
	if h.IsZero() {
		return 0, errors.New("cudamem: cannot open zero IPC handle")
	}
	var raw driver.IpcMemHandle
	copy(raw.Reserved[:], h[:])
	ptr, err := ipcOpenMemHandleFn(raw, driver.IpcMemLazyEnablePeerAccess)
	if err != nil {
		return 0, fmt.Errorf("cudamem: open IPC handle: %w", err)
	}
	return ptr, nil
}

// CloseIPC unmaps a device pointer previously opened via [OpenIPC].
func CloseIPC(ptr driver.DevicePtr) error {
	if ptr == 0 {
		return errors.New("cudamem: cannot close zero IPC device pointer")
	}
	if err := ipcCloseMemHandleFn(ptr); err != nil {
		return fmt.Errorf("cudamem: close IPC handle: %w", err)
	}
	return nil
}
