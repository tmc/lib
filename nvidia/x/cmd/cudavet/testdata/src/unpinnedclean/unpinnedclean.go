// Package unpinnedclean holds the host buffer still for the transfer, and
// must produce no diagnostics.
package unpinnedclean

import (
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

func upload(d driver.DevicePtr, host []byte, s driver.Stream) error {
	var pinner runtime.Pinner
	pinner.Pin(&host[0])
	defer pinner.Unpin()
	return d.CopyFromHostAsyncPtr(unsafe.Pointer(&host[0]), uint(len(host)), s)
}

// fromDevice copies between device allocations, so no Go memory is involved.
func fromDevice(d driver.DevicePtr, staging unsafe.Pointer, s driver.Stream) error {
	return d.CopyToHostAsyncPtr(staging, 64, s)
}

func uploadSlice(d driver.DevicePtr, host []float32, s driver.Stream) error {
	var pinner runtime.Pinner
	pinner.Pin(&host[0])
	defer pinner.Unpin()
	return d.CopyFromHostAsync(host, s)
}

// fromHostAlloc views memory the driver allocated, which does not move.
func fromHostAlloc(d driver.DevicePtr, staging unsafe.Pointer, s driver.Stream) error {
	return d.CopyToHostAsync(unsafe.Slice((*float32)(staging), 16), s)
}
