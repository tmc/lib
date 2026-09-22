package unpinned

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

func upload(d driver.DevicePtr, host []byte, s driver.Stream) error {
	return d.MemcpyHtoDAsync(unsafe.Pointer(&host[0]), uint(len(host)), s) // want `MemcpyHtoDAsync takes the address of Go memory; pin it with a runtime.Pinner held until the stream completes`
}

func download(d driver.DevicePtr, host []float32, s driver.Stream) error {
	return d.MemcpyDtoHAsync(unsafe.Pointer(&host[0]), uint(len(host))*4, s) // want `MemcpyDtoHAsync takes the address of Go memory; pin it with a runtime.Pinner held until the stream completes`
}
