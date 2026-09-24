package unpinned

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

func upload(d driver.DevicePtr, host []byte, s driver.Stream) error {
	return d.CopyFromHostAsync(unsafe.Pointer(&host[0]), uint(len(host)), s) // want `CopyFromHostAsync takes the address of Go memory; pin it with a runtime.Pinner held until the stream completes`
}

func download(d driver.DevicePtr, host []float32, s driver.Stream) error {
	return d.CopyToHostAsync(unsafe.Pointer(&host[0]), uint(len(host))*4, s) // want `CopyToHostAsync takes the address of Go memory; pin it with a runtime.Pinner held until the stream completes`
}
