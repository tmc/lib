package unpinned

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

func upload(d driver.DevicePtr, host []byte, s driver.Stream) error {
	return d.CopyFromHostAsyncPtr(unsafe.Pointer(&host[0]), uint(len(host)), s) // want `CopyFromHostAsyncPtr takes the address of Go memory; pin it with a runtime.Pinner held until the stream completes`
}

func download(d driver.DevicePtr, host []float32, s driver.Stream) error {
	return d.CopyToHostAsyncPtr(unsafe.Pointer(&host[0]), uint(len(host))*4, s) // want `CopyToHostAsyncPtr takes the address of Go memory; pin it with a runtime.Pinner held until the stream completes`
}

// toArray passes the host pointer second, after the array offset.
func toArray(a driver.Array, host []byte, s driver.Stream) error {
	return a.CopyFromHostAsyncPtr(0, unsafe.Pointer(&host[0]), uint(len(host)), s) // want `CopyFromHostAsyncPtr takes the address of Go memory; pin it with a runtime.Pinner held until the stream completes`
}

func uploadSlice(d driver.DevicePtr, host []float32, s driver.Stream) error {
	return d.CopyFromHostAsync(host, s) // want `CopyFromHostAsync takes a Go slice; pin its first element with a runtime.Pinner held until the stream completes`
}
