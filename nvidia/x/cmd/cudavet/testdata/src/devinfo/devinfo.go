package devinfo

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cusolver"
	"github.com/tmc/lib/nvidia/driver"
)

func unchecked(h cusolver.Handle, a, work, ipiv driver.DevicePtr, n int32) error {
	info, err := driver.MemAlloc[int32](1)
	if err != nil {
		return err
	}
	defer info.Free()
	return h.Dgetrf(n, n, a, n, work, ipiv, info) // want `info from Dgetrf is never copied back and checked; a negative value is a bad parameter and a positive one a failure such as a singular pivot`
}

func checked(h cusolver.Handle, a, work, ipiv driver.DevicePtr, n int32) error {
	info, err := driver.MemAlloc[int32](1)
	if err != nil {
		return err
	}
	defer info.Free()
	if err := h.Dgetrf(n, n, a, n, work, ipiv, info); err != nil {
		return err
	}
	host := make([]int32, 1)
	if err := info.CopyToHost(host); err != nil {
		return err
	}
	if host[0] != 0 {
		return fmt.Errorf("getrf: info %d", host[0])
	}
	return nil
}

func handedBack(h cusolver.Handle, a, work, ipiv, info driver.DevicePtr, n int32) (driver.DevicePtr, error) {
	return info, h.Dgetrf(n, n, a, n, work, ipiv, info)
}
