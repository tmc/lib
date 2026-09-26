package linkoptions

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

func escaping() (driver.LinkState, error) {
	logBuf := make([]byte, 4096)
	opts := []driver.JITOption{driver.JITInfoLogBuffer, driver.JITInfoLogBufferSizeBytes}
	vals := []uintptr{uintptr(unsafe.Pointer(&logBuf[0])), uintptr(len(logBuf))}
	l, err := driver.LinkCreate(2, &opts[0], &vals[0]) // want `LinkCreate is given option buffers local to this function, but l outlives it; the linker writes them until LinkComplete`
	if err != nil {
		return 0, err
	}
	return l, nil
}

func local() (uintptr, error) {
	logBuf := make([]byte, 4096)
	opts := []driver.JITOption{driver.JITInfoLogBuffer, driver.JITInfoLogBufferSizeBytes}
	vals := []uintptr{uintptr(unsafe.Pointer(&logBuf[0])), uintptr(len(logBuf))}
	l, err := driver.LinkCreate(2, &opts[0], &vals[0])
	if err != nil {
		return 0, err
	}
	defer l.Close()
	cubin, _, err := l.Complete()
	return cubin, err
}
