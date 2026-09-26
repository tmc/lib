package jitlog

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

func whole(image []byte) (driver.Module, error) {
	logBuf := make([]byte, 4096)
	opts := []driver.JITOption{driver.JITInfoLogBuffer, driver.JITInfoLogBufferSizeBytes}
	vals := []uintptr{uintptr(unsafe.Pointer(&logBuf[0])), uintptr(len(logBuf))}
	m, err := driver.ModuleLoadDataEx(image, 2, &opts[0], &vals[0])
	if err != nil {
		return 0, fmt.Errorf("load module: %v: %s", err, string(logBuf)) // want `string\(logBuf\) takes the whole log buffer; only the prefix the JIT reports filling holds the log`
	}
	return m, nil
}

func prefix(image []byte) (driver.Module, error) {
	logBuf := make([]byte, 4096)
	opts := []driver.JITOption{driver.JITInfoLogBuffer, driver.JITInfoLogBufferSizeBytes}
	vals := []uintptr{uintptr(unsafe.Pointer(&logBuf[0])), uintptr(len(logBuf))}
	m, err := driver.ModuleLoadDataEx(image, 2, &opts[0], &vals[0])
	if err != nil {
		return 0, fmt.Errorf("load module: %v: %s", err, string(logBuf[:vals[1]]))
	}
	return m, nil
}
