package nvrtcx

import (
	"bytes"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// JITLog contains driver diagnostics for one selected artifact. Messages are
// limited to 16 KiB each and may be truncated by the driver.
type JITLog struct {
	Output OutputKind
	Info   string
	Error  string
}

// LoadOutputWithLog loads one artifact and returns its driver JIT diagnostics,
// including when loading fails. It never falls back to another format.
// A nonzero returned module belongs to the caller even when err is non-nil
// (for example, if restoring the previous CUDA context fails). The caller must
// unload it with dev's context current before closing dev.
func (cm *CompiledModule) LoadOutputWithLog(dev *cuda.Device, kind OutputKind) (mod driver.Module, log JITLog, err error) {
	log.Output = kind
	if cm == nil {
		return 0, log, fmt.Errorf("nvrtcx: nil compiled module")
	}
	var image []byte
	switch kind {
	case OutputPTX:
		if cm.ptx != "" {
			image = append([]byte(cm.ptx), 0)
		}
	case OutputCUBIN:
		image = cm.cubin
	case OutputTileIR:
		image = cm.tileir
	default:
		return 0, log, fmt.Errorf("nvrtcx: output %d cannot be loaded directly", kind)
	}
	if len(image) == 0 {
		return 0, log, fmt.Errorf("nvrtcx: output %d is unavailable", kind)
	}
	if dev == nil {
		return 0, log, fmt.Errorf("nvrtcx: nil device passed to Load")
	}
	info, errors := make([]byte, 16<<10), make([]byte, 16<<10)
	options := []driver.JITOption{driver.JITInfoLogBuffer, driver.JITInfoLogBufferSizeBytes, driver.JITErrorLogBuffer, driver.JITErrorLogBufferSizeBytes, driver.JITLogVerbose}
	values := []uintptr{uintptr(unsafe.Pointer(&info[0])), uintptr(len(info)), uintptr(unsafe.Pointer(&errors[0])), uintptr(len(errors)), 1}
	err = dev.WithContext(func() error {
		var loadErr error
		mod, loadErr = driver.NewModuleLoadDataEx(unsafe.Pointer(&image[0]), uint32(len(options)), &options[0], &values[0])
		return loadErr
	})
	runtime.KeepAlive(image)
	runtime.KeepAlive(info)
	runtime.KeepAlive(errors)
	log.Info, log.Error = jitLogString(info), jitLogString(errors)
	if err != nil {
		err = fmt.Errorf("nvrtcx: load output %d: %w", kind, err)
	}
	return mod, log, err
}

func jitLogString(b []byte) string {
	if i := bytes.IndexByte(b, 0); i >= 0 {
		b = b[:i]
	}
	return string(b)
}

// LoadOwnedOutput loads one artifact and transfers it to a CUDA module owner.
// The owner retains its context independently of dev and must be closed.
// Keep dev open until this call returns. On failure no module is returned;
// if cleanup also fails, the error implements io.Closer for retry before dev
// closes. JIT diagnostics are returned on both success and failure.
func (cm *CompiledModule) LoadOwnedOutput(dev *cuda.Device, kind OutputKind) (*cuda.Module, JITLog, error) {
	raw, log, err := cm.LoadOutputWithLog(dev, kind)
	if err == nil {
		var owner *cuda.Module
		owner, err = cuda.OwnModule(dev, raw)
		if err == nil {
			return owner, log, nil
		}
	}
	pending := raw
	_, err = finishModuleLoad(raw, err, func() error {
		if pending == 0 {
			return nil
		}
		return dev.WithContext(func() error {
			if err := pending.Unload(); err != nil {
				return err
			}
			pending = 0
			return nil
		})
	})
	return nil, log, err
}
