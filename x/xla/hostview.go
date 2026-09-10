package xla

import (
	"errors"
	"fmt"
	"unsafe"
)

// ErrHostAccessUnavailable reports that direct CPU host access was not enabled
// or the buffer does not have a supported CPU layout. Callers may use ToHost.
var ErrHostAccessUnavailable = errors.New("xla: direct CPU host access unavailable")

// EnableCPUHostAccess asserts that this engine's plugin returns host-addressable
// CPU memory from OpaqueDeviceMemoryDataPointer while an external reference is
// held. PJRT does not guarantee this for arbitrary plugins. Call this only after
// qualifying the plugin, before using its buffers. It does not enable mutation.
func (e *Engine) EnableCPUHostAccess() error {
	if e == nil {
		return fmt.Errorf("xla: engine is closed")
	}
	e.viewMu.Lock()
	defer e.viewMu.Unlock()
	if e.client == nil {
		return fmt.Errorf("xla: engine is closed")
	}
	e.cpuHostAccess = true
	return nil
}

// WithReadOnlyFloats calls read with a direct view of a dense F32 CPU buffer.
// The plugin must have been qualified and EnableCPUHostAccess called first.
// read must neither mutate nor retain the slice. The buffer cannot be closed
// during the callback; Close returns an error and leaves it open. Engine.Close
// also returns an error during a view. The caller must serialize other native
// buffer operations.
// The external reference is released on return, including callback errors.
func (b *Buffer) WithReadOnlyFloats(read func([]float32) error) (err error) {
	if b == nil {
		return fmt.Errorf("xla: buffer is closed")
	}
	b.viewMu.Lock()
	if b.wrapper == nil {
		b.viewMu.Unlock()
		return fmt.Errorf("xla: buffer is closed")
	}
	if b.viewing {
		b.viewMu.Unlock()
		return fmt.Errorf("xla: buffer has an active host view")
	}
	if read == nil {
		b.viewMu.Unlock()
		return fmt.Errorf("xla: nil host view callback")
	}
	if b.engine == nil || b.elemTyp != F32 {
		b.viewMu.Unlock()
		return ErrHostAccessUnavailable
	}
	engine := b.engine
	engine.viewMu.Lock()
	if engine.client == nil {
		engine.viewMu.Unlock()
		b.viewMu.Unlock()
		return fmt.Errorf("xla: engine is closed")
	}
	if !engine.cpuHostAccess {
		engine.viewMu.Unlock()
		b.viewMu.Unlock()
		return ErrHostAccessUnavailable
	}
	engine.hostViews++
	engine.viewMu.Unlock()
	defer func() {
		engine.viewMu.Lock()
		engine.hostViews--
		engine.viewMu.Unlock()
	}()
	b.viewing = true
	wrapper := b.wrapper
	b.viewMu.Unlock()
	defer func() {
		b.viewMu.Lock()
		b.viewing = false
		b.viewMu.Unlock()
	}()
	cpu, err := wrapper.IsOnCPU()
	if err != nil {
		return err
	}
	if !cpu {
		return ErrHostAccessUnavailable
	}
	dense, err := wrapper.IsDenseRowMajor()
	if err != nil {
		return err
	}
	if !dense {
		return ErrHostAccessUnavailable
	}
	size, err := bufferSize(b.elemTyp, b.dims)
	if err != nil {
		return err
	}
	extent, err := wrapper.OnDeviceSizeInBytes()
	if err != nil {
		return err
	}
	if extent != uint64(size) {
		return fmt.Errorf("xla: host view storage has %d bytes, want %d", extent, size)
	}
	if err := wrapper.IncreaseExternalReferenceCount(); err != nil {
		return err
	}
	defer func() { err = errors.Join(err, wrapper.DecreaseExternalReferenceCount()) }()
	if err := wrapper.WaitReady(); err != nil {
		return err
	}
	data, err := wrapper.OpaqueDeviceMemoryDataPointer()
	if err != nil {
		return err
	}
	if size != 0 && data == nil {
		return fmt.Errorf("xla: nil host view pointer")
	}
	if uintptr(data)%unsafe.Alignof(float32(0)) != 0 {
		return fmt.Errorf("xla: unaligned host view pointer")
	}
	return read(unsafe.Slice((*float32)(data), size/4))
}
