package gomlx

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
)

// NewSharedBuffer allocates mutable CPU input storage. Callers must finish
// mutations before Execute and must not donate this buffer. HasSharedBuffers
// remains false because native outputs do not expose owned shared storage.
func (b *Backend) NewSharedBuffer(device compute.DeviceNum, s shapes.Shape) (compute.Buffer, any, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.engine == nil {
		return nil, nil, fmt.Errorf("backend closed")
	}
	if device != 0 {
		return nil, nil, unsupported
	}
	if err := validate(s); err != nil {
		return nil, nil, err
	}
	info, ok := element(s.DType)
	if !ok {
		return nil, nil, unsupported
	}
	devices, err := b.engine.Devices()
	if err != nil {
		return nil, nil, err
	}
	if len(devices) == 0 {
		return nil, nil, fmt.Errorf("no addressable devices")
	}
	dims := make([]int64, s.Rank())
	for i, n := range s.Dimensions {
		dims[i] = int64(n)
	}
	raw, data, err := b.engine.NewSharedBuffer(info.pjrt, dims, devices[0])
	if err != nil {
		return nil, nil, err
	}
	out := &buffer{backend: b, raw: raw, shape: s.Clone()}
	b.buffers[out] = struct{}{}
	return out, sharedFlat(s, data), nil
}

func sharedFlat(s shapes.Shape, data []byte) any {
	return reflect.SliceAt(s.DType.GoType(), unsafe.Pointer(unsafe.SliceData(data)), s.Size()).Interface()
}

func (b *buffer) Data() (any, error) {
	b.backend.mu.Lock()
	defer b.backend.mu.Unlock()
	if b.raw == nil {
		return nil, fmt.Errorf("buffer closed")
	}
	if !b.raw.IsShared() {
		return nil, unsupported
	}
	data, err := b.raw.SharedData()
	if err != nil {
		return nil, err
	}
	return sharedFlat(b.shape, data), nil
}
