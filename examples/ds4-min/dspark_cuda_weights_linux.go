//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaDSparkWeights owns resident immutable support bytes on the target
// backend's device. It borrows the device context, not target State, streams,
// model mappings, or arenas. The caller must close every consuming request
// before closing this owner. No native operation may fall back to CPU reads.
type cudaDSparkWeights struct {
	mu            sync.Mutex
	device        *cuda.Device
	model         *dsparkModel
	tensors       map[string]cudaModelTensor
	storage       []cudamem.Slice[byte]
	closed        bool
	module        driver.Module
	moduleCleanup []driver.Module
	matrix        driver.Function
}

func newCUDADSparkWeights(device *cuda.Device, model *dsparkModel) (_ *cudaDSparkWeights, err error) {
	if device == nil || model == nil || model.reader == nil || len(model.sha256) != 64 || model.stages < 3 || len(model.targets) == 0 {
		return nil, fmt.Errorf("cuda dspark: verified support model and device required")
	}
	names := make([]string, 0, len(model.tensors))
	var total uint64
	for name, tensor := range model.tensors {
		bytes, err := tensor.Bytes()
		if err != nil || bytes == 0 || bytes > math.MaxInt32 || total > 8<<30-bytes {
			return nil, fmt.Errorf("cuda dspark: support allocation exceeds 8 GiB budget")
		}
		if _, err := dsparkDataOffset(model.dataOffset, tensor.Offset, bytes); err != nil {
			return nil, err
		}
		total += bytes
		names = append(names, name)
	}
	sort.Strings(names)
	w := &cudaDSparkWeights{device: device, model: model, tensors: make(map[string]cudaModelTensor)}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, w)
		}
	}()
	// A bounded staging buffer avoids a second host copy of the support file.
	host := make([]byte, 1<<20)
	for _, name := range names {
		tensor, _ := model.tensor(name)
		bytes, e := tensor.Bytes()
		if e != nil {
			return nil, e
		}
		memory, e := cudamem.Alloc[byte](device, int(bytes))
		if memory.Ptr() != 0 {
			w.storage = append(w.storage, memory)
		}
		if e != nil {
			return nil, fmt.Errorf("cuda dspark allocate %s: %w", name, e)
		}
		for start := uint64(0); start < bytes; {
			length := min(uint64(len(host)), bytes-start)
			chunk := host[:length]
			offset, e := dsparkDataOffset(model.dataOffset, tensor.Offset, start)
			if e != nil {
				return nil, e
			}
			n, e := model.reader.ReadAt(chunk, offset)
			if e != nil {
				return nil, fmt.Errorf("cuda dspark read %s: %w", name, e)
			}
			if n != len(chunk) {
				return nil, fmt.Errorf("cuda dspark read %s: %w", name, io.ErrUnexpectedEOF)
			}
			if e := memory.Slice(int(start), int(start+length)).CopyFromHost(chunk); e != nil {
				return nil, fmt.Errorf("cuda dspark upload %s: %w", name, e)
			}
			start += length
		}
		w.tensors[name] = cudaModelTensor{name: name, typeID: tensor.TypeID, dimensions: append([]uint64(nil), tensor.Dimensions...), device: memory.Ptr(), bytes: bytes}
	}
	return w, nil
}

func (w *cudaDSparkWeights) Close() error { return w.close() }

func (w *cudaDSparkWeights) tensor(name string) (cudaModelTensor, error) {
	if w == nil {
		return cudaModelTensor{}, fmt.Errorf("cuda dspark: support weights closed")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return cudaModelTensor{}, fmt.Errorf("cuda dspark: support weights closed")
	}
	t, ok := w.tensors[name]
	if !ok {
		return cudaModelTensor{}, fmt.Errorf("cuda dspark: missing tensor %s", name)
	}
	t.dimensions = append([]uint64(nil), t.dimensions...)
	return t, nil
}

func (w *cudaDSparkWeights) close() error {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed && len(w.storage) == 0 && w.module == 0 && len(w.moduleCleanup) == 0 {
		return nil
	}
	w.closed = true
	if err := w.device.WithContext(driver.CtxSynchronize); err != nil {
		return err
	}
	if w.module != 0 {
		if err := w.device.WithContext(func() error {
			if err := w.module.Close(); err != nil {
				return err
			}
			w.module, w.matrix = 0, 0
			return nil
		}); err != nil {
			return err
		}
	}
	var remaining []driver.Module
	var cleanupErrors []error
	for _, module := range w.moduleCleanup {
		err := w.device.WithContext(func() error {
			if err := module.Close(); err != nil {
				return err
			}
			module = 0
			return nil
		})
		if module != 0 {
			remaining = append(remaining, module)
		}
		cleanupErrors = append(cleanupErrors, err)
	}
	w.moduleCleanup = remaining
	cleanupErr := errors.Join(cleanupErrors...)
	if cleanupErr != nil {
		return cleanupErr
	}
	var err error
	w.storage, err = releaseDSparkHandles(w.storage, func(memory cudamem.Slice[byte]) error { return memory.Free() })
	if err != nil {
		return err
	}
	w.storage = nil
	w.tensors = nil
	w.model = nil
	return nil
}
