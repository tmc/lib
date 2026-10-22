//go:build darwin && arm64

package main

import (
	"fmt"
	"io"
	"math"
	"sort"
	"sync"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// metalDSparkWeights owns support bytes, never target request state. Consumers
// must release their lease after their command buffers are terminal.
type metalDSparkWeights struct {
	mu      sync.Mutex
	device  *libmetal.Device
	model   *dsparkModel
	tensors map[string]boundMetalTensor
	storage []metalmem.Slice[byte]
	leases  int
	closed  bool
}

func metalDSparkWeightPlan(model *dsparkModel) ([]string, error) {
	if model == nil || model.reader == nil || len(model.sha256) != 64 || model.stages < 3 || len(model.targets) == 0 {
		return nil, fmt.Errorf("metal dspark: verified support model required")
	}
	names := make([]string, 0, len(model.tensors))
	var total uint64
	for name, t := range model.tensors {
		n, err := t.Bytes()
		if err != nil || n == 0 || n > math.MaxInt32 || n > 8<<30 || total > 8<<30-n {
			return nil, fmt.Errorf("metal dspark: support exceeds 8 GiB budget")
		}
		if _, err := dsparkDataOffset(model.dataOffset, t.Offset, n); err != nil {
			return nil, err
		}
		total += n
		names = append(names, name)
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("metal dspark: empty support tensor directory")
	}
	sort.Strings(names)
	return names, nil
}

func newMetalDSparkWeights(device *libmetal.Device, model *dsparkModel) (_ *metalDSparkWeights, err error) {
	if device == nil {
		return nil, fmt.Errorf("metal dspark: device required")
	}
	names, err := metalDSparkWeightPlan(model)
	if err != nil {
		return nil, err
	}
	w := &metalDSparkWeights{device: device, model: model, tensors: make(map[string]boundMetalTensor)}
	defer func() {
		if err != nil {
			for _, v := range w.storage {
				v.Free()
			}
		}
	}()
	for _, name := range names {
		t, _ := model.tensor(name)
		n, err := t.Bytes()
		if err != nil {
			return nil, err
		}
		memory, err := metalmem.Alloc[byte](device, int(n))
		if err != nil {
			return nil, fmt.Errorf("metal dspark allocate %s: %w", name, err)
		}
		w.storage = append(w.storage, memory)
		offset, err := dsparkDataOffset(model.dataOffset, t.Offset, 0)
		if err != nil {
			return nil, err
		}
		read, err := model.reader.ReadAt(memory.HostSlice(), offset)
		if err != nil {
			return nil, fmt.Errorf("metal dspark read %s: %w", name, err)
		}
		if read != int(n) {
			return nil, fmt.Errorf("metal dspark read %s: %w", name, io.ErrUnexpectedEOF)
		}
		w.tensors[name] = boundMetalTensor{name: name, typeID: t.TypeID, dimensions: append([]uint64(nil), t.Dimensions...), data: memory}
	}
	return w, nil
}

func (w *metalDSparkWeights) acquire() error {
	if w == nil {
		return fmt.Errorf("metal dspark: weights closed")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return fmt.Errorf("metal dspark: weights closed")
	}
	w.leases++
	return nil
}

func (w *metalDSparkWeights) release() error {
	if w == nil {
		return fmt.Errorf("metal dspark: missing weight owner")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.leases <= 0 {
		return fmt.Errorf("metal dspark: unowned weight release")
	}
	w.leases--
	return nil
}

func (w *metalDSparkWeights) tensor(name string) (boundMetalTensor, error) {
	if w == nil {
		return boundMetalTensor{}, fmt.Errorf("metal dspark: weights closed")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed || w.leases == 0 {
		return boundMetalTensor{}, fmt.Errorf("metal dspark: consumer lease required")
	}
	t, ok := w.tensors[name]
	if !ok {
		return boundMetalTensor{}, fmt.Errorf("metal dspark: missing %s", name)
	}
	t.dimensions = append([]uint64(nil), t.dimensions...)
	return t, nil
}

func (w *metalDSparkWeights) Close() error {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.leases != 0 {
		return fmt.Errorf("metal dspark: active consumers retain support weights")
	}
	if w.closed {
		return nil
	}
	for _, v := range w.storage {
		v.Free()
	}
	w.storage = nil
	w.tensors = nil
	w.closed = true
	return nil
}
