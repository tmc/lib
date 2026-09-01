// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
)

// SSDOffloader provides secondary SSD storage offloading for NVIDIA CUDA KV buffers.
type SSDOffloader struct {
	dir   string
	mu    sync.RWMutex
	files map[int]string // layerIndex -> file path
}

// NewSSDOffloader creates a new SSDOffloader rooted at dir.
func NewSSDOffloader(dir string) (*SSDOffloader, error) {
	if dir == "" {
		dir = filepath.Join(os.TempDir(), "cudakv-ssd-offload")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("cudakv: create offload dir: %w", err)
	}
	return &SSDOffloader{
		dir:   dir,
		files: make(map[int]string),
	}, nil
}

// writeFloats writes vals to w as little-endian float32.
func writeFloats(w io.Writer, vals []float32) error {
	buf := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	_, err := w.Write(buf)
	return err
}

// readFloats fills vals from r. It reports an error unless r yields exactly
// len(vals) little-endian float32 values, so a buffer whose shape no longer
// matches the offloaded layer cannot be silently half-restored.
func readFloats(r io.Reader, vals []float32) error {
	buf := make([]byte, len(vals)*4)
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	for i := range vals {
		vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return nil
}

// EvictLayer writes a layer's K and V buffers from host/device memory to SSD storage.
func (o *SSDOffloader) EvictLayer(ctx context.Context, layer LayerCache) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	kData := make([]float32, layer.K.Len())
	if err := layer.K.CopyToHost(kData); err != nil {
		return fmt.Errorf("cudakv: copy K to host: %w", err)
	}

	vData := make([]float32, layer.V.Len())
	if err := layer.V.CopyToHost(vData); err != nil {
		return fmt.Errorf("cudakv: copy V to host: %w", err)
	}

	filePath := filepath.Join(o.dir, fmt.Sprintf("cuda_layer_%d.bin", layer.LayerIndex))
	f, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("cudakv: create layer file: %w", err)
	}
	defer f.Close()

	if err := writeFloats(f, kData); err != nil {
		return fmt.Errorf("cudakv: write layer %d K: %w", layer.LayerIndex, err)
	}
	if err := writeFloats(f, vData); err != nil {
		return fmt.Errorf("cudakv: write layer %d V: %w", layer.LayerIndex, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("cudakv: close layer file: %w", err)
	}

	o.files[layer.LayerIndex] = filePath
	return nil
}

// RestoreLayer restores K and V buffers from SSD back into GPU memory.
func (o *SSDOffloader) RestoreLayer(ctx context.Context, layer LayerCache) error {
	o.mu.RLock()
	filePath, ok := o.files[layer.LayerIndex]
	o.mu.RUnlock()

	if !ok {
		return fmt.Errorf("cudakv: layer %d not found on ssd", layer.LayerIndex)
	}

	f, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("cudakv: read layer file: %w", err)
	}
	defer f.Close()

	kData := make([]float32, layer.K.Len())
	vData := make([]float32, layer.V.Len())

	// The evicted file holds K then V back to back. A size mismatch means the
	// destination buffers are not the ones that were offloaded.
	fi, err := f.Stat()
	if err != nil {
		return fmt.Errorf("cudakv: stat layer file: %w", err)
	}
	if want := int64(len(kData)+len(vData)) * 4; fi.Size() != want {
		return fmt.Errorf("cudakv: layer %d file holds %d bytes, want %d", layer.LayerIndex, fi.Size(), want)
	}

	if err := readFloats(f, kData); err != nil {
		return fmt.Errorf("cudakv: read layer %d K: %w", layer.LayerIndex, err)
	}
	if err := layer.K.CopyFromHost(kData); err != nil {
		return fmt.Errorf("cudakv: copy K to device: %w", err)
	}

	if err := readFloats(f, vData); err != nil {
		return fmt.Errorf("cudakv: read layer %d V: %w", layer.LayerIndex, err)
	}
	if err := layer.V.CopyFromHost(vData); err != nil {
		return fmt.Errorf("cudakv: copy V to device: %w", err)
	}

	return nil
}

// Close removes the layer files the offloader spilled and forgets them.
// Restoring a layer after Close reports that the layer is not on the SSD.
func (o *SSDOffloader) Close() error {
	o.mu.Lock()
	defer o.mu.Unlock()

	var firstErr error
	for layer, path := range o.files {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) && firstErr == nil {
			firstErr = fmt.Errorf("cudakv: remove layer %d file: %w", layer, err)
		}
		delete(o.files, layer)
	}
	return firstErr
}
