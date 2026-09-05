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
// Instances may share a directory; each owns distinct spill files.
// Spills are process-local scratch storage, not crash-durable checkpoints.
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
	n, err := w.Write(buf)
	if err == nil && n != len(buf) {
		err = io.ErrShortWrite
	}
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
	if err := ctx.Err(); err != nil {
		return err
	}

	kData := make([]float32, layer.K.Len())
	if err := layer.K.CopyToHost(kData); err != nil {
		return fmt.Errorf("cudakv: copy K to host: %w", err)
	}

	vData := make([]float32, layer.V.Len())
	if err := layer.V.CopyToHost(vData); err != nil {
		return fmt.Errorf("cudakv: copy V to host: %w", err)
	}

	return o.spill(ctx, layer.LayerIndex, kData, vData)
}

// spill publishes a complete replacement while the caller holds o.mu.
// Each offloader owns distinct files even when instances share a directory.
func (o *SSDOffloader) spill(ctx context.Context, index int, kData, vData []float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	f, err := os.CreateTemp(o.dir, "cuda_layer_*.bin")
	if err != nil {
		return fmt.Errorf("cudakv: create layer file: %w", err)
	}
	defer f.Close()
	path := f.Name()
	published := false
	defer func() {
		if !published {
			os.Remove(path)
		}
	}()

	if err := writeFloats(f, kData); err != nil {
		return fmt.Errorf("cudakv: write layer %d K: %w", index, err)
	}
	if err := writeFloats(f, vData); err != nil {
		return fmt.Errorf("cudakv: write layer %d V: %w", index, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("cudakv: close layer file: %w", err)
	}

	if err := ctx.Err(); err != nil {
		return err
	}
	if old, ok := o.files[index]; ok {
		if err := os.Rename(path, old); err != nil {
			return fmt.Errorf("cudakv: replace layer %d: %w", index, err)
		}
		path = old
	}
	o.files[index] = path
	published = true
	return nil
}

// RestoreLayer restores K and V buffers from SSD back into GPU memory.
func (o *SSDOffloader) RestoreLayer(ctx context.Context, layer LayerCache) error {
	o.mu.RLock()
	defer o.mu.RUnlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	filePath, ok := o.files[layer.LayerIndex]

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
// Failed removals remain owned so a later Close can retry them.
// The offloader can be reused after Close.
func (o *SSDOffloader) Close() error {
	o.mu.Lock()
	defer o.mu.Unlock()

	var firstErr error
	for layer, path := range o.files {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			if firstErr == nil {
				firstErr = fmt.Errorf("cudakv: remove layer %d file: %w", layer, err)
			}
			continue
		}
		delete(o.files, layer)
	}
	return firstErr
}
