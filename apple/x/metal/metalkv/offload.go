// SPDX-License-Identifier: Apache-2.0

package metalkv

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

// SSDOffloader provides secondary NVMe storage offloading for Apple Silicon Metal KV buffers.
type SSDOffloader struct {
	dir   string
	mu    sync.RWMutex
	files map[int]string // layerIndex -> file path
}

// NewSSDOffloader creates a new SSDOffloader rooted at dir.
func NewSSDOffloader(dir string) (*SSDOffloader, error) {
	if dir == "" {
		dir = filepath.Join(os.TempDir(), "metalkv-ssd-offload")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("metalkv: create offload dir: %w", err)
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

// EvictLayer writes a layer's K and V buffers to SSD storage.
func (o *SSDOffloader) EvictLayer(ctx context.Context, layer LayerCache) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	kData := layer.K.HostSlice()
	vData := layer.V.HostSlice()

	filePath := filepath.Join(o.dir, fmt.Sprintf("layer_%d.bin", layer.LayerIndex))
	f, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("metalkv: create layer file: %w", err)
	}
	defer f.Close()

	if err := writeFloats(f, kData); err != nil {
		return fmt.Errorf("metalkv: write layer %d K: %w", layer.LayerIndex, err)
	}
	if err := writeFloats(f, vData); err != nil {
		return fmt.Errorf("metalkv: write layer %d V: %w", layer.LayerIndex, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("metalkv: close layer file: %w", err)
	}

	o.files[layer.LayerIndex] = filePath
	return nil
}

// RestoreLayer restores K and V buffers from SSD into unified memory.
func (o *SSDOffloader) RestoreLayer(ctx context.Context, layer LayerCache) error {
	o.mu.RLock()
	filePath, ok := o.files[layer.LayerIndex]
	o.mu.RUnlock()

	if !ok {
		return fmt.Errorf("metalkv: layer %d not found on ssd", layer.LayerIndex)
	}

	f, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("metalkv: read layer file: %w", err)
	}
	defer f.Close()

	kData := layer.K.HostSlice()
	vData := layer.V.HostSlice()

	// The evicted file holds K then V back to back. A size mismatch means the
	// destination buffers are not the ones that were offloaded.
	fi, err := f.Stat()
	if err != nil {
		return fmt.Errorf("metalkv: stat layer file: %w", err)
	}
	if want := int64(len(kData)+len(vData)) * 4; fi.Size() != want {
		return fmt.Errorf("metalkv: layer %d file holds %d bytes, want %d", layer.LayerIndex, fi.Size(), want)
	}

	if err := readFloats(f, kData); err != nil {
		return fmt.Errorf("metalkv: read layer %d K: %w", layer.LayerIndex, err)
	}
	if err := readFloats(f, vData); err != nil {
		return fmt.Errorf("metalkv: read layer %d V: %w", layer.LayerIndex, err)
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
			firstErr = fmt.Errorf("metalkv: remove layer %d file: %w", layer, err)
		}
		delete(o.files, layer)
	}
	return firstErr
}
