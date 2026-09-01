// SPDX-License-Identifier: Apache-2.0

package metalkv

import (
	"context"
	"encoding/binary"
	"fmt"
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

	// Write K bytes
	kBytes := make([]byte, len(kData)*4)
	for i, v := range kData {
		binary.LittleEndian.PutUint32(kBytes[i*4:(i+1)*4], math.Float32bits(v))
	}
	if _, err := f.Write(kBytes); err != nil {
		return err
	}

	// Write V bytes
	vBytes := make([]byte, len(vData)*4)
	for i, v := range vData {
		binary.LittleEndian.PutUint32(vBytes[i*4:(i+1)*4], math.Float32bits(v))
	}
	if _, err := f.Write(vBytes); err != nil {
		return err
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

	kBytes := make([]byte, len(kData)*4)
	if _, err := f.Read(kBytes); err != nil {
		return err
	}
	for i := range kData {
		u := binary.LittleEndian.Uint32(kBytes[i*4 : (i+1)*4])
		kData[i] = math.Float32frombits(u)
	}

	vBytes := make([]byte, len(vData)*4)
	if _, err := f.Read(vBytes); err != nil {
		return err
	}
	for i := range vData {
		u := binary.LittleEndian.Uint32(vBytes[i*4 : (i+1)*4])
		vData[i] = math.Float32frombits(u)
	}

	return nil
}

// Close releases the offloader resources.
func (o *SSDOffloader) Close() error {
	return nil
}
