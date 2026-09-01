// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"context"
	"encoding/binary"
	"fmt"
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
	kBytes := make([]byte, len(kData)*4)
	if _, err := f.Read(kBytes); err != nil {
		return err
	}
	for i := range kData {
		u := binary.LittleEndian.Uint32(kBytes[i*4 : (i+1)*4])
		kData[i] = math.Float32frombits(u)
	}
	if err := layer.K.CopyFromHost(kData); err != nil {
		return fmt.Errorf("cudakv: copy K to device: %w", err)
	}

	vData := make([]float32, layer.V.Len())
	vBytes := make([]byte, len(vData)*4)
	if _, err := f.Read(vBytes); err != nil {
		return err
	}
	for i := range vData {
		u := binary.LittleEndian.Uint32(vBytes[i*4 : (i+1)*4])
		vData[i] = math.Float32frombits(u)
	}
	if err := layer.V.CopyFromHost(vData); err != nil {
		return fmt.Errorf("cudakv: copy V to device: %w", err)
	}

	return nil
}

// Close releases the offloader resources.
func (o *SSDOffloader) Close() error {
	return nil
}
