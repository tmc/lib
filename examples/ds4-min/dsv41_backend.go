package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/gguf"
)

// v41CPUBackend owns the file borrowed by its model and native request states.
// Common dispatch admission remains separate from this constructor.
type v41CPUBackend struct {
	*dsv41.Backend
	mu   sync.Mutex
	file v41ModelFile
}
type v41ModelFile interface {
	io.ReaderAt
	Stat() (os.FileInfo, error)
	Close() error
}
type v41CPUOptions struct {
	// Zero permits ordinary execution, but not portable payloads. Nonzero
	// must come from an independent complete-asset receipt.
	AssetSHA256 [32]byte
}

func newV41CPUBackend(path string, options v41CPUOptions) (*v41CPUBackend, error) {
	return openV41CPUBackend(path, options, func(path string) (v41ModelFile, error) { return os.Open(path) })
}

// If construction and cleanup both fail, the nonnil result retains the file
// for a retry of Close. Callers must close that result even when err != nil.
func openV41CPUBackend(path string, options v41CPUOptions, open func(string) (v41ModelFile, error)) (owner *v41CPUBackend, err error) {
	f, err := open(path)
	if err != nil {
		return nil, err
	}
	if f == nil {
		return nil, fmt.Errorf("open V4.1 model: missing file owner")
	}
	ok := false
	defer func() {
		if !ok {
			if closeErr := f.Close(); closeErr != nil {
				owner = &v41CPUBackend{file: f}
				err = errors.Join(err, fmt.Errorf("close V4.1 model: %w", closeErr))
			}
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	g, err := gguf.Read(f, info.Size())
	if err != nil {
		return nil, fmt.Errorf("read V4.1 model: %w", err)
	}
	var model *dsv41.Model
	if options.AssetSHA256 == ([32]byte{}) {
		model, err = dsv41.Bind(f, info.Size(), g)
	} else {
		model, err = dsv41.BindVerified(f, info.Size(), g, options.AssetSHA256)
	}
	if err != nil {
		return nil, fmt.Errorf("bind V4.1 model: %w", err)
	}
	b, err := dsv41.NewBackend(model)
	if err != nil {
		return nil, err
	}
	ok = true
	return &v41CPUBackend{Backend: b, file: f}, nil
}
func (b *v41CPUBackend) Close() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Backend != nil {
		if err := b.Backend.Close(); err != nil {
			return err
		}
	}
	if b.file == nil {
		return nil
	}
	if err := b.file.Close(); err != nil {
		return err
	}
	b.file = nil
	return nil
}
