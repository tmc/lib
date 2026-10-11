//go:build linux && native

package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

const sidecarImageLimit = 16
const sidecarImageBytes = 8 << 20

// This finite native-fixture recorder observes only delegated successful loads.
// Compilation images alone are never recorded as loaded generations.
type sidecarImageLoader struct {
	cudahotswap.Loader
	mu               sync.Mutex
	referenceSource  string
	referenceSidecar string
	attempts         int
	pins             map[*cudahotswap.Pin]bool
	compiled         map[*cudahotswap.Compilation]sidecarImageSource
	loaded           map[driver.Function]*sidecarImageKernel
	image            func(*cudahotswap.Compilation) ([]byte, error)
	attribute        func(context.Context, string, string, sidecarImageRecord, []byte, sidecarImageSource) error
}

type sidecarImageSource struct{ Digest, Key, Entry, Filename, Code, PTX string }
type sidecarImageRecord struct {
	State                                         cudahotswap.State
	Function                                      driver.Function
	SourceSHA256, CompilerKey, Entry, CUBINSHA256 string
	ReferenceSourceSHA256, ReferenceSidecarSHA256 string
	Bytes                                         int
}
type sidecarImageKernel struct {
	cudahotswap.Kernel
	owner    *sidecarImageLoader
	source   sidecarImageSource
	body     []byte
	function driver.Function
}

func newSidecarImageLoader(loader cudahotswap.Loader, reference string, metadata []byte) *sidecarImageLoader {
	return &sidecarImageLoader{Loader: loader, referenceSource: fmt.Sprintf("%x", sha256.Sum256([]byte(reference))), referenceSidecar: fmt.Sprintf("%x", sha256.Sum256(metadata)), pins: make(map[*cudahotswap.Pin]bool), compiled: make(map[*cudahotswap.Compilation]sidecarImageSource), loaded: make(map[driver.Function]*sidecarImageKernel), image: func(c *cudahotswap.Compilation) ([]byte, error) { return c.CUBIN() }}
}
func (l *sidecarImageLoader) Compile(ctx context.Context, s cudahotswap.Source) (*cudahotswap.Compilation, error) {
	l.mu.Lock()
	if l.attempts >= sidecarImageLimit {
		l.mu.Unlock()
		return nil, errors.New("native image compilation capacity exceeded")
	}
	l.attempts++
	l.mu.Unlock()
	c, err := l.Loader.Compile(ctx, s)
	if err != nil {
		return c, err
	}
	if c == nil {
		return nil, errors.New("native image compilation absent")
	}
	metadata := sidecarImageSource{Digest: fmt.Sprintf("%x", sha256.Sum256([]byte(s.Code))), Key: c.Key, Entry: c.Entry, Filename: s.Filename, Code: s.Code, PTX: c.PTX}
	if metadata.Key == "" || metadata.Entry != s.Entry {
		return c, errors.New("native image compilation metadata differs")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, exists := l.compiled[c]; exists {
		return c, errors.New("native image compilation owner reused")
	}
	l.compiled[c] = metadata
	return c, nil
}
func (l *sidecarImageLoader) Load(ctx context.Context, c *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	l.mu.Lock()
	metadata, ok := l.compiled[c]
	l.mu.Unlock()
	if !ok || c == nil || c.Key != metadata.Key || c.Entry != metadata.Entry {
		return nil, errors.New("native image lacks exact compiled source association")
	}
	body, err := l.image(c)
	if err != nil {
		return nil, err
	}
	if len(body) == 0 || len(body) > sidecarImageBytes {
		return nil, errors.New("native image size outside bound")
	}
	body = append([]byte(nil), body...)
	k, err := l.Loader.Load(ctx, c)
	if err != nil {
		return k, err
	}
	if k == nil {
		return nil, errors.New("native image successful load owner absent")
	}
	function := k.Raw()
	if function == 0 {
		return k, errors.New("native image successful load has no function")
	}
	wrapped := &sidecarImageKernel{Kernel: k, owner: l, source: metadata, body: body, function: function}
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, exists := l.loaded[function]; exists {
		return k, errors.New("native image active function reused")
	}
	delete(l.compiled, c)
	l.loaded[function] = wrapped
	return wrapped, nil
}
func (k *sidecarImageKernel) Close() error {
	if err := k.Kernel.Close(); err != nil {
		return err
	}
	k.owner.mu.Lock()
	defer k.owner.mu.Unlock()
	if k.owner.loaded[k.function] == k {
		delete(k.owner.loaded, k.function)
	}
	return nil
}
func (l *sidecarImageLoader) snapshot(state cudahotswap.State, function driver.Function) (sidecarImageRecord, []byte, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	k := l.loaded[function]
	if k == nil || function == 0 || state.Kind != "kernel" || state.Generation == 0 || state.Version == 0 || state.SourceDigest != k.source.Digest || state.CompilerKey != k.source.Key {
		return sidecarImageRecord{}, nil, errors.New("native held generation differs from loaded image")
	}
	record := sidecarImageRecord{State: state, Function: function, SourceSHA256: k.source.Digest, CompilerKey: k.source.Key, Entry: k.source.Entry, CUBINSHA256: fmt.Sprintf("%x", sha256.Sum256(k.body)), ReferenceSourceSHA256: l.referenceSource, ReferenceSidecarSHA256: l.referenceSidecar, Bytes: len(k.body)}
	return record, append([]byte(nil), k.body...), nil
}
func sidecarRetainImage(ctx context.Context, dir, name string, service *cudahotswap.Service, loader *sidecarImageLoader, state cudahotswap.State) (err error) {
	pin, err := service.Pin(state.Name)
	if err != nil {
		return err
	}
	loader.mu.Lock()
	loader.pins[pin] = true
	loader.mu.Unlock()
	defer func() { err = errors.Join(err, loader.releasePin(ctx, pin)) }()
	if pin.State() != state {
		return errors.New("native image pin differs from observed state")
	}
	record, body, err := loader.snapshot(pin.State(), pin.Function())
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(dir, name+".cubin"), body, 0600); err != nil {
		return err
	}
	if err := sidecarWriteImageRecord(filepath.Join(dir, name+".json"), record); err != nil {
		return err
	}
	if loader.attribute != nil {
		loader.mu.Lock()
		image := loader.loaded[pin.Function()]
		if image == nil {
			loader.mu.Unlock()
			return errors.New("native attribution lost held image owner")
		}
		compiled := image.source
		loader.mu.Unlock()
		return loader.attribute(ctx, dir, name, record, body, compiled)
	}
	return nil
}

func (l *sidecarImageLoader) releasePin(ctx context.Context, p *cudahotswap.Pin) error {
	if err := p.Release(ctx); err != nil {
		return err
	}
	l.mu.Lock()
	delete(l.pins, p)
	l.mu.Unlock()
	return nil
}
func (l *sidecarImageLoader) releasePins(ctx context.Context) error {
	l.mu.Lock()
	pins := make([]*cudahotswap.Pin, 0, len(l.pins))
	for p := range l.pins {
		pins = append(pins, p)
	}
	l.mu.Unlock()
	var err error
	for _, p := range pins {
		err = errors.Join(err, l.releasePin(ctx, p))
	}
	return err
}
func sidecarWriteImageRecord(path string, record sidecarImageRecord) error {
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

// Early constructor refusal has no session cleanup owner. Retain that fact;
// never turn a nil cleanup into a closure that calls a nil function.
func sidecarImageCleanup(close func() error, images *sidecarImageLoader) func() error {
	if close == nil && images == nil {
		return nil
	}
	return func() error {
		if images != nil {
			if err := images.releasePins(context.Background()); err != nil {
				return err
			}
		}
		if close != nil {
			return close()
		}
		return nil
	}
}
