//go:build linux && native

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

// watchNativeRecorder observes the original loader and owns only its receipt
// data. Loaded kernel ownership is forwarded to the same Service unchanged.
type watchNativeRecorder struct {
	mu       sync.Mutex
	evidence watchEvidence
	dir      string
	proof    *nativereceipt.Session
	compiled map[*cudahotswap.Compilation]int
}

type watchNativeLoader struct {
	cudahotswap.Loader
	dev      *cuda.Device
	recorder *watchNativeRecorder
}

func watchWriteExclusive(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	return errors.Join(err, f.Close())
}

func (l *watchNativeLoader) Compile(ctx context.Context, source cudahotswap.Source) (*cudahotswap.Compilation, error) {
	c, err := l.Loader.Compile(ctx, source)
	if c == nil || err != nil {
		return c, err
	}
	image, err := c.CUBIN()
	if err != nil {
		return c, err
	}
	if len(source.Code) > 8<<20 || len(c.PTX) > 8<<20 || len(image) > 64<<20 {
		return c, errors.New("watch compile receipt: body limit")
	}
	r := l.recorder
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.evidence.Compilations) >= 4 {
		return c, errors.New("watch compile receipt: unexpected compilation")
	}
	index := len(r.evidence.Compilations)
	source.Flags = append([]string(nil), source.Flags...)
	row := watchCompilation{Source: source, SourceSHA256: watchDigest([]byte(source.Code)), PTXSHA256: watchDigest([]byte(c.PTX)), CUBINSHA256: watchDigest(image), Key: c.Key, PTXBytes: len(c.PTX), CUBINBytes: len(image)}
	r.evidence.Compilations = append(r.evidence.Compilations, row)
	r.compiled[c] = index
	for _, artifact := range []struct {
		suffix string
		body   []byte
	}{{"cu", []byte(source.Code)}, {"ptx", []byte(c.PTX)}, {"cubin", image}} {
		if err := watchWriteExclusive(filepath.Join(r.dir, fmt.Sprintf("compile-%d.%s", index, artifact.suffix)), artifact.body); err != nil {
			return c, err
		}
	}
	if err := r.proof.Capture(fmt.Sprintf("watch-compile-%d", index), row.SourceSHA256, row.CUBINSHA256); err != nil {
		return c, err
	}
	return c, nil
}

func (l *watchNativeLoader) Load(ctx context.Context, c *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	// Always forward the original result, including a partial nonnil owner.
	k, loadErr := l.Loader.Load(ctx, c)
	if k == nil {
		return nil, loadErr
	}
	r := l.recorder
	r.mu.Lock()
	defer r.mu.Unlock()
	index, ok := r.compiled[c]
	row := watchLoad{Compilation: index, Function: uint64(k.Raw())}
	var current driver.Context
	observeErr := l.dev.WithContext(func() error { var err error; current, err = driver.CtxGetCurrent(); return err })
	row.Context = uint64(current)
	if loadErr != nil {
		row.LoadError = loadErr.Error()
	}
	slot := len(r.evidence.Loads)
	r.evidence.Loads = append(r.evidence.Loads, row)

	if !ok {
		return k, errors.Join(loadErr, observeErr, errors.New("watch load receipt: compilation absent"))
	}
	compiled := r.evidence.Compilations[index]
	observeErr = errors.Join(observeErr, r.proof.Capture(fmt.Sprintf("watch-load-%d", slot), compiled.SourceSHA256, compiled.CUBINSHA256))
	return k, errors.Join(loadErr, observeErr)
}

func (r *watchNativeRecorder) cleanup(service *cudahotswap.Service, close func() error) error {
	err := close()
	r.mu.Lock()
	defer r.mu.Unlock()
	text := ""
	if err != nil {
		text = err.Error()
	}
	r.evidence.Cleanup.Attempts = append(r.evidence.Cleanup.Attempts, text)
	if err == nil {
		pin, pinErr := service.Pin("kernel")
		if pin != nil {
			err = errors.Join(err, pin.Release(context.Background()))
		}
		r.evidence.Cleanup.ClosedService = pin == nil && pinErr != nil && pinErr.Error() == "service is closed"
	}
	r.evidence.Cleanup.Attempted = true
	r.evidence.Cleanup.Complete = err == nil
	r.evidence.Cleanup.Error = text
	return err
}

func (r *watchNativeRecorder) finish(path string) error {
	proof, err := nativereceipt.Read(path)
	if err != nil {
		return err
	}
	own, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		return err
	}
	observed := proof.Process
	observed.ExecutableSHA256 = ""
	if observed != own {
		return errors.New("watch evidence: mapped receipt is not this process")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.evidence.Process = proof.Process
	data, err := json.MarshalIndent(r.evidence, "", "  ")
	if err != nil {
		return err
	}
	if err := watchWriteExclusive(filepath.Join(r.dir, "watch-evidence.json"), append(data, '\n')); err != nil {
		return err
	}
	return errors.Join(validateWatchEvidence(r.evidence), validateWatchArtifacts(r.dir, r.evidence))
}
