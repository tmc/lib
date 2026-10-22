//go:build linux

package main

import (
	"fmt"
	"os"
	"syscall"

	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/nvidia/x/cuda"
)

type cudaFoundationConstructorOperations struct {
	open       func(string) (*os.File, error)
	stat       func(*os.File) (os.FileInfo, error)
	direct     func(string, os.FileInfo) (*os.File, int)
	read       func(*os.File, int64) (*gguf.File, error)
	validate   func(*gguf.File, *os.File, int64) error
	mapModel   func(*os.File, int64) ([]byte, error)
	device     func() (*cuda.Device, error)
	describe   func(*cudaFoundation) error
	stream     func(*cuda.Device) (*cuda.Stream, error)
	warm       func(*cudaFoundation, cudaRuntimeOptions) error
	streaming  func(*cudaFoundation, streamingOptions, uint64) error
	probe      func(*cudaFoundation) error
	probeModel func(*cudaFoundation) error
	close      *cudaFoundationCloseOperations
}

func defaultCUDAFoundationConstructorOperations() cudaFoundationConstructorOperations {
	return cudaFoundationConstructorOperations{
		open: os.Open,
		stat: func(f *os.File) (os.FileInfo, error) { return f.Stat() },
		direct: func(path string, info os.FileInfo) (*os.File, int) {
			if cudaDirectStaging() {
				return openDirectModel(path, info)
			}
			return nil, 0
		},
		read:     func(f *os.File, size int64) (*gguf.File, error) { return gguf.Read(f, size) },
		validate: func(g *gguf.File, f *os.File, size int64) error { return validateProfileFile(g, f, size) },
		mapModel: func(f *os.File, size int64) ([]byte, error) {
			return syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
		},
		device: func() (*cuda.Device, error) { return cuda.NewDevice(0) },
		describe: func(r *cudaFoundation) (err error) {
			if r.name, err = r.device.Name(); err != nil {
				return err
			}
			if r.major, r.minor, err = r.device.ComputeCapability(); err != nil {
				return err
			}
			r.hmm, err = cudaSupportsHostPageTables(r.device)
			return err
		},
		stream: func(d *cuda.Device) (*cuda.Stream, error) { return d.NewStream() },
		warm: func(r *cudaFoundation, options cudaRuntimeOptions) error {
			switch {
			case r.hmm && cudaResidentModel():
				return r.loadResidentSet()
			case cudaWarmModelSelected(options.warmWeightsSet, options.warmWeights):
				return r.warmModel()
			}
			return nil
		},
		streaming: func(r *cudaFoundation, o streamingOptions, size uint64) error {
			return r.loadStreamingExperts(o, size)
		},
		probe:      func(r *cudaFoundation) error { return r.probe() },
		probeModel: func(r *cudaFoundation) error { return r.probeModel(r.gguf) },
	}
}

// Each acquired owner is published before its accompanying error is checked.
// A failed cleanup is returned as an io.Closer, not discarded by a defer.
func newCUDAFoundationWithOperations(path string, options cudaRuntimeOptions, ops cudaFoundationConstructorOperations) (_ *cudaFoundation, err error) {
	if err := options.streaming.validate("cuda"); err != nil {
		return nil, err
	}
	r := &cudaFoundation{stateIdentity: new(modelStateIdentityCache), closeOps: ops.close, releaseMapping: cudaReleaseMappedArtifacts()}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, r)
		}
	}()
	r.file, err = ops.open(path)
	if err != nil {
		return nil, fmt.Errorf("cuda: open model: %w", err)
	}
	info, err := ops.stat(r.file)
	if err != nil {
		return nil, fmt.Errorf("cuda: stat model: %w", err)
	}
	r.direct, r.directAlign = ops.direct(path, info)
	r.gguf, err = ops.read(r.file, info.Size())
	if err != nil {
		return nil, fmt.Errorf("cuda: read model: %w", err)
	}
	if err = ops.validate(r.gguf, r.file, info.Size()); err != nil {
		return nil, fmt.Errorf("cuda: model profile: %w", err)
	}
	r.model, err = ops.mapModel(r.file, info.Size())
	if err != nil {
		return nil, fmt.Errorf("cuda: map model: %w", err)
	}
	r.device, err = ops.device()
	if err != nil {
		return nil, fmt.Errorf("cuda: device: %w", err)
	}
	if err = ops.describe(r); err != nil {
		return nil, err
	}
	r.workStream, err = ops.stream(r.device)
	if err != nil {
		return nil, fmt.Errorf("cuda: create work stream: %w", err)
	}
	if options.streaming.enabled {
		if ops.streaming == nil {
			return nil, fmt.Errorf("cuda: streaming loader unavailable")
		}
		err = ops.streaming(r, options.streaming, uint64(info.Size()))
	} else {
		err = ops.warm(r, options)
	}
	if err != nil {
		return nil, err
	}
	if err = ops.probe(r); err != nil {
		return nil, err
	}
	if err = ops.probeModel(r); err != nil {
		return nil, err
	}
	return r, nil
}
