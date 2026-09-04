//go:build linux

package main

import (
	"errors"
	"fmt"
	"os"
	"syscall"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const cudaProbeSource = `
extern "C" __global__ void ds4_probe(float *p) {
  if (blockIdx.x == 0 && threadIdx.x == 0) p[0] = 4.0f;
}`

// cudaFoundation owns the resources needed to begin a CUDA model runtime.
// It deliberately does not implement sessionBackend until token evaluation is
// available on CUDA.
type cudaFoundation struct {
	device *cuda.Device
	file   *os.File
	model  []byte
	name   string
	major  int
	minor  int
}

func newCUDAFoundation(path string) (_ *cudaFoundation, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cuda: open model: %w", err)
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("cuda: stat model: %w", err)
	}
	if info.Size() != pinnedModelBytes {
		return nil, fmt.Errorf("cuda: model size %d, want %d", info.Size(), int64(pinnedModelBytes))
	}
	g, err := readGGUF(f, info.Size())
	if err != nil {
		return nil, fmt.Errorf("cuda: read model: %w", err)
	}
	if err := validatePinnedProfile(g); err != nil {
		return nil, fmt.Errorf("cuda: model profile: %w", err)
	}
	model, err := syscall.Mmap(int(f.Fd()), 0, int(info.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("cuda: map model: %w", err)
	}
	defer func() {
		if err != nil {
			syscall.Munmap(model)
		}
	}()

	dev, err := cuda.NewDevice(0)
	if err != nil {
		return nil, fmt.Errorf("cuda: device: %w", err)
	}
	defer func() {
		if err != nil {
			dev.Close()
		}
	}()
	name, err := dev.Name()
	if err != nil {
		return nil, err
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	r := &cudaFoundation{device: dev, file: f, model: model, name: name, major: major, minor: minor}
	if err := r.probe(); err != nil {
		return nil, err
	}
	return r, nil
}

func (r *cudaFoundation) probe() error {
	compiled, err := nvrtcx.Compile(cudaProbeSource,
		nvrtcx.WithName("ds4_probe.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		return fmt.Errorf("cuda: compile probe: %w", err)
	}
	return r.device.WithContext(func() error {
		mod, err := compiled.Load(r.device)
		if err != nil {
			return fmt.Errorf("cuda: load probe: %w", err)
		}
		defer mod.Unload()
		fn, err := mod.GetFunction("ds4_probe")
		if err != nil {
			return fmt.Errorf("cuda: find probe: %w", err)
		}
		buf, err := cudamem.Alloc[float32](r.device, 1)
		if err != nil {
			return fmt.Errorf("cuda: allocate probe: %w", err)
		}
		defer buf.Free()
		if err := cudalaunch.Kernel(fn).Launch(buf); err != nil {
			return fmt.Errorf("cuda: launch probe: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize probe: %w", err)
		}
		var got [1]float32
		if err := buf.CopyToHost(got[:]); err != nil {
			return fmt.Errorf("cuda: read probe: %w", err)
		}
		if got[0] != 4 {
			return fmt.Errorf("cuda: probe result %g, want 4", got[0])
		}
		return nil
	})
}

func (r *cudaFoundation) Close() error {
	if r == nil {
		return nil
	}
	var errs []error
	if r.device != nil {
		errs = append(errs, r.device.Close())
		r.device = nil
	}
	if r.model != nil {
		errs = append(errs, syscall.Munmap(r.model))
		r.model = nil
	}
	if r.file != nil {
		errs = append(errs, r.file.Close())
		r.file = nil
	}
	return errors.Join(errs...)
}
