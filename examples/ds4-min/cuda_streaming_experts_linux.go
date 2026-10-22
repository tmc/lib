//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// cudaStreamingExperts shares only the admitted source/cache and immutable
// kernels. It owns no whole-expert tensor or HMM weight view. The source reader
// remains the foundation's responsibility and must outlive this owner.
type cudaStreamingExperts struct {
	coordinator     cudaStreamingCoordinator
	selections      *dsv4expertcache.CUDASelections
	module          *cuda.Module
	gate, down      *cuda.Kernel
	partial         []io.Closer
	closeSelections func() error
}

// Both the returned error and the foundation can discover this owner. Each
// successful release is consumed once even when outer cleanup visits both.
type cudaStreamingLoadError struct {
	cause  error
	owners []io.Closer
}

func (e *cudaStreamingLoadError) Error() string { return e.cause.Error() }
func (e *cudaStreamingLoadError) Unwrap() error { return e.cause }
func (e *cudaStreamingLoadError) Close() error {
	for len(e.owners) > 0 {
		if err := e.owners[0].Close(); err != nil {
			return err
		}
		e.owners = e.owners[1:]
	}
	return nil
}

// Ownership transfers immediately, including when construction returns an
// error. The caller keeps the returned owner until Close succeeds.
func newCUDAStreamingExperts(device *cuda.Device, selections *dsv4expertcache.CUDASelections) (*cudaStreamingExperts, error) {
	return newCUDAStreamingExpertsWithLoader(device, selections, func() (*cuda.Module, error) {
		compiled, err := nvrtcx.Compile(cudaProbeSource+cudaExpertSource+cudaExpertAddressSource, nvrtcx.WithDeviceArch(device))
		if err != nil {
			return nil, fmt.Errorf("compile CUDA streaming experts: %w", err)
		}
		return compiled.LoadModule(device)
	})
}

func newCUDAStreamingExpertsWithLoader(device *cuda.Device, selections *dsv4expertcache.CUDASelections, load func() (*cuda.Module, error)) (*cudaStreamingExperts, error) {
	if device == nil || selections == nil {
		return nil, errors.New("missing CUDA streaming resources")
	}
	o := &cudaStreamingExperts{selections: selections}
	o.closeSelections = selections.Close
	o.coordinator.closeCache = o.closeResources
	var err error
	o.module, err = load()
	if err != nil {
		if owners := backendErrorOwners(err); len(owners) > 0 {
			retained := &cudaStreamingLoadError{cause: err, owners: owners}
			o.partial = []io.Closer{retained}
			err = retained
		}
		return o, fmt.Errorf("load CUDA streaming experts: %w", err)
	}
	o.gate, err = o.module.Function("ds4_iq2_xxs_pair_address_f32")
	if err == nil {
		o.down, err = o.module.Function("ds4_q2_k_address_f32")
	}
	if err != nil {
		return o, fmt.Errorf("bind CUDA streaming experts: %w", err)
	}
	return o, nil
}

func (o *cudaStreamingExperts) closeResources() error {
	for len(o.partial) > 0 {
		if err := o.partial[0].Close(); err != nil {
			return err
		}
		o.partial = o.partial[1:]
	}
	if o.selections != nil {
		close := o.selections.Close
		if o.closeSelections != nil {
			close = o.closeSelections
		}
		if err := close(); err != nil {
			return err
		}
		o.selections = nil
		o.closeSelections = nil
	}
	if o.module != nil {
		if err := o.module.Close(); err != nil {
			return err
		}
		o.module, o.gate, o.down = nil, nil, nil
	}
	return nil
}

func (o *cudaStreamingExperts) Close() error {
	if o == nil {
		return nil
	}
	return o.coordinator.Close()
}

type cudaStreamingConsumer struct {
	selection   *dsv4expertcache.CUDASelection
	stream      *cuda.Stream
	launch      *cudaExpertAddressLaunch
	failedOwner *dsv4expertcache.CUDASelections
}

func (c *cudaStreamingConsumer) Close() error {
	if c.failedOwner != nil {
		return c.failedOwner.Close()
	}
	if c.launch != nil {
		return c.launch.Close()
	}
	return c.selection.Finish(c.stream)
}

// evalStreamingExperts is the scalar inference consumer, before all resident,
// aligned and Q8_1 branches. It preserves GPU routing weights through the
// existing weighted SwiGLU, but deliberately uses the qualified F32 activation
// consumers. That arithmetic route needs its own full-model oracle gate.
func (r *cudaRuntime) evalStreamingExperts(s *cudaTokenState, index int) error {
	o := r.streamingExperts
	if o == nil || r.streamingRequest == nil {
		return errors.New("CUDA streaming kernels unavailable")
	}
	l := &r.model.layers[index]
	if l.expertGate.device != 0 || l.expertUp.device != 0 || l.expertDown.device != 0 {
		return errors.New("CUDA streaming expert retains whole tensor address")
	}
	if err := cudaExpertShape(l.expertGate, typeIQ2XXS, 66); err != nil {
		return err
	}
	if err := cudaExpertShape(l.expertUp, typeIQ2XXS, 66); err != nil {
		return err
	}
	if err := cudaExpertShape(l.expertDown, typeQ2K, 84); err != nil {
		return err
	}
	if l.expertGate.dimensions[0] != cudaWidth || l.expertGate.dimensions[1] != 2048 ||
		l.expertUp.dimensions[0] != cudaWidth || l.expertUp.dimensions[1] != 2048 ||
		l.expertDown.dimensions[0] != 2048 || l.expertDown.dimensions[1] != cudaWidth ||
		s.expertIDs.Len() != 6 || s.expertW.Len() != 6 {
		return errors.New("CUDA streaming expert shape differs")
	}
	return r.streamingRequest.run(func() (io.Closer, error) {
		if o.gate == nil || o.down == nil || o.selections == nil {
			return nil, errors.New("CUDA streaming kernels unavailable")
		}
		// Router results must be complete before the bounded host ID read.
		if err := syncStream(r.stream()); err != nil {
			return nil, err
		}
		var ids32 [6]int32
		if err := s.expertIDs.CopyToHost(ids32[:]); err != nil {
			return nil, err
		}
		ids := make([]int, len(ids32))
		for i, id := range ids32 {
			ids[i] = int(id)
		}
		selection, err := o.selections.Select(index, ids, false)
		if err != nil {
			if selection == nil {
				return nil, err
			}
			// Upload uncertainty quarantines the whole shared cache. Its owner,
			// not a fabricated consumer event, resolves the unexposed lease.
			o.coordinator.closing = true
			return &cudaStreamingConsumer{failedOwner: o.selections}, err
		}
		consumer := &cudaStreamingConsumer{selection: selection, stream: r.stream()}
		consumer.launch, err = newCUDAExpertAddressLaunch(selection, r.stream(), len(ids), cudaWidth, 2048)
		return consumer, err
	}, func(owner io.Closer) error {
		consumer := owner.(*cudaStreamingConsumer)
		if err := consumer.launch.gateUp(o.gate, s.ffnNorm, s.expertGate, s.expertUp); err != nil {
			return err
		}
		if err := r.encodeSwiGLUSlots(s.expertGate, s.expertUp, s.expertW, s.expertMid, 6, 1); err != nil {
			return err
		}
		return consumer.launch.down(o.down, s.expertMid, s.expertDown)
	})
}
