//go:build linux

package dsv4expertcache

import (
	"context"
	"errors"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// NewCUDASelections takes exclusive ownership of an unused CUDAUpload on
// success. Source's immutable reader must outlive the returned owner. Every
// layer must have the uploader's exact component strides. No device storage
// address is exposed before a selection has committed.
func NewCUDASelections(source *Source, upload *CUDAUpload) (*CUDASelections, error) {
	if source == nil || upload == nil || upload.closed || upload.closing || upload.failed || upload.lease != nil || upload.starts != 0 {
		return nil, errors.New("invalid CUDA selection owner")
	}
	b, ok := upload.backend.(*cudaUploadDevice)
	if !ok || b.device == nil {
		return nil, errors.New("CUDA upload lacks device views")
	}
	cache, err := cudaSelectionCache(source, upload.slots, upload.strides)
	if err != nil {
		return nil, err
	}
	o := &CUDASelections{source: source, cache: cache, upload: upload}
	o.view = func(slot, component int) (uint64, uint64, error) {
		if component < 0 || component >= 3 || slot < 0 || slot >= upload.slots {
			return 0, 0, errors.New("invalid CUDA component")
		}
		n := upload.strides[component]
		slab := b.slabs[component]
		if n <= 0 || slab.Len() < n || slot > (slab.Len()-n)/n {
			return 0, 0, errors.New("CUDA slab extent differs")
		}
		v := slab.Slice(slot*n, (slot+1)*n)
		return uint64(v.Ptr()), uint64(v.Len()), nil
	}
	return o, nil
}

type cudaSelectionConsumer struct {
	stream             *cuda.Stream
	event              *cuda.Event
	recorded, complete bool
}

// CheckStream admits a live committed selection for the uploader's exact
// device, without recording completion or exposing operands. Callers still
// serialize submission with Finish; this check does not extend view lifetime.
func (s *CUDASelection) CheckStream(stream *cuda.Stream) error {
	if stream == nil {
		return errors.New("missing CUDA consumer stream")
	}
	return s.checkStream(func() error {
		u, ok := s.owner.upload.(*CUDAUpload)
		if !ok || u.closed || u.closing || u.failed {
			return errors.New("CUDA uploader unavailable")
		}
		b, ok := u.backend.(*cudaUploadDevice)
		if !ok || b.device == nil || stream.Device() != b.device {
			return errors.New("CUDA consumer device differs")
		}
		return stream.WithHandle(func(raw driver.Stream) error {
			return b.device.WithContext(func() error {
				status, err := raw.IsCapturing()
				if err != nil {
					return err
				}
				if status != driver.StreamCaptureStatusNone {
					return errors.New("CUDA selection graph capture unsupported")
				}
				return nil
			})
		})
	})
}

func (c *cudaSelectionConsumer) Wait() error {
	if c.complete {
		return nil
	}
	var err error
	if c.recorded {
		err = c.event.Synchronize(context.Background())
	} else {
		err = c.stream.Synchronize(context.Background())
	}
	if err == nil {
		c.complete = true
	}
	return err
}
func (c *cudaSelectionConsumer) close() error {
	if c.event == nil {
		return nil
	}
	if err := c.event.Close(); err != nil {
		return err
	}
	c.event = nil
	return nil
}

// Finish records a terminal event after all readers on the request stream.
// On recording failure it retains a stream-drain completion and the event.
// Retry Finish or Close to drain; never discard a selection on error.
func (s *CUDASelection) Finish(stream *cuda.Stream) error {
	if s == nil || s.owner == nil || stream == nil {
		return errors.New("missing CUDA consumer stream")
	}
	return s.finish(func() (cudaConsumer, error) {
		u, ok := s.owner.upload.(*CUDAUpload)
		if !ok {
			return nil, errors.New("CUDA consumer lacks uploader")
		}
		b, ok := u.backend.(*cudaUploadDevice)
		if !ok || stream.Device() != b.device {
			return nil, errors.New("CUDA consumer device differs")
		}
		c := &cudaSelectionConsumer{stream: stream}
		e, err := b.device.NewEventWithFlags(driver.EventDisableTiming)
		c.event = e
		if err != nil {
			return c, err
		}
		if err = e.Record(stream); err != nil {
			return c, err
		}
		c.recorded = true
		return c, nil
	})
}
