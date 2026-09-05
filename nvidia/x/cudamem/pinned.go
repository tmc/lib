package cudamem

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// CopyFromPinnedAsync copies src into s using CUDA-page-locked host memory.
// Both owners are retained until completion; Free returns ErrInUse meanwhile.
// Do not modify src until completion. The allocation must be usable from the
// stream's context. Graph capture is unsupported; callers must serialize capture
// transitions with this call. Close any non-nil completion, including on error.
func (s Slice[T]) CopyFromPinnedAsync(src HostAllocation[T], stream *cuda.Stream) (*Completion, error) {
	if src.len > s.len {
		return nil, errors.New("cudamem: pinned source exceeds destination length")
	}
	return s.copyPinnedAsync(src, stream, src.len, true)
}

// CopyToPinnedAsync copies s into dst using CUDA-page-locked host memory.
// Both owners are retained until completion. Do not access dst until completion.
// Context, capture and cleanup requirements are the same as CopyFromPinnedAsync.
func (s Slice[T]) CopyToPinnedAsync(dst HostAllocation[T], stream *cuda.Stream) (*Completion, error) {
	if dst.len < s.len {
		return nil, errors.New("cudamem: pinned destination is too short")
	}
	return s.copyPinnedAsync(dst, stream, s.len, false)
}

func (s Slice[T]) copyPinnedAsync(host HostAllocation[T], stream *cuda.Stream, count int, fromHost bool) (*Completion, error) {
	if stream == nil {
		return nil, errors.New("cudamem: nil transfer stream")
	}
	if count == 0 {
		return &Completion{}, nil
	}
	dev := stream.Device()
	releaseDevice, err := s.retain(dev)
	if err != nil {
		return nil, err
	}
	releaseHost, err := host.retain()
	if err != nil {
		releaseDevice()
		return nil, err
	}
	release := func() { releaseHost(); releaseDevice() }
	event, err := dev.NewEventWithFlags(uint32(driver.EventDisableTiming))
	if err != nil {
		release()
		return nil, fmt.Errorf("cudamem: create transfer event: %w", err)
	}
	state := &transferState{release: release, destroy: event.Destroy, wait: stream.Synchronize}
	completion := &Completion{state: state}
	attempted := false
	err = stream.WithHandle(func(handle driver.Stream) error {
		return dev.WithContext(func() error {
			capture, err := handle.IsCapturing()
			if err != nil {
				return err
			}
			if capture != driver.StreamCaptureStatusNone {
				return errors.New("pinned transfer during graph capture")
			}
			attempted = true
			ptr := unsafe.Pointer(&host.slice[0])
			n := uint(uintptr(count) * s.elemSize)
			if fromHost {
				err = s.ptr.MemcpyHtoDAsync(ptr, n, handle)
			} else {
				err = s.ptr.MemcpyDtoHAsync(ptr, n, handle)
			}
			if err != nil {
				return fmt.Errorf("enqueue: %w", err)
			}
			// Record through the raw handle within the same stream lifetime guard.
			// Calling Event.Record here would recursively acquire the stream guard.
			if err := event.Raw().Record(handle); err != nil {
				return fmt.Errorf("record completion: %w", err)
			}
			state.wait = event.Synchronize
			return nil
		})
	})
	if !attempted {
		state.complete()
		return completion, errors.Join(err, completion.Close())
	}
	// On enqueue/record errors CUDA may report earlier asynchronous failures.
	// Conservatively retain owners until successful stream synchronization.
	stream.AddCleanup(state.complete)
	if err != nil {
		return completion, fmt.Errorf("cudamem: pinned transfer: %w", err)
	}
	return completion, nil
}
