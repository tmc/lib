package capturesync

import (
	"context"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

// syncInCapture waits for the stream it is capturing.
func syncInCapture(s driver.Stream) (driver.Graph, error) {
	if err := s.BeginCapture(driver.StreamCaptureModeGlobal); err != nil {
		return 0, err
	}
	s.Synchronize() // want `s.Synchronize waits for stream s while it is being captured, which invalidates the capture`
	return s.EndCapture()
}

// syncAfterCapture waits once the capture has ended.
func syncAfterCapture(s driver.Stream) (driver.Graph, error) {
	if err := s.BeginCapture(driver.StreamCaptureModeGlobal); err != nil {
		return 0, err
	}
	g, err := s.EndCapture()
	if err != nil {
		return 0, err
	}
	s.Synchronize()
	driver.CtxSynchronize()
	return g, nil
}

// syncOther waits for a stream that is not being captured.
func syncOther(s, t driver.Stream) (driver.Graph, error) {
	s.BeginCapture(driver.StreamCaptureModeGlobal)
	t.Synchronize()
	return s.EndCapture()
}

// syncOnBranch waits on one path through the capture.
func syncOnBranch(s driver.Stream, check bool) (driver.Graph, error) {
	s.BeginCapture(driver.StreamCaptureModeRelaxed)
	if check {
		driver.CtxSynchronize() // want `CtxSynchronize waits for every stream of the context while s is being captured, which invalidates the capture`
	}
	return s.EndCapture()
}

// deferredSync waits when the function returns, after the capture ends.
func deferredSync(s driver.Stream) (driver.Graph, error) {
	s.BeginCapture(driver.StreamCaptureModeGlobal)
	defer s.Synchronize()
	return s.EndCapture()
}

// endedElsewhere begins a capture that a caller ends.
func endedElsewhere(s driver.Stream) {
	s.BeginCapture(driver.StreamCaptureModeGlobal)
	driver.CtxSynchronize()
}

// copies copies synchronously and asynchronously during a capture.
func copies(s driver.Stream, d, e driver.DevicePtr, host []float32) (driver.Graph, error) {
	s.BeginCapture(driver.StreamCaptureModeGlobal)
	d.CopyToHost(host) // want `CopyToHost is synchronous and runs on the legacy stream, which Global capture mode prohibits while s is being captured; use the Async form on s`
	d.Memcpy(e, 4)     // want `Memcpy is synchronous and runs on the legacy stream, which Global capture mode prohibits while s is being captured; use the Async form on s`
	d.MemsetD8(0, 4)   // want `MemsetD8 is synchronous and runs on the legacy stream, which Global capture mode prohibits while s is being captured; use the Async form on s`
	d.MemcpyAsync(e, 4, s)
	d.CopyToHostAsync(host, s)
	d.MemsetD8Async(0, 4, s)
	g, err := s.EndCapture()
	d.CopyToHost(host)
	return g, err
}

// relaxedCopy copies synchronously in relaxed mode, which allows it on a
// non-blocking stream.
func relaxedCopy(s driver.Stream, d, e driver.DevicePtr) (driver.Graph, error) {
	s.BeginCapture(driver.StreamCaptureModeRelaxed)
	d.Memcpy(e, 4)
	return s.EndCapture()
}

// eventSync waits for an event recorded during the capture.
func eventSync(s driver.Stream, e, f driver.Event) (driver.Graph, error) {
	s.BeginCapture(driver.StreamCaptureModeThreadLocal)
	e.Record(s)
	f.Synchronize()
	e.Synchronize() // want `e.Synchronize waits for an event recorded on s while it is being captured, which invalidates the capture`
	return s.EndCapture()
}

// hostDot reads a dot product into host memory on the captured stream.
func hostDot(s driver.Stream, x, y driver.DevicePtr, n int32) (driver.Graph, error) {
	h, err := cublas.Create()
	if err != nil {
		return 0, err
	}
	defer h.Close()
	h.SetStream(s)
	s.BeginCapture(driver.StreamCaptureModeGlobal)
	h.Sdot(n, x, 1, y, 1) // want `Sdot writes its result to host memory, which forces a synchronization while s is being captured; set device pointer mode and write the result to device memory`
	var r float32
	h.DotEx(n, x, cublas.R32F, 1, y, cublas.R32F, 1, unsafe.Pointer(&r), cublas.R32F, cublas.R32F) // want `DotEx writes its result to host memory`
	g, err := s.EndCapture()
	h.Sdot(n, x, 1, y, 1)
	return g, err
}

// deviceDot writes a dot product to device memory.
func deviceDot(s driver.Stream, x, y, r driver.DevicePtr, n int32) (driver.Graph, error) {
	h, err := cublas.Create()
	if err != nil {
		return 0, err
	}
	defer h.Close()
	h.SetStream(s)
	h.SetPointerMode(cublas.PointerModeDevice)
	s.BeginCapture(driver.StreamCaptureModeGlobal)
	h.DotEx(n, x, cublas.R32F, 1, y, cublas.R32F, 1, unsafe.Pointer(uintptr(r)), cublas.R32F, cublas.R32F)
	return s.EndCapture()
}

// passedHandle has a handle in an unknown mode: a result returned as a Go
// value is host memory, one written through an unsafe.Pointer may not be.
func passedHandle(h cublas.Handle, s driver.Stream, x, y driver.DevicePtr, res unsafe.Pointer, n int32) (driver.Graph, error) {
	h.SetStream(s)
	s.BeginCapture(driver.StreamCaptureModeGlobal)
	h.Isamax(n, x, 1) // want `Isamax writes its result to host memory`
	var a, b, c, d float32
	h.Srotg(&a, &b, &c, &d) // want `Srotg writes its result to host memory`
	h.DotEx(n, x, cublas.R32F, 1, y, cublas.R32F, 1, res, cublas.R32F, cublas.R32F)
	return s.EndCapture()
}

// otherStream reduces on a handle bound to a stream not being captured.
func otherStream(h cublas.Handle, s, t driver.Stream, x, y driver.DevicePtr, n int32) (driver.Graph, error) {
	h.SetStream(t)
	s.BeginCapture(driver.StreamCaptureModeGlobal)
	h.Sdot(n, x, 1, y, 1)
	return s.EndCapture()
}

// inLoop synchronizes each time round a loop inside the capture.
func inLoop(s driver.Stream, n int) (driver.Graph, error) {
	s.BeginCapture(driver.StreamCaptureModeGlobal)
	for range n {
		s.Synchronize() // want `s.Synchronize waits for stream s`
	}
	return s.EndCapture()
}

// graphCapture synchronizes inside the function cudagraph.Capture records.
func graphCapture(ctx context.Context, s *cuda.Stream, c *cuda.Context, h cublas.Handle, x, y driver.DevicePtr, host []float32) error {
	h.SetStream(s.Raw())
	_, err := cudagraph.Capture(s, func() {
		s.Synchronize(ctx)    // want `s.Synchronize waits for stream s`
		c.Synchronize()       // want `Synchronize waits for every stream of the context while s is being captured`
		x.CopyToHost(host)    // want `CopyToHost is synchronous`
		h.Sdot(4, x, 1, y, 1) // want `Sdot writes its result to host memory`
		x.CopyToHostAsync(host, s.Raw())
		go s.Synchronize(ctx)
	})
	s.Synchronize(ctx)
	return err
}

// rawCapture captures an x/cuda stream through its driver handle.
func rawCapture(ctx context.Context, s *cuda.Stream) (driver.Graph, error) {
	if err := s.Raw().BeginCapture(driver.StreamCaptureModeGlobal); err != nil {
		return 0, err
	}
	s.Synchronize(ctx) // want `s.Synchronize waits for stream s`
	return s.Raw().EndCapture()
}
