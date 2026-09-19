//go:build linux

package cudagraph_test

import (
	"runtime"
	"syscall"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// TestCaptureStaysOnOneThread covers a capture that began on one OS thread and
// ended on another.
//
// A stream capture sequence belongs to the thread that began it, so ending it
// elsewhere fails with CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD. cuda.Context.Run
// locks the thread for the duration of one call and releases it on return,
// which left the capture function -- the longest part of a capture, and the
// part that yields -- running unpinned between the begin and end calls. It
// needs the scheduler to actually move the goroutine, so it presented as a
// flake in the work being captured rather than as a capture that never worked:
// one sweep saw it in about 4% of runs, aborting 8-55 seconds in.
//
// The capture function is where the only observation point is, so the test
// churns the scheduler there and asserts the thread did not change. Without
// the pin the test is not guaranteed to fail on any single run, which is the
// nature of the bug; with enough yields it fails nearly always.
func TestCaptureStaysOnOneThread(t *testing.T) {
	dev := cudatest.Require(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	var first, last int
	var moved int
	g, err := cudagraph.Capture(stream, func() {
		first = syscall.Gettid()
		last = first
		for i := 0; i < 200_000; i++ {
			runtime.Gosched()
			if tid := syscall.Gettid(); tid != last {
				moved++
				last = tid
			}
		}
	})
	if err != nil {
		t.Fatalf("capture: %v", err)
	}
	defer dev.WithContext(g.Destroy)

	if moved != 0 {
		t.Errorf("goroutine changed OS thread %d times during capture (%d -> %d): the capture is not pinned and its end call can land on another thread", moved, first, last)
	}
}
