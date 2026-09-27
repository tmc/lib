package uncheckedsync

import (
	"context"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func dropped(s driver.Stream) {
	s.Synchronize() // want `the error of Synchronize is discarded; it is where failures of earlier asynchronous work are reported`
}

func ctx() {
	driver.CtxSynchronize() // want `the error of CtxSynchronize is discarded; it is where failures of earlier asynchronous work are reported`
}

func blank(s driver.Stream) {
	_ = s.Synchronize()
	_ = s.Close()
}

func fatal(s driver.Stream, d driver.DevicePtr, host []float32) {
	if err := d.CopyToHost(host); err != nil {
		s.Close()
		panic(err)
	}
}

func release(s driver.Stream) error {
	s.Close() // want `the error of Close is discarded; it can report a failure of earlier asynchronous work`
	return nil
}

func deferred(s driver.Stream) error {
	defer s.Close()
	return s.Synchronize()
}

func errorPath(s driver.Stream, d driver.DevicePtr, host []float32) error {
	if err := d.CopyToHost(host); err != nil {
		s.Close()
		return err
	}
	return s.Close()
}

func wrapped(s *cuda.Stream) error {
	s.Close() // want `the error of Close is discarded; it can report a failure of earlier asynchronous work`
	return nil
}

// thenSync returns the result of more work, not an error it already had.
func thenSync(e *cuda.Event, s *cuda.Stream) error {
	e.Close() // want `the error of Close is discarded; it can report a failure of earlier asynchronous work`
	return s.Synchronize(context.Background())
}

// failed returns an error it makes, so Close is on an error path.
func failed(s *cuda.Stream, n int) error {
	if n < 0 {
		s.Close()
		return fmt.Errorf("negative count %d", n)
	}
	return s.Close()
}

// catalog wraps no device handle, so its Close is not a release.
func catalog(c *cuptirange.Catalog) {
	c.Close()
}

// cleanup releases d when the test ends, the way defer d.Free() would.
func cleanup(t *testing.T, d driver.DevicePtr) {
	t.Cleanup(func() { d.Free() })
}

func deferredLit(d driver.DevicePtr) {
	defer func() { d.Free() }()
}

// loop releases d in a closure that runs now, not as a cleanup.
func loop(d driver.DevicePtr) {
	func() {
		d.Free() // want `the error of Free is discarded; it can report a failure of earlier asynchronous work`
	}()
}

// unexpected releases what a call that should have failed returned, then
// fails the test.
func unexpected(t *testing.T) {
	s, err := driver.NewStream(0)
	if err == nil {
		if s != 0 {
			s.Close()
		}
		t.Fatal("NewStream succeeded")
	}
}

// loopBody releases s in a loop that goes around again; the t.Fatal after
// the loop does not make it an error path.
func loopBody(t *testing.T, streams []driver.Stream) {
	for _, s := range streams {
		if s != 0 {
			s.Close() // want `the error of Close is discarded; it can report a failure of earlier asynchronous work`
		}
	}
	t.Fatal("done")
}
