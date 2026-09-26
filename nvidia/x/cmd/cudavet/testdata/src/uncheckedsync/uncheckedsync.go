package uncheckedsync

import (
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaring"
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

// ring is not a binding handle, so its Close is not a release.
func ring(r cudaring.Ring) {
	r.Close()
}
