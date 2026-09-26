package uncheckedsync

import "github.com/tmc/lib/nvidia/driver"

func dropped(s driver.Stream) {
	s.Synchronize() // want `the error of Synchronize is discarded; it is where failures of earlier asynchronous work are reported`
}

func ctx() {
	driver.CtxSynchronize() // want `the error of CtxSynchronize is discarded; it is where failures of earlier asynchronous work are reported`
}

func blank(s driver.Stream) {
	_ = s.Synchronize()
	_ = s.Destroy()
}

func fatal(s driver.Stream, d driver.DevicePtr, host []float32) {
	if err := d.CopyToHost(host); err != nil {
		s.Destroy()
		panic(err)
	}
}

func release(s driver.Stream) error {
	s.Destroy() // want `the error of Destroy is discarded; it can report a failure of earlier asynchronous work`
	return nil
}

func deferred(s driver.Stream) error {
	defer s.Destroy()
	return s.Synchronize()
}

func errorPath(s driver.Stream, d driver.DevicePtr, host []float32) error {
	if err := d.CopyToHost(host); err != nil {
		s.Destroy()
		return err
	}
	return s.Destroy()
}
