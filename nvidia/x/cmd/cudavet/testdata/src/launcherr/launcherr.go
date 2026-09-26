package launcherr

import "github.com/tmc/lib/nvidia/driver"

func discarded(f driver.Function, s driver.Stream, out driver.DevicePtr, host []float32) error {
	if err := f.LaunchKernel(driver.Dims(1, 1, 1), driver.Dims(32, 1, 1), 0, s, nil); err != nil { // want `LaunchKernel reports only that the launch was enqueued, and the error of Synchronize, the first call to wait for it, is discarded; check that error before reading results`
		return err
	}
	s.Synchronize()
	return out.CopyToHost(host)
}

func graph(exec driver.GraphExec, s driver.Stream) {
	exec.Launch(s) // want `Launch reports only that the launch was enqueued, and the error of CtxSynchronize, the first call to wait for it, is discarded`
	_ = driver.CtxSynchronize()
}

func checked(f driver.Function, s driver.Stream, out driver.DevicePtr, host []float32) error {
	if err := f.LaunchKernel(driver.Dims(1, 1, 1), driver.Dims(32, 1, 1), 0, s, nil); err != nil {
		return err
	}
	if err := s.Synchronize(); err != nil {
		return err
	}
	return out.CopyToHost(host)
}

func copied(f driver.Function, out driver.DevicePtr, host []float32) error {
	if err := f.LaunchKernel(driver.Dims(1, 1, 1), driver.Dims(32, 1, 1), 0, 0, nil); err != nil {
		return err
	}
	return out.CopyToHost(host)
}

// wait reports the error its own way; it has none to discard.
func wait(s driver.Stream) {
	if err := s.Synchronize(); err != nil {
		panic(err)
	}
}

func helper(f driver.Function, s driver.Stream) error {
	if err := f.LaunchKernel(driver.Dims(1, 1, 1), driver.Dims(32, 1, 1), 0, s, nil); err != nil {
		return err
	}
	wait(s)
	return nil
}
