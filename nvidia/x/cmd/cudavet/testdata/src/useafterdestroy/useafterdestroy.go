package useafterdestroy

import "github.com/tmc/lib/nvidia/driver"

func streams() error {
	s, err := driver.NewStream(0)
	if err != nil {
		return err
	}
	s.Destroy()
	return s.Synchronize() // want `s is used after s.Destroy`
}

func memory(d driver.DevicePtr, host []float32) error {
	if err := d.Free(); err != nil {
		return err
	}
	return d.CopyToHost(host) // want `d is used after d.Free`
}

func reassigned() error {
	s, err := driver.NewStream(0)
	if err != nil {
		return err
	}
	s.Destroy()
	s, err = driver.NewStream(0)
	if err != nil {
		return err
	}
	defer s.Destroy()
	return s.Synchronize()
}
