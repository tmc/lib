package useafterdestroy

import "github.com/tmc/lib/nvidia/driver"

func streams() error {
	s, err := driver.NewStream(0)
	if err != nil {
		return err
	}
	s.Close()
	return s.Synchronize() // want `s is used after s.Close`
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
	s.Close()
	s, err = driver.NewStream(0)
	if err != nil {
		return err
	}
	defer s.Close()
	return s.Synchronize()
}
