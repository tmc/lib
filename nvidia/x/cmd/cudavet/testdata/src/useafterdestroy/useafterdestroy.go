package useafterdestroy

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
)

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

// zeroed checks that Close zeroed s; comparing it with 0 does not use it.
func zeroed() (bool, error) {
	s, err := driver.NewStream(0)
	if err != nil {
		return false, err
	}
	if err := s.Close(); err != nil {
		return false, err
	}
	if s != 0 {
		return false, fmt.Errorf("stream = %#x after Close, want 0", uintptr(s))
	}
	return true, nil
}

// printed hands s to a formatting call inside another call, which is a use.
func printed() error {
	s, err := driver.NewStream(0)
	if err != nil {
		return err
	}
	s.Close()
	return fmt.Errorf("sync: %v", s.Synchronize()) // want `s is used after s.Close`
}

// compared calls s in a comparison, which is a use.
func compared() error {
	s, err := driver.NewStream(0)
	if err != nil {
		return err
	}
	s.Close()
	if s.Synchronize() != nil { // want `s is used after s.Close`
		return nil
	}
	return nil
}

// refused frees d only when the first Free is refused, so d is still live.
func refused(d driver.DevicePtr) error {
	if err := d.Free(); !errors.Is(err, driver.ErrInvalidValue) {
		return errors.New("Free accepted a busy allocation")
	}
	return d.Free()
}

func refusedNil(d driver.DevicePtr) error {
	if err := d.Free(); err == nil {
		panic("Free succeeded")
	}
	return d.Free()
}

// checked frees d, then frees it again: the first Free succeeded.
func checked(d driver.DevicePtr) error {
	if err := d.Free(); err != nil {
		return err
	}
	return d.Free() // want `d is used after d.Free`
}
