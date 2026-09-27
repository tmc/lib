package useafterdestroy

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func wrapped(dev *cuda.Device) error {
	s, err := dev.NewStream()
	if err != nil {
		return err
	}
	s.Close()
	return s.Synchronize(context.Background()) // want `s is used after s.Close`
}

var errBusy = errors.New("busy")

// closedRejects tests the wrapper's record of its release: a second Close,
// a call expected to fail, and a stream that was never made.
func closedRejects(dev *cuda.Device) {
	s, err := dev.NewStream()
	if err != nil {
		panic(err)
	}
	if err := s.Close(); err != nil {
		panic(err)
	}
	if err := s.Synchronize(context.Background()); err == nil {
		panic("Synchronize succeeded after Close")
	}
	if err := s.Close(); err != nil {
		panic(err)
	}
	busy, err := dev.NewStream()
	if err != nil {
		panic(err)
	}
	if err := busy.Close(); !errors.Is(err, errBusy) {
		panic(err)
	}
	busy.Synchronize(context.Background())
	var never *cuda.Stream
	never.Close()
	never.Synchronize(context.Background())
}

// doubleClose releases a driver handle twice, which the driver cannot detect.
func doubleClose() error {
	s, err := driver.NewStream(0)
	if err != nil {
		return err
	}
	s.Close()
	return s.Close() // want `s is used after s.Close`
}

func wrappedClean(dev *cuda.Device) error {
	s, err := dev.NewStream()
	if err != nil {
		return err
	}
	defer s.Close()
	return s.Synchronize(context.Background())
}

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

// sliceFreedTwice tests that a Slice's copies share one record of its
// release, so a second Free releases nothing.
func sliceFreedTwice(s cudamem.Slice[float32]) error {
	if err := s.Free(); err != nil {
		return err
	}
	if err := s.Free(); err != nil {
		return fmt.Errorf("second Free: %w", err)
	}
	return nil
}

// valueFreedTwice frees a bare driver handle twice, as sliceFreedTwice does a
// Slice; nothing records the first release.
func valueFreedTwice(d driver.DevicePtr) error {
	if err := d.Free(); err != nil {
		return err
	}
	if err := d.Free(); err != nil { // want `d is used after d.Free`
		return fmt.Errorf("second Free: %w", err)
	}
	return nil
}

// checked frees d, then frees it again: the first Free succeeded.
func checked(d driver.DevicePtr) error {
	if err := d.Free(); err != nil {
		return err
	}
	return d.Free() // want `d is used after d.Free`
}
