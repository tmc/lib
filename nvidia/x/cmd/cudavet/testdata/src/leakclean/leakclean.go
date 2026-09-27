// Package leakclean destroys what it creates or hands it back, and must
// produce no diagnostics.
package leakclean

import (
	"context"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func wrapped(dev *cuda.Device) error {
	s, err := dev.NewStream()
	if err != nil {
		return err
	}
	defer s.Close()
	return s.Synchronize(context.Background())
}

// borrowed is the device's own stream, which the device releases.
func borrowed(dev *cuda.Device) error {
	s := dev.DefaultStream()
	return s.Synchronize(context.Background())
}

// catalog wraps no device handle, so its Close is not a release.
func catalog(chip string, avail []byte) error {
	c, err := cuptirange.NewCatalog(chip, avail)
	if err != nil {
		return err
	}
	_ = c
	return nil
}

func work() error {
	s, err := driver.NewStream(0)
	if err != nil {
		return err
	}
	defer s.Close()
	return nil
}

// open gives the stream to its caller, whose lifetime it becomes.
func open() (driver.Stream, error) {
	s, err := driver.NewStream(0)
	if err != nil {
		return 0, err
	}
	return s, nil
}
