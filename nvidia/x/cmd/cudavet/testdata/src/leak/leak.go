package leak

import (
	"github.com/tmc/lib/nvidia/cusolver"
	"github.com/tmc/lib/nvidia/driver"
)

func work() error {
	s, err := driver.NewStream(0) // want `s is created and never released; defer s.Close\(\)`
	if err != nil {
		return err
	}
	e, err := driver.NewEvent(0) // want `e is created and never released; defer e.Close\(\)`
	if err != nil {
		return err
	}
	_ = s
	_ = e
	return nil
}

func handles() error {
	d, err := driver.MemAlloc[float32](4) // want `d is created and never released; defer d.Free\(\)`
	if err != nil {
		return err
	}
	h, err := cusolver.Create() // want `h is created and never released; defer h.Close\(\)`
	if err != nil {
		return err
	}
	_ = d
	_ = h
	return nil
}

// captured assigns a stream of the function around it, which releases it.
func captured() error {
	var s driver.Stream
	create := func() error {
		var err error
		s, err = driver.NewStream(0)
		return err
	}
	if err := create(); err != nil {
		return err
	}
	return s.Close()
}
