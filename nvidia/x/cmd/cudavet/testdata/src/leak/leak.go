package leak

import "github.com/tmc/lib/nvidia/driver"

func work() error {
	s, err := driver.NewStream(0) // want `s is created and never destroyed; defer s.Destroy\(\)`
	if err != nil {
		return err
	}
	e, err := driver.NewEvent(0) // want `e is created and never destroyed; defer e.Destroy\(\)`
	if err != nil {
		return err
	}
	_ = s
	_ = e
	return nil
}
