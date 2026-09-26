// Package leakclean destroys what it creates or hands it back, and must
// produce no diagnostics.
package leakclean

import "github.com/tmc/lib/nvidia/driver"

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
