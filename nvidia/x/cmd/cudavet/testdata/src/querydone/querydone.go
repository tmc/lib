package querydone

import (
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func poll(e *cuda.Event) error {
	_, err := e.Query() // want `the done result of Event.Query is discarded; a nil error does not mean the work is done`
	return err
}

func drop(s *cuda.Stream) {
	s.Query() // want `the done result of Stream.Query is discarded; a nil error does not mean the work is done`
}

func ready(e *cuda.Event) (bool, error) {
	return e.Query()
}

func spin(s *cuda.Stream) error {
	for {
		done, err := s.Query()
		if err != nil || done {
			return err
		}
	}
}

// binding is the driver's Query, which reports not done as ErrNotReady and
// has no done result to discard.
func binding(s driver.Stream) error {
	return s.Query()
}
