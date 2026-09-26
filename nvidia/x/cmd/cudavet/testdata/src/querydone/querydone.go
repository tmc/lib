package querydone

import "github.com/tmc/lib/nvidia/driver"

func poll(e driver.Event) error {
	_, err := e.Query() // want `the done result of Event.Query is discarded; a nil error does not mean the work is done`
	return err
}

func drop(s driver.Stream) {
	s.Query() // want `the done result of Stream.Query is discarded; a nil error does not mean the work is done`
}

func ready(e driver.Event) (bool, error) {
	return e.Query()
}

func spin(s driver.Stream) error {
	for {
		done, err := s.Query()
		if err != nil || done {
			return err
		}
	}
}
