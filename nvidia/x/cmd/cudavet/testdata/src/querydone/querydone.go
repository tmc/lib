package querydone

import (
	"errors"

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

// afterClose tests the closed handle, where only the error means anything.
func afterClose(e *cuda.Event) error {
	if err := e.Close(); err != nil {
		return err
	}
	if _, err := e.Query(); err == nil {
		return errors.New("Query succeeded after Close")
	}
	return nil
}

// duringClose queries from a goroutine while the handle is closed.
func duringClose(s *cuda.Stream) error {
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			if _, err := s.Query(); err != nil {
				return
			}
		}
	}()
	err := s.Close()
	<-done
	return err
}

// deferredClose closes only on the way out, so the Query polls a live handle.
func deferredClose(s *cuda.Stream) error {
	defer s.Close()
	_, err := s.Query() // want `the done result of Stream.Query is discarded; a nil error does not mean the work is done`
	return err
}

// closedLater closes after the Query on the same goroutine.
func closedLater(e *cuda.Event) error {
	_, err := e.Query() // want `the done result of Event.Query is discarded; a nil error does not mean the work is done`
	return errors.Join(err, e.Close())
}

// dropDriver throws away the driver's whole answer.
func dropDriver(s driver.Stream, e driver.Event) {
	s.Query() // want `the result of Stream.Query is discarded; it is nil when the work is done and ErrNotReady when it is not`
	e.Query() // want `the result of Event.Query is discarded; it is nil when the work is done and ErrNotReady when it is not`
}

// blankDriver discards the answer on purpose.
func blankDriver(s driver.Stream) {
	_ = s.Query()
}

// spinDriver waits on any error, so a failed stream spins forever.
func spinDriver(s driver.Stream, e driver.Event) {
	for s.Query() != nil { // want `the loop waits while Stream.Query returns any error; only ErrNotReady means not done, and a failure spins it forever`
	}
	for err := e.Query(); err != nil; err = e.Query() { // want `the loop waits while Event.Query returns any error; only ErrNotReady means not done, and a failure spins it forever`
	}
}

// waitDriver stops on an error other than ErrNotReady.
func waitDriver(s driver.Stream) error {
	err := s.Query()
	for err != nil {
		if !errors.Is(err, driver.ErrNotReady) {
			return err
		}
		err = s.Query()
	}
	for errors.Is(s.Query(), driver.ErrNotReady) {
	}
	return nil
}
