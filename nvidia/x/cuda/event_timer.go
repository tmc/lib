package cuda

import (
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"sync"
	"time"
)

// EventTimer measures completed GPU work across a fixed set of streams. The
// streams and device must outlive it. Its zero value is not usable. Do not copy
// a timer. Concurrent Measure and Close calls fail.
type EventTimer struct {
	mu          sync.Mutex
	api         timerAPI
	streams     []*Stream
	control     *Stream
	start, stop *Event
	ends        []*Event
	closed      bool
}

type timerAPI struct {
	stream        func() (*Stream, error)
	event         func(bool) (*Event, error)
	record        func(*Event, *Stream) error
	wait          func(*Stream, *Event) error
	syncEvent     func(*Event) error
	syncStream    func(*Stream) error
	elapsed       func(*Event, *Event) (time.Duration, error)
	destroyEvent  func(*Event) error
	destroyStream func(*Stream) error
}

// NewEventTimer creates a timer for streams on dev. Its nonblocking coordination
// stream does not order participating streams against each other. Call Close on
// a nonnil timer returned with an error to retry cleanup. Callers must exclude
// unrelated work submitted to these streams.
func NewEventTimer(dev *Device, streams []*Stream) (*EventTimer, error) {
	if dev == nil || len(streams) == 0 {
		return nil, errors.New("cuda: event timer requires a device and streams")
	}
	seen := make(map[driver.Stream]bool)
	for _, s := range streams {
		if s == nil || s.Device() != dev {
			return nil, errors.New("cuda: event timer stream belongs to another device")
		}
		if seen[s.Raw()] {
			return nil, errors.New("cuda: duplicate event timer stream")
		}
		seen[s.Raw()] = true
	}
	return newEventTimer(streams, timerAPI{
		stream: func() (*Stream, error) { return dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking)) },
		event: func(timing bool) (*Event, error) {
			if timing {
				return dev.NewEvent()
			}
			return dev.NewEventWithFlags(uint32(driver.EventDisableTiming))
		},
		record: (*Event).Record, wait: (*Stream).WaitEvent,
		syncEvent:  func(e *Event) error { return e.Synchronize(nil) },
		syncStream: func(s *Stream) error { return s.Synchronize(nil) },
		elapsed:    (*Event).ElapsedSince, destroyEvent: (*Event).Destroy, destroyStream: (*Stream).Destroy,
	})
}

func newEventTimer(streams []*Stream, api timerAPI) (t *EventTimer, err error) {
	t = &EventTimer{api: api, streams: append([]*Stream(nil), streams...)}
	defer func() {
		if err != nil {
			err = errors.Join(err, t.Close())
		}
	}()
	t.control, err = api.stream()
	if err != nil {
		return t, fmt.Errorf("cuda: timer stream: %w", err)
	}
	t.start, err = api.event(true)
	if err != nil {
		return t, fmt.Errorf("cuda: timer start: %w", err)
	}
	t.stop, err = api.event(true)
	if err != nil {
		return t, fmt.Errorf("cuda: timer stop: %w", err)
	}
	for range streams {
		e, eerr := api.event(false)
		if e != nil {
			t.ends = append(t.ends, e)
		}
		if eerr != nil {
			return t, fmt.Errorf("cuda: timer completion: %w", eerr)
		}
	}
	return t, nil
}

// Measure runs fn once and measures the event interval covering its GPU work.
// Synchronize prior work and preparation before calling Measure. fn must submit
// all measured work to the declared streams before returning. The timer joins
// completion even when fn fails or panics. Duration is usable only without an
// error; zero means the device clock did not resolve the interval.
func (t *EventTimer) Measure(fn func() error) (duration time.Duration, err error) {
	if t == nil || fn == nil {
		return 0, errors.New("cuda: invalid event timer measurement")
	}
	if !t.mu.TryLock() {
		return 0, errors.New("cuda: event timer busy")
	}
	defer t.mu.Unlock()
	if t.closed || t.start == nil || t.stop == nil || len(t.ends) != len(t.streams) {
		return 0, errors.New("cuda: event timer is not open")
	}
	if err = t.api.record(t.start, t.control); err != nil {
		return 0, fmt.Errorf("cuda: timer start: %w", err)
	}
	for _, s := range t.streams {
		if err = t.api.wait(s, t.start); err != nil {
			return 0, fmt.Errorf("cuda: timer start dependency: %w", err)
		}
	}
	defer func() {
		var finishErr error
		for i, s := range t.streams {
			if e := t.api.record(t.ends[i], s); e != nil {
				finishErr = errors.Join(finishErr, e)
				continue
			}
			finishErr = errors.Join(finishErr, t.api.wait(t.control, t.ends[i]))
		}
		if finishErr == nil {
			finishErr = t.api.record(t.stop, t.control)
		}
		if finishErr == nil {
			finishErr = t.api.syncEvent(t.stop)
		}
		if finishErr == nil {
			duration, finishErr = t.api.elapsed(t.stop, t.start)
		}
		if finishErr != nil {
			for _, stream := range t.streams {
				finishErr = errors.Join(finishErr, t.api.syncStream(stream))
			}
			err = errors.Join(err, fmt.Errorf("cuda: timer completion: %w", finishErr))
		}
		if err != nil {
			duration = 0
		}
	}()
	err = fn()
	return
}

// Close synchronizes participating streams and releases owned resources. It does
// not destroy caller streams. Failed cleanup can be retried. Close is nil-safe;
// repeated successful calls do nothing.
func (t *EventTimer) Close() error {
	if t == nil {
		return nil
	}
	if !t.mu.TryLock() {
		return errors.New("cuda: event timer busy")
	}
	defer t.mu.Unlock()
	if t.closed {
		return nil
	}
	var err error
	for _, s := range t.streams {
		err = errors.Join(err, t.api.syncStream(s))
	}
	if err != nil {
		return fmt.Errorf("cuda: timer drain: %w", err)
	}
	for _, p := range []**Event{&t.start, &t.stop} {
		if *p != nil {
			if e := t.api.destroyEvent(*p); e != nil {
				err = errors.Join(err, e)
			} else {
				*p = nil
			}
		}
	}
	for i, e := range t.ends {
		if e != nil {
			if eerr := t.api.destroyEvent(e); eerr != nil {
				err = errors.Join(err, eerr)
			} else {
				t.ends[i] = nil
			}
		}
	}
	if err != nil {
		return fmt.Errorf("cuda: timer events: %w", err)
	}
	if t.control != nil {
		if err = t.api.destroyStream(t.control); err != nil {
			return fmt.Errorf("cuda: timer stream: %w", err)
		}
		t.control = nil
	}
	t.closed = true
	return nil
}
