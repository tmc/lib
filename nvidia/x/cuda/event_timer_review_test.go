package cuda

import (
	"errors"
	"slices"
	"testing"
	"time"
)

func TestEventTimerCompletionFallbackDrainsEveryStream(t *testing.T) {
	for _, stage := range []string{"end-record", "end-wait", "stop-record", "stop-sync", "elapsed"} {
		t.Run(stage, func(t *testing.T) {
			timer, _, api := timerFixture(t)
			failure := errors.New("injected " + stage)
			drainFailure := errors.New("first stream drain")
			record, wait, syncEvent, elapsed := api.record, api.wait, api.syncEvent, api.elapsed
			api.record = func(e *Event, s *Stream) error {
				if stage == "end-record" && e == timer.ends[0] || stage == "stop-record" && e == timer.stop {
					return failure
				}
				return record(e, s)
			}
			api.wait = func(s *Stream, e *Event) error {
				if stage == "end-wait" && s == timer.control && e == timer.ends[0] {
					return failure
				}
				return wait(s, e)
			}
			api.syncEvent = func(e *Event) error {
				if stage == "stop-sync" {
					return failure
				}
				return syncEvent(e)
			}
			api.elapsed = func(a, b *Event) (time.Duration, error) {
				if stage == "elapsed" {
					return 0, failure
				}
				return elapsed(a, b)
			}
			var drained []*Stream
			api.syncStream = func(s *Stream) error {
				drained = append(drained, s)
				if s == timer.streams[0] {
					return drainFailure
				}
				return nil
			}
			ran := false
			duration, err := timer.Measure(func() error { ran = true; return nil })
			if !ran || duration != 0 || !errors.Is(err, failure) || !errors.Is(err, drainFailure) {
				t.Fatalf("Measure ran=%v duration=%v error=%v", ran, duration, err)
			}
			if !slices.Equal(drained, timer.streams) {
				t.Fatalf("drained=%v want every stream in %v", drained, timer.streams)
			}
			// Recovery is explicit; a failed first synchronization cannot free events.
			if err := timer.Close(); !errors.Is(err, drainFailure) {
				t.Fatalf("Close=%v", err)
			}
			if timer.start == nil || timer.closed {
				t.Fatal("failed drain released owned events")
			}
			api.syncStream = func(*Stream) error { return nil }
			if err := timer.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEventTimerSuccessfulJoinDoesNotFallback(t *testing.T) {
	timer, _, api := timerFixture(t)
	drains := 0
	api.syncStream = func(*Stream) error { drains++; return errors.New("unexpected fallback") }
	duration, err := timer.Measure(func() error { return nil })
	if err != nil || duration != 7*time.Microsecond || drains != 0 {
		t.Fatalf("Measure=%v,%v drains=%d", duration, err, drains)
	}
	api.syncStream = func(*Stream) error { return nil }
	if err := timer.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestEventTimerPanicWithBrokenCompletionStillDrains(t *testing.T) {
	timer, _, api := timerFixture(t)
	record := api.record
	api.record = func(e *Event, s *Stream) error {
		if e == timer.ends[0] {
			return errors.New("completion failed")
		}
		return record(e, s)
	}
	var drained []*Stream
	api.syncStream = func(s *Stream) error { drained = append(drained, s); return nil }
	panicValue := errors.New("workload panic")
	func() {
		defer func() {
			if got := recover(); got != panicValue {
				t.Errorf("panic=%v want %v", got, panicValue)
			}
		}()
		_, _ = timer.Measure(func() error { panic(panicValue) })
		t.Error("panic lost")
	}()
	if !slices.Equal(drained, timer.streams) {
		t.Fatal("panic bypassed fallback stream drain")
	}
	if err := timer.Close(); err != nil {
		t.Fatal(err)
	}
}
