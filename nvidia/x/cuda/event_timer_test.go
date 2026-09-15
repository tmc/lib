package cuda

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"
)

func timerFixture(t *testing.T) (*EventTimer, *[]string, *timerAPI) {
	t.Helper()
	var log []string
	streams := []*Stream{{}, {}}
	control := &Stream{}
	events := map[*Event]int{}
	next := 0
	streamName := func(s *Stream) string {
		if s == control {
			return "control"
		}
		if s == streams[0] {
			return "a"
		}
		return "b"
	}
	api := timerAPI{
		stream: func() (*Stream, error) { return control, nil },
		event:  func(bool) (*Event, error) { e := &Event{}; events[e] = next; next++; return e, nil },
		record: func(e *Event, s *Stream) error {
			log = append(log, fmt.Sprintf("record %d %s", events[e], streamName(s)))
			return nil
		},
		wait: func(s *Stream, e *Event) error {
			log = append(log, fmt.Sprintf("wait %s %d", streamName(s), events[e]))
			return nil
		},
		syncEvent:     func(e *Event) error { log = append(log, fmt.Sprintf("sync %d", events[e])); return nil },
		syncStream:    func(s *Stream) error { log = append(log, "drain "+streamName(s)); return nil },
		elapsed:       func(*Event, *Event) (time.Duration, error) { return 7 * time.Microsecond, nil },
		destroyEvent:  func(e *Event) error { log = append(log, fmt.Sprintf("destroy %d", events[e])); return nil },
		destroyStream: func(s *Stream) error { log = append(log, "destroy "+streamName(s)); return nil },
	}
	timer, err := newEventTimer(streams, api)
	if err != nil {
		t.Fatal(err)
	}
	return timer, &log, &timer.api
}

func TestEventTimerJoinsStreams(t *testing.T) {
	timer, log, _ := timerFixture(t)
	got, err := timer.Measure(func() error { *log = append(*log, "work"); return nil })
	if err != nil || got != 7*time.Microsecond {
		t.Fatalf("Measure = %v, %v", got, err)
	}
	want := []string{"record 0 control", "wait a 0", "wait b 0", "work", "record 2 a", "wait control 2", "record 3 b", "wait control 3", "record 1 control", "sync 1"}
	if !reflect.DeepEqual(*log, want) {
		t.Fatalf("events = %v; want %v", *log, want)
	}
	if err := timer.Close(); err != nil {
		t.Fatal(err)
	}
	for _, s := range *log {
		if s == "destroy a" || s == "destroy b" {
			t.Fatal("destroyed caller stream")
		}
	}
	n := len(*log)
	if err := timer.Close(); err != nil || len(*log) != n {
		t.Fatal("close not idempotent")
	}
}

func TestEventTimerFailureAndPanicJoin(t *testing.T) {
	for _, panicWork := range []bool{false, true} {
		t.Run(fmt.Sprint(panicWork), func(t *testing.T) {
			timer, log, _ := timerFixture(t)
			failure := errors.New("work failed")
			func() {
				defer func() {
					v := recover()
					if panicWork && v != failure {
						t.Errorf("panic=%v", v)
					}
				}()
				d, err := timer.Measure(func() error {
					if panicWork {
						panic(failure)
					}
					return failure
				})
				if !errors.Is(err, failure) || d != 0 {
					t.Errorf("Measure=%v,%v", d, err)
				}
			}()
			if (*log)[len(*log)-1] != "sync 1" {
				t.Fatalf("did not join: %v", *log)
			}
			if err := timer.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEventTimerRetryCleanup(t *testing.T) {
	timer, _, api := timerFixture(t)
	original := api.destroyEvent
	failure := errors.New("destroy failed")
	once := true
	api.destroyEvent = func(e *Event) error {
		if once {
			once = false
			return failure
		}
		return original(e)
	}
	if err := timer.Close(); !errors.Is(err, failure) {
		t.Fatalf("Close=%v", err)
	}
	if timer.closed || timer.start == nil {
		t.Fatal("lost failed resource")
	}
	if err := timer.Close(); err != nil {
		t.Fatal(err)
	}
	if !timer.closed {
		t.Fatal("not closed")
	}
}

func TestEventTimerBusy(t *testing.T) {
	timer, _, _ := timerFixture(t)
	_, err := timer.Measure(func() error {
		if err := timer.Close(); err == nil {
			t.Error("concurrent Close succeeded")
		}
		if _, err := timer.Measure(func() error { return nil }); err == nil {
			t.Error("nested Measure succeeded")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := timer.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestEventTimerCompletionFailure(t *testing.T) {
	timer, _, api := timerFixture(t)
	fail := errors.New("event failed")
	api.syncEvent = func(*Event) error { return fail }
	if d, err := timer.Measure(func() error { return nil }); d != 0 || !errors.Is(err, fail) {
		t.Fatalf("Measure=%v,%v", d, err)
	}
	if err := timer.Close(); err != nil {
		t.Fatal(err)
	}
}

func ExampleNewEventTimer() {
	// Device and stream creation normally happens in benchmark setup.
	_, err := NewEventTimer(nil, nil)
	fmt.Println(err)
	// Output: cuda: event timer requires a device and streams
}

func ExampleEventTimer_Measure() {
	var timer *EventTimer
	_, err := timer.Measure(func() error { return nil })
	fmt.Println(err)
	// Output: cuda: invalid event timer measurement
}

func ExampleEventTimer_Close() {
	var timer *EventTimer
	fmt.Println(timer.Close())
	// Output: <nil>
}
