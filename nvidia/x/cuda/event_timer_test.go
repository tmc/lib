package cuda

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
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

// TestEventTimerRejectsBusyStreams covers Measure's entry check. The contract
// it enforces was previously documentation only, and a caller who violated it
// got a plausible duration rather than an error -- the interval silently
// included whatever was already pending. Nothing downstream can tell such a
// number from a correct one, which is why this is checked and not just stated.
//
// The assertion that matters is not only the error: it is that NOTHING was
// recorded. A check that ran after the start event would still report the
// violation while having already moved the clock.
func TestEventTimerRejectsBusyStreams(t *testing.T) {
	for _, tt := range []struct {
		name string
		busy string // which stream reports pending work
		want string
	}{
		{"control", "control", "control stream has pending work"},
		{"declared", "a", "stream 0 has pending work"},
		{"second declared", "b", "stream 1 has pending work"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			timer, log, api := timerFixture(t)
			names := map[*Stream]string{timer.control: "control", timer.streams[0]: "a", timer.streams[1]: "b"}
			api.query = func(s *Stream) (bool, error) {
				*log = append(*log, "query "+names[s])
				return names[s] != tt.busy, nil
			}
			_, err := timer.Measure(func() error { *log = append(*log, "work"); return nil })
			if err == nil {
				t.Fatalf("Measure succeeded with %s busy: a violated precondition produced a number", tt.busy)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %q, want it to name the stream (%q)", err, tt.want)
			}
			for _, e := range *log {
				if strings.HasPrefix(e, "record") || e == "work" {
					t.Fatalf("log = %v: the check ran too late, %q already happened", *log, e)
				}
			}
		})
	}
}

// TestEventTimerChecksEveryStreamBeforeStarting pins that a passing check costs
// one query per stream and still produces the unchanged measurement sequence.
func TestEventTimerChecksEveryStreamBeforeStarting(t *testing.T) {
	timer, log, api := timerFixture(t)
	api.query = func(*Stream) (bool, error) { *log = append(*log, "query"); return true, nil }
	if !timer.Checked() {
		t.Error("Checked = false with a query available")
	}
	if _, err := timer.Measure(func() error { *log = append(*log, "work"); return nil }); err != nil {
		t.Fatal(err)
	}
	want := []string{"query", "query", "query", "record 0 control", "wait a 0", "wait b 0", "work",
		"record 2 a", "wait control 2", "record 3 b", "wait control 3", "record 1 control", "sync 1"}
	if !reflect.DeepEqual(*log, want) {
		t.Fatalf("events = %v; want %v", *log, want)
	}
}

// TestEventTimerUncheckedIsNotTheSameAsPassing pins the degradation. When the
// loaded driver has no cuStreamQuery the timer cannot check, and Measure then
// returns no error whether or not the contract held. Reporting that as success
// would rebuild the original defect one level up, so Checked distinguishes them.
func TestEventTimerUncheckedIsNotTheSameAsPassing(t *testing.T) {
	timer, _, api := timerFixture(t)
	api.query = nil
	if timer.Checked() {
		t.Error("Checked = true with no query available")
	}
	if _, err := timer.Measure(func() error { return nil }); err != nil {
		t.Fatalf("Measure failed with the check unavailable: %v", err)
	}
	var nilTimer *EventTimer
	if nilTimer.Checked() {
		t.Error("Checked = true on a nil timer")
	}
}

// TestEventTimerReportsQueryFailure separates "the stream is busy" from "the
// query itself failed". Treating a broken query as idle would make the guard
// pass in exactly the case where it knows least.
func TestEventTimerReportsQueryFailure(t *testing.T) {
	timer, _, api := timerFixture(t)
	api.query = func(*Stream) (bool, error) { return false, errors.New("driver exploded") }
	_, err := timer.Measure(func() error { return nil })
	if err == nil || !strings.Contains(err.Error(), "driver exploded") {
		t.Fatalf("Measure error = %v, want the query failure surfaced", err)
	}
}
