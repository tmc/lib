package main

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
)

// testEvents returns an Xid and a P-state change on device 0 and a
// power source change on device 1.
func testEvents() []event {
	return []event{
		{at: testNow.Add(-time.Minute), dev: 0, kind: nvml.EventTypePState},
		{at: testNow.Add(-30 * time.Second), dev: 1, kind: nvml.EventTypePowerSourceChange},
		{at: testNow, dev: 0, kind: nvml.EventTypeXidCriticalError, data: 79},
	}
}

func TestEventString(t *testing.T) {
	for _, tt := range []struct {
		e     event
		want  string
		fault bool
	}{
		{event{kind: nvml.EventTypeXidCriticalError, data: 79}, "Xid 79: GPU has fallen off the bus", true},
		{event{kind: nvml.EventTypeXidCriticalError, data: 1000}, "Xid 1000", true},
		{event{kind: nvml.EventTypePState}, "P-state change", false},
		{event{kind: nvml.EventMigConfigChange}, "MIG configuration change", false},
		{event{kind: nvml.EventTypeDoubleBitEccError}, "double-bit ECC error", true},
		{event{kind: 1 << 40}, "event 0x10000000000", false},
	} {
		if got := tt.e.String(); got != tt.want {
			t.Errorf("String() = %q, want %q", got, tt.want)
		}
		if got := tt.e.fault(); got != tt.fault {
			t.Errorf("%q: fault() = %v, want %v", tt.want, got, tt.fault)
		}
	}
}

func TestWatchedEvents(t *testing.T) {
	if watchedEvents&nvml.EventTypeClock != 0 {
		t.Errorf("watching clock changes")
	}
	if watchedEvents&nvml.EventTypeXidCriticalError == 0 {
		t.Errorf("not watching Xid errors")
	}
}

func TestAddEvents(t *testing.T) {
	u := testUI(options{ext: true})
	u.addEvents(testEvents())
	if !slices.Equal(u.eventCount, []int{2, 1}) || !slices.Equal(u.faulted, []bool{true, false}) {
		t.Errorf("counts %v, faults %v; want [2 1], [true false]", u.eventCount, u.faulted)
	}
	// An event from a device nvtop does not know is logged, not counted.
	u.addEvents([]event{{dev: -1, kind: nvml.EventTypePState}})
	if len(u.events) != 4 || !slices.Equal(u.eventCount, []int{2, 1}) {
		t.Errorf("unknown device: %d events, counts %v", len(u.events), u.eventCount)
	}
	many := make([]event, maxEvents)
	for i := range many {
		many[i] = event{dev: 1, kind: nvml.EventTypePState, data: uint64(i)}
	}
	u.addEvents(many)
	if len(u.events) != maxEvents || u.events[0].data != 0 || u.events[maxEvents-1].data != maxEvents-1 {
		t.Errorf("log of %d events, first %d, last %d; want the newest %d", len(u.events), u.events[0].data, u.events[len(u.events)-1].data, maxEvents)
	}
}

func TestEventsView(t *testing.T) {
	u := testUI(options{ext: true})
	u.addEvents(testEvents())
	s := newScreen(120, 40, true)
	u.render(s)
	u.handle('e')
	if u.view != viewEvents {
		t.Fatalf("e: view %d, want the events", u.view)
	}
	// Kill and sort are for the process list.
	u.handle(keyF9)
	u.handle(keyF6)
	if u.overlay != overlayNone {
		t.Errorf("F9, F6 in the events view opened overlay %d", u.overlay)
	}
	u.render(s)
	golden(t, "render/events.golden", []byte(s.String()))
	checkColors(t, s, []colorCheck{
		{"Xid 79", 0, red},
		{"P-state change", 0, noColor},
	})
	out := s.String()
	if strings.Index(out, "Xid 79") > strings.Index(out, "P-state") {
		t.Errorf("events not newest first:\n%s", out)
	}
	u.handle(keyEsc)
	if u.view != viewProcesses || u.quit {
		t.Errorf("Esc: view %d, quit %v; want the process list", u.view, u.quit)
	}

	// Without -x, e does nothing.
	u = testUI(options{})
	u.handle('e')
	if u.view != viewProcesses {
		t.Errorf("e without -x: view %d", u.view)
	}
}
