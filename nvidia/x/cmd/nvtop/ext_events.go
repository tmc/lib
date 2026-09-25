package main

import (
	"fmt"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
)

// An event is something NVML reported about a device as it happened,
// such as an Xid error.
type event struct {
	at   time.Time
	dev  int    // device index
	kind uint64 // an nvml.EventType* bit
	data uint64 // for an Xid error, the Xid
	msg  string // for a lost host, why
}

// An eventSource is a source that also reports events.
type eventSource interface {
	// events returns the events since the last call.
	events() []event
}

// eventKinds names the NVML event types, and says which are faults.
var eventKinds = []struct {
	kind  uint64
	name  string
	fault bool
}{
	{nvml.EventTypeSingleBitEccError, "single-bit ECC error", false},
	{nvml.EventTypeDoubleBitEccError, "double-bit ECC error", true},
	{nvml.EventTypePState, "P-state change", false},
	{nvml.EventTypeXidCriticalError, "Xid", true},
	{nvml.EventTypeClock, "clock change", false},
	{nvml.EventTypePowerSourceChange, "power source change", false},
	{nvml.EventMigConfigChange, "MIG configuration change", false},
	{nvml.EventTypeSingleBitEccErrorStorm, "single-bit ECC error storm", true},
	{nvml.EventTypeDramRetirementEvent, "DRAM page retirement", false},
	{nvml.EventTypeDramRetirementFailure, "DRAM page retirement failure", true},
	{nvml.EventTypeNonFatalPoisonError, "non-fatal poison error", false},
	{nvml.EventTypeFatalPoisonError, "fatal poison error", true},
	{nvml.EventTypeGpuUnavailableError, "GPU unavailable", true},
	{nvml.EventTypeGpuRecoveryAction, "GPU recovery action", true},
	{eventHostLost, "connection to host lost", true},
}

// watchedEvents are the event types nvtop listens for: all but clock
// changes, which the header already shows and which come in floods.
const watchedEvents = nvml.EventTypeAll &^ nvml.EventTypeClock

// xids describes the commonest Xid errors, from NVIDIA's Xid catalog.
var xids = map[uint64]string{
	13:  "graphics engine exception",
	31:  "GPU memory page fault",
	32:  "invalid or corrupted push buffer stream",
	38:  "driver firmware error",
	43:  "GPU stopped processing",
	45:  "preemptive cleanup after previous errors",
	48:  "double-bit ECC error",
	61:  "internal micro-controller breakpoint",
	62:  "internal micro-controller halt",
	63:  "ECC page retirement or row remapping",
	64:  "ECC page retirement or row remapping failure",
	68:  "video decoder exception",
	69:  "graphics engine class error",
	74:  "NVLink error",
	79:  "GPU has fallen off the bus",
	92:  "high single-bit ECC error rate",
	94:  "contained ECC error",
	95:  "uncontained ECC error",
	109: "context switch timeout",
	119: "GSP RPC timeout",
	120: "GSP error",
}

// fault reports whether e is a hardware or driver fault.
func (e event) fault() bool {
	for _, k := range eventKinds {
		if k.kind == e.kind {
			return k.fault
		}
	}
	return false
}

// String describes e, without its time or device.
func (e event) String() string {
	if e.kind == nvml.EventTypeXidCriticalError {
		if d, ok := xids[e.data]; ok {
			return fmt.Sprintf("Xid %d: %s", e.data, d)
		}
		return fmt.Sprintf("Xid %d", e.data)
	}
	for _, k := range eventKinds {
		if k.kind == e.kind {
			if e.msg != "" {
				return k.name + ": " + e.msg
			}
			return k.name
		}
	}
	return fmt.Sprintf("event %#x", e.kind)
}

// An eventWatcher waits for NVML events on a goroutine of its own.
type eventWatcher struct {
	set     nvml.EventSet
	handles []nvml.Device
	stop    chan struct{}
	done    chan struct{}

	mu      sync.Mutex
	pending []event
}

// watchEvents starts waiting for the events of the given devices. It
// returns nil if none of the devices reports any.
func watchEvents(handles []nvml.Device) *eventWatcher {
	set, err := nvml.NewEventSet()
	if err != nil {
		return nil
	}
	registered := false
	for _, hd := range handles {
		types, err := hd.SupportedEventTypes()
		if err != nil || types&watchedEvents == 0 {
			continue
		}
		if hd.RegisterEvents(types&watchedEvents, set) == nil {
			registered = true
		}
	}
	if !registered {
		set.Free()
		return nil
	}
	w := &eventWatcher{set: set, handles: handles, stop: make(chan struct{}), done: make(chan struct{})}
	go w.run()
	return w
}

// waitTimeout is how long one wait for an event lasts, and so how long
// closing the watcher can take.
const waitTimeout = 200 // ms

func (w *eventWatcher) run() {
	defer close(w.done)
	for {
		select {
		case <-w.stop:
			return
		default:
		}
		var data nvml.EventData
		if err := w.set.Wait(&data, waitTimeout); err != nil {
			continue
		}
		e := event{at: time.Now(), dev: -1, kind: data.EventType, data: data.EventData}
		for i, hd := range w.handles {
			if hd == data.Device {
				e.dev = i
			}
		}
		w.mu.Lock()
		w.pending = append(w.pending, e)
		w.mu.Unlock()
	}
}

// events returns the events since the last call.
func (w *eventWatcher) events() []event {
	w.mu.Lock()
	defer w.mu.Unlock()
	e := w.pending
	w.pending = nil
	return e
}

// close stops the watcher and frees its event set. It must be called
// before NVML is shut down.
func (w *eventWatcher) close() {
	close(w.stop)
	<-w.done
	w.set.Free()
}

// maxEvents bounds the event log.
const maxEvents = 1000

// addEvents adds evs to the log, dropping the oldest past maxEvents,
// and counts them against their devices.
func (u *ui) addEvents(evs []event) {
	for _, e := range evs {
		if e.dev >= 0 && e.dev < len(u.devs) {
			u.eventCount[e.dev]++
			u.faulted[e.dev] = u.faulted[e.dev] || e.fault()
		}
	}
	u.events = append(u.events, evs...)
	if n := len(u.events) - maxEvents; n > 0 {
		u.events = append(u.events[:0], u.events[n:]...)
	}
}

// drawEvents draws the event log in place of the process list, newest
// first.
func (u *ui) drawEvents(s *screen, r rect) {
	w := s.window(r.x, r.y, r.w, r.h)
	w.print(0, 0, fmt.Sprintf("%-8s %3s %s", "TIME", "DEV", "EVENT"), plain)
	w.chgat(0, 0, -1, standout(green))
	avail := w.h - 1
	u.eventOffset = max(0, min(u.eventOffset, len(u.events)-avail))
	if len(u.events) == 0 {
		w.print(0, 1, "No events since nvtop started.", plain)
		return
	}
	for y := 1; y <= avail; y++ {
		i := len(u.events) - u.eventOffset - y
		if i < 0 {
			break
		}
		e := u.events[i]
		dev := "?"
		if e.dev >= 0 {
			dev = fmt.Sprint(e.dev)
		}
		x := w.printf(0, y, plain, "%-8s %3s ", e.at.Format(time.TimeOnly), dev)
		st := plain
		if e.fault() {
			st = fg(red)
		}
		w.print(x, y, e.String(), st)
	}
}
