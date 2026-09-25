package main

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
)

func TestOptJSON(t *testing.T) {
	for _, tt := range []struct {
		in   any
		want string
	}{
		{opt[uint32]{}, `null`},
		{some[uint32](7), `7`},
		{some(""), `""`},
		{some([2]uint32{1, 2}), `[1,2]`},
		{struct {
			A opt[int] `json:"a,omitzero"`
			B opt[int] `json:"b,omitzero"`
		}{B: some(0)}, `{"b":0}`},
	} {
		b, err := json.Marshal(tt.in)
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != tt.want {
			t.Errorf("Marshal(%v) = %s, want %s", tt.in, b, tt.want)
		}
	}
	var o opt[uint32]
	if err := json.Unmarshal([]byte(`12`), &o); err != nil || o != some[uint32](12) {
		t.Errorf("Unmarshal(12) = %v, %v", o, err)
	}
	if err := json.Unmarshal([]byte(`null`), &o); err != nil || o.ok {
		t.Errorf("Unmarshal(null) = %v, %v", o, err)
	}
	if err := json.Unmarshal([]byte(`"x"`), &o); err == nil {
		t.Errorf("Unmarshal(\"x\") into opt[uint32] succeeded")
	}
}

// TestWireFields checks that a wireDevice has a field for every field of
// a device but its index, counting the nvlink struct as its two fields,
// so that a field added to one is not forgotten in the other.
func TestWireFields(t *testing.T) {
	d := reflect.TypeFor[device]().NumField() // less index, plus nvlink's second field
	w := reflect.TypeFor[wireDevice]().NumField()
	if d != w {
		t.Errorf("device has %d recordable fields, wireDevice %d", d, w)
	}
	p := reflect.TypeFor[process]().NumField()
	wp := reflect.TypeFor[wireProcess]().NumField()
	if p != wp {
		t.Errorf("process has %d fields, wireProcess %d", p, wp)
	}
}

func TestWireRoundTrip(t *testing.T) {
	devs := append(extDevices(), datacenterDevices()...)
	devs[0].nvlinkErrs = some(nvlinkErrors{flit: 1, crc: 2, ecc: 3})
	devs[0].mig = some(false)
	for i, d := range devs {
		d.index = i
		w := toWire(d)
		b, err := json.Marshal(&w)
		if err != nil {
			t.Fatal(err)
		}
		var back wireDevice
		if err := json.Unmarshal(b, &back); err != nil {
			t.Fatal(err)
		}
		got := fromWire(&back, i)
		if !reflect.DeepEqual(&got, d) {
			t.Errorf("device %d: round trip\n got %+v\nwant %+v\njson %s", i, got, *d, b)
		}
	}
}

// testRecording records three frames of extDevices a second apart, with
// an event in the second.
func testRecording(t *testing.T) []byte {
	var b bytes.Buffer
	rec := newRecorder(&b)
	devs := extDevices()
	for i := range 3 {
		devs[0].memUtil = some(uint32(10 * i))
		var evs []event
		if i == 1 {
			evs = []event{{at: testNow, dev: 0, kind: nvml.EventTypeXidCriticalError, data: 79}}
		}
		if err := rec.write(testNow.Add(time.Duration(i)*time.Second), devs, evs); err != nil {
			t.Fatal(err)
		}
	}
	return b.Bytes()
}

func TestReplay(t *testing.T) {
	data := testRecording(t)
	if n := bytes.Count(data, []byte("\n")); n != 3 {
		t.Fatalf("recording has %d lines, want 3:\n%s", n, data)
	}
	s, err := newReplay(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	devs := s.devices()
	if len(devs) != 2 || devs[1].name != some("NVIDIA GB10") || devs[1].index != 1 {
		t.Fatalf("devices = %+v", devs)
	}
	var utils []uint32
	var events []event
	for i := 0; ; i++ {
		if at := s.at(); !at.Equal(testNow.Add(time.Duration(i) * time.Second)) {
			t.Errorf("frame %d at %v", i, at)
		}
		if g, want := s.gap(), time.Second; i < 2 && g != want || i == 2 && g != 0 {
			t.Errorf("frame %d gap = %v", i, g)
		}
		gather(s, nil, devs, true)
		utils = append(utils, devs[0].memUtil.v)
		events = append(events, s.events()...)
		events = append(events, s.events()...) // taken once
		if !s.step() {
			break
		}
	}
	if !reflect.DeepEqual(utils, []uint32{0, 10, 20}) {
		t.Errorf("memory utilization by frame = %v", utils)
	}
	if len(events) != 1 || events[0].String() != "Xid 79: GPU has fallen off the bus" || !events[0].at.Equal(testNow) {
		t.Errorf("events = %v", events)
	}
	if len(devs[0].procs) != 3 || devs[0].energy != some[uint64](45990058) {
		t.Errorf("last frame device 0 = %+v", devs[0])
	}
	if err := s.close(); err != nil {
		t.Error(err)
	}
}

func TestReplayErrors(t *testing.T) {
	for _, tt := range []struct {
		name, data, err string
	}{
		{"empty", "", "no frames"},
		{"blank", "\n\n", "no frames"},
		{"garbage", "{\n", "unexpected end of JSON"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newReplay(strings.NewReader(tt.data))
			if err == nil || !strings.Contains(err.Error(), tt.err) {
				t.Errorf("newReplay error = %v, want %q", err, tt.err)
			}
		})
	}

	// A bad frame after the first ends the replay, and close reports it.
	data := append(testRecording(t), "nonsense\n"...)
	s, err := newReplay(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	for s.step() {
	}
	if err := s.close(); err == nil {
		t.Error("close after a bad frame succeeded")
	}
}

func TestReplayKeys(t *testing.T) {
	s, err := newReplay(bytes.NewReader(testRecording(t)))
	if err != nil {
		t.Fatal(err)
	}
	u := newUI(s.devices(), options{ext: true, encodeHide: 30 * time.Second}, 0, s.at)
	u.player = s
	if got := u.interval(); got != time.Second {
		t.Errorf("interval = %v, want the gap between frames", got)
	}
	u.handle(' ')
	if !u.paused || u.kick {
		t.Errorf("after Space: paused %v, kick %v", u.paused, u.kick)
	}
	u.handle('n')
	if !u.kick {
		t.Errorf("n while paused did not ask for a frame")
	}
	u.kick = false
	u.handle(keyF9)
	if u.overlay != overlayNone {
		t.Errorf("F9 opened the kill overlay while replaying")
	}
	u.handle(' ')
	if u.paused || !u.kick {
		t.Errorf("after second Space: paused %v, kick %v", u.paused, u.kick)
	}
	u.opts.delay = 300 * time.Millisecond
	if got := u.interval(); got != u.opts.delay {
		t.Errorf("interval with -d = %v", got)
	}
}
