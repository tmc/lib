package main

import (
	"testing"
	"time"
)

func TestExtCountersUpdate(t *testing.T) {
	sec := uint64(time.Second)
	held := func(pw, therm uint64) [numReasons]opt[uint64] {
		var h [numReasons]opt[uint64]
		h[2], h[6] = some(pw), some(therm) // sw-power, hw-therm
		return h
	}
	var c extCounters
	steps := []struct {
		held [numReasons]opt[uint64]
		at   time.Duration
		ok   bool
		pw   uint32
		th   uint32
	}{
		{held(10*sec, 0), 0, false, 0, 0},                 // first read: nothing to compare
		{held(10*sec+sec/2, 0), time.Second, true, 50, 0}, // half of the second
		{held(10*sec+sec/2, sec/10), 2 * time.Second, true, 0, 10},
		{held(12*sec+sec/2, sec/10), 3 * time.Second, true, 100, 0}, // capped
		{held(0, 0), 4 * time.Second, false, 0, 0},                  // counters went back
		{[numReasons]opt[uint64]{}, 5 * time.Second, false, 0, 0},   // not read
	}
	for i, s := range steps {
		got := c.update(s.held, testNow.Add(s.at))
		if got.ok != s.ok || got.v[2] != s.pw || got.v[6] != s.th {
			t.Errorf("step %d: update = %v, %v; want ok %v, sw-power %d%%, hw-therm %d%%", i, got.ok, got.v, s.ok, s.pw, s.th)
		}
	}
}

func TestExtCountersEnergy(t *testing.T) {
	var c extCounters
	for _, tt := range []struct{ total, want uint64 }{
		{5000, 0},
		{7500, 2500},
		{100, 0}, // the driver was reloaded
		{600, 500},
	} {
		if got := c.energy(tt.total); !got.ok || got.v != tt.want {
			t.Errorf("energy(%d) = %v, want %d", tt.total, got, tt.want)
		}
	}
}

func TestEnergyString(t *testing.T) {
	for _, tt := range []struct {
		mj   uint64
		want string
	}{
		{0, "0.000 J"},
		{1500, "1.500 J"},
		{45990058, "45.99 kJ"},
		{999_000_000, "999.0 kJ"},
		{3_600_000_000, "3.600 MJ"},
	} {
		if got := energyString(tt.mj); got != tt.want {
			t.Errorf("energyString(%d) = %q, want %q", tt.mj, got, tt.want)
		}
	}
}

// extDevices returns testDevices with the -x fields set: the first
// device held down by its power cap and by heat, the second idle.
func extDevices() []*device {
	devs := testDevices()
	var pct [numReasons]uint32
	pct[2], pct[6] = 37, 2
	devs[0].reasons = some[uint64](0x4 | 0x40)
	devs[0].throttle = some(pct)
	devs[0].energy = some[uint64](45990058)
	devs[1].reasons = some[uint64](0x1)
	return devs
}

func TestRenderExt(t *testing.T) {
	for _, tt := range []struct {
		name string
		opts options
	}{
		{"ext", options{ext: true}},
		{"ext-info", options{ext: true, infoBar: true}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tt.opts.encodeHide = 30 * time.Second
			u := newUI(extDevices(), tt.opts, 1, func() time.Time { return testNow })
			u.addEvents(testEvents())
			s := newScreen(120, 40, true)
			u.render(s)
			golden(t, "render/"+tt.name+".golden", []byte(s.String()))
			checkColors(t, s, []colorCheck{
				{"NRG 45.99 kJ", 0, cyan},
				{"NRG 45.99 kJ", 4, noColor},
				{"THR sw-power 37% hw-therm 2%", 0, cyan},
				{"THR sw-power 37% hw-therm 2%", 4, yellow},
				{"THR sw-power 37% hw-therm 2%", 17, red},
				{"EVT 2", 0, cyan},
				{"EVT 2", 4, red},
				{"EVT 1", 4, noColor},
				{"NRG N/A", 0, cyan},
				{"THR idle", 4, noColor},
			})
		})
	}
}
