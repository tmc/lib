package main

import (
	"fmt"
	"strconv"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
)

// A reason is a clock event reason: why the clocks are below their
// maximum.
type reason struct {
	bit   uint64
	name  string
	field uint32 // the field counting the time it held the clocks, or 0
	color color  // how serious it is
}

var reasons = [...]reason{
	{nvml.ClocksEventReasonGpuIdle, "idle", 0, noColor},
	{nvml.ClocksEventReasonApplicationsClocksSetting, "app-clocks", 0, noColor},
	{nvml.ClocksEventReasonSwPowerCap, "sw-power", nvml.FieldDevClocksEventReasonSwPowerCap, yellow},
	{nvml.ClocksThrottleReasonHwSlowdown, "hw-slow", 0, red},
	{nvml.ClocksEventReasonSyncBoost, "sync-boost", nvml.FieldDevClocksEventReasonSyncBoost, noColor},
	{nvml.ClocksEventReasonSwThermalSlowdown, "sw-therm", nvml.FieldDevClocksEventReasonSwThermSlowdown, yellow},
	{nvml.ClocksThrottleReasonHwThermalSlowdown, "hw-therm", nvml.FieldDevClocksEventReasonHwThermSlowdown, red},
	{nvml.ClocksThrottleReasonHwPowerBrakeSlowdown, "hw-brake", nvml.FieldDevClocksEventReasonHwPowerBrakeSlowdown, red},
	{nvml.ClocksEventReasonDisplayClockSetting, "display", 0, noColor},
	// Bits package nvml does not define.
	{0x200, "board-limit", 0, yellow},
	{0x400, "reliability", 0, yellow},
}

const numReasons = len(reasons)

// extCounters is what the NVML source keeps about a device between
// refreshes for -x.
type extCounters struct {
	held   [numReasons]uint64 // ns each reason has held the clocks, as last read
	heldOK [numReasons]bool
	at     time.Time

	energyBase uint64 // mJ at the first read
	energyRead bool
}

// read reads the -x fields of d.
func (c *extCounters) read(hd nvml.Device, d *device, now time.Time) {
	d.reasons = get(hd.CurrentClocksEventReasons())
	if e, err := hd.TotalEnergyConsumption(); err == nil {
		d.energy = c.energy(e)
	}
	var v []nvml.FieldValue
	for _, r := range reasons {
		if r.field != 0 {
			v = append(v, nvml.FieldValue{FieldId: r.field})
		}
	}
	var held [numReasons]opt[uint64]
	if err := hd.FieldValues(int32(len(v)), &v[0]); err == nil {
		j := 0
		for i, r := range reasons {
			if r.field == 0 {
				continue
			}
			if v[j].NvmlReturn == nvml.Success {
				held[i] = some(v[j].Value.Uint64())
			}
			j++
		}
	}
	d.throttle = c.update(held, now)
}

// energy returns the energy used since the first call, given the
// device's total in mJ.
func (c *extCounters) energy(total uint64) opt[uint64] {
	if !c.energyRead {
		c.energyBase, c.energyRead = total, true
	}
	if total < c.energyBase {
		// The driver was reloaded.
		c.energyBase = total
	}
	return some(total - c.energyBase)
}

// update takes the time each reason has held the clocks, as read at
// now, and returns the percentage of the time since the previous read
// that each held them. Reasons without a counter show 0.
func (c *extCounters) update(held [numReasons]opt[uint64], now time.Time) opt[[numReasons]uint32] {
	var pct [numReasons]uint32
	dt := now.Sub(c.at)
	counted := false
	for i, h := range held {
		if h.ok && c.heldOK[i] && !c.at.IsZero() && dt > 0 && h.v >= c.held[i] {
			pct[i] = uint32(min(100, 100*float64(h.v-c.held[i])/float64(dt.Nanoseconds())))
			counted = true
		}
		c.held[i], c.heldOK[i] = h.v, h.ok
	}
	c.at = now
	if !counted {
		return opt[[numReasons]uint32]{}
	}
	return some(pct)
}

// energyString formats an energy in mJ as joules at a readable scale.
func energyString(mj uint64) string {
	f := float64(mj) / 1000
	prefix := [...]string{"", "k", "M", "G", "T"}
	i := 0
	for ; i < len(prefix)-1 && f >= 1000; i++ {
		f /= 1000
	}
	prec := 3
	switch {
	case f >= 100:
		prec = 1
	case f >= 10:
		prec = 2
	}
	return strconv.FormatFloat(f, 'f', prec, 64) + " " + prefix[i] + "J"
}

// Widths of the fields on the -x line.
const (
	energyWidth = 13
	eventsWidth = 9
)

// drawExtLine draws a device's -x line in w: the energy used since
// nvtop started, then the reasons the clocks are held down, each with
// the share of the last interval it held them, if counted.
func (u *ui) drawExtLine(w *window, d *device) {
	x := w.print(0, 0, "NRG ", fg(cyan))
	if d.energy.ok {
		w.print(x, 0, energyString(d.energy.v), plain)
	} else {
		w.print(x, 0, "N/A", plain)
	}
	x = energyWidth + 1
	x = w.print(x, 0, "EVT ", fg(cyan))
	st := plain
	if u.faulted[d.index] {
		st = fg(red)
	}
	w.print(x, 0, fmt.Sprint(u.eventCount[d.index]), st)
	x = w.print(energyWidth+1+eventsWidth+1, 0, "THR", fg(cyan))
	if !d.reasons.ok && !d.throttle.ok {
		w.print(x, 0, " N/A", plain)
		return
	}
	shown := false
	for i, r := range reasons {
		active := d.reasons.ok && d.reasons.v&r.bit != 0
		pct := d.throttle.v[i]
		if !active && pct == 0 {
			continue
		}
		text := " " + r.name
		if r.field != 0 && d.throttle.ok {
			text += fmt.Sprintf(" %d%%", pct)
		}
		x = w.print(x, 0, text, fg(r.color))
		shown = true
	}
	if !shown {
		w.print(x, 0, " none", plain)
	}
}
