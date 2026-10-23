package cudaphase

import (
	"bufio"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"
)

// Header is cudaphase.h, the device side: CUDAPHASE_MARK, cudaphase_now
// and cudaphase_smid.
//
//go:embed cudaphase.h
var Header string

// Python is cudaphase.py, which defines cudaphase_summary for a submission
// to print a Summary line from inside a judge run.
//
//go:embed cudaphase.py
var Python string

// A Phase is the time from one slot's stamp to another's.
type Phase struct {
	Name     string
	From, To int // slots
}

// A Summary describes one traced launch. Times are in microseconds.
type Summary struct {
	Name       string             `json:"name"`
	Blocks     int                `json:"blocks"` // blocks that wrote any slot
	SpanUS     float64            `json:"span_us"`
	QuantumUS  float64            `json:"quantum_us"` // smallest gap between distinct globaltimer stamps: the timer step seen
	SMs        int                `json:"sms,omitempty"`
	GHz        float64            `json:"ghz,omitempty"`         // SM clock measured by SummarizeClock
	RateBlocks int                `json:"rate_blocks,omitempty"` // blocks the clock rate was measured over
	Migrated   int                `json:"migrated,omitempty"`    // blocks whose SM changed, left out
	Phases     []PhaseStat        `json:"phases"`
	Extra      map[string]float64 `json:"extra,omitempty"`
}

// A PhaseStat is one phase over the blocks that stamped both its ends. The
// median is the lower median.
type PhaseStat struct {
	Name     string  `json:"name"`
	Blocks   int     `json:"blocks"`
	MedianUS float64 `json:"median_us"`
	MinUS    float64 `json:"min_us"`
	MaxUS    float64 `json:"max_us"`
}

// Summarize summarizes a trace buffer of slots values per block, with
// times in nanoseconds (cudaphase_now). A zero in any slot is unset.
// smSlot names the slot that holds cudaphase_smid, or is -1.
func Summarize(name string, buf []uint64, slots int, phases []Phase, smSlot int) (Summary, error) {
	return summarize(name, buf, slots, phases, smSlot, nil)
}

// A Clock says which slots of a SummarizeClock buffer anchor its
// %clock64 stamps (cudaphase_clock) to %globaltimer (cudaphase_now).
type Clock struct {
	Start, End           int // slots holding cudaphase_clock at the block's start and end
	TimerStart, TimerEnd int // slots holding cudaphase_now at the same points
	SMEnd                int // slot holding cudaphase_smid at the end, compared to smSlot's; -1 if none

	// MinSpanNS is the globaltimer span a block needs for its clock
	// rate to count; 0 means 2000. Shorter spans are a few timer steps,
	// too coarse to measure a rate.
	MinSpanNS uint64
}

// SummarizeClock summarizes a trace buffer whose phase slots hold SM
// cycles from cudaphase_clock, which resolves single instructions where
// globaltimer moves in coarse steps. The SM clock rate is the lower
// median, over blocks whose globaltimer span is at least c.MinSpanNS, of
// cycles per nanosecond between c.Start and c.End; phases are cycle
// counts converted at that rate. Each block is placed on globaltimer's
// timeline at its c.TimerStart stamp, for the launch's span: clock64 is
// per SM, so cycles from different blocks are never compared directly.
// A block whose SM at the end differs from its SM at the start (smSlot
// and c.SMEnd) has cycles from two clocks; it is left out and counted.
func SummarizeClock(name string, buf []uint64, slots int, phases []Phase, smSlot int, c Clock) (Summary, error) {
	for _, k := range []int{c.Start, c.End, c.TimerStart, c.TimerEnd} {
		if k < 0 || k >= slots {
			return Summary{}, fmt.Errorf("clock slot %d out of range [0, %d)", k, slots)
		}
	}
	if c.SMEnd >= slots || c.SMEnd >= 0 && smSlot < 0 {
		return Summary{}, fmt.Errorf("SM end slot %d needs an SM slot in range [0, %d)", c.SMEnd, slots)
	}
	if c.MinSpanNS == 0 {
		c.MinSpanNS = 2000
	}
	return summarize(name, buf, slots, phases, smSlot, &c)
}

func summarize(name string, buf []uint64, slots int, phases []Phase, smSlot int, c *Clock) (Summary, error) {
	if slots <= 0 || len(buf)%slots != 0 {
		return Summary{}, fmt.Errorf("buffer of %d values is not a whole number of %d-slot rows", len(buf), slots)
	}
	if smSlot >= slots {
		return Summary{}, fmt.Errorf("SM slot %d out of range [0, %d)", smSlot, slots)
	}
	for _, p := range phases {
		if p.From < 0 || p.From >= slots || p.To < 0 || p.To >= slots {
			return Summary{}, fmt.Errorf("phase %s: slots %d, %d out of range [0, %d)", p.Name, p.From, p.To, slots)
		}
	}
	var rows [][]uint64
	for i := 0; i < len(buf); i += slots {
		r := buf[i : i+slots]
		if slices.ContainsFunc(r, func(v uint64) bool { return v != 0 }) {
			rows = append(rows, r)
		}
	}
	s := Summary{Name: name, Blocks: len(rows), Phases: []PhaseStat{}}
	timers := map[uint64]bool{} // globaltimer stamps, for the quantum
	var placed []float64        // stamps on globaltimer's timeline, in ns
	rate := 1.0                 // ns per stamp unit
	if c != nil {
		var kept, ratios [][]uint64
		for _, r := range rows {
			if c.SMEnd >= 0 && r[smSlot] > 0 && r[c.SMEnd] > 0 && r[smSlot] != r[c.SMEnd] {
				s.Migrated++
				continue
			}
			kept = append(kept, r)
			if r[c.TimerStart] > 0 && r[c.TimerEnd] >= r[c.TimerStart]+c.MinSpanNS && r[c.Start] > 0 && r[c.End] > r[c.Start] {
				ratios = append(ratios, r)
			}
			for _, k := range []int{c.TimerStart, c.TimerEnd} {
				if r[k] > 0 {
					timers[r[k]] = true
				}
			}
		}
		if len(ratios) == 0 {
			return Summary{}, fmt.Errorf("no block spans %d ns with clock and timer stamps at both ends, to measure the clock rate", c.MinSpanNS)
		}
		rs := make([]float64, len(ratios))
		for i, r := range ratios {
			rs[i] = float64(r[c.TimerEnd]-r[c.TimerStart]) / float64(r[c.End]-r[c.Start])
		}
		slices.Sort(rs)
		rate = rs[(len(rs)-1)/2]
		s.GHz, s.RateBlocks = 1/rate, len(rs)
		rows = kept
	}
	for _, p := range phases {
		var d []float64
		for _, r := range rows {
			if r[p.From] > 0 && r[p.To] > 0 {
				d = append(d, float64(int64(r[p.To]-r[p.From]))*rate)
			}
			for _, k := range []int{p.From, p.To} {
				switch {
				case r[k] == 0:
				case c == nil:
					timers[r[k]] = true
					placed = append(placed, float64(r[k]))
				case r[c.TimerStart] > 0 && r[c.Start] > 0:
					placed = append(placed, float64(r[c.TimerStart])+float64(int64(r[k]-r[c.Start]))*rate)
				}
			}
		}
		slices.Sort(d)
		st := PhaseStat{Name: p.Name, Blocks: len(d)}
		if len(d) > 0 {
			st.MedianUS = d[(len(d)-1)/2] / 1e3
			st.MinUS, st.MaxUS = d[0]/1e3, d[len(d)-1]/1e3
		}
		s.Phases = append(s.Phases, st)
	}
	if len(placed) > 0 {
		s.SpanUS = (slices.Max(placed) - slices.Min(placed)) / 1e3
	}
	ts := make([]uint64, 0, len(timers))
	for v := range timers {
		ts = append(ts, v)
	}
	slices.Sort(ts)
	for i := 1; i < len(ts); i++ {
		if g := float64(ts[i]-ts[i-1]) / 1e3; i == 1 || g < s.QuantumUS {
			s.QuantumUS = g
		}
	}
	if smSlot >= 0 {
		sms := map[uint64]bool{}
		for _, r := range rows {
			if r[smSlot] > 0 {
				sms[r[smSlot]] = true
			}
		}
		s.SMs = len(sms)
	}
	return s, nil
}

// Prefix starts every summary line.
const Prefix = "CUDAPHASE "

// Parse returns the summaries in r: lines that start with Prefix, after
// any leading spaces, as a judge log or cudaphase_summary prints them.
func Parse(r io.Reader) ([]Summary, error) {
	var out []Summary
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimLeft(sc.Text(), " \t")
		rest, ok := strings.CutPrefix(line, Prefix)
		if !ok {
			continue
		}
		var s Summary
		if err := json.Unmarshal([]byte(rest), &s); err != nil {
			return out, fmt.Errorf("line %d: %w", n, err)
		}
		out = append(out, s)
	}
	return out, sc.Err()
}
