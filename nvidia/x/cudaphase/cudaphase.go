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
	Name      string             `json:"name"`
	Blocks    int                `json:"blocks"` // blocks that wrote any slot
	SpanUS    float64            `json:"span_us"`
	QuantumUS float64            `json:"quantum_us"` // smallest gap between distinct stamps: the timer step seen
	SMs       int                `json:"sms,omitempty"`
	Phases    []PhaseStat        `json:"phases"`
	Extra     map[string]float64 `json:"extra,omitempty"`
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
// times in nanoseconds. A zero in any slot is unset. smSlot names the slot
// that holds cudaphase_smid, or is -1.
func Summarize(name string, buf []uint64, slots int, phases []Phase, smSlot int) (Summary, error) {
	if slots <= 0 || len(buf)%slots != 0 {
		return Summary{}, fmt.Errorf("buffer of %d values is not a whole number of %d-slot rows", len(buf), slots)
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
	stamps := map[uint64]bool{}
	for _, p := range phases {
		var d []uint64
		for _, r := range rows {
			if r[p.From] > 0 && r[p.To] > 0 {
				d = append(d, r[p.To]-r[p.From])
			}
			for _, v := range []uint64{r[p.From], r[p.To]} {
				if v > 0 {
					stamps[v] = true
				}
			}
		}
		slices.Sort(d)
		st := PhaseStat{Name: p.Name, Blocks: len(d)}
		if len(d) > 0 {
			st.MedianUS = us(d[(len(d)-1)/2])
			st.MinUS, st.MaxUS = us(d[0]), us(d[len(d)-1])
		}
		s.Phases = append(s.Phases, st)
	}
	ss := make([]uint64, 0, len(stamps))
	for v := range stamps {
		ss = append(ss, v)
	}
	slices.Sort(ss)
	if len(ss) > 0 {
		s.SpanUS = us(ss[len(ss)-1] - ss[0])
	}
	for i := 1; i < len(ss); i++ {
		if g := us(ss[i] - ss[i-1]); i == 1 || g < s.QuantumUS {
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

func us(ns uint64) float64 { return float64(ns) / 1e3 }

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
