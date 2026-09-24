package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"
)

// A violations log records the processes a watcher saw loading the host
// while they took no part in the campaign, and when the watcher was looking.
// Each line is a violation or a heartbeat:
//
//	time pid comm cpu gpu holder
//	time heartbeat holder
//
// with fields separated by spaces or tabs. In both, time is when the watcher
// sampled, in RFC 3339 with optional fractional seconds, and holder is the
// label of whoever held the host lock at the time, or "-" when nobody did. In
// a violation, pid is a positive integer; comm is the process name, which
// must not contain whitespace; cpu is its CPU use in percent of one core, so
// above 100 on several; and gpu is the device memory it held in MiB. A
// heartbeat has exactly three fields, the second the literal word
// "heartbeat". For example:
//
//	2026-09-24T03:12:07.5Z	heartbeat	ds4-qualify
//	2026-09-24T03:12:07.5Z	41822	python3	327.0	0	ds4-qualify
//
// The watcher writes a heartbeat every sampling interval, whether or not it
// saw a violation, because an absent violation is evidence only while the
// watcher was running: a log with violations and no heartbeats checks
// nothing. Blank lines and lines starting with # are ignored. Any other line
// that does not parse is an error naming its line number: a log the parser
// cannot read is not a clean one.
type hostLog struct {
	Violations []hostViolation
	Heartbeats []time.Time // in increasing order
}

// hostViolation is one violation line of a violations log.
type hostViolation struct {
	Time   time.Time `json:"time"`
	PID    int       `json:"pid"`
	Comm   string    `json:"comm"`
	CPUPct float64   `json:"cpu_pct"`
	GPUMiB float64   `json:"gpu_mib"`
	Holder string    `json:"lock_holder"`
}

func (v hostViolation) String() string {
	return fmt.Sprintf("%s pid %d %s %.1f%% CPU %.0f MiB GPU, lock holder %s", v.Time.Format(time.RFC3339Nano), v.PID, v.Comm, v.CPUPct, v.GPUMiB, v.Holder)
}

// parseHostLog reads a violations log.
func parseHostLog(r io.Reader) (hostLog, error) {
	var log hostLog
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		f := strings.Fields(line)
		heartbeat := len(f) == 3 && f[1] == "heartbeat"
		if !heartbeat && len(f) != 6 {
			return hostLog{}, fmt.Errorf("line %d: want \"time pid comm cpu gpu holder\" or \"time heartbeat holder\", have %d fields", n, len(f))
		}
		t, err := time.Parse(time.RFC3339Nano, f[0])
		if err != nil {
			return hostLog{}, fmt.Errorf("line %d: time %q is not RFC 3339", n, f[0])
		}
		if heartbeat {
			log.Heartbeats = append(log.Heartbeats, t)
			continue
		}
		pid, err := strconv.Atoi(f[1])
		if err != nil || pid <= 0 {
			return hostLog{}, fmt.Errorf("line %d: invalid pid %q", n, f[1])
		}
		cpu, ok := parseAmount(f[3])
		if !ok {
			return hostLog{}, fmt.Errorf("line %d: invalid cpu percent %q", n, f[3])
		}
		gpu, ok := parseAmount(f[4])
		if !ok {
			return hostLog{}, fmt.Errorf("line %d: invalid gpu MiB %q", n, f[4])
		}
		log.Violations = append(log.Violations, hostViolation{Time: t, PID: pid, Comm: f[2], CPUPct: cpu, GPUMiB: gpu, Holder: f[5]})
	}
	if err := sc.Err(); err != nil {
		return hostLog{}, err
	}
	slices.SortFunc(log.Heartbeats, time.Time.Compare)
	return log, nil
}

// uncovered returns why the heartbeats do not cover [lo, hi], or "" when
// they do: a heartbeat at or before lo, one at or after hi, and no two
// consecutive heartbeats between them further apart than maxGap.
func (log hostLog) uncovered(lo, hi time.Time, maxGap time.Duration) string {
	hb := log.Heartbeats
	// first is the last heartbeat at or before lo; last the first at or
	// after hi.
	first, _ := slices.BinarySearchFunc(hb, lo, func(t, lo time.Time) int {
		if t.After(lo) {
			return 1
		}
		return -1
	})
	first--
	last, _ := slices.BinarySearchFunc(hb, hi, time.Time.Compare)
	switch {
	case first < 0:
		return fmt.Sprintf("watcher did not cover the window start: no heartbeat at or before %s", lo.Format(time.RFC3339Nano))
	case last == len(hb):
		return fmt.Sprintf("watcher did not cover the window end: no heartbeat at or after %s", hi.Format(time.RFC3339Nano))
	}
	for i := first; i < last; i++ {
		if gap := hb[i+1].Sub(hb[i]); gap > maxGap {
			return fmt.Sprintf("watcher did not cover %s to %s: %s between heartbeats, more than the %s margin", hb[i].Format(time.RFC3339Nano), hb[i+1].Format(time.RFC3339Nano), gap, maxGap)
		}
	}
	return ""
}

// heartbeatInterval returns the median time between consecutive heartbeats,
// or 0 with fewer than two.
func (log hostLog) heartbeatInterval() time.Duration {
	if len(log.Heartbeats) < 2 {
		return 0
	}
	gaps := make([]float64, len(log.Heartbeats)-1)
	for i := range gaps {
		gaps[i] = float64(log.Heartbeats[i+1].Sub(log.Heartbeats[i]))
	}
	return time.Duration(median(gaps))
}

// parseAmount parses a finite, nonnegative number.
func parseAmount(s string) (float64, bool) {
	x, err := strconv.ParseFloat(s, 64)
	if err != nil || !(x >= 0) || math.IsInf(x, 0) {
		return 0, false
	}
	return x, true
}

// A qualifyWindow is the wall-clock interval an arm ran its capture range.
type qualifyWindow struct {
	Start time.Time `json:"start"`
	Stop  time.Time `json:"stop"`
}

// armWindow returns the wall-clock window of the capture whose status is s,
// from its first capture-range edge to its last, placed in wall time through
// the capture's wall clock. When the status cannot place it, armWindow
// returns the reason instead.
func armWindow(s *captureResult) (*qualifyWindow, string) {
	r := s.PhaseReceipt
	switch {
	case s.WallClock == nil:
		return nil, "capture records no wall clock"
	case r == nil || len(r.Edges) == 0:
		return nil, "capture records no capture-range edges"
	case r.OmittedEdges > 0:
		return nil, fmt.Sprintf("capture omitted %d capture-range edges, so its last is unknown", r.OmittedEdges)
	}
	first, last := r.Edges[0], r.Edges[0]
	for _, e := range r.Edges[1:] {
		if e.Timestamp < first.Timestamp {
			first = e
		}
		if e.Timestamp > last.Timestamp {
			last = e
		}
	}
	switch {
	case first.Timestamp == 0:
		return nil, "a capture-range edge has no timestamp"
	case last.Edge != "stop" && last.Edge != "redundant_stop":
		return nil, fmt.Sprintf("the last capture-range edge is %q, so the range never closed", last.Edge)
	}
	return &qualifyWindow{Start: s.WallClock.at(first.Timestamp), Stop: s.WallClock.at(last.Timestamp)}, ""
}

// Host load dispositions of an arm.
const (
	hostClean     = "clean"
	hostLoaded    = "loaded"
	hostUnchecked = "unchecked"
)

// qualifyHostLoad summarizes the host load check. Checked is false when no
// violations log was given. HeartbeatIntervalMS is the median time between
// the log's heartbeats, the watcher's sampling interval as the log reports
// it.
type qualifyHostLoad struct {
	Checked             bool    `json:"checked"`
	Log                 string  `json:"log,omitempty"`
	MarginMS            float64 `json:"margin_ms,omitempty"`
	Heartbeats          int     `json:"heartbeats"`
	HeartbeatIntervalMS float64 `json:"heartbeat_interval_ms"`
	Violations          int     `json:"violations"`
	InWindows           int     `json:"violations_in_windows"`
	Clean               int     `json:"clean_arms"`
	Loaded              int     `json:"loaded_arms"`
	Unchecked           int     `json:"unchecked_arms"`
}

// markHostLoad sets the host load of each entry from log. A violation
// counts against an arm when it lies in the arm's window widened by margin
// on each side, edges included: the watcher sees a process only when it
// samples, so a load that spans a window is recorded within one sampling
// interval of it, and margin should be at least that interval. An arm is
// checked only when the log's heartbeats cover its widened window, with no
// gap longer than margin; otherwise it is unchecked.
func markHostLoad(entries []qualifyEntry, log hostLog, margin time.Duration) qualifyHostLoad {
	h := qualifyHostLoad{
		Checked:             true,
		MarginMS:            float64(margin) / 1e6,
		Heartbeats:          len(log.Heartbeats),
		HeartbeatIntervalMS: float64(log.heartbeatInterval()) / 1e6,
		Violations:          len(log.Violations),
	}
	counted := make([]bool, len(log.Violations))
	for i := range entries {
		e := &entries[i]
		e.Violations = nil
		if e.Window == nil {
			e.HostLoad = hostUnchecked
			h.Unchecked++
			continue
		}
		lo, hi := e.Window.Start.Add(-margin), e.Window.Stop.Add(margin)
		if why := log.uncovered(lo, hi, margin); why != "" {
			e.HostLoad, e.Unchecked = hostUnchecked, why
			h.Unchecked++
			continue
		}
		for j, v := range log.Violations {
			if v.Time.Before(lo) || v.Time.After(hi) {
				continue
			}
			e.Violations = append(e.Violations, v)
			if !counted[j] {
				counted[j] = true
				h.InWindows++
			}
		}
		if len(e.Violations) > 0 {
			e.HostLoad = hostLoaded
			h.Loaded++
		} else {
			e.HostLoad = hostClean
			h.Clean++
		}
	}
	return h
}

// writeHostLoad writes the host load check and names every excluded arm.
func (q *Qualification) writeHostLoad(w io.Writer) {
	h := q.HostLoad
	if !h.Checked {
		fmt.Fprintf(w, "host load: not checked: no -violations log was given\n")
		return
	}
	fmt.Fprintf(w, "host load: %d violations in %s, %d within %g ms of an arm; %d heartbeats, median interval %g ms: %d arms clean, %d loaded, %d unchecked\n",
		h.Violations, h.Log, h.InWindows, h.MarginMS, h.Heartbeats, h.HeartbeatIntervalMS, h.Clean, h.Loaded, h.Unchecked)
	for _, e := range q.Captures {
		switch e.HostLoad {
		case hostLoaded:
			for _, v := range e.Violations {
				fmt.Fprintf(w, "  %s %s excluded: %s\n", e.Block, e.Arm, v)
			}
		case hostUnchecked:
			fmt.Fprintf(w, "  %s %s excluded, cannot check: %s\n", e.Block, e.Arm, e.Unchecked)
		}
	}
}
