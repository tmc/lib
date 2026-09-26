package perfscript

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// A Sample is one perf sample.
type Sample struct {
	Time   int64    // CLOCK_MONOTONIC nanoseconds
	Thread int      // thread ID
	Comm   string   // thread name
	Stack  []string // function names, leaf first
}

// Parse reads the output of perf script run with --ns and at least the
// tid and time fields, as in
//
//	perf script --ns -F comm,tid,time,ip,sym
//
// A sample is a header line, "comm tid time:" (or "pid/tid"), followed
// by one indented line per frame, "addr sym (dso)" with the dso optional,
// and ends at a blank line. Frames whose symbol perf could not resolve
// are kept as "[unknown]".
func Parse(r io.Reader) ([]Sample, error) {
	var samples []Sample
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	line := 0
	var cur *Sample
	for sc.Scan() {
		line++
		text := sc.Text()
		switch {
		case strings.TrimSpace(text) == "":
			cur = nil
		case text[0] == ' ' || text[0] == '\t':
			if cur == nil {
				return nil, fmt.Errorf("line %d: frame outside a sample", line)
			}
			cur.Stack = append(cur.Stack, frame(text))
		default:
			s, err := header(text)
			if err != nil {
				return nil, fmt.Errorf("line %d: %w", line, err)
			}
			samples = append(samples, s)
			cur = &samples[len(samples)-1]
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return samples, nil
}

// header parses "comm tid time:" and ignores anything after the time,
// such as an event name.
func header(text string) (Sample, error) {
	f := strings.Fields(text)
	t := -1
	for i, w := range f {
		if strings.HasSuffix(w, ":") && strings.Contains(w, ".") {
			if _, err := parseTime(strings.TrimSuffix(w, ":")); err == nil {
				t = i
				break
			}
		}
	}
	if t < 1 {
		return Sample{}, fmt.Errorf("no tid and time in sample header %q (run perf script with --ns -F tid,time)", text)
	}
	var s Sample
	s.Time, _ = parseTime(strings.TrimSuffix(f[t], ":"))
	tid := f[t-1]
	if _, after, ok := strings.Cut(tid, "/"); ok {
		tid = after
	}
	n, err := strconv.Atoi(tid)
	if err != nil {
		return Sample{}, fmt.Errorf("bad tid %q in sample header %q", f[t-1], text)
	}
	s.Thread = n
	s.Comm = strings.Join(f[:t-1], " ")
	return s, nil
}

// parseTime parses seconds with a fraction, "64745.949787732", as
// nanoseconds without rounding.
func parseTime(s string) (int64, error) {
	sec, frac, _ := strings.Cut(s, ".")
	if len(frac) > 9 {
		return 0, fmt.Errorf("bad time %q", s)
	}
	n, err := strconv.ParseInt(sec, 10, 64)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("bad time %q", s)
	}
	var f int64
	if frac != "" {
		f, err = strconv.ParseInt(frac+strings.Repeat("0", 9-len(frac)), 10, 64)
		if err != nil || f < 0 {
			return 0, fmt.Errorf("bad time %q", s)
		}
	}
	return n*1e9 + f, nil
}

// frame returns the symbol of an "addr sym (dso)" line.
func frame(text string) string {
	text = strings.TrimSpace(text)
	if _, rest, ok := strings.Cut(text, " "); ok {
		text = strings.TrimSpace(rest)
	}
	if strings.HasSuffix(text, ")") {
		if i := strings.LastIndex(text, " ("); i >= 0 {
			text = text[:i]
		}
	}
	if i := strings.LastIndex(text, "+0x"); i > 0 {
		text = text[:i]
	}
	return text
}
