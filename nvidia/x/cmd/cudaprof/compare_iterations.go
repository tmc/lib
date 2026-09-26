package main

import (
	"errors"
	"flag"
	"fmt"
	"regexp"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// iterationWindow narrows a compare window to whole iterations: from the
// launch of the marker kernel that begins iteration skip to the one that
// begins iteration skip+count.
type iterationWindow struct {
	marker *regexp.Regexp
	skip   int
	// count is the number of iterations, or 0 for every iteration after
	// skip but the last, which has no end.
	count int
}

// iterationFlags are the compare flags that set an iterationWindow.
type iterationFlags struct {
	marker      string
	skip, count int
}

func (f *iterationFlags) register(fs *flag.FlagSet) {
	fs.StringVar(&f.marker, "iteration-marker", "", "compare whole iterations, each beginning at a kernel matching `RE`, demangled with template arguments")
	fs.IntVar(&f.skip, "skip-iterations", 1, "with -iteration-marker, exclude the first `N` iterations as warm-up")
	fs.IntVar(&f.count, "iteration-count", 0, "with -iteration-marker, compare the first `N` steady iterations; required when the sides' counts differ")
}

// window returns the iterationWindow the flags describe, or nil when
// -iteration-marker is unset.
func (f *iterationFlags) window(set map[string]bool) (*iterationWindow, error) {
	if f.marker == "" {
		if set["skip-iterations"] || set["iteration-count"] {
			return nil, errors.New("-skip-iterations and -iteration-count need -iteration-marker")
		}
		return nil, nil
	}
	if set["warmup-ms"] || set["warmup-until"] {
		return nil, errors.New("-iteration-marker and -warmup-ms or -warmup-until are exclusive")
	}
	if f.skip < 0 {
		return nil, fmt.Errorf("invalid -skip-iterations %d", f.skip)
	}
	if f.count < 0 {
		return nil, fmt.Errorf("invalid -iteration-count %d", f.count)
	}
	re, err := regexp.Compile(f.marker)
	if err != nil {
		return nil, fmt.Errorf("-iteration-marker: %w", err)
	}
	return &iterationWindow{marker: re, skip: f.skip, count: f.count}, nil
}

// cutIterations narrows the window to the iterations w selects among the
// marker launches inside it.
func (s *CompareSide) cutIterations(records timeline.Records, w iterationWindow) error {
	var marks []int64
	match := make(map[string]bool)
	for _, a := range records.Activities {
		if a.Subtype != "kernel" || a.Start < s.StartTimestamp || a.End > s.StopTimestamp {
			continue
		}
		m, ok := match[a.Name]
		if !ok {
			m = w.marker.MatchString(iteration.KernelName(a.Name)) || w.marker.MatchString(a.Name)
			match[a.Name] = m
		}
		if m {
			marks = append(marks, a.Start)
		}
	}
	sort.Slice(marks, func(i, j int) bool { return marks[i] < marks[j] })
	have := len(marks) - 1 - w.skip
	if have < 1 {
		return fmt.Errorf("-iteration-marker %s fired %d times in the window: no iteration left after skipping %d", w.marker, len(marks), w.skip)
	}
	n := have
	if w.count > 0 {
		if w.count > have {
			return fmt.Errorf("-iteration-count %d: the window holds %d steady iterations", w.count, have)
		}
		n = w.count
	}
	start, stop := marks[w.skip], marks[w.skip+n]
	s.Iterations = n
	s.Warmup = fmt.Sprintf("%d iterations by -iteration-marker %s; window is the next %d of %d", w.skip, w.marker, n, have)
	s.WarmupNanos = start - s.StartTimestamp
	s.StartTimestamp, s.StopTimestamp = start, stop
	return nil
}

// checkIterations refuses sides that hold different numbers of
// iterations, and notes the per-iteration deltas when they agree.
func (c *Comparison) checkIterations(explicit bool) error {
	b, t := c.Base, c.Target
	if b.Iterations != t.Iterations {
		if explicit {
			return fmt.Errorf("base holds %d iterations, target %d", b.Iterations, t.Iterations)
		}
		return fmt.Errorf("base holds %d steady iterations, target %d; pass -iteration-count to compare the same number", b.Iterations, t.Iterations)
	}
	n := int64(b.Iterations)
	per := func(name string) string {
		d := c.term(name)
		return fmt.Sprintf("%s %.3f -> %.3f ms (%+.3f)", name, float64(d.Base)/float64(n)/1e6, float64(d.Target)/float64(n)/1e6, float64(d.Delta)/float64(n)/1e6)
	}
	c.Notes = append(c.Notes, fmt.Sprintf("window is %d iterations per side; per iteration: %s, %s, %s", n, per("window"), per("busy"), per("idle")))
	return nil
}
