package main

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// allowUnknownCoverageFlag lets a caller render a ranking whose denominator the
// profile cannot state. It exists so the refusal below is recoverable by
// someone who knows what they are reading, not so it can be passed by habit.
const allowUnknownCoverageFlag = "-allow_unknown_coverage"

// coverage states what fraction of a capture's measured span the profile
// accounts for, and names the denominator of every share rendered from it.
//
// The distinction the type exists to keep is that a PC-sampling share is a
// share of samples, never of elapsed time. A profiler that could observe only
// 40% of the span prints an identically confident 98.34%, and a five-item
// optimization list built on those shares topped out at 3.0x against a measured
// 8.05x gap: the missing 2.7x was the part of the wall clock the instrument
// cannot see, and nothing in the output said such a part existed.
type coverage struct {
	Span    time.Duration // wall-clock span the capture reports
	Busy    time.Duration // device time the profile measures, zero when it measures none
	Samples int64         // total PC samples: the denominator of every share

	// Reason is empty exactly when the span's covered fraction is determined.
	// A non-empty Reason means the fraction is unknown, not that it is small.
	Reason string
}

// Determined reports whether the covered fraction of the span is known.
func (c coverage) Determined() bool { return c.Reason == "" }

// Rankable reports whether the profile can state the denominator of the shares
// a ranking would print. Device time can: it is measured in the same unit as
// the quantities it is a share of, so "75.95% of 868.79us" names both. A count
// of samples cannot, and a ranking whose denominator is unknown is not a
// ranking. Coverage of the wall-clock span is a further question, and its being
// unknown weakens a ranking without invalidating it.
func (c coverage) Rankable() bool { return c.Busy > 0 }

// Fraction is the share of the measured span the profile observed. It is
// meaningful only when Determined reports true.
func (c coverage) Fraction() float64 {
	if c.Span <= 0 {
		return 0
	}
	return float64(c.Busy) / float64(c.Span)
}

// String renders the statement that must accompany any share drawn from this
// profile, in one line, in the units each quantity was measured in.
func (c coverage) String() string {
	var b strings.Builder
	if c.Determined() {
		fmt.Fprintf(&b, "coverage: the profile measures %s of device time across a %s span (%.1f%%)",
			c.Busy.Round(time.Millisecond), c.Span.Round(time.Millisecond), 100*c.Fraction())
	} else {
		fmt.Fprintf(&b, "coverage: unknown -- %s", c.Reason)
	}
	// The sample count is named as the denominator only where it is one. A
	// profile carrying device time is ranked by that, and calling its shares
	// shares of samples would mislabel them in the act of labelling them.
	if c.Samples > 0 && !c.Rankable() {
		fmt.Fprintf(&b, "; every share below is of %d observed samples, not of elapsed time", c.Samples)
	}
	return b.String()
}

// captureCoverage measures what the profile in cap accounts for.
//
// Device time is the only term here that can close against the span: it is
// measured in the same unit the span is measured in. Sample counts cannot,
// because a sample says where execution was and never for how long, so a
// sample-only profile leaves the fraction undetermined rather than assuming it
// is 1.
func captureCoverage(cap *cudacapture.Capture) coverage {
	var c coverage
	if cap == nil || cap.Profile == nil {
		c.Reason = "the capture carries no profile"
		return c
	}
	p := cap.Profile
	c.Span = time.Duration(p.DurationNanos)
	for i, vt := range p.SampleType {
		switch vt.Type {
		case "kernel_time", "gpu_time":
			for _, s := range p.Sample {
				if i < len(s.Value) && s.Value[i] > 0 {
					c.Busy += time.Duration(s.Value[i])
				}
			}
		case "pc_samples", "samples":
			for _, s := range p.Sample {
				if i < len(s.Value) && s.Value[i] > 0 {
					c.Samples += s.Value[i]
				}
			}
		}
	}
	switch {
	case c.Busy <= 0:
		c.Reason = "the profile carries no device time, only sample counts, and a sample count " +
			"has no duration: it says where execution was, never for how long"
	case c.Span <= 0:
		c.Reason = "the capture records no measured span for the " + c.Busy.Round(time.Microsecond).String() +
			" of device time to be a fraction of"
	}
	return c
}

// rankingRequested reports the first argument asking for a ranking of the
// profile. Interactive and graphical views are not rankings and are not gated:
// a reader who opens the web UI is looking at the profile, not reading a
// ordered list off it.
func rankingRequested(args []string) (string, bool) {
	for _, a := range args {
		name, _, _ := strings.Cut(a, "=")
		switch name {
		case "-top", "--top", "-text", "--text", "-tree", "--tree":
			return name, true
		}
	}
	return "", false
}

// stripCoverageFlag removes allowUnknownCoverageFlag from args and reports
// whether it was present. pprof's own driver parses everything else, and an
// unknown flag there is a usage error.
func stripCoverageFlag(args []string) ([]string, bool) {
	out := args[:0:0]
	found := false
	for _, a := range args {
		if a == allowUnknownCoverageFlag || strings.HasPrefix(a, allowUnknownCoverageFlag+"=") {
			found = true
			continue
		}
		out = append(out, a)
	}
	return out, found
}

// preflightCoverage prints the capture's coverage statement and refuses a
// ranking whose denominator the profile cannot state. A ranking whose
// denominator is unknown is not a ranking; the errata entry saying so has
// existed for longer than the misranking it failed to prevent, which is the
// argument for enforcing it here rather than recording it in a document.
func preflightCoverage(w io.Writer, cap *cudacapture.Capture, args []string, allow bool) error {
	c := captureCoverage(cap)
	fmt.Fprintf(w, "cudaprof: %s\n", c)
	if c.Determined() {
		return nil
	}
	flag, ranking := rankingRequested(args)
	if !ranking || allow || c.Rankable() {
		return nil
	}
	return fmt.Errorf("%s would rank this profile by a total that has no duration.\n"+
		"\t%s\n"+
		"\tCapture activity tracing alongside PC sampling so the profile carries device time,\n"+
		"\tor pass %s to read the ranking as a share of samples and nothing more.",
		flag, c.Reason, allowUnknownCoverageFlag)
}
