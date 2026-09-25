package main

import (
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
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
	Rows    int           // sample rows in the profile; zero means nothing was observed
	// Counters names the profile's other sample types: range counters,
	// neither time nor samples.
	Counters []string

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
//
// The unit is not the whole of it. A PC sample says where execution was, never
// how fast, and the share a kernel collects is a share of resident warp-cycles,
// not of wall-clock. A kernel occupying one block of a 48-SM device holds the
// machine for its whole duration while contributing the warp-cycles of a single
// block, so it collects samples in proportion to how much of the device it
// wastes, inverted. That bias runs in the worst direction available: smallest
// for the kernels that fill the device, largest for the ones that starve it,
// which are exactly the ones a profile is usually opened to find. Measured on
// GB10 (48 SMs, 32 decode steps under gpu-lock with no other compute app on the
// device, CUPTI ActivityKindConcurrentKernel) at 76d0df0b: the decode step holds
// 5.26ms/token in kernels launched with a grid narrower than 48, 7.5% of summed
// device span across 914 launches a token, and a sample ranking of that same
// workload put every one of them in the tail. The vivid case was historical --
// ds4_indexer_select at 552us/token on grid 1, one SM of 48 for the whole of it,
// as it stood at 9fb244e4 and before it was rewritten -- but the aggregate is
// the load-bearing number and does not depend on any one kernel surviving. The
// failure is unfalsifiable from inside the profile: no sample count would reveal
// it, because the undersampling is the finding.
//
// None of this is an argument against PC sampling, which answers a different
// question well: which source lines a kernel's warps sit on, and on what stall
// reason. The refusal is specific to using sample share as a time ranking
// across kernels of differing occupancy.
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
	c.Rows = len(p.Sample)
	for i, vt := range p.SampleType {
		switch {
		case deviceTimeType(vt.Type):
			for _, s := range p.Sample {
				if i < len(s.Value) && s.Value[i] > 0 {
					c.Busy += time.Duration(s.Value[i])
				}
			}
		case vt.Type == "pc_samples", vt.Type == "samples":
			for _, s := range p.Sample {
				if i < len(s.Value) && s.Value[i] > 0 {
					c.Samples += s.Value[i]
				}
			}
		default:
			c.Counters = append(c.Counters, vt.Type)
		}
	}
	// A span of zero is a fact about the pprof projection, which is an
	// aggregate, and not about the capture: the per-record journal carries
	// every activity's start and end. Saying the capture records no span
	// while holding the file that does is how a reader gets told to re-run
	// work already on disk.
	if c.Span <= 0 {
		if events, _, err := timeline.JournalEvents(cap); err == nil {
			var first, last uint64
			var have bool
			for _, ev := range events {
				if ev.End == 0 || ev.End <= ev.Start {
					continue
				}
				if !have || ev.Start < first {
					first, have = ev.Start, true
				}
				if ev.End > last {
					last = ev.End
				}
			}
			if have && last > first {
				c.Span = time.Duration(last - first)
			}
		}
	}
	switch {
	case c.Rows == 0:
		// Ordered before the device-time case, which would otherwise diagnose
		// an empty capture as a sample-only one and prescribe its remedy.
		// Those are opposite captures: one observed the wrong quantity, the
		// other observed nothing, and sending someone to add activity tracing
		// to a capture that collected zero records costs a run under GPU lock
		// and produces the same empty file.
		c.Reason = "the profile holds no samples at all, so this capture observed nothing: " +
			"an empty ranking drawn from it means the capture was empty, not that the workload ran no such kernels"
	case c.Busy <= 0 && c.Samples > 0:
		c.Reason = "the profile carries no device time, only sample counts, and a sample count " +
			"has no duration: it says where execution was, never for how long"
	case c.Busy <= 0:
		// A counter capture (cuptibench's export) holds per-range hardware
		// counters. They are not samples and they are not time: each totals
		// what one range did, so there is no device-time total for a ranking
		// to be a share of.
		c.Reason = "the profile holds range counters (" + counterList(c.Counters) + "), not device time or PC samples: " +
			"a counter totals what one range did, so there is no device-time total to rank by"
	case c.Span <= 0:
		c.Reason = "neither the profile nor a per-record journal gives a span for the " + c.Busy.Round(time.Microsecond).String() +
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
	if !ranking {
		return nil
	}
	// An empty profile is refused ahead of the override, and the override does
	// not reach it. The flag exists so a reader who knows what a sample share
	// is can read one anyway; there is no share of zero samples to read, so
	// letting it through here would print the clean empty table this refusal
	// exists to prevent -- a table that reads as "those kernels are not in
	// this workload" when the capture in fact observed nothing.
	if c.Rows == 0 {
		return fmt.Errorf("%s has nothing to rank: %s.\n"+
			"\tNo flag makes this file rankable. Read the capture headline for why it is empty --\n"+
			"\ta target that exited before the collector finalized is the usual cause -- and re-capture.",
			flag, c.Reason)
	}
	if allow || c.Rankable() {
		return nil
	}
	if c.Samples > 0 {
		return fmt.Errorf("%s would rank this profile by a total that has no duration.\n"+
			"\t%s\n"+
			"\tCapture activity tracing alongside PC sampling so the profile carries device time,\n"+
			"\tor pass %s to read the ranking as a share of samples and nothing more.",
			flag, c.Reason, allowUnknownCoverageFlag)
	}
	return fmt.Errorf("%s cannot rank this profile: it has no device-time total.\n"+
		"\t%s\n"+
		"\tCapture activity tracing alongside the counters so the profile carries device time,\n"+
		"\tor pass %s to rank by the counter's value and nothing more.",
		flag, c.Reason, allowUnknownCoverageFlag)
}

// counterList names up to three counters and how many more there are.
func counterList(names []string) string {
	const max = 3
	if len(names) == 0 {
		return "no named type"
	}
	if len(names) <= max {
		return strings.Join(names, ", ")
	}
	return fmt.Sprintf("%s and %d more", strings.Join(names[:max], ", "), len(names)-max)
}

// deviceTimeType reports whether a sample type carries device time. It is the
// only quantity in a cudaprof profile measured in the same unit as the span,
// so it is the only one captureCoverage can close against it, and the only one
// the coverage statement above describes.
func deviceTimeType(name string) bool {
	return name == "kernel_time" || name == "gpu_time"
}

// sampleIndexRequested returns the -sample_index value in args. pprof parses
// the flag with the flag package, so both spellings and both forms -- joined by
// = or followed by a separate word -- have to be recognized here or the
// refusal below silently does not apply to half the ways of writing it.
func sampleIndexRequested(args []string) (string, bool) {
	for i, a := range args {
		name, value, hasValue := strings.Cut(a, "=")
		switch name {
		case "-sample_index", "--sample_index":
			if hasValue {
				return value, true
			}
			if i+1 < len(args) {
				return args[i+1], true
			}
			return "", true
		}
	}
	return "", false
}

// resolveSampleIndex reports which sample type value names, accepting exactly
// what pprof's profile.SampleIndexByName accepts: a numeric index in range or
// a type name, including the legacy inuse_ prefix. The legacy rule is mirrored
// even though no cudaprof profile emits such a type, because the cost of the
// two forms disagreeing is a refusal here of a value pprof would have taken --
// a preflight that rejects working input is worse than the confusion it fixes.
func resolveSampleIndex(p *profile.Profile, value string) (int, bool) {
	if i, err := strconv.Atoi(value); err == nil {
		if i < 0 || i >= len(p.SampleType) {
			return 0, false
		}
		return i, true
	}
	noInuse := strings.TrimPrefix(value, "inuse_")
	for i, vt := range p.SampleType {
		if vt.Type == value || vt.Type == noInuse {
			return i, true
		}
	}
	return 0, false
}

// sampleTypeMenu names the quantities a profile emits, each with its unit and
// the default marked.
func sampleTypeMenu(p *profile.Profile) string {
	parts := make([]string, 0, len(p.SampleType))
	for _, vt := range p.SampleType {
		s := fmt.Sprintf("%s (%s)", vt.Type, vt.Unit)
		if vt.Type == p.DefaultSampleType {
			s += " [default]"
		}
		parts = append(parts, s)
	}
	return strings.Join(parts, ", ")
}

// preflightSampleIndex refuses a -sample_index this capture does not measure,
// and says what it does measure in the unit each quantity is measured in.
//
// pprof already rejects an unknown value and already lists the type names:
// `sample_index "activity_ns" must be one of: [kernel_time copy_time
// copy_bytes activity]`. The list was never the missing part. The unit is.
// activity_ns is an operator asking for a count of activity in nanoseconds,
// and a list of four bare names cannot tell them that kernel_time is the
// nanoseconds and activity is the count -- it reads as four alternative
// spellings of one quantity rather than as three quantities in three units.
//
// The second half is the same denominator discipline the coverage statement
// enforces. That statement is about device time, and a reader who selects
// copy_bytes or activity gets it printed directly above a table it does not
// describe: the header says what fraction of the span the device was busy for,
// and the rows below rank bytes moved or launches counted. Naming the mismatch
// costs a line and removes the only reading of the pair that is wrong.
func preflightSampleIndex(w io.Writer, cap *cudacapture.Capture, args []string) error {
	value, ok := sampleIndexRequested(args)
	if !ok || value == "" {
		return nil
	}
	if cap == nil || cap.Profile == nil || len(cap.Profile.SampleType) == 0 {
		return nil
	}
	p := cap.Profile
	idx, ok := resolveSampleIndex(p, value)
	if !ok {
		return fmt.Errorf("-sample_index=%s names a quantity this capture does not measure.\n"+
			"\tIt measures: %s.\n"+
			"\tThe unit decides what the ranking means: nanoseconds ranks device time, bytes ranks traffic, count ranks launches.",
			value, sampleTypeMenu(p))
	}
	if vt := p.SampleType[idx]; !deviceTimeType(vt.Type) {
		fmt.Fprintf(w, "cudaprof: -sample_index=%s ranks %s in %s, not device time: the coverage statement above is about device time and is not the denominator of the table below\n",
			value, vt.Type, vt.Unit)
	}
	return nil
}
