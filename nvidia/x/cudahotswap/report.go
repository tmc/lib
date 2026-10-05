package cudahotswap

import (
	"fmt"
	"sort"
	"strings"
)

// Report is the whole result of one candidate. It is what a refinement loop
// reads. Every number says how it was obtained, and Unmeasured names what is
// not here so that an absent field is visible rather than read as a zero.
type Report struct {
	// Generation and ConfigRevision identify the kernel and settings used by
	// a Service check. Non-service harness reports omit these fields.
	Generation     uint64  `json:"generation,omitempty"`
	ConfigRevision uint64  `json:"config_revision,omitempty"`
	Task           string  `json:"task"`
	TaskDef        string  `json:"task_definition"`
	Candidate      string  `json:"candidate"`
	Entry          string  `json:"entry"`
	Outcome        Outcome `json:"outcome"`
	// Reason is the underlying error text for any outcome but ok. It is the
	// compiler log for compile_error and the driver message otherwise.
	Reason string `json:"reason,omitempty"`
	// ContextPoisoned reports that the CUDA context is unusable after this
	// candidate. The driver documents the fault codes as leaving the process in
	// an inconsistent state, so a loop must re-exec the harness, not reuse it.
	ContextPoisoned bool `json:"context_poisoned"`
	// Recognized is false when the harness met an error it does not classify.
	// The outcome is then harness_error, never ok.
	Recognized bool `json:"error_recognized"`

	Device      *DeviceReport      `json:"device,omitempty"`
	Compile     *CompileReport     `json:"compile,omitempty"`
	Correctness *CorrectnessReport `json:"correctness,omitempty"`
	Timing      *TimingReport      `json:"timing,omitempty"`
	Baseline    *TimingReport      `json:"baseline_timing,omitempty"`
	Speedup     *SpeedupReport     `json:"speedup,omitempty"`
	Replay      *ReplayReport      `json:"replay,omitempty"`

	// Unmeasured names quantities this harness does not measure. A consumer
	// that wants one of them must go get it; it must not infer it from here.
	Unmeasured []string `json:"unmeasured"`
}

// DeviceReport records what the numbers were taken on. A timing without the
// device and driver it was taken on is not reproducible.
type DeviceReport struct {
	Ordinal int    `json:"ordinal"`
	Name    string `json:"name"`
	Compute string `json:"compute_capability"`
	SMCount int    `json:"multiprocessors"`
}

// CompileReport records the compilation. The digests are recorded by the
// producer, in the same call that produced the artifact: this report binds the
// PTX it names to the source it names. It binds nothing else -- no CUBIN or
// SASS digest is recorded here, and nothing downstream consumes these.
type CompileReport struct {
	SourceSHA256 string `json:"source_sha256"`
	SourceBytes  int    `json:"source_bytes"`
	PTXSHA256    string `json:"ptx_sha256,omitempty"`
	PTXBytes     int    `json:"ptx_bytes,omitempty"`
	Log          string `json:"compiler_log,omitempty"`
	Arch         string `json:"arch"`
	// WallNanos is host wall clock. Compilation is host work, so no device
	// timer applies; the name says which clock this is.
	WallNanos int64 `json:"compile_wall_nanos"`
}

// TrialReport is one correctness trial against freshly generated inputs.
type TrialReport struct {
	Seed       int64      `json:"seed"`
	Comparison Comparison `json:"comparison"`
}

// CorrectnessReport records every trial. Inputs are regenerated per trial from
// a distinct seed, so passing requires being right on several draws rather than
// on one memorized one.
type CorrectnessReport struct {
	Reference     string        `json:"reference"`
	Tolerance     Tolerance     `json:"tolerance"`
	Elements      int           `json:"elements"`
	Trials        []TrialReport `json:"trials"`
	Passed        bool          `json:"passed"`
	OutputPrefill string        `json:"output_prefill"`
}

// TimingReport is one arm of a measurement. SampleNanos holds every sample, so
// a reader can see the spread instead of trusting a single reported figure.
type TimingReport struct {
	// Source names the clock. "cuda_event_device_span" is the interval between
	// two CUDA events bracketing the submitted work on the timed stream.
	Source            string `json:"source"`
	Warmup            int    `json:"warmup_launches_discarded"`
	LaunchesPerSample int    `json:"launches_per_sample"`
	// Resolved is false when the device clock returned a zero interval. No rate
	// is derived in that case: an absent column is visible, a zero is not.
	Resolved    bool    `json:"resolved"`
	SampleNanos []int64 `json:"sample_device_nanos"`
	// MedianNanosPerLaunch is derived: the median sample divided by
	// LaunchesPerSample. The median, not the mean: on a kernel this short one
	// scheduling hiccup moves a mean further than the effect being measured.
	MedianNanosPerLaunch float64 `json:"median_device_nanos_per_launch"`
	MinNanosPerLaunch    float64 `json:"min_device_nanos_per_launch"`
	MaxNanosPerLaunch    float64 `json:"max_device_nanos_per_launch"`
	// WallNanos is the host span over the whole timed section, including
	// submission and synchronization. It is reported for contrast and is not
	// the kernel's time.
	WallNanos int64 `json:"wall_nanos"`
	// InputPolicy states what the timed launches read. It is recorded because
	// reusing one uploaded input across samples leaves it in device cache, and
	// a memory-bound kernel then reads fast for a reason that is not its own.
	InputPolicy string `json:"input_policy"`
}

// SpeedupReport compares the candidate against the task's baseline kernel.
type SpeedupReport struct {
	// Method names the interleaving. Arms are run ABBA rather than one after
	// the other, because a fixed order gives the first arm a different clock
	// and cache state from the second and the difference reads as an effect.
	Method string `json:"method"`
	// Ratio is baseline median divided by candidate median. Greater than 1
	// means the candidate is faster. Both medians are in the arms above.
	Ratio float64 `json:"ratio"`
	Note  string  `json:"note"`
}

// median returns the median of xs. It sorts a copy.
func median(xs []int64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]int64(nil), xs...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	mid := len(s) / 2
	if len(s)%2 == 1 {
		return float64(s[mid])
	}
	return float64(s[mid-1])/2 + float64(s[mid])/2
}

// Summarize fills the derived per-launch figures from the collected samples.
func (t *TimingReport) Summarize() {
	t.Resolved = false
	t.MedianNanosPerLaunch = 0
	t.MinNanosPerLaunch = 0
	t.MaxNanosPerLaunch = 0
	if len(t.SampleNanos) == 0 || t.LaunchesPerSample <= 0 {
		return
	}
	lo, hi := t.SampleNanos[0], t.SampleNanos[0]
	for _, v := range t.SampleNanos {
		if v <= 0 {
			// The device clock did not resolve one sample. Deriving a rate from
			// the rest would quietly drop it, so no rate is derived at all.
			return
		}
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	n := float64(t.LaunchesPerSample)
	t.Resolved = true
	t.MedianNanosPerLaunch = median(t.SampleNanos) / n
	t.MinNanosPerLaunch = float64(lo) / n
	t.MaxNanosPerLaunch = float64(hi) / n
}

// Text renders the report for a human. The JSON form is the one a loop should
// consume; this one exists so a failure is readable without a parser.
func (r *Report) Text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "task      %s  (%s)\n", r.Task, r.TaskDef)
	fmt.Fprintf(&b, "candidate %s\n", r.Candidate)
	fmt.Fprintf(&b, "outcome   %s\n", r.Outcome)
	if r.Reason != "" {
		fmt.Fprintf(&b, "reason    %s\n", strings.TrimRight(r.Reason, "\n"))
	}
	if r.ContextPoisoned {
		b.WriteString("WARNING   the CUDA context is poisoned; re-exec the harness before the next candidate\n")
	}
	if !r.Recognized {
		b.WriteString("WARNING   the harness did not recognize this error; the outcome above is not a verdict on the candidate\n")
	}
	if d := r.Device; d != nil {
		fmt.Fprintf(&b, "device    %d %s %s, %d SMs\n", d.Ordinal, d.Name, d.Compute, d.SMCount)
	}
	if c := r.Compile; c != nil {
		fmt.Fprintf(&b, "compile   source sha256 %s\n", shortDigest(c.SourceSHA256))
		if c.PTXSHA256 != "" {
			fmt.Fprintf(&b, "          ptx    sha256 %s (%d bytes), %.1fms wall clock\n",
				shortDigest(c.PTXSHA256), c.PTXBytes, float64(c.WallNanos)/1e6)
		}
		if log := strings.TrimSpace(c.Log); log != "" {
			fmt.Fprintf(&b, "          compiler log:\n%s\n", log)
		}
	}
	if c := r.Correctness; c != nil {
		fmt.Fprintf(&b, "oracle    %s\n", c.Reference)
		fmt.Fprintf(&b, "          tolerance abs %g rel %g over %d elements, %d trials on fresh inputs\n",
			c.Tolerance.Abs, c.Tolerance.Rel, c.Elements, len(c.Trials))
		if n := len(c.Trials); n > 0 {
			last := c.Trials[n-1].Comparison
			if w := last.Worst; w != nil {
				fmt.Fprintf(&b, "          worst element %d: got %g want %g (abs %g, rel %g)\n",
					w.Index, w.Got, w.Want, w.AbsErr, w.RelErr)
			}
		}
	}
	writeTiming := func(label string, t *TimingReport) {
		if t == nil {
			return
		}
		if !t.Resolved {
			fmt.Fprintf(&b, "%-9s device clock did not resolve the interval; no rate reported\n", label)
			return
		}
		fmt.Fprintf(&b, "%-9s %.1f ns/launch median (%s), min %.1f max %.1f over %d samples of %d launches\n",
			label, t.MedianNanosPerLaunch, t.Source, t.MinNanosPerLaunch, t.MaxNanosPerLaunch,
			len(t.SampleNanos), t.LaunchesPerSample)
	}
	writeTiming("candidate", r.Timing)
	writeTiming("baseline", r.Baseline)
	if s := r.Speedup; s != nil {
		fmt.Fprintf(&b, "speedup   %.3fx  (%s)\n", s.Ratio, s.Method)
	}
	b.WriteString("unmeasured:\n")
	for _, u := range r.Unmeasured {
		fmt.Fprintf(&b, "  - %s\n", u)
	}
	return b.String()
}

func shortDigest(s string) string {
	if len(s) > 16 {
		return s[:16]
	}
	return s
}

// Fail records a classified error and its context-poisoning status in r.
// A nil error leaves the report unchanged.
func (r *Report) Fail(err error) *Report {
	if err == nil {
		return r
	}
	outcome, recognized := Classify(err)
	if outcome == OutcomeOK {
		outcome, recognized = OutcomeHarnessError, false
	}
	r.Outcome = outcome
	r.Recognized = recognized
	r.ContextPoisoned = outcome.PoisonsContext()
	r.Reason = err.Error()
	return r
}
