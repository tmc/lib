package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func main() {
	taskFlag := flag.String("task", "axpby", "task name; -list shows the built-in tasks")
	kernelFlag := flag.String("kernel", "", "path to the candidate CUDA source file")
	jsonFlag := flag.Bool("json", false, "print the report as JSON")
	listFlag := flag.Bool("list", false, "list the built-in tasks and the required kernel signature, then exit")
	ordinalFlag := flag.Int("device", 0, "CUDA device ordinal")
	trialsFlag := flag.Int("trials", 0, "correctness trials, each on freshly generated inputs (0 uses the default)")
	samplesFlag := flag.Int("samples", 0, "event-timed samples per arm (0 uses the default)")
	baselineFlag := flag.Bool("baseline", true, "time the task's baseline kernel as an interleaved second arm")
	timeoutFlag := flag.Duration("timeout", 0, "deadline for one correctness launch (0 uses the default)")
	flag.Parse()

	if *listFlag {
		fmt.Printf("kernel signature required of every candidate:\n\n  %s\n\n", KernelABI)
		fmt.Println("tasks:")
		for _, name := range TaskNames() {
			t := tasks[name]
			fmt.Printf("  %-12s entry %-12s %s\n", t.Name, t.Entry, t.Summary)
		}
		return
	}

	if *kernelFlag == "" {
		fmt.Fprintln(os.Stderr, "kernelharness: -kernel is required; -list shows the tasks")
		os.Exit(4)
	}
	task, err := LookupTask(*taskFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "kernelharness:", err)
		os.Exit(4)
	}
	source, err := os.ReadFile(*kernelFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "kernelharness:", err)
		os.Exit(4)
	}

	cfg := DefaultConfig()
	if *trialsFlag > 0 {
		cfg.Trials = *trialsFlag
	}
	if *samplesFlag > 0 {
		cfg.Samples = *samplesFlag
	}
	if *timeoutFlag > 0 {
		cfg.Timeout = *timeoutFlag
	}
	cfg.Baseline = *baselineFlag

	rep := runOnDevice(*ordinalFlag, task, *kernelFlag, string(source), cfg)
	if *jsonFlag {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(os.Stderr, "kernelharness:", err)
			os.Exit(4)
		}
	} else {
		fmt.Print(rep.Text())
	}
	os.Exit(rep.Outcome.exitCode())
}

// runOnDevice opens the device and runs one candidate. A device that cannot be
// opened yields outcome no_device, which is neither a pass nor a failed
// candidate: exit status 3 keeps it distinguishable from both.
func runOnDevice(ordinal int, task *Task, name, source string, cfg Config) *Report {
	base := &Report{
		Task: task.Name, TaskDef: task.Summary, Candidate: name, Entry: task.Entry,
		Unmeasured: unmeasured(),
	}
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		rep := base.fail(err)
		if rep.Outcome == OutcomeHarnessError {
			// A driver that will not load at all is not a driver code the
			// classifier can read. It is still an absent device from the
			// caller's point of view, and recognizing it here is the only
			// place that knows the call was cuda.NewDevice.
			rep.Outcome, rep.Recognized = OutcomeNoDevice, true
			rep.Reason = "no usable CUDA device: " + err.Error()
		}
		return rep
	}
	defer func() {
		if cerr := dev.Close(); cerr != nil {
			fmt.Fprintln(os.Stderr, "kernelharness: closing device:", cerr)
		}
	}()

	rep, err := Run(dev, task, name, source, cfg)
	if err != nil {
		if rep == nil {
			rep = base
		}
		rep.Outcome, rep.Recognized = OutcomeHarnessError, false
		rep.Reason = err.Error()
	}
	return rep
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
		fmt.Fprintf(&b, "compile   source sha256 %s\n", c.SourceSHA256[:16])
		if c.PTXSHA256 != "" {
			fmt.Fprintf(&b, "          ptx    sha256 %s (%d bytes), %.1fms wall clock\n",
				c.PTXSHA256[:16], c.PTXBytes, float64(c.WallNanos)/1e6)
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
