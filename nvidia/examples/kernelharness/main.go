package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
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
	os.Exit(rep.Outcome.ExitCode())
}

// runOnDevice opens the device and runs one candidate. A device that cannot be
// opened yields outcome no_device, which is neither a pass nor a failed
// candidate: exit status 3 keeps it distinguishable from both.
func runOnDevice(ordinal int, task *Task, name, source string, cfg Config) *cudahotswap.Report {
	base := &cudahotswap.Report{
		Task: task.Name, TaskDef: task.Summary, Candidate: name, Entry: task.Entry,
		Unmeasured: unmeasured(),
	}
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		rep := base.Fail(err)
		if rep.Outcome == cudahotswap.OutcomeHarnessError {
			// A driver that will not load at all is not a driver code the
			// classifier can read. It is still an absent device from the
			// caller's point of view, and recognizing it here is the only
			// place that knows the call was cuda.NewDevice.
			rep.Outcome, rep.Recognized = cudahotswap.OutcomeNoDevice, true
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
		rep.Outcome, rep.Recognized = cudahotswap.OutcomeHarnessError, false
		rep.Reason = err.Error()
	}
	return rep
}
