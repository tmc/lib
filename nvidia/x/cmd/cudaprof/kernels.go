package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

const kernelsUsage = "usage: cudaprof kernels [-grep RE] [-phase i] [-warmup-until=RE | -warmup-ms=N] [-templates] [-max-records=N] [-journal-bytes=N] [--capture-file-bytes=N] [--capture-total-bytes=N] [-json] capture.pprof"

// KernelStat is one kernel's launches inside a capture's window.
type KernelStat struct {
	Kernel            string              `json:"kernel"`
	Launches          int                 `json:"launches"`
	SummedNanos       int64               `json:"summed_ns"`
	ExclusiveNanos    int64               `json:"exclusive_ns"`
	MeanNanos         int64               `json:"mean_ns"`
	MedianNanos       int64               `json:"median_ns"`
	P90Nanos          int64               `json:"p90_ns"`
	Configs           []LaunchConfigCount `json:"configs,omitempty"`
	UnrecordedConfigs int                 `json:"unrecorded_configs,omitempty"`
}

// KernelReport is the output of "cudaprof kernels".
type KernelReport struct {
	Capture string       `json:"capture"`
	Side    *CompareSide `json:"window"`
	Grep    string       `json:"grep,omitempty"`
	Kernels []KernelStat `json:"kernels"`
}

// runKernelsCommand implements "cudaprof kernels".
func runKernelsCommand(rawArgs []string, stdout io.Writer) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return fmt.Errorf("kernels: %w", err)
	}
	opts := compareOptions{limits: limits, allowUnresolved: true}
	var grep string
	var warmup gapOptions
	fs := flag.NewFlagSet("kernels", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVar(&grep, "grep", "", "list only kernels whose match key matches `RE`")
	fs.IntVar(&opts.phase, "phase", -1, "index of the capture-range phase, required when the capture has several")
	fs.BoolVar(&opts.json, "json", false, "write the report as JSON")
	fs.BoolVar(&opts.templates, "templates", false, "keep template arguments in kernel names")
	fs.Int64Var(&opts.maxRecords, "max-records", 0, fmt.Sprintf("maximum journal records to decode (default: the larger of %d and what the capture records)", controllerMaxJournalRecords))
	fs.Int64Var(&opts.journalBytes, "journal-bytes", 0, fmt.Sprintf("maximum decompressed bytes per journal (default: the larger of %d and what the capture records)", cudacapture.MaxCaptureInputBytes))
	fs.Float64Var(&warmup.warmupMS, "warmup-ms", 0, "start the window `N` ms after its first device activity, as gaps does")
	fs.StringVar(&warmup.warmupUntil, "warmup-until", "", "start the window at its first kernel matching `RE`, as gaps does")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprintln(stdout, kernelsUsage)
			fs.SetOutput(stdout)
			fs.PrintDefaults()
			return nil
		}
		return fmt.Errorf("kernels: %w\n%s", err, kernelsUsage)
	}
	if fs.NArg() != 1 {
		return errors.New(kernelsUsage)
	}
	re, err := regexp.Compile(grep)
	if err != nil {
		return fmt.Errorf("kernels: -grep: %w", err)
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	switch {
	case set["warmup-ms"] && set["warmup-until"]:
		return errors.New("kernels: -warmup-ms and -warmup-until are exclusive")
	case set["warmup-ms"] && warmup.warmupMS < 0:
		return fmt.Errorf("kernels: invalid -warmup-ms %g", warmup.warmupMS)
	case set["warmup-until"] && warmup.warmupUntil == "":
		return errors.New("kernels: empty -warmup-until pattern")
	case set["warmup-until"]:
		if _, err := regexp.Compile(warmup.warmupUntil); err != nil {
			return fmt.Errorf("kernels: -warmup-until: %w", err)
		}
	}
	if set["warmup-ms"] || set["warmup-until"] {
		opts.warmup = &warmup
	}
	if set["max-records"] && opts.maxRecords <= 0 {
		return fmt.Errorf("kernels: invalid -max-records %d", opts.maxRecords)
	}
	if set["journal-bytes"] && opts.journalBytes <= 0 {
		return fmt.Errorf("kernels: invalid -journal-bytes %d", opts.journalBytes)
	}
	path, err := filepath.Abs(fs.Arg(0))
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, opts.limits)
	if err != nil {
		return fmt.Errorf("kernels: read capture %s: %w", path, err)
	}
	r, err := kernelReport(cap, re, opts)
	if err != nil {
		return fmt.Errorf("kernels: %w", err)
	}
	r.Capture, r.Grep = path, grep
	if opts.json {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(r)
	}
	return r.write(stdout)
}

// kernelReport measures cap's window, as compare measures a side, and
// returns the kernels whose match key re matches, most summed time first.
func kernelReport(cap *cudacapture.Capture, re *regexp.Regexp, opts compareOptions) (*KernelReport, error) {
	side, err := measureCompareSide(cap, nil, nil, opts)
	if err != nil {
		return nil, err
	}
	r := &KernelReport{Side: side, Kernels: []KernelStat{}}
	for key, t := range side.ops {
		if !re.MatchString(key) {
			continue
		}
		l := side.launches[key]
		slices.Sort(l.durations)
		r.Kernels = append(r.Kernels, KernelStat{
			Kernel:            key,
			Launches:          t.launches,
			SummedNanos:       t.summed,
			ExclusiveNanos:    t.exclusive,
			MeanNanos:         t.mean(),
			MedianNanos:       percentile(l.durations, 50),
			P90Nanos:          percentile(l.durations, 90),
			Configs:           l.configCounts(),
			UnrecordedConfigs: l.unrecorded,
		})
	}
	sort.Slice(r.Kernels, func(i, j int) bool {
		a, b := r.Kernels[i], r.Kernels[j]
		if a.SummedNanos != b.SummedNanos {
			return a.SummedNanos > b.SummedNanos
		}
		return a.Kernel < b.Kernel
	})
	return r, nil
}

func (r *KernelReport) write(w io.Writer) error {
	fmt.Fprintf(w, "capture %s\n        %s\n", r.Capture, r.Side.windowLine())
	if r.Side.UnresolvedNanos > 0 {
		fmt.Fprintf(w, "note: %d ns of graph replay is covered by no resolved record; kernels inside it are not listed\n", r.Side.UnresolvedNanos)
	}
	fmt.Fprintln(w, "ns. summed: raw launch durations added up; exclusive: each busy instant charged once, as compare does.")
	fmt.Fprintln(w, "configs: distinct launch configurations, the most launched first; - when not recorded.")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "launches\tsummed\texclusive\tmean\tmedian\tp90\tconfigs\t  kernel")
	for _, k := range r.Kernels {
		configs := "-"
		if len(k.Configs) > 0 {
			configs = fmt.Sprintf("%d: %s", len(k.Configs), k.Configs[0].LaunchConfig)
		}
		fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%d\t%d\t%s\t  %s\n", k.Launches, k.SummedNanos, k.ExclusiveNanos,
			k.MeanNanos, k.MedianNanos, k.P90Nanos, configs, k.Kernel)
	}
	return tw.Flush()
}
