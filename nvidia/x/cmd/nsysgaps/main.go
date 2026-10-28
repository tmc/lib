package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/nsightsystems"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

const usage = "usage: nsysgaps [-end NAME] [-skip N] [-floor DUR] [-top N] [-api API.csv] GPU.csv"

type durations []time.Duration

func (d *durations) String() string { return fmt.Sprint(*d) }

func (d *durations) Set(s string) error {
	v, err := time.ParseDuration(s)
	if err != nil {
		return err
	}
	*d = append(*d, v)
	return nil
}

// run executes the command line args and returns the exit status.
func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("nsysgaps", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintln(stderr, usage)
		fs.PrintDefaults()
	}
	end := fs.String("end", "Argmax", "a step ends with a kernel whose name contains `name`")
	skip := fs.Int("skip", 1, "drop the first `n` steps")
	top := fs.Int("top", 8, "list the `n` kernels with the most idle time before them")
	api := fs.String("api", "", "cuda_api_trace CSV `file` of the same capture")
	var floors durations
	fs.Var(&floors, "floor", "model the step with its kernels taking `dur` (repeatable; default 2ms)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return 2
	}
	if len(floors) == 0 {
		floors = durations{2 * time.Millisecond}
	}
	if err := report(stdout, fs.Arg(0), *api, *end, *skip, *top, floors); err != nil {
		fmt.Fprintf(stderr, "nsysgaps: %v\n", err)
		return 1
	}
	return 0
}

func report(w io.Writer, gpuFile, apiFile, end string, skip, top int, floors durations) error {
	f, err := os.Open(gpuFile)
	if err != nil {
		return err
	}
	kernels, err := nsightsystems.ReadGPUTraceCSV(f)
	f.Close()
	if err != nil {
		return fmt.Errorf("%s: %w", gpuFile, err)
	}
	steps := nsightsystems.SplitSteps(kernels, end)
	if len(steps) <= skip {
		return fmt.Errorf("%s: %d steps ending in %q, need more than %d", gpuFile, len(steps), end, skip)
	}
	fmt.Fprintf(w, "%d kernels, %d steps ending in %q; skipping %d\n", len(kernels), len(steps), end, skip)
	fmt.Fprintf(w, "%4s %8s %10s %10s %10s %8s %7s\n", "step", "launches", "span", "busy", "idle", "idle/ln", "idle%")
	var gs []nsightsystems.LaunchGaps
	for i, s := range steps {
		g := nsightsystems.AnalyzeLaunchGaps(s)
		fmt.Fprintf(w, "%4d %8d %10s %10s %10s %8s %6.1f%%\n", i, g.Launches, ns(g.SpanNS), ns(g.BusyNS), ns(g.IdleNS),
			ns(int64(g.GapPerLaunchNS())), 100*float64(g.IdleNS)/float64(max(g.SpanNS, 1)))
		if i >= skip {
			gs = append(gs, g)
		}
	}
	slices.SortFunc(gs, func(a, b nsightsystems.LaunchGaps) int { return int(a.SpanNS - b.SpanNS) })
	med := gs[len(gs)/2]
	fmt.Fprintf(w, "\nmedian step: %d launches, span %s, busy %s, idle %s (%s per launch)\n",
		med.Launches, ns(med.SpanNS), ns(med.BusyNS), ns(med.IdleNS), ns(int64(med.GapPerLaunchNS())))
	fmt.Fprintf(w, "most idle before (median step):\n")
	for _, k := range med.Kernels[:min(top, len(med.Kernels))] {
		fmt.Fprintf(w, "  %10s idle %4d launches %10s busy  %s\n", ns(k.GapNS), k.Launches, ns(k.BusyNS), short(k.Name))
	}
	fmt.Fprintf(w, "\nMODELLED (floor + measured idle; launch gaps assumed unchanged):\n")
	for _, f := range floors {
		p := med.Project(f.Nanoseconds())
		fmt.Fprintf(w, "  floor %8s -> step %8s, idle %5.1f%% of step\n", f, ns(p), 100*float64(med.IdleNS)/float64(p))
	}
	if apiFile == "" {
		return nil
	}
	f, err = os.Open(apiFile)
	if err != nil {
		return err
	}
	api, err := nsightsystems.ReadAPITraceCSV(f)
	f.Close()
	if err != nil {
		return fmt.Errorf("%s: %w", apiFile, err)
	}
	fmt.Fprintf(w, "\nhost API (whole capture):\n")
	for _, n := range []string{"cuLaunchKernel", "cuGraphLaunch", "cudaLaunchKernel", "cudaGraphLaunch"} {
		calls, total := nsightsystems.APILaunches(api, n)
		if calls == 0 {
			continue
		}
		fmt.Fprintf(w, "  %-16s %6d calls %10s total %8s mean\n", n, calls, ns(total), ns(total/int64(calls)))
	}
	return nil
}

func ns(v int64) string {
	d := time.Duration(v)
	if d >= time.Millisecond {
		return d.Round(time.Microsecond).String()
	}
	return d.Round(100 * time.Nanosecond).String()
}

// short trims a demangled kernel name to its name and template arguments.
func short(name string) string {
	if i := strings.IndexByte(name, '('); i > 0 {
		name = name[:i]
	}
	if len(name) > 70 {
		name = name[:70]
	}
	return name
}
