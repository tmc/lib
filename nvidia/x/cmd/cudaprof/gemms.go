package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/cublaslog"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

const gemmsUsage = "usage: cudaprof gemms [-phase i] [-warmup-until=RE | -warmup-ms=N] [-max-records=N] [-journal-bytes=N] [--capture-file-bytes=N] [--capture-total-bytes=N] [-json] capture.pprof"

// A GEMMCensus is the cuBLASLt calls of one capture window, grouped by
// what the call asked for and how it ran.
type GEMMCensus struct {
	Capture string       `json:"capture"`
	Side    *CompareSide `json:"window"`
	// Calls counts the calls in the window, Joined those linked to at
	// least one kernel record, and Unparsed those whose log line Parse
	// refused (they are left out of Rows).
	Calls    int       `json:"calls"`
	Joined   int       `json:"joined"`
	Unparsed int       `json:"unparsed"`
	Rows     []GEMMRow `json:"rows"`
}

// A GEMMRow is the calls that share a shape, types, epilogue, beta and
// algorithm.
type GEMMRow struct {
	M        int64  `json:"m"`
	N        int64  `json:"n"`
	K        int64  `json:"k"`
	Batch    int64  `json:"batch"`
	Op       string `json:"op"` // transa, transb, as "TN"
	Types    string `json:"types"`
	Compute  string `json:"compute"`
	Scale    string `json:"scale"`
	Epilogue string `json:"epilogue"`
	Beta     string `json:"beta"`
	Algo     string `json:"algo"`
	API      string `json:"api"`

	Calls  int `json:"calls"`
	Joined int `json:"joined"`
	// KernelNanos is the device time of the joined calls' kernels.
	KernelNanos int64 `json:"kernel_ns"`
	// Kernels counts the joined calls' kernels by name.
	Kernels map[string]int `json:"kernels"`
}

// PerCall is the mean kernel time of a joined call, or zero with none.
func (r *GEMMRow) PerCall() int64 {
	if r.Joined == 0 {
		return 0
	}
	return r.KernelNanos / int64(r.Joined)
}

// Shape returns the shape as "MxNxK", with a batch count when there is one.
func (r *GEMMRow) Shape() string {
	s := fmt.Sprintf("%dx%dx%d", r.M, r.N, r.K)
	if r.Batch > 1 {
		s += fmt.Sprintf(" b%d", r.Batch)
	}
	return s
}

// Kernel names the row's most frequent kernel, with a count of the others.
func (r *GEMMRow) Kernel() string {
	if len(r.Kernels) == 0 {
		return "-"
	}
	names := make([]string, 0, len(r.Kernels))
	for n := range r.Kernels {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool {
		if r.Kernels[names[i]] != r.Kernels[names[j]] {
			return r.Kernels[names[i]] > r.Kernels[names[j]]
		}
		return names[i] < names[j]
	})
	// Template arguments are kept: a CUTLASS kernel's name without them
	// is cutlass::Kernel2, whatever it computes.
	s := opKey(names[0], true)
	if len(s) > 160 {
		s = s[:157] + "..."
	}
	if len(names) > 1 {
		s += fmt.Sprintf(" (+%d more)", len(names)-1)
	}
	return s
}

func runGEMMsCommand(args []string) error {
	opts, paths, err := parseGEMMsArgs(args[1:], os.Stdout, "gemms", 1)
	if err != nil || paths == nil {
		return err
	}
	c, err := readGEMMCensus(paths[0], opts)
	if err != nil {
		return fmt.Errorf("gemms: %w", err)
	}
	if opts.json {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(c)
	}
	return c.write(os.Stdout)
}

// parseGEMMsArgs parses the flags gemms and compare -gemms share. It
// returns nil paths and a nil error when help was requested and printed.
func parseGEMMsArgs(rawArgs []string, stdout io.Writer, name string, npaths int) (compareOptions, []string, error) {
	var opts compareOptions
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return opts, nil, fmt.Errorf("%s: %w", name, err)
	}
	opts.limits = limits
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.IntVar(&opts.phase, "phase", -1, "index of the capture-range phase to read, required when a capture has several")
	fs.BoolVar(&opts.json, "json", false, "write the census as JSON")
	fs.Int64Var(&opts.maxRecords, "max-records", controllerMaxJournalRecords, "maximum journal records to decode per capture, as capture --max-records allowed it to write")
	fs.Int64Var(&opts.journalBytes, "journal-bytes", cudacapture.MaxCaptureInputBytes, "maximum decompressed bytes per journal")
	var warmup gapOptions
	fs.Float64Var(&warmup.warmupMS, "warmup-ms", 0, "start the window `N` ms after its first device activity, as gaps does")
	fs.StringVar(&warmup.warmupUntil, "warmup-until", "", "start the window at its first kernel matching `RE`, as gaps does")
	usage := gemmsUsage
	if name != "gemms" {
		usage = compareGEMMsUsage
		fs.Bool("gemms", true, "compare the captures' cuBLASLt calls")
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprintln(stdout, usage)
			fs.SetOutput(stdout)
			fs.PrintDefaults()
			return opts, nil, nil
		}
		return opts, nil, fmt.Errorf("%s: %w\n%s", name, err, usage)
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	switch {
	case set["warmup-ms"] && set["warmup-until"]:
		return opts, nil, fmt.Errorf("%s: -warmup-ms and -warmup-until are exclusive", name)
	case set["warmup-ms"] && warmup.warmupMS < 0:
		return opts, nil, fmt.Errorf("%s: invalid -warmup-ms %g", name, warmup.warmupMS)
	case set["warmup-until"]:
		if _, err := regexp.Compile(warmup.warmupUntil); err != nil || warmup.warmupUntil == "" {
			return opts, nil, fmt.Errorf("%s: invalid -warmup-until %q", name, warmup.warmupUntil)
		}
	}
	if set["warmup-ms"] || set["warmup-until"] {
		opts.warmup = &warmup
	}
	if opts.maxRecords <= 0 {
		return opts, nil, fmt.Errorf("%s: invalid -max-records %d", name, opts.maxRecords)
	}
	if opts.journalBytes <= 0 {
		return opts, nil, fmt.Errorf("%s: invalid -journal-bytes %d", name, opts.journalBytes)
	}
	if fs.NArg() != npaths {
		return opts, nil, errors.New(usage)
	}
	return opts, fs.Args(), nil
}

func readGEMMCensus(path string, opts compareOptions) (*GEMMCensus, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	cap, err := readCapturePathLimits(abs, opts.limits)
	if err != nil {
		return nil, fmt.Errorf("read capture %s: %w", path, err)
	}
	side, records, err := openCompareSide(cap, opts)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(records.GEMMs) == 0 {
		return nil, fmt.Errorf("%s recorded no cuBLASLt calls; capture with --gemms", path)
	}
	c := censusGEMMs(side, records.GEMMs)
	c.Capture = abs
	return c, nil
}

// censusGEMMs groups the calls in the side's window. A call is in the
// window when its first kernel starts there, or, with no kernel, when it
// was logged there.
func censusGEMMs(side *CompareSide, gemms []timeline.GEMM) *GEMMCensus {
	c := &GEMMCensus{Side: side}
	rows := make(map[string]*GEMMRow)
	for _, g := range gemms {
		at := g.Time
		for i, k := range g.Kernels {
			if i == 0 || k.Start < at {
				at = k.Start
			}
		}
		if at < side.StartTimestamp || at >= side.StopTimestamp {
			continue
		}
		c.Calls++
		if len(g.Kernels) > 0 {
			c.Joined++
		}
		m, err := cublaslog.Parse(g.Log)
		if err != nil {
			c.Unparsed++
			continue
		}
		r := &GEMMRow{
			M: m.M, N: m.N, K: m.K, Batch: m.Batch,
			Op:       m.TransA + m.TransB,
			Types:    m.Types(),
			Compute:  strings.TrimPrefix(m.Compute, "COMPUTE_"),
			Scale:    m.Scale,
			Epilogue: m.Epilogue,
			Beta:     m.Beta,
			Algo:     m.Algo.String(),
			API:      g.API,
		}
		key := r.groupKey()
		if old, ok := rows[key]; ok {
			r = old
		} else {
			r.Kernels = make(map[string]int)
			rows[key] = r
		}
		r.Calls++
		if len(g.Kernels) > 0 {
			r.Joined++
		}
		for _, k := range g.Kernels {
			r.KernelNanos += k.End - k.Start
			r.Kernels[k.Name]++
		}
	}
	for _, r := range rows {
		c.Rows = append(c.Rows, *r)
	}
	sort.Slice(c.Rows, func(i, j int) bool {
		a, b := &c.Rows[i], &c.Rows[j]
		if a.KernelNanos != b.KernelNanos {
			return a.KernelNanos > b.KernelNanos
		}
		if a.Calls != b.Calls {
			return a.Calls > b.Calls
		}
		return a.groupKey() < b.groupKey()
	})
	return c
}

func (r *GEMMRow) groupKey() string {
	return strings.Join([]string{r.shapeKey(), r.Compute, r.Scale, r.Epilogue, r.Beta, r.Algo, r.API}, "|")
}

// shapeKey is what compare -gemms joins two captures' rows by.
func (r *GEMMRow) shapeKey() string {
	return strings.Join([]string{r.Shape(), r.Op, r.Types}, "|")
}

func (c *GEMMCensus) write(w io.Writer) error {
	var total int64
	for i := range c.Rows {
		total += c.Rows[i].KernelNanos
	}
	fmt.Fprintf(w, "capture: %s\n%s\n", c.Capture, c.Side.windowLine())
	fmt.Fprintf(w, "calls: %d in window, %d joined to kernels, %d unparsed; kernel time %s ms\n\n", c.Calls, c.Joined, c.Unparsed, gemmMS(total))
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "calls\ttotal ms\tus/call\tshape MxNxK\top\ttypes\tcompute\tepilogue\tbeta\talgo\t")
	for i := range c.Rows {
		r := &c.Rows[i]
		fmt.Fprintf(tw, "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n", r.Calls, gemmMS(r.KernelNanos), gemmUS(r.PerCall()), r.Shape(), r.Op, r.Types, r.Compute, r.Epilogue, r.Beta, r.Algo)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	// Kernel names are long and would push the table off the screen, so
	// they are listed after it, keyed by row.
	fmt.Fprintln(w, "\nkernels:")
	for i := range c.Rows {
		r := &c.Rows[i]
		joined := ""
		if r.Joined < r.Calls {
			joined = fmt.Sprintf(" [%d of %d calls unjoined]", r.Calls-r.Joined, r.Calls)
		}
		fmt.Fprintf(w, "%3d  %-18s %s  %s%s\n", i+1, r.Shape(), r.API, r.Kernel(), joined)
	}
	return nil
}

func gemmMS(ns int64) string { return strconv.FormatFloat(float64(ns)/1e6, 'f', 3, 64) }
func gemmUS(ns int64) string { return strconv.FormatFloat(float64(ns)/1e3, 'f', 1, 64) }
