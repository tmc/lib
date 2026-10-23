// Command pairedtiming scores paired ABBA timings of a candidate against a
// baseline and prints per-shape ratios, per-load ratios, the weighted
// geometric-mean score, a block bootstrap interval and a sign test.
//
// Usage:
//
//	pairedtiming [-estimator geometric|sum] [-bootstrap N] [-seed S] [-format blocks|okc] [-kernels] [-json] [file.json]
//
// The default blocks format is
//
//	{"series": [{
//	  "name": "decode-1", "load": "decode", "weight": 46,
//	  "intercept": 4.29,                   // optional; or:
//	  "control": [[one1, two1, two2, one2], ...],
//	  "blocks": [[a1, b1, b2, a2], ...]
//	}]}
//
// where a is the baseline, b the candidate, all in one time unit, in the
// order they ran. The okc format is an Open Kernel Challenge judge profile
// (benchmarks[].baseline_runs_ns, runs_ns, replay_overhead.overhead_ns,
// case.load, case.weight); consecutive sample pairs form one block. With
// -kernels, the judge's per-case kernel tables are summed into one
// case-weighted table per load (see package okcprofile).
//
// The command reads samples only; it never runs a workload.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/okcprofile"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "pairedtiming:", err)
		os.Exit(1)
	}
}

type input struct {
	Series []struct {
		Name      string
		Load      string
		Weight    float64
		Intercept float64
		Control   [][4]float64
		Blocks    [][4]float64
	}
}

type shapeReport struct {
	Name      string    `json:"name"`
	Load      string    `json:"load"`
	Weight    float64   `json:"weight"`
	Intercept float64   `json:"intercept"`
	Blocks    int       `json:"blocks"`
	Ratio     float64   `json:"ratio"`
	MinRatio  float64   `json:"min_ratio"`
	Baseline  float64   `json:"baseline"`
	Candidate float64   `json:"candidate"`
	Ratios    []float64 `json:"ratios"`
}

type report struct {
	Estimator string                   `json:"estimator"`
	Shapes    []shapeReport            `json:"shapes"`
	Loads     []pairedtiming.LoadScore `json:"loads"`
	Speedup   float64                  `json:"speedup"`
	Interval  *pairedtiming.Interval   `json:"interval,omitempty"`
	Draws     int                      `json:"draws,omitempty"`
	Seed      uint64                   `json:"seed,omitempty"`
	Wins      int                      `json:"block_wins"`
	Losses    int                      `json:"block_losses"`
	SignP     float64                  `json:"sign_p"`
	Kernels   *okcprofile.Table        `json:"kernels,omitempty"`
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("pairedtiming", flag.ContinueOnError)
	fs.SetOutput(stderr)
	estimator := fs.String("estimator", "geometric", "block ratio: geometric or sum")
	draws := fs.Int("bootstrap", 10000, "bootstrap resamples; 0 disables the interval")
	seed := fs.Uint64("seed", 1, "bootstrap seed")
	format := fs.String("format", "blocks", "input format: blocks or okc")
	kernels := fs.Bool("kernels", false, "with -format okc, also print the weighted per-kernel table from the judge's profile reports")
	asJSON := fs.Bool("json", false, "print JSON")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: pairedtiming [flags] [file.json]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() > 1 {
		return errors.New("expected at most one input file")
	}
	e, err := pairedtiming.ParseEstimator(*estimator)
	if err != nil {
		return err
	}
	r := stdin
	if fs.NArg() == 1 {
		f, err := os.Open(fs.Arg(0))
		if err != nil {
			return err
		}
		defer f.Close()
		r = f
	}
	if *kernels && *format != "okc" {
		return errors.New("-kernels needs -format okc")
	}
	series, profile, err := decode(r, *format)
	if err != nil {
		return err
	}
	rep, err := analyze(series, e, *draws, *seed)
	if err != nil {
		return err
	}
	if *kernels {
		t, err := okcprofile.Kernels(profile)
		if err != nil {
			return err
		}
		rep.Kernels = &t
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(rep)
	}
	return printText(stdout, rep)
}

// decode returns the series in r and, for the okc format, the judge profile
// they came from.
func decode(r io.Reader, format string) ([]pairedtiming.Series, *okcprofile.File, error) {
	dec := json.NewDecoder(r)
	switch format {
	case "blocks":
		dec.DisallowUnknownFields()
		var in input
		if err := dec.Decode(&in); err != nil {
			return nil, nil, fmt.Errorf("decode blocks: %w", err)
		}
		var out []pairedtiming.Series
		for i, s := range in.Series {
			name := s.Name
			if name == "" {
				name = fmt.Sprint(i)
			}
			h := s.Intercept
			if len(s.Control) > 0 {
				if h != 0 {
					return nil, nil, fmt.Errorf("series %s: both intercept and control given", name)
				}
				var cb []pairedtiming.ControlBlock
				for _, c := range s.Control {
					cb = append(cb, pairedtiming.ControlBlock{One1: c[0], Two1: c[1], Two2: c[2], One2: c[3]})
				}
				var err error
				if h, err = pairedtiming.Intercept(cb); err != nil {
					return nil, nil, fmt.Errorf("series %s: %w", name, err)
				}
			}
			ps := pairedtiming.Series{Name: name, Load: s.Load, Weight: s.Weight, Intercept: h}
			for _, b := range s.Blocks {
				ps.Blocks = append(ps.Blocks, pairedtiming.Block{A1: b[0], B1: b[1], B2: b[2], A2: b[3]})
			}
			out = append(out, ps)
		}
		return out, nil, nil
	case "okc":
		f, err := okcprofile.Decode(r)
		if err != nil {
			return nil, nil, err
		}
		series, err := f.Series()
		return series, f, err
	}
	return nil, nil, fmt.Errorf("unknown format %q", format)
}

func analyze(series []pairedtiming.Series, e pairedtiming.Estimator, draws int, seed uint64) (*report, error) {
	sc, results, err := pairedtiming.ScoreSeries(series, e)
	if err != nil {
		return nil, err
	}
	rep := &report{Estimator: e.String(), Loads: sc.Loads, Speedup: sc.Speedup}
	for i, s := range series {
		r := results[i]
		rep.Shapes = append(rep.Shapes, shapeReport{
			Name: s.Name, Load: s.Load, Weight: s.Weight, Intercept: s.Intercept,
			Blocks: len(s.Blocks), Ratio: r.Ratio, MinRatio: r.MinRatio,
			Baseline: r.Baseline, Candidate: r.Candidate, Ratios: r.Ratios,
		})
		for _, x := range r.Ratios {
			switch {
			case x > 1:
				rep.Wins++
			case x < 1:
				rep.Losses++
			}
		}
	}
	rep.SignP = pairedtiming.SignTest(rep.Wins, rep.Losses)
	if draws > 0 {
		iv, err := pairedtiming.Bootstrap(series, e, draws, seed)
		if err != nil {
			return nil, err
		}
		rep.Interval, rep.Draws, rep.Seed = &iv, draws, seed
	}
	return rep, nil
}

func printText(w io.Writer, rep *report) error {
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "shape\tload\tweight\tblocks\tintercept\tbaseline\tcandidate\tratio\tmin\t")
	for _, s := range rep.Shapes {
		fmt.Fprintf(tw, "%s\t%s\t%g\t%d\t%.4g\t%.4g\t%.4g\t%.4f\t%.4f\t\n",
			s.Name, s.Load, s.Weight, s.Blocks, s.Intercept, s.Baseline, s.Candidate, s.Ratio, s.MinRatio)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	fmt.Fprintln(w)
	tw = tabwriter.NewWriter(w, 0, 4, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "load\tweighted baseline\tweighted candidate\tspeedup\t")
	for _, l := range rep.Loads {
		fmt.Fprintf(tw, "%s\t%.6g\t%.6g\t%.4f\t\n", l.Load, l.Baseline, l.Candidate, l.Speedup)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	fmt.Fprintf(w, "\nscore %.4f (%s estimator, geometric mean over %d loads)\n", rep.Speedup, rep.Estimator, len(rep.Loads))
	if rep.Interval != nil {
		fmt.Fprintf(w, "95%% block bootstrap [%.4f, %.4f] (%d draws, seed %d; within this campaign only)\n",
			rep.Interval.Lo, rep.Interval.Hi, rep.Draws, rep.Seed)
	}
	fmt.Fprintf(w, "blocks won %d, lost %d, sign test p=%.3g\n", rep.Wins, rep.Losses, rep.SignP)
	if rep.Kernels != nil {
		return printKernels(w, rep.Kernels)
	}
	return nil
}

func printKernels(w io.Writer, t *okcprofile.Table) error {
	fmt.Fprintln(w)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "load\tweighted µs\tshare\tcases\tcalls\tkernel")
	for _, k := range t.Kernels {
		name := k.Kernel
		if k.Truncated {
			name += "..."
		}
		if k.Merged {
			name += " (merged rows)"
		}
		fmt.Fprintf(tw, "%s\t%.1f\t%.1f%%\t%d\t%d\t%s\n", k.Load, float64(k.Weighted)/1e3, 100*k.Share, k.Cases, k.Calls, name)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	if t.Hidden > 0 {
		fmt.Fprintf(w, "%d kernel groups hidden by the judge; calls are lower bounds\n", t.Hidden)
	}
	if t.Overlap {
		fmt.Fprintln(w, "kernels overlapped; rows include waits and shares can sum past 100%")
	}
	return nil
}
