package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"text/tabwriter"
	"time"
)

// Ab runs a matched A/B: it captures a base and a target command back to
// back with the same capture flags, optionally captures the base command a
// second time as a control, and compares base with target and base with
// control over the same window.
//
// A delta between two captures is the change plus whatever the two runs did
// differently on their own: clocks, temperature, another process, the
// allocator's luck. The control is the same program run twice, so its delta
// is that run-to-run drift with no change in it, measured under the same
// conditions and by the same arithmetic as the delta that matters. A
// base-to-target delta is worth reading only when it clears the control's.
// The control runs last by default, so the target lies between the two
// runs of the base and the control also carries drift across the whole run.
//
// The base and target commands are split into words as a shell would split
// them, honoring quotes and backslashes, but nothing is expanded, and each
// runs as the capture's target process with no shell in between: capture
// reads the range edges from the target process's own journal, so a shell
// wrapping the workload would leave every capture without a range and
// without the clocks read at its edges. Use env to set variables.
//
// Ab does not lock the GPU. It is meant to run inside one hold of whatever
// lock the GPU's users share, so that no other work runs between or during
// its captures; captures taken in separate holds are separated by other
// agents' work and by the GPU's thermal history. Because a lock keeps out
// only the processes that take it, the -quiet command, when given, runs
// through sh -c before each capture, and a nonzero exit stops the run
// before that capture: a script that waits until no other compute process
// holds the GPU is the usual check.
//
// Each capture's range edges carry the SM and memory clocks, temperature
// and power NVML reported. Ab prints them per capture and warns when the
// target or control ran at an SM clock more than -clock-tolerance percent
// or a temperature more than -temp-tolerance degrees away from the base.

const abUsage = "usage: cudaprof ab -o dir -base 'command' -target 'command' [-control] [-order base,target,control] [-quiet 'command'] [-compare 'compare flags'] [-clock-tolerance=pct] [-temp-tolerance=C] [-json] [-- capture flags...]"

// The arms of an A/B.
const (
	armBase    = "base"
	armTarget  = "target"
	armControl = "control"
)

type abOptions struct {
	dir     string
	base    []string
	target  []string
	control bool
	order   []string
	quiet   string
	json    bool

	// clockTolerance is a percentage of the base SM clock, tempTolerance
	// degrees Celsius.
	clockTolerance float64
	tempTolerance  float64

	// captureFlags are passed to every capture; compareArgs are compare's
	// flags. Compare reads the limits each capture records, so capture's
	// limit flags need no counterpart here.
	captureFlags []string
	compareArgs  []string

	// capture captures command to out, writing its output to log. It is a
	// field so tests can run without a GPU.
	capture func(flags []string, out, log string, command []string) error
}

// runABCommand implements "cudaprof ab".
func runABCommand(args []string, stdout, stderr io.Writer) error {
	opts, err := parseABArgs(args, stdout)
	if err != nil || opts == nil {
		return err
	}
	opts.capture = captureSelf
	return runAB(opts, stdout, stderr)
}

// parseABArgs returns nil options and a nil error when help was requested
// and printed.
func parseABArgs(args []string, stdout io.Writer) (*abOptions, error) {
	opts := &abOptions{}
	fs := flag.NewFlagSet("ab", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVar(&opts.dir, "o", "", "write captures, logs and comparisons to `dir`")
	base := fs.String("base", "", "base `command`, split into words and run without a shell")
	target := fs.String("target", "", "target `command`, split into words and run without a shell")
	fs.BoolVar(&opts.control, "control", false, "capture the base command again as a control")
	order := fs.String("order", "", "comma-separated run `order` of base, target and control (default base,target[,control])")
	fs.StringVar(&opts.quiet, "quiet", "", "`command` run by sh -c before each capture; a nonzero exit stops the run")
	compare := fs.String("compare", "", "compare `flags`, separated by spaces")
	fs.Float64Var(&opts.clockTolerance, "clock-tolerance", 2, "warn when an SM clock differs from the base by more than `pct` percent")
	fs.Float64Var(&opts.tempTolerance, "temp-tolerance", 5, "warn when a temperature differs from the base by more than `C` degrees")
	fs.BoolVar(&opts.json, "json", false, "write the summary as JSON")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprintln(stdout, abUsage)
			fs.SetOutput(stdout)
			fs.PrintDefaults()
			return nil, nil
		}
		return nil, fmt.Errorf("ab: %w\n%s", err, abUsage)
	}
	if opts.dir == "" || *base == "" || *target == "" {
		return nil, errors.New(abUsage)
	}
	var err error
	if opts.base, err = splitWords(*base); err != nil {
		return nil, fmt.Errorf("ab: -base: %w", err)
	}
	if opts.target, err = splitWords(*target); err != nil {
		return nil, fmt.Errorf("ab: -target: %w", err)
	}
	if len(opts.base) == 0 || len(opts.target) == 0 {
		return nil, errors.New(abUsage)
	}
	if opts.clockTolerance < 0 || opts.tempTolerance < 0 {
		return nil, errors.New("ab: negative tolerance")
	}
	opts.captureFlags = fs.Args()
	for _, f := range opts.captureFlags {
		name, _, _ := strings.Cut(f, "=")
		switch name {
		case "-o", "--":
			return nil, fmt.Errorf("ab: capture flag %s is set by ab", f)
		}
	}

	opts.order = []string{armBase, armTarget}
	if opts.control {
		opts.order = append(opts.order, armControl)
	}
	if *order != "" {
		opts.order = strings.Split(*order, ",")
		for i, arm := range opts.order {
			if arm != armBase && arm != armTarget && arm != armControl {
				return nil, fmt.Errorf("ab: -order: unknown arm %q", arm)
			}
			if slices.Contains(opts.order[:i], arm) {
				return nil, fmt.Errorf("ab: -order: %s appears twice", arm)
			}
		}
		if !slices.Contains(opts.order, armBase) || !slices.Contains(opts.order, armTarget) {
			return nil, errors.New("ab: -order must name base and target")
		}
		if slices.Contains(opts.order, armControl) {
			opts.control = true
		} else if opts.control {
			return nil, errors.New("ab: -control given but -order has no control")
		}
	}

	opts.compareArgs = strings.Fields(*compare)
	if _, _, err := parseCompareArgs(append(slices.Clone(opts.compareArgs), "b", "t"), io.Discard); err != nil {
		return nil, fmt.Errorf("ab: -compare: %w", err)
	}
	return opts, nil
}

// abReport is what an A/B run reports.
type abReport struct {
	Dir          string   `json:"dir"`
	Order        []string `json:"order"`
	CaptureFlags []string `json:"capture_flags"`
	CompareFlags []string `json:"compare_flags"`
	Runs         []*abRun `json:"runs"`

	// Target is base to target; Control, base to control, is the noise
	// floor Target is read against.
	Target  *abDelta `json:"target"`
	Control *abDelta `json:"control,omitempty"`

	// NoiseRatio is |Δwindow| of Target over |Δwindow| of Control, absent
	// without a control or when the control's delta is zero.
	NoiseRatio *float64 `json:"noise_ratio,omitempty"`
	Warnings   []string `json:"warnings,omitempty"`
}

// abRun is one capture of an A/B.
type abRun struct {
	Arm       string    `json:"arm"`
	Command   []string  `json:"command"`
	Capture   string    `json:"capture"`
	Log       string    `json:"log"`
	Started   time.Time `json:"started"`
	WallNanos int64     `json:"wall_ns"`

	WindowNanos int64          `json:"window_ns"`
	BusyNanos   int64          `json:"busy_ns"`
	KernelNanos int64          `json:"kernel_exclusive_ns"`
	IdleNanos   int64          `json:"idle_ns"`
	Clocks      *compareClocks `json:"clocks"`
}

// abDelta is one comparison of an A/B: its identity terms and the files
// holding the whole of it.
type abDelta struct {
	Name  string        `json:"name"`
	Terms []GapIntDelta `json:"terms"`
	Text  string        `json:"text"`
	JSON  string        `json:"json"`
}

func runAB(opts *abOptions, stdout, stderr io.Writer) error {
	dir, err := filepath.Abs(opts.dir)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("ab: %w", err)
	}
	rep := &abReport{Dir: dir, Order: opts.order, CaptureFlags: opts.captureFlags, CompareFlags: opts.compareArgs}
	for _, arm := range opts.order {
		r := &abRun{Arm: arm, Command: opts.base,
			Capture: filepath.Join(dir, arm+".pprof"), Log: filepath.Join(dir, arm+".log")}
		if arm == armTarget {
			r.Command = opts.target
		}
		// Refuse before spending GPU time, not after the first capture.
		if _, err := os.Stat(r.Capture); err == nil {
			return fmt.Errorf("ab: %s exists", r.Capture)
		}
		rep.Runs = append(rep.Runs, r)
	}

	for i, r := range rep.Runs {
		if opts.quiet != "" {
			fmt.Fprintf(stderr, "ab: quiet check before %s\n", r.Arm)
			if err := runQuiet(opts.quiet, filepath.Join(dir, "quiet.log"), r.Arm); err != nil {
				return fmt.Errorf("ab: quiet check before %s: %w", r.Arm, err)
			}
		}
		fmt.Fprintf(stderr, "ab: capture %d/%d %s: %s\n", i+1, len(rep.Runs), r.Arm, strings.Join(r.Command, " "))
		r.Started = time.Now()
		err := opts.capture(opts.captureFlags, r.Capture, r.Log, r.Command)
		r.WallNanos = int64(time.Since(r.Started))
		if err != nil {
			return fmt.Errorf("ab: capture %s: %w (log %s)", r.Arm, err, r.Log)
		}
	}

	copts, _, err := parseCompareArgs(append(slices.Clone(opts.compareArgs), "b", "t"), io.Discard)
	if err != nil {
		return fmt.Errorf("ab: %w", err)
	}
	rules, err := readOpRules(copts.opsPath)
	if err != nil {
		return fmt.Errorf("ab: %w", err)
	}
	stages, err := readStageBoundaries(copts.stagesPath)
	if err != nil {
		return fmt.Errorf("ab: %w", err)
	}
	// Measure each capture once and drop it: a large capture's journals
	// take far more memory than its measured side.
	sides := map[string]*CompareSide{}
	for _, r := range rep.Runs {
		cap, err := readCapturePathLimits(r.Capture, copts.limits)
		if err != nil {
			return fmt.Errorf("ab: read capture %s: %w", r.Capture, err)
		}
		s, err := measureCompareSide(cap, rules, stages, copts)
		if err != nil {
			return fmt.Errorf("ab: %s: %w", r.Arm, err)
		}
		s.Capture = r.Capture
		sides[r.Arm] = s
		r.WindowNanos, r.BusyNanos, r.KernelNanos, r.IdleNanos = s.WindowNanos, s.BusyNanos, s.KernelNanos, s.IdleNanos
		r.Clocks = s.Clocks
	}
	arms := []string{armTarget}
	if opts.control {
		arms = append(arms, armControl)
	}
	for _, arm := range arms {
		c, err := compareSides(sides[armBase], sides[arm], rules, stages, copts)
		if err != nil {
			return fmt.Errorf("ab: compare base with %s: %w", arm, err)
		}
		d := &abDelta{Name: "base -> " + arm, Terms: c.Terms,
			Text: filepath.Join(dir, "base-"+arm+".txt"), JSON: filepath.Join(dir, "base-"+arm+".json")}
		if err := writeComparison(c, d.Text, d.JSON); err != nil {
			return fmt.Errorf("ab: %w", err)
		}
		if arm == armTarget {
			rep.Target = d
		} else {
			rep.Control = d
		}
	}
	if rep.Control != nil {
		if n := abs64(rep.Control.window()); n != 0 {
			ratio := float64(abs64(rep.Target.window())) / float64(n)
			rep.NoiseRatio = &ratio
		}
	}
	rep.Warnings = abWarnings(rep.Runs, opts)

	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "ab.json"), append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("ab: %w", err)
	}
	var text strings.Builder
	if err := rep.write(&text); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "ab.txt"), []byte(text.String()), 0o644); err != nil {
		return fmt.Errorf("ab: %w", err)
	}
	if opts.json {
		_, err = stdout.Write(append(data, '\n'))
		return err
	}
	_, err = io.WriteString(stdout, text.String())
	return err
}

// captureSelf runs "cudaprof capture" in a child process of this
// executable, so each capture is exactly what the capture command makes.
func captureSelf(flags []string, out, log string, command []string) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	f, err := os.Create(log)
	if err != nil {
		return err
	}
	defer f.Close()
	args := append([]string{"capture"}, flags...)
	args = append(args, "--json", "-o", out, "--")
	args = append(args, command...)
	cmd := exec.Command(exe, args...)
	cmd.Stdout, cmd.Stderr = f, f
	if err := cmd.Run(); err != nil {
		return err
	}
	return f.Close()
}

// splitWords splits s into words at unquoted white space. Single quotes
// keep everything up to the next single quote; a backslash outside them
// keeps the next character, and inside double quotes keeps a following
// double quote or backslash.
func splitWords(s string) ([]string, error) {
	var words []string
	var w strings.Builder
	inWord := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n':
			if inWord {
				words = append(words, w.String())
				w.Reset()
				inWord = false
			}
			continue
		case c == '\'':
			j := strings.IndexByte(s[i+1:], '\'')
			if j < 0 {
				return nil, errors.New("unterminated single quote")
			}
			w.WriteString(s[i+1 : i+1+j])
			i += j + 1
		case c == '"':
			i++
			for ; i < len(s) && s[i] != '"'; i++ {
				if s[i] == '\\' && i+1 < len(s) && (s[i+1] == '"' || s[i+1] == '\\') {
					i++
				}
				w.WriteByte(s[i])
			}
			if i == len(s) {
				return nil, errors.New("unterminated double quote")
			}
		case c == '\\':
			if i+1 == len(s) {
				return nil, errors.New("trailing backslash")
			}
			i++
			w.WriteByte(s[i])
		default:
			w.WriteByte(c)
		}
		inWord = true
	}
	if inWord {
		words = append(words, w.String())
	}
	return words, nil
}

// runQuiet runs the quiet check, appending its output to log.
func runQuiet(command, log, arm string) error {
	f, err := os.OpenFile(log, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Fprintf(f, "%s before %s: %s\n", time.Now().Format(time.RFC3339), arm, command)
	cmd := exec.Command("sh", "-c", command)
	cmd.Stdout, cmd.Stderr = f, f
	if err := cmd.Run(); err != nil {
		return err
	}
	return f.Close()
}

// writeComparison writes c as compare's text and JSON.
func writeComparison(c *Comparison, text, js string) error {
	var b strings.Builder
	if err := c.write(&b); err != nil {
		return err
	}
	if err := os.WriteFile(text, []byte(b.String()), 0o644); err != nil {
		return err
	}
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(js, append(data, '\n'), 0o644)
}

func (d *abDelta) term(name string) int64 {
	for _, t := range d.Terms {
		if t.Name == name {
			return t.Delta
		}
	}
	return 0
}

func (d *abDelta) window() int64 { return d.term("window") }

// firstReading returns field of the first device that reported it at s.
func firstReading(s *gpuEdgeSample, field func(gpuReading) *uint32) (uint32, bool) {
	if s == nil {
		return 0, false
	}
	for _, d := range s.Devices {
		if v := field(d); v != nil {
			return *v, true
		}
	}
	return 0, false
}

var (
	smClockField  = func(r gpuReading) *uint32 { return r.SMClockMHz }
	memClockField = func(r gpuReading) *uint32 { return r.MemClockMHz }
	tempField     = func(r gpuReading) *uint32 { return r.TemperatureC }
	powerField    = func(r gpuReading) *uint32 { return r.PowerMilliwatts }
)

// abWarnings compares the GPU state of the target and control with that of
// the base, edge by edge.
func abWarnings(runs []*abRun, opts *abOptions) []string {
	var warnings []string
	var base *abRun
	for _, r := range runs {
		if r.Clocks != nil && r.Clocks.NotRecorded != "" {
			warnings = append(warnings, fmt.Sprintf("%s: clocks not recorded: %s", r.Arm, r.Clocks.NotRecorded))
		}
		if r.Arm == armBase {
			base = r
		}
	}
	for _, r := range runs {
		if r == base || base.Clocks == nil || r.Clocks == nil {
			continue
		}
		arm := r.Arm
		for _, e := range []struct {
			name    string
			base, s *gpuEdgeSample
		}{{"start", base.Clocks.Start, r.Clocks.Start}, {"stop", base.Clocks.Stop, r.Clocks.Stop}} {
			if bc, ok := firstReading(e.base, smClockField); ok {
				if c, ok := firstReading(e.s, smClockField); ok {
					if pct := 100 * math.Abs(float64(c)-float64(bc)) / float64(bc); pct > opts.clockTolerance {
						warnings = append(warnings, fmt.Sprintf("SM clock at the %s edge was %d MHz in base and %d MHz in %s, %.1f%% apart (tolerance %g%%)", e.name, bc, c, arm, pct, opts.clockTolerance))
					}
				}
			}
			if bt, ok := firstReading(e.base, tempField); ok {
				if t, ok := firstReading(e.s, tempField); ok {
					if d := math.Abs(float64(t) - float64(bt)); d > opts.tempTolerance {
						warnings = append(warnings, fmt.Sprintf("temperature at the %s edge was %d C in base and %d C in %s, %g C apart (tolerance %g C)", e.name, bt, t, arm, d, opts.tempTolerance))
					}
				}
			}
		}
	}
	return warnings
}

// edgeCell formats a reading at the start and stop edges as "start/stop".
func edgeCell(c *compareClocks, field func(gpuReading) *uint32, scale uint32) string {
	cell := func(s *gpuEdgeSample) string {
		v, ok := firstReading(s, field)
		if !ok {
			return "-"
		}
		return fmt.Sprint(v / scale)
	}
	if c == nil {
		return "-"
	}
	return cell(c.Start) + "/" + cell(c.Stop)
}

func abMillis(ns int64) string { return fmt.Sprintf("%.3f", float64(ns)/1e6) }

func (rep *abReport) write(w io.Writer) error {
	fmt.Fprintf(w, "ab: %d captures in %s, order %s\n", len(rep.Runs), rep.Dir, strings.Join(rep.Order, ", "))
	if len(rep.CaptureFlags) > 0 {
		fmt.Fprintf(w, "capture flags: %s\n", strings.Join(rep.CaptureFlags, " "))
	}
	if len(rep.CompareFlags) > 0 {
		fmt.Fprintf(w, "compare flags: %s\n", strings.Join(rep.CompareFlags, " "))
	}
	fmt.Fprintln(w)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "arm\twindow ms\tbusy ms\tkernel ms\tidle ms\tsm MHz\tmem MHz\ttemp C\tpower W\tcapture s\t")
	for _, r := range rep.Runs {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.1f\t\n", r.Arm,
			abMillis(r.WindowNanos), abMillis(r.BusyNanos), abMillis(r.KernelNanos), abMillis(r.IdleNanos),
			edgeCell(r.Clocks, smClockField, 1), edgeCell(r.Clocks, memClockField, 1),
			edgeCell(r.Clocks, tempField, 1), edgeCell(r.Clocks, powerField, 1000),
			time.Duration(r.WallNanos).Seconds())
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	fmt.Fprintln(w, "(gpu columns are start/stop range edges)")
	fmt.Fprintln(w)
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "\tΔwindow ms\tΔbusy ms\tΔkernel ms\tΔidle ms\t\t")
	for _, d := range []*abDelta{rep.Target, rep.Control} {
		if d == nil {
			continue
		}
		note := ""
		if d == rep.Control {
			note = "noise floor"
		}
		fmt.Fprintf(tw, "%s\t%+.3f\t%+.3f\t%+.3f\t%+.3f\t%s\t\n", d.Name,
			float64(d.window())/1e6, float64(d.term("busy"))/1e6, float64(d.term("kernel"))/1e6, float64(d.term("idle"))/1e6, note)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	switch {
	case rep.Control == nil:
		fmt.Fprintln(w, "no control: the delta has no noise floor to be read against (-control)")
	case rep.NoiseRatio == nil:
		fmt.Fprintln(w, "the control's window delta is zero, so the ratio is undefined")
	case *rep.NoiseRatio <= 1:
		fmt.Fprintf(w, "|Δwindow| is %.2fx the control's: within run-to-run noise\n", *rep.NoiseRatio)
	default:
		fmt.Fprintf(w, "|Δwindow| is %.1fx the control's\n", *rep.NoiseRatio)
	}
	for _, s := range rep.Warnings {
		fmt.Fprintf(w, "warning: %s\n", s)
	}
	fmt.Fprintf(w, "\ncomparisons: %s", rep.Target.Text)
	if rep.Control != nil {
		fmt.Fprintf(w, ", %s", rep.Control.Text)
	}
	fmt.Fprintln(w)
	return nil
}
