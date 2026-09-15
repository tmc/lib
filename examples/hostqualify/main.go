package main

import (
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"time"
)

const usage = `usage: hostqualify [-max-load x] [-reference file -ctx n [-prefill t/s] [-gen t/s]]
       hostqualify -baseline [-samples n] [-interval d] [-max-load x]

Refuse to record a throughput number from a degraded host.

  -max-load float
        highest 1-minute load average that still qualifies
        (default: one quarter of the CPU count)
  -reference file
        ds4-bench CSV of published rates for this machine
  -ctx int
        context length whose reference row to compare against
  -prefill float
        measured prefill rate, in tokens per second
  -gen float
        measured generation rate, in tokens per second
  -tolerance float
        lowest measured/published ratio that still qualifies (default 0.85)
  -baseline
        sample the idle load and report whether the limit is attainable
  -samples int
        baseline samples (default 16)
  -interval duration
        baseline sampling interval (default 15s)
`

type config struct {
	maxLoad   float64
	reference string
	ctx       int
	prefill   float64
	gen       float64
	tolerance float64
	baseline  bool
	samples   int
	interval  time.Duration
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		fmt.Fprintln(os.Stderr, "hostqualify:", err)
		os.Exit(1)
	}
}

// errDegraded reports a host that does not qualify. It is distinct from
// a usage or I/O error so that a caller can tell "the host is busy" from
// "the gate could not run".
var errDegraded = errors.New("host does not qualify")

func run(args []string, stdout, stderr io.Writer) error {
	cfg, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	if cfg.baseline {
		return baseline(cfg, stdout)
	}
	load, err := loadAverage()
	if err != nil {
		return err
	}
	ok := true
	fmt.Fprintf(stdout, "load average %.2f over %d CPUs, limit %.2f\n", load, runtime.NumCPU(), cfg.maxLoad)
	if load > cfg.maxLoad {
		fmt.Fprintln(stdout, "  DEGRADED: another workload is competing for this machine")
		ok = false
	}
	if cfg.reference != "" {
		row, err := readReference(cfg.reference, cfg.ctx)
		if err != nil {
			return err
		}
		for _, c := range []struct {
			name      string
			measured  float64
			published float64
		}{
			{"prefill", cfg.prefill, row.prefill},
			{"generation", cfg.gen, row.gen},
		} {
			if c.measured == 0 || c.published == 0 {
				continue
			}
			ratio := c.measured / c.published
			fmt.Fprintf(stdout, "%s %.2f t/s against published %.2f t/s: %.2f of reference\n",
				c.name, c.measured, c.published, ratio)
			if ratio < cfg.tolerance {
				fmt.Fprintf(stdout, "  DEGRADED: below the %.2f tolerance\n", cfg.tolerance)
				ok = false
			}
		}
	}
	if !ok {
		return errDegraded
	}
	fmt.Fprintln(stdout, "host qualifies")
	return nil
}

func parseConfig(args []string, stderr io.Writer) (config, error) {
	var cfg config
	fs := flag.NewFlagSet("hostqualify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	fs.Float64Var(&cfg.maxLoad, "max-load", float64(runtime.NumCPU())/4, "highest qualifying load average")
	fs.StringVar(&cfg.reference, "reference", "", "ds4-bench reference CSV")
	fs.IntVar(&cfg.ctx, "ctx", 0, "reference context length")
	fs.Float64Var(&cfg.prefill, "prefill", 0, "measured prefill rate")
	fs.Float64Var(&cfg.gen, "gen", 0, "measured generation rate")
	fs.Float64Var(&cfg.tolerance, "tolerance", 0.85, "lowest qualifying ratio")
	fs.BoolVar(&cfg.baseline, "baseline", false, "sample the idle load instead of gating a run")
	fs.IntVar(&cfg.samples, "samples", 16, "baseline samples")
	fs.DurationVar(&cfg.interval, "interval", 15*time.Second, "baseline sampling interval")
	if err := fs.Parse(args); err != nil {
		return config{}, err
	}
	if fs.NArg() != 0 {
		return config{}, errors.New("unexpected arguments")
	}
	if cfg.maxLoad <= 0 {
		return config{}, errors.New("max-load must be positive")
	}
	if cfg.tolerance <= 0 || cfg.tolerance > 1 {
		return config{}, errors.New("tolerance must be in (0,1]")
	}
	if cfg.reference != "" && cfg.ctx <= 0 {
		return config{}, errors.New("reference needs a positive ctx")
	}
	if cfg.reference == "" && (cfg.prefill != 0 || cfg.gen != 0) {
		return config{}, errors.New("a measured rate needs a reference")
	}
	if cfg.baseline && cfg.reference != "" {
		return config{}, errors.New("baseline takes no reference")
	}
	if cfg.samples <= 0 {
		return config{}, errors.New("samples must be positive")
	}
	if cfg.interval < 0 {
		return config{}, errors.New("interval must not be negative")
	}
	return cfg, nil
}

// A referenceRow is one published measurement.
type referenceRow struct {
	prefill float64
	gen     float64
}

// readReference reads the row for ctx from a ds4-bench CSV. Columns are
// found by header name, because the Metal and CUDA reference files
// publish different column sets.
func readReference(path string, ctx int) (referenceRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return referenceRow{}, fmt.Errorf("open reference: %w", err)
	}
	defer f.Close()
	records, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return referenceRow{}, fmt.Errorf("read reference: %w", err)
	}
	if len(records) < 2 {
		return referenceRow{}, fmt.Errorf("reference %s has no rows", path)
	}
	column := make(map[string]int, len(records[0]))
	for i, name := range records[0] {
		column[name] = i
	}
	for _, name := range []string{"ctx_tokens", "prefill_tps", "gen_tps"} {
		if _, ok := column[name]; !ok {
			return referenceRow{}, fmt.Errorf("reference %s has no %s column", path, name)
		}
	}
	for _, record := range records[1:] {
		if len(record) != len(records[0]) {
			return referenceRow{}, fmt.Errorf("reference %s has a short row", path)
		}
		if record[column["ctx_tokens"]] != strconv.Itoa(ctx) {
			continue
		}
		var row referenceRow
		if row.prefill, err = strconv.ParseFloat(record[column["prefill_tps"]], 64); err != nil {
			return referenceRow{}, fmt.Errorf("reference %s: prefill_tps: %w", path, err)
		}
		if row.gen, err = strconv.ParseFloat(record[column["gen_tps"]], 64); err != nil {
			return referenceRow{}, fmt.Errorf("reference %s: gen_tps: %w", path, err)
		}
		return row, nil
	}
	return referenceRow{}, fmt.Errorf("reference %s has no row for ctx %d", path, ctx)
}
