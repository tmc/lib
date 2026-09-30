package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// runCorrelateCommand exposes retained relationships without guessing missing
// submission or operation identity from names or timestamp proximity.
func runCorrelateCommand(args []string) error {
	args, limits, err := parseReadLimits(args)
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet("correlate", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)
	fs.Usage = func() {
		fmt.Fprintln(os.Stdout, "usage: cudaprof correlate [-domain driver|runtime] [--max-records N] [--journal-bytes N] capture.pprof")
		fs.PrintDefaults()
	}
	domain := fs.String("domain", "driver", "API domain: driver or runtime")
	maxRecords := fs.Int64("max-records", 0, "journal record limit")
	journalBytes := fs.Int64("journal-bytes", 0, "decompressed journal byte limit")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: cudaprof correlate [-domain driver|runtime] capture.pprof")
	}
	if *domain != "driver" && *domain != "runtime" {
		return fmt.Errorf("correlate: invalid API domain %q", *domain)
	}
	cap, err := readCapturePathLimits(fs.Arg(0), limits)
	if err != nil {
		return err
	}
	_, records, err := decodeCompleteCapture(cap, *maxRecords, *journalBytes)
	if err != nil {
		return err
	}
	report := struct {
		Version         int                    `json:"version"`
		Domain          string                 `json:"domain"`
		Activities      []timeline.Activity    `json:"activities"`
		Calls           []timeline.Call        `json:"calls"`
		Ranges          []timeline.Range       `json:"ranges"`
		Links           []timeline.Correlation `json:"links"`
		UnpairedMarkers int                    `json:"unpaired_markers"`
		Limits          string                 `json:"limits"`
	}{1, *domain, records.Activities, records.Calls, records.Ranges, timeline.Correlations(records, *domain), records.UnpairedMarkers, "links identify retained submission context, not causes of kernel duration; graph provenance and host stacks are unsupported"}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}
