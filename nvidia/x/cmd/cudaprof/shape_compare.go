package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"

	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// runShapeCompareCommand preserves completeness and stationarity independently
// of matching explicit operation identity. It never changes a failed gate.
func runShapeCompareCommand(args []string) error {
	return runShapeCompare(args, os.Stdout)
}

func runShapeCompare(args []string, stdout io.Writer) error {
	args, limits, err := parseReadLimits(args)
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet("shape-compare", flag.ContinueOnError)
	fs.SetOutput(stdout)
	fs.Usage = func() {
		fmt.Fprintln(stdout, "usage: cudaprof shape-compare [-domain driver|runtime] [-iteration-marker RE] base.pprof target.pprof")
		fs.PrintDefaults()
	}
	domain := fs.String("domain", "driver", "submission API domain: driver or runtime")
	pattern := fs.String("iteration-marker", "arg_reduce", "kernel marker for the separate stationarity gate")
	maxRecords := fs.Int64("max-records", 0, "maximum journal records; 0 uses recorded bounds")
	journalBytes := fs.Int64("journal-bytes", 0, "maximum decompressed journal bytes; 0 uses recorded bounds")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("usage: cudaprof shape-compare [-domain driver|runtime] [-iteration-marker RE] base.pprof target.pprof")
	}
	if *domain != "driver" && *domain != "runtime" {
		return fmt.Errorf("shape-compare: invalid API domain %q", *domain)
	}
	if *maxRecords < 0 || *journalBytes < 0 {
		return fmt.Errorf("shape-compare: negative decode limit")
	}
	re, err := regexp.Compile(*pattern)
	if err != nil {
		return fmt.Errorf("shape-compare: iteration marker: %w", err)
	}
	type arm struct {
		Isolation       isolation.Result `json:"gpu_isolation"`
		Path            string           `json:"path"`
		SHA256          string           `json:"sha256"`
		Completeness    string           `json:"completeness"`
		Stationarity    iteration.Gate   `json:"stationarity"`
		Markers         int              `json:"inferred_iteration_markers"`
		UnpairedMarkers int              `json:"unpaired_nvtx_markers"`
	}
	var arms [2]arm
	var records [2]timeline.Records
	for i, path := range fs.Args() {
		cap, digest, err := readAlignmentCapture(path, limits)
		if err != nil {
			return fmt.Errorf("shape-compare %s: %w", path, err)
		}
		_, records[i], err = decodeCompleteCapture(cap, *maxRecords, *journalBytes)
		if err != nil {
			return fmt.Errorf("shape-compare %s: %w", path, err)
		}
		iterations, err := iteration.Analyze(records[i], iteration.Options{Marker: re})
		if err != nil {
			iterations = &iteration.Report{Gate: iteration.Gate{Failures: []string{"stationarity unevaluable: " + err.Error()}}}
		}
		arms[i] = arm{captureIsolation(cap), path, digest, "passed", iterations.Gate, iterations.Markers, records[i].UnpairedMarkers}
	}
	report := struct {
		Version         int          `json:"version"`
		Domain          string       `json:"domain"`
		IterationMarker string       `json:"inferred_iteration_marker"`
		Arms            [2]arm       `json:"arms"`
		Shape           shape.Report `json:"shape_qualification"`
		Limits          string       `json:"limits"`
	}{1, *domain, *pattern, arms, shape.Compare(records[0], records[1], *domain), "explicit NVTX fields qualify shapes; kernel iteration boundaries are inferred and stationarity remains separate; launch duration distributions are not independent run samples; no causal speedup claimed"}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}
