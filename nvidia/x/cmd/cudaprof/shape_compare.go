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
	"github.com/tmc/lib/nvidia/x/cudaprof/traceanalysis"
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
	scope := fs.String("scope", "model", "model, component or geometry; component requires explicit request mappings")
	requestPairs := fs.String("request-pairs", "", "JSON array declaring process/journal/request/role equivalence for v2/v3 metadata")
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
	if *scope != "model" && *scope != "component" && *scope != "geometry" {
		return fmt.Errorf("shape-compare: invalid scope %q", *scope)
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
	var pairs []shape.RequestPair
	if *requestPairs != "" {
		pairs, err = readShapeRequestPairs(*requestPairs)
		if err != nil {
			return err
		}
	}
	if *scope == "component" && len(pairs) == 0 {
		return fmt.Errorf("shape-compare: component scope requires -request-pairs")
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
		iterations := &iteration.Report{Gate: iteration.Gate{Failures: []string{"unassessed: non-model scope requires explicit component batch analysis"}}}
		err = nil
		if *scope == "model" {
			iterations, err = iteration.Analyze(records[i], iteration.Options{Marker: re})
		}
		if err != nil {
			iterations = &iteration.Report{Gate: iteration.Gate{Failures: []string{"stationarity unevaluable: " + err.Error()}}}
		}
		arms[i] = arm{captureIsolation(cap), path, digest, "passed", iterations.Gate, iterations.Markers, records[i].UnpairedMarkers}
	}
	if *scope != "model" {
		geometry := [2]any{}
		for i := range records {
			session, err := traceanalysis.New(records[i], traceanalysis.Options{CollectorComplete: true, Domain: *domain})
			if err != nil {
				return err
			}
			raw, err := session.Query("kernel-shapes")
			if err != nil {
				return err
			}
			shapes := raw.(traceanalysis.KernelShapes)
			geometry[i] = struct {
				Distributions          []traceanalysis.LaunchDistribution `json:"raw_launch_distributions"`
				ApplicationEquivalence string                             `json:"application_equivalence"`
				Grouping               string                             `json:"grouping"`
			}{shapes.Distributions, "unavailable from geometry alone", "journal/device/raw-symbol/grid/block; shared-memory grouping unavailable in this query"}
		}
		var component *shape.ComponentReport
		var regex any
		if *scope == "component" {
			c := shape.CompareComponents(records[0], records[1], *domain, pairs)
			component = &c
			regex = componentRegexDiagnostic(c, records, re)
		}
		schema := "cudaprof.component-shape-comparison.v1"
		if *scope == "geometry" {
			schema = "cudaprof.kernel-geometry-comparison.v1"
		}
		report := struct {
			Version   int                    `json:"version"`
			Schema    string                 `json:"schema"`
			Scope     string                 `json:"scope"`
			Domain    string                 `json:"domain"`
			Arms      [2]arm                 `json:"arms"`
			Component *shape.ComponentReport `json:"component_qualification,omitempty"`
			Geometry  [2]any                 `json:"observed_geometry"`
			Regex     any                    `json:"inferred_boundary_diagnostic,omitempty"`
			Limits    string                 `json:"limits"`
		}{2, schema, *scope, *domain, arms, component, geometry, regex, "geometry is diagnostic; component contracts and observed root coverage are separate from correctness, stationarity, completion and isolation; no causal timing or occurrence pairing"}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(report)
	}
	report := struct {
		Version         int          `json:"version"`
		Domain          string       `json:"domain"`
		IterationMarker string       `json:"inferred_iteration_marker"`
		Arms            [2]arm       `json:"arms"`
		Shape           shape.Report `json:"shape_qualification"`
		Limits          string       `json:"limits"`
	}{1, *domain, *pattern, arms, shape.CompareScoped(records[0], records[1], *domain, pairs), "explicit NVTX fields qualify shapes; kernel iteration boundaries are inferred and stationarity remains separate; launch duration distributions are not independent run samples; no causal speedup claimed"}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func readShapeRequestPairs(path string) ([]shape.RequestPair, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("shape-compare request pairs: %w", err)
	}
	data, readErr := io.ReadAll(io.LimitReader(f, 1<<20+1))
	if err := errors.Join(readErr, f.Close()); err != nil {
		return nil, fmt.Errorf("shape-compare request pairs: %w", err)
	}
	if len(data) > 1<<20 {
		return nil, errors.New("shape-compare request pairs exceed 1 MiB")
	}
	var pairs []shape.RequestPair
	if err := decodeCounterJSON(data, &pairs); err != nil {
		return nil, fmt.Errorf("shape-compare request pairs: %w", err)
	}
	return pairs, nil
}
