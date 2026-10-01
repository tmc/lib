package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"io"
	"os"
	"regexp"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/traceanalysis"
)

type traceViewResult struct {
	Version        int               `json:"version"`
	Consumer       buildStamp        `json:"consumer"`
	Options        map[string]any    `json:"options,omitempty"`
	Schema         string            `json:"schema"`
	Input          string            `json:"input"`
	SHA256         string            `json:"sha256"`
	Kind           string            `json:"detected_type"`
	Backend        string            `json:"backend"`
	View           string            `json:"view"`
	State          string            `json:"state"`
	Qualifications map[string]string `json:"qualifications,omitempty"`
	Evidence       any               `json:"evidence"`
	Limits         []string          `json:"limits"`
}

func runTraceSummaryCommand(args []string) error { return runTraceSummary(args, os.Stdout) }
func runTraceSummary(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("trace summary", flag.ContinueOnError)
	fs.SetOutput(out)
	resourceLimits := registerTraceViewLimits(fs)
	jsonOutput := fs.Bool("json", false, "write versioned JSON")
	htmlPath := fs.String("html", "", "write standalone escaped HTML")
	shapePath := fs.String("shape", "", "bound shape receipt and original captures for comparison view")
	campaignPath := fs.String("campaign", "", "bound campaign receipt and retained controls for comparison view")
	view := fs.String("view", "overview", "overview (compact), detailed, tokens, operations, gaps, dependencies or comparison")
	top := fs.Int("top", 10, "maximum compact request, operation, shape and delta-group rows (1-100)")
	processor := fs.String("perfetto", "", "optional local Trace Processor v58.2 executable for exported trace summaries")
	domain := fs.String("domain", "driver", "submission API domain: driver or runtime")
	pattern := fs.String("iteration-marker", "", "raw kernel regex for separately inferred intervals")
	fs.Usage = func() {
		fmt.Fprintln(out, "Usage: cudaprof trace summary [-json] [-html FILE] [-view overview|detailed|tokens|operations|gaps|dependencies|comparison] [-domain driver|runtime] [-iteration-marker RE] INPUT")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if err := resourceLimits.validate(); err != nil {
		return err
	}
	out = &limitedExportWriter{Writer: out, remaining: resourceLimits.outputBytes}
	if fs.NArg() != 1 {
		return errors.New("usage: cudaprof trace summary [options] INPUT")
	}
	switch *view {
	case "overview", "detailed", "tokens", "operations", "gaps", "dependencies", "comparison":
	default:
		return fmt.Errorf("trace summary: invalid view %q", *view)
	}
	if *top < 1 || *top > 100 {
		return errors.New("trace summary: -top must be between 1 and 100")
	}
	if (*shapePath != "" || *campaignPath != "") && *view != "comparison" {
		return errors.New("trace summary: -shape and -campaign require -view comparison")
	}
	if *campaignPath != "" && *shapePath == "" {
		return errors.New("trace summary: -campaign requires -shape")
	}
	var inferred *regexp.Regexp
	var err error
	if *pattern != "" {
		inferred, err = regexp.Compile(*pattern)
		if err != nil {
			return fmt.Errorf("trace summary: iteration marker: %w", err)
		}
	}
	input, err := loadTraceSessionLimits(fs.Arg(0), *domain, resourceLimits.input)
	if err != nil {
		return err
	}
	if (*shapePath != "" || *campaignPath != "") && input.Kind != "comparison_archive" {
		return errors.New("trace summary: bound comparison evidence requires a comparison archive")
	}
	result := traceViewResult{Consumer: currentBuild(), Options: map[string]any{"resource_limits": resourceLimits.receipt(), "api_domain": *domain, "iteration_marker": *pattern, "view": *view, "top": *top, "shape_receipt": *shapePath, "campaign_receipt": *campaignPath}, Version: 1, Schema: "cudaprof.trace-summary.v1", Input: input.Path, SHA256: input.SHA256, Kind: input.Kind, Backend: "native", View: *view, State: "observed", Limits: []string{"collector, correctness, workload coverage, stationarity, shape and dependency completeness are independent", "native v1 markers lack request identity; native v2 request metrics require validated lifecycle endpoints and counters"}}
	var failure error
	var backendFailure error
	if input.Kind == "comparison_archive" && *view == "comparison" {
		var bound bool
		result.Evidence, bound, failure = traceComparisonView(input)
		result.State = "partial"
		result.Backend = "native_receipts"
		if !bound {
			result.State = "unavailable"
		} else if *shapePath != "" {
			result.Evidence, result.Qualifications, err = qualifiedTraceComparison(input, *shapePath, *campaignPath, resourceLimits.input)
			if err != nil {
				return err
			}
		}

	} else if input.Capture == nil {
		result.State = "unavailable"
		result.Backend = "trace_processor_v58.2"
		name := "inventory"
		if *view == "gaps" {
			name = "recorded-gaps"
		}
		if *view != "overview" && *view != "detailed" && *view != "gaps" {
			result.Evidence = map[string]string{"reason": "view requires native metadata/dependency/bound comparison receipts"}
			failure = errors.New("trace summary: native evidence unavailable")
		} else {
			sql, err := traceanalysis.SQL(name)
			if err != nil {
				return err
			}
			receipt, backendError := queryTraceProcessorLimits(input.Data, *processor, sql, resourceLimits.backend)
			result.Evidence = receipt
			result.Qualifications = map[string]string{"collector": "unavailable: unbound exported provenance", "correctness": "unverified", "stationarity": "unassessed", "dependency": "partial", "request_scope": "unavailable"}
			if backendError != nil {
				failure = backendError
				var unavailable *traceBackendUnavailable
				if !errors.As(backendError, &unavailable) {
					backendFailure = backendError
				}
			} else if len(receipt.Rows) <= 1 {
				failure = errors.New("trace summary: required role schema unavailable")
			} else {
				result.State = "partial"
				result.Limits = append(result.Limits, "exported overview is role inventory only; native token/operation views require retained metadata and bound receipts")
			}
		}
	} else {
		options := traceanalysis.Options{CollectorComplete: input.CollectorError == nil, Domain: *domain, IterationMarker: inferred}
		if input.CollectorError != nil {
			options.CollectorFailure = input.CollectorError.Error()
			failure = input.CollectorError
			result.State = "unavailable"
		}
		session, err := traceanalysis.New(input.Records, options)
		if err != nil {
			return err
		}
		summary := session.Summary()
		result.Qualifications = summary.Gates
		switch *view {
		case "overview":
			result.Version = 2
			result.Schema = "cudaprof.trace-summary.v2"
			var shapes traceanalysis.KernelShapes
			if raw, queryErr := session.Query("kernel-shapes"); queryErr == nil {
				shapes, _ = raw.(traceanalysis.KernelShapes)
			}
			result.Evidence = compactTraceSummary(summary, shapes, *top)
		case "detailed":
			result.Evidence = summary
		case "tokens":
			result.Evidence = struct {
				Observed  any `json:"observed_tokens"`
				Inferred  any `json:"inferred_intervals"`
				Endpoints any `json:"observed_endpoints"`
				Gates     any `json:"qualifications"`
			}{summary.Overview.Tokens, summary.Overview.Inferred, summary.Endpoints, summary.Gates}
		case "operations":
			result.Evidence = summary.Overview.Hierarchy
		case "gaps":
			result.Evidence = summary.Windows
		case "dependencies":
			result.Evidence = summary.Dependencies
		case "comparison":
			result.State = "unavailable"
			result.Evidence = map[string]string{"reason": "comparison requires two bound arm identities and qualified comparison/campaign receipts; individual capture is not a comparison"}
			failure = errors.New("trace summary: comparison evidence unavailable")
		}
	}
	if failure != nil {
		result.Limits = append(result.Limits, failure.Error())
	}
	if result.Qualifications == nil {
		result.Qualifications = map[string]string{}
	}
	if _, exists := result.Qualifications["isolation"]; !exists {
		result.Qualifications["isolation"] = "unchecked: no bound native observer receipt"
	}
	if input.Capture != nil {
		observed := captureIsolation(input.Capture)
		result.Qualifications["isolation"] = observed.State + ": " + strings.Join(observed.Reasons, "; ")
	}
	if *htmlPath != "" {
		if err := writeTraceSummaryHTML(*htmlPath, result, resourceLimits.outputBytes); err != nil {
			return err
		}
	}
	if err := writeTraceView(out, result, *jsonOutput); err != nil {
		return err
	}
	return backendFailure
}
func writeTraceView(out io.Writer, result traceViewResult, jsonOutput bool) error {
	if jsonOutput {
		encoder := json.NewEncoder(out)
		encoder.SetIndent("", "  ")
		return encoder.Encode(result)
	}
	if _, err := fmt.Fprintf(out, "%s v%d: %s\ninput: %s\nsha256: %s\nbackend: %s (%s)\n", result.Schema, result.Version, result.State, result.Input, result.SHA256, result.Backend, result.Kind); err != nil {
		return err
	}
	encoder := json.NewEncoder(out)
	encoder.SetIndent("", "  ")
	return encoder.Encode(result.Evidence)
}
func writeTraceSummaryHTML(path string, result traceViewResult, limits ...int64) error {
	outputBytes := defaultExportBytes
	if len(limits) > 0 {
		outputBytes = limits[0]
	}
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	page := template.Must(template.New("trace").Parse(`<!doctype html><html lang="en"><meta charset="utf-8"><title>cudaprof trace summary</title><style>body{font-family:system-ui;max-width:1100px;margin:2em auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:1.5em}</style><h1>{{.Result.View}} — {{.Result.State}}</h1><p>Input: {{.Result.Input}}<br>SHA256: {{.Result.SHA256}}</p><p>Collector completeness does not establish workload coverage, correctness, stationarity or dependency completeness. Raw source references remain in the evidence below.</p><pre>{{.JSON}}</pre></html>`))
	return atomicExport(path, []string{result.Input}, outputBytes, func(file io.Writer) error {
		return page.Execute(file, struct {
			Result traceViewResult
			JSON   string
		}{result, string(data)})
	})
}
