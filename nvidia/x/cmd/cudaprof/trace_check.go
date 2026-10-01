package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
)

type traceEvidence struct {
	Disposition string `json:"disposition"`
	Reason      string `json:"reason"`
	Provenance  string `json:"provenance"`
}

type traceCheckReport struct {
	Schema    string                   `json:"schema"`
	Consumer  buildStamp               `json:"consumer_build"`
	Options   map[string]any           `json:"normalized_options"`
	Processor *traceProcessorReceipt   `json:"trace_processor,omitempty"`
	Version   int                      `json:"version"`
	Command   string                   `json:"command"`
	Input     string                   `json:"input"`
	SHA256    string                   `json:"sha256"`
	Kind      string                   `json:"detected_type"`
	Backend   string                   `json:"backend"`
	Required  []string                 `json:"required"`
	Checks    map[string]traceEvidence `json:"checks"`
	Qualified bool                     `json:"requested_checks_passed"`
	Limits    map[string]int64         `json:"limits"`
}

func runTraceCheckCommand(args []string) error { return runTraceCheck(args, os.Stdout) }

func runTraceCheck(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("trace check", flag.ContinueOnError)
	fs.SetOutput(out)
	fs.Usage = func() {
		for _, c := range traceCommands() {
			if c.name == "check" {
				traceHelp(c, "trace check")
			}
		}
		fs.PrintDefaults()
	}
	jsonOutput := fs.Bool("json", false, "write one versioned JSON receipt")
	required := fs.String("require", "", "additional checks: integrity,collector,import,coverage,stationarity,shape,correctness,isolation,reconciliation")
	marker := fs.String("iteration-marker", "", "recorded marker kernel expression")
	expect := fs.Int("expect-markers", 0, "required marker count for independent workload coverage")
	domain := fs.String("domain", "driver", "submission API domain: driver or runtime")
	inputBytes := fs.Int64("input-bytes", 512<<20, "maximum input bytes")
	journalBytes := fs.Int64("journal-bytes", 256<<20, "maximum decoded journal bytes")
	maxRecords := fs.Int64("max-records", 1<<20, "maximum decoded records per journal")
	archiveBytes := fs.Int64("archive-bytes", 512<<20, "maximum expanded archive bytes")
	archiveMembers := fs.Int("archive-members", 4096, "maximum archive members")
	backendTimeout := fs.Duration("backend-timeout", 30*time.Second, "Trace Processor timeout per invocation")
	backendOutput := fs.Int("backend-output-bytes", 4<<20, "maximum Trace Processor stdout/stderr bytes each")
	backendRows := fs.Int("backend-rows", 10000, "maximum Trace Processor result rows")
	shapeReceipt := fs.String("shape-receipt", "", "bound shape-compare receipt; original captures are revalidated")
	correctnessReceipt := fs.String("correctness-receipt", "", "campaign receipt with retained stdout and capture artifacts")
	processor := fs.String("perfetto", "", "explicit optional Trace Processor executable (import backend)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if *backendTimeout <= 0 || *backendOutput <= 0 || *backendRows <= 0 {
		return errors.New("trace check: backend resource limits must be positive")
	}
	if fs.NArg() != 1 {
		return errors.New("usage: cudaprof trace check [options] INPUT")
	}
	if *expect < 0 || (*expect > 0 && *marker == "") {
		return errors.New("trace check: -expect-markers requires -iteration-marker and a nonnegative count")
	}
	var re *regexp.Regexp
	var err error
	if *marker != "" {
		re, err = regexp.Compile(*marker)
		if err != nil {
			return fmt.Errorf("trace check: marker: %w", err)
		}
	}
	names := []string{"integrity", "collector", "import", "coverage", "stationarity", "shape", "correctness", "isolation", "reconciliation"}
	wants := []string{"integrity", "collector"}
	for _, name := range strings.Split(*required, ",") {
		if name == "" {
			continue
		}
		found := false
		for _, known := range names {
			if name == known {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("trace check: unknown check %q", name)
		}
		wants = append(wants, name)
	}
	limits := traceInputLimits{InputBytes: *inputBytes, JournalBytes: *journalBytes, Records: *maxRecords, ArchiveBytes: *archiveBytes, ArchiveMembers: *archiveMembers}
	s, err := loadTraceSessionLimits(fs.Arg(0), *domain, limits)
	if err != nil {
		return err
	}
	r := traceCheckReport{Schema: "cudaprof.trace.check.v1", Consumer: currentBuild(), Options: map[string]any{"api_domain": *domain, "iteration_marker": *marker, "expected_markers": *expect, "perfetto": *processor}, Version: 1, Command: "trace check", Input: s.Path, SHA256: s.SHA256, Kind: s.Kind, Backend: "native", Required: wants, Checks: make(map[string]traceEvidence), Limits: map[string]int64{"input_bytes": *inputBytes, "journal_records": *maxRecords, "journal_bytes": *journalBytes, "archive_bytes": *archiveBytes, "archive_members": int64(*archiveMembers), "backend_timeout_ns": int64(*backendTimeout), "backend_output_bytes": int64(*backendOutput), "backend_rows": int64(*backendRows)}}
	for _, name := range names {
		r.Checks[name] = traceEvidence{"unavailable", "no supported independent evidence in this input", "native input"}
	}
	if s.Capture != nil {
		observed := captureIsolation(s.Capture)
		state := "unavailable"
		if observed.State == "clear" {
			state = "passed"
		}
		if observed.State == "contaminated" {
			state = "failed"
			wants = append(wants, "isolation")
			r.Required = wants
		}
		r.Checks["isolation"] = traceEvidence{state, observed.State + ": " + strings.Join(observed.Reasons, "; "), "hash-bound sampled observer receipt"}
	}
	r.Checks["integrity"] = traceEvidence{"passed", "bounded input container parsed", "input bytes sha256"}
	if s.CollectorError == nil {
		r.Checks["collector"] = traceEvidence{"passed", "complete collector status and finalized journals", "capture-status.json and journals"}
	} else {
		r.Checks["collector"] = traceEvidence{"failed", s.CollectorError.Error(), "capture-status.json and journals"}
	}
	r.Checks["import"] = traceEvidence{"unchecked", "native capture checks do not require a Trace Processor", "native backend"}
	if *processor != "" {
		r.Checks["import"] = traceEvidence{"unavailable", "external import adapter is not available for native capture input; export the capture first", "explicit -perfetto"}
	}
	if re != nil && s.CollectorError == nil {
		report, err := iteration.Analyze(s.Records, iteration.Options{Marker: re})
		if err != nil {
			r.Checks["coverage"] = traceEvidence{"failed", err.Error(), "recorded marker kernel"}
			r.Checks["stationarity"] = traceEvidence{"unavailable", err.Error(), "native iteration analyzer"}
		} else {
			if *expect > 0 {
				state := "passed"
				if report.Markers != *expect {
					state = "failed"
				}
				r.Checks["coverage"] = traceEvidence{state, fmt.Sprintf("observed %d markers; expected %d", report.Markers, *expect), "recorded marker invariant"}
			}
			state := "passed"
			reason := "native iteration gate passed"
			if !report.Gate.Pass {
				state = "failed"
				reason = strings.Join(report.Gate.Failures, "; ")
			}
			r.Checks["stationarity"] = traceEvidence{state, reason, "native iteration gate"}
		}
	}

	var invocationError error
	if s.Kind != "capture" {
		wants = []string{"integrity", "import"}
		for _, name := range strings.Split(*required, ",") {
			if name != "" {
				wants = append(wants, name)
			}
		}
		r.Required = wants
		r.Backend = "trace_processor"
		r.Checks["collector"] = traceEvidence{"unavailable", "standalone trace has no bound collector receipt", "external input"}
		sql := "SELECT name, severity, value FROM stats WHERE value > 0 AND severity IN ('error', 'data_loss') ORDER BY name"
		backend, backendErr := queryTraceProcessorLimits(s.Data, *processor, sql, traceProcessorLimits{*backendTimeout, *backendOutput, *backendRows})
		r.Processor = &backend
		if backendErr == nil {
			backendErr = validateTraceErrorStats(backend.Rows)
		}
		if backendErr != nil {
			var unavailable *traceBackendUnavailable
			state := "unavailable"
			if !errors.As(backendErr, &unavailable) {
				invocationError = backendErr
				state = "failed"
			}
			r.Checks["import"] = traceEvidence{state, backendErr.Error(), "Trace Processor v58.2 adapter"}
		} else {
			state := "passed"
			reason := "no positive importer error or data-loss statistics; collector completeness remains unavailable"
			if len(backend.Rows) > 1 {
				state = "failed"
				reason = fmt.Sprintf("%d positive importer error or data-loss statistics", len(backend.Rows)-1)
			}
			r.Checks["import"] = traceEvidence{state, reason, "Trace Processor stats"}
		}
	}
	if s.Kind != "capture" && r.Checks["import"].Disposition == "passed" {
		evidence, err := reconcileTraceImport(s.Data, *processor, traceProcessorLimits{*backendTimeout, *backendOutput, *backendRows})
		if err != nil {
			invocationError = err
			evidence = traceEvidence{"failed", err.Error(), "Trace Processor reconciliation adapter"}
		}
		r.Checks["reconciliation"] = evidence
		if evidence.Disposition == "failed" {
			wants = append(wants, "reconciliation")
			r.Required = wants
		}
	}
	if *shapeReceipt != "" {
		e, err := checkTraceShape(s, *shapeReceipt, limits)
		if err != nil {
			return err
		}
		r.Checks["shape"] = e
	}
	if *correctnessReceipt != "" {
		e, err := checkTraceCorrectness(s, *correctnessReceipt, limits)
		if err != nil {
			return err
		}
		r.Checks["correctness"] = e
	}
	seen := map[string]bool{}
	unique := make([]string, 0, len(wants))
	for _, name := range wants {
		if !seen[name] {
			seen[name] = true
			unique = append(unique, name)
		}
	}
	wants = unique
	r.Required = wants
	r.Qualified = true
	for _, name := range wants {
		if r.Checks[name].Disposition != "passed" {
			r.Qualified = false
		}
	}
	if *jsonOutput {
		err = json.NewEncoder(out).Encode(r)
	} else {
		for _, name := range names {
			e := r.Checks[name]
			if _, err = fmt.Fprintf(out, "%s: %s (%s)\n", name, e.Disposition, e.Reason); err != nil {
				break
			}
		}
	}
	if err != nil {
		return err
	}
	if invocationError != nil {
		return invocationError
	}
	if !r.Qualified {
		return &exitCodeError{Code: 2, Err: errors.New("trace check: requested evidence failed or unavailable")}
	}
	return nil
}

func validateTraceErrorStats(rows [][]string) error {
	if len(rows) == 0 || len(rows[0]) != 3 || rows[0][0] != "name" || rows[0][1] != "severity" || rows[0][2] != "value" {
		return errors.New("trace processor returned incompatible error-statistics schema")
	}
	for _, row := range rows[1:] {
		if len(row) != 3 || (row[1] != "error" && row[1] != "data_loss") {
			return errors.New("trace processor returned invalid statistics row")
		}
		value, err := strconv.ParseInt(row[2], 10, 64)
		if err != nil || value <= 0 {
			return errors.New("trace processor returned invalid statistics value")
		}
	}
	return nil
}
