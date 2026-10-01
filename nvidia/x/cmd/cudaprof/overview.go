package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"io"
	"os"
	"regexp"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

type overviewRawRecord struct {
	Sequence        uint64             `json:"raw_record_sequence,omitempty"`
	Reference       overview.Reference `json:"reference"`
	Name            string             `json:"name"`
	Start           int64              `json:"start_ns"`
	End             int64              `json:"end_ns"`
	Process, Thread uint32
}

type overviewResult struct {
	Isolation        isolation.Result       `json:"gpu_isolation"`
	Version          int                    `json:"version"`
	Capture          string                 `json:"capture"`
	SHA256           string                 `json:"capture_sha256"`
	Completeness     string                 `json:"collector_completeness"`
	CollectorReceipt *captureResult         `json:"collector_receipt,omitempty"`
	Stationarity     string                 `json:"stationarity"`
	Correctness      string                 `json:"correctness"`
	Domain           string                 `json:"api_domain"`
	Overview         overview.Report        `json:"overview"`
	Submission       []timeline.Correlation `json:"submission_context"`
	Raw              []overviewRawRecord    `json:"referenced_raw_records"`
}

func runOverviewCommand(args []string) error { return runOverview(args, os.Stdout) }

func runOverview(args []string, out io.Writer) error {
	args, limits, err := parseReadLimits(args)
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet("overview", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: cudaprof overview [-html overview.html] [-iteration-marker RE] capture.pprof")
		fs.PrintDefaults()
	}
	htmlPath := fs.String("html", "", "write standalone token and hierarchy HTML")
	domain := fs.String("domain", "driver", "submission API domain: driver or runtime")
	pattern := fs.String("iteration-marker", "", "raw kernel regex for separately labeled inferred intervals")
	maxRecords := fs.Int64("max-records", 0, "journal record limit")
	journalBytes := fs.Int64("journal-bytes", 0, "decompressed journal byte limit")
	if err = fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 1 || *maxRecords < 0 || *journalBytes < 0 || (*domain != "driver" && *domain != "runtime") {
		return fmt.Errorf("usage: cudaprof overview [-html FILE] [-iteration-marker RE] capture.pprof")
	}
	var inferred *regexp.Regexp
	if *pattern != "" {
		inferred, err = regexp.Compile(*pattern)
		if err != nil {
			return fmt.Errorf("overview: iteration marker: %w", err)
		}
	}
	cap, sha, err := readAlignmentCapture(fs.Arg(0), limits)
	if err != nil {
		return err
	}
	result := overviewResult{Isolation: captureIsolation(cap), Version: 1, Capture: fs.Arg(0), SHA256: sha, Completeness: "unknown", Stationarity: "unassessed", Correctness: "unverified", Domain: *domain}
	if data, ok := cap.Files["capture-status.json"]; ok {
		var envelope struct {
			Status captureResult `json:"status"`
		}
		if err = json.Unmarshal(data, &envelope); err != nil {
			return fmt.Errorf("overview: collector receipt: %w", err)
		}
		result.CollectorReceipt = &envelope.Status
		result.Completeness = envelope.Status.RunCompleteness
		if result.Completeness == "" {
			s := envelope.Status
			result.Completeness = runCompletenessOf(s.TransportReady, s.CollectorFinalized, s.UnqualifiedGenerations, s.DroppedRecords, s.ActivityRecords)
		}
	}
	journals, analysisErr := requireJournals(cap)
	var records timeline.Records
	if analysisErr == nil {
		*maxRecords, *journalBytes = decodeLimits(cap, *maxRecords, *journalBytes)
		records, analysisErr = timeline.Decode(journals, *maxRecords, *journalBytes)
		if analysisErr != nil {
			records = timeline.Records{}
			analysisErr = decodeRemedy(cap, analysisErr, *maxRecords, *journalBytes)
		}
	}
	result.Overview = overview.Analyze(records, *domain, inferred)
	result.Overview.Limits = append(result.Overview.Limits, "overview is diagnostic: collector status, target outcome, stationarity and correctness remain separate; missing journal records cannot be reconstructed")
	if analysisErr != nil {
		result.Overview.Limits = append(result.Overview.Limits, "timeline reconstruction refused: "+analysisErr.Error())
	}
	result.Submission = timeline.Correlations(records, *domain)
	result.Raw = overviewRaw(records, result.Overview)
	if *htmlPath != "" {
		f, e := os.Create(*htmlPath)
		if e != nil {
			return fmt.Errorf("overview: create HTML: %w", e)
		}
		e = errors.Join(writeOverviewHTML(f, result), f.Close())
		if e != nil {
			return fmt.Errorf("overview: write HTML: %w", e)
		}
	}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	return errors.Join(analysisErr, enc.Encode(result))
}

func overviewRaw(records timeline.Records, report overview.Report) []overviewRawRecord {
	refs := map[overview.Reference]bool{}
	var walk func(*overview.Node)
	walk = func(n *overview.Node) {
		for _, ref := range n.References {
			refs[ref] = true
		}
		for _, c := range n.Children {
			walk(c)
		}
	}
	walk(report.Hierarchy)
	for _, tokens := range [][]overview.Token{report.Tokens, report.Inferred} {
		for _, t := range tokens {
			for _, items := range [][]overview.Reference{t.References, t.Crossing, t.Materialization} {
				for _, ref := range items {
					refs[ref] = true
				}
			}
		}
	}
	out := []overviewRawRecord{}
	for ref := range refs {
		r := overviewRawRecord{Reference: ref}
		switch ref.Kind {
		case "activity":
			a := records.Activities[ref.Index]
			r.Sequence = a.Sequence
			r.Name = a.Name
			r.Start = a.Start
			r.End = a.End
		case "range":
			a := records.Ranges[ref.Index]
			r.Name = a.Name
			r.Start = a.Start
			r.End = a.End
			r.Process = a.Process
			r.Thread = a.Thread
		case "call":
			a := records.Calls[ref.Index]
			r.Sequence = a.Sequence
			r.Name = a.Name()
			r.Start = a.Start
			r.End = a.End
			r.Process = a.Process
			r.Thread = a.Thread
		case "marker":
			a := records.Markers[ref.Index]
			r.Sequence = a.Sequence
			r.Name = a.Name
			r.Start = a.Time
			r.End = a.Time
			r.Process = a.Process
			r.Thread = a.Thread
		}
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i].Reference, out[j].Reference
		if a.Journal != b.Journal {
			return a.Journal < b.Journal
		}
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		return a.Index < b.Index
	})
	return out
}
func overviewRefID(r overview.Reference) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s\x00%s\x00%d", r.Journal, r.Kind, r.Index)))
	return "raw-" + hex.EncodeToString(sum[:12])
}

var overviewHTML = template.Must(template.New("overview").Funcs(template.FuncMap{"refID": overviewRefID}).Parse(`<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Token and operation overview</title>
<style>body{font:15px system-ui;margin:2rem auto;max-width:1400px;padding:0 1rem}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccc;padding:.4rem;text-align:left;overflow-wrap:anywhere}details{margin:.5rem;padding:.4rem;border-left:2px solid #ccc}code{overflow-wrap:anywhere}.gates{background:#fff2cc;padding:1rem}</style>
<h1>Token and operation overview</h1><p>Capture: <code>{{.Capture}}</code><br>SHA-256: <code>{{.SHA256}}</code><br>Timestamp domain: {{.Overview.TimestampDomain}}</p>
<p class="gates">GPU isolation: <strong>{{.Isolation.State}}</strong>; {{.Isolation.Limits}}<br>Collector completeness: <strong>{{.Completeness}}</strong>; stationarity: {{.Stationarity}}; correctness: {{.Correctness}}. Summary generation establishes no causal attribution or workload qualification.</p>
{{with .CollectorReceipt}}<p>Target outcome: {{.Outcome}}; exit code: {{.TargetExitCode}}; signal: {{.TargetSignal}}; collector finalized: {{.CollectorFinalized}}; dropped records: {{.DroppedRecords}}.</p>{{else}}<p>No retained collector receipt.</p>{{end}}
<ul>{{range .Overview.Limits}}<li>{{.}}</li>{{end}}</ul><p>Unmatched activity records: {{len .Overview.Unmatched}}; unpaired markers: {{.Overview.UnpairedMarkers}}.</p>
<h2>Observed tokens</h2>{{template "tokens" .Overview.Tokens}}
<h2>Inferred kernel-marker intervals</h2><p>These are separate diagnostic intervals, with no observed delivery or materialization.</p>{{template "tokens" .Overview.Inferred}}
<h2>Phase → token → layer → operation → activity</h2><p>Each parent computes unique contributing records; do not sum parents and children. Activity union intervals remain separate by device and journal.</p>{{template "node" .Overview.Hierarchy}}
<h2>Referenced raw records</h2><table><tr><th>Journal / kind / index</th><th>Raw record sequence</th><th>Name / original metadata</th><th>Start ns</th><th>End ns</th><th>Process / thread</th></tr>{{range .Raw}}<tr id="{{refID .Reference}}"><td>{{.Reference.Journal}} / {{.Reference.Kind}} / {{.Reference.Index}}</td><td>{{if .Sequence}}{{.Sequence}}{{else}}unavailable for derived range{{end}}</td><td>{{.Name}}</td><td>{{.Start}}</td><td>{{.End}}</td><td>{{.Process}} / {{.Thread}}</td></tr>{{end}}</table>
{{define "refs"}}{{range .}}<a href="#{{refID .}}">{{.Journal}}:{{.Kind}}:{{.Index}}</a> {{end}}{{end}}
{{define "metrics"}}<p>Summed kernel work: {{.KernelSumNanos}} ns / {{.KernelCount}} kernels; sum of per-journal host unions: {{.HostUnionNanos}} ns. {{with .GPUSpan}}Single-device/journal GPU span: {{.Start}}–{{.End}} ns.{{else}}Single-device/journal GPU span unavailable.{{end}}</p><details><summary>Recorded activity unions and host intervals</summary>{{range $device,$spans:=.GPUUnion}}<p>Device {{$device}}: {{range $spans}}{{.Journal}} {{.Start}}–{{.End}} ns; {{end}}</p>{{end}}<p>Host scopes: {{range .HostRanges}}{{.Journal}} {{.Start}}–{{.End}} ns; {{end}}</p></details>{{end}}
{{define "node"}}<details><summary>{{.Kind}}: {{.Label}} [{{.Provenance}}]</summary>{{template "metrics" .Metrics}}<p>Sources: {{template "refs" .References}}</p>{{range .Children}}{{template "node" .}}{{end}}</details>{{end}}
{{define "tokens"}}{{range .}}<details><summary>{{.Journal}} {{.Phase}} token {{.Index}} [{{.Provenance}}]</summary>{{template "metrics" .Metrics}}<p>Delivered timestamp: {{with .Delivery}}{{.}}{{else}}unavailable{{end}}; host scope to delivery: {{with .DeliveryLatency}}{{.}} ns{{else}}unavailable{{end}}; termination: {{.EndReason}}</p><p>Unavailable: {{range .Availability}}{{.}}; {{end}}</p><p>Sources: {{template "refs" .References}}</p><p>Work crossing host/inferred boundary: {{template "refs" .Crossing}}</p><p>Observed host materialization-scope union (not GPU completion): {{range .MaterializationRanges}}{{.Journal}} {{.Start}}–{{.End}} ns; {{else}}unavailable{{end}}</p><p>Materialization sources: {{template "refs" .Materialization}}</p></details>{{else}}<p>No intervals with this provenance.</p>{{end}}{{end}}
</html>`))

func writeOverviewHTML(w io.Writer, result overviewResult) error {
	return overviewHTML.Execute(w, result)
}
