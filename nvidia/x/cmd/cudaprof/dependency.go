package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"io"
	"os"

	"github.com/tmc/lib/nvidia/x/cudaprof/dependency"
	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

type dependencyEndpoint struct {
	Record   string          `json:"record"`
	Journal  string          `json:"journal"`
	Process  uint32          `json:"process"`
	Thread   uint32          `json:"thread"`
	Time     int64           `json:"time_ns"`
	Metadata marker.Metadata `json:"metadata"`
	Status   string          `json:"status"`
	Reason   string          `json:"reason,omitempty"`
}

type dependencyResult struct {
	Version      int                    `json:"version"`
	Capture      string                 `json:"capture"`
	SHA256       string                 `json:"capture_sha256"`
	Completeness string                 `json:"collector_completeness"`
	Dependency   dependency.Report      `json:"dependency"`
	Submission   []timeline.Correlation `json:"submission_context"`
	Ranges       []timeline.Range       `json:"application_ranges"`
	Endpoints    []dependencyEndpoint   `json:"delivery_endpoint_observations"`
	Limits       string                 `json:"limits"`
}

// runDependencyCommand inventories retained nodes without manufacturing edges
// from timestamp adjacency, graph membership or submission correlations.
func runDependencyCommand(args []string) error {
	args, limits, err := parseReadLimits(args)
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet("dependency", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: cudaprof dependency [-html report.html] [-perfetto trace.pftrace] capture.pprof")
		fs.PrintDefaults()
	}
	perfettoPath := fs.String("perfetto", "", "write raw timeline and explicit dependency evidence to Perfetto")
	htmlPath := fs.String("html", "", "write standalone dependency inventory HTML")
	maxRecords := fs.Int64("max-records", 0, "journal record limit")
	journalBytes := fs.Int64("journal-bytes", 0, "decompressed journal byte limit")
	if err = fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 1 || *maxRecords < 0 || *journalBytes < 0 {
		return fmt.Errorf("usage: cudaprof dependency capture.pprof")
	}
	cap, sha, err := readAlignmentCapture(fs.Arg(0), limits)
	if err != nil {
		return err
	}
	status, records, err := decodeCompleteCapture(cap, *maxRecords, *journalBytes)
	if err != nil {
		return err
	}
	graph, endpoints := dependencyInventory(records)
	report, err := dependency.Analyze(graph)
	if err != nil {
		return err
	}
	result := dependencyResult{1, fs.Arg(0), sha, status.RunCompleteness, report, timeline.Correlations(records, "driver"), records.Ranges, endpoints, "submission context, delivery markers and application ranges are observations, not dependency edges; no complete critical path or causal attribution is established"}
	if *htmlPath != "" {
		f, err := os.Create(*htmlPath)
		if err != nil {
			return fmt.Errorf("dependency: create HTML: %w", err)
		}
		renderErr := writeDependencyHTML(f, result)
		if err := errors.Join(renderErr, f.Close()); err != nil {
			return fmt.Errorf("dependency: write HTML: %w", err)
		}
	}
	if *perfettoPath != "" {
		journals, err := requireJournals(cap)
		if err != nil {
			return err
		}
		recordsLimit, bytesLimit := decodeLimits(cap, *maxRecords, *journalBytes)
		raw, _, _, err := decodeTraceJournals(journals, recordsLimit, bytesLimit)
		if err != nil {
			return err
		}
		f, err := os.Create(*perfettoPath)
		if err != nil {
			return fmt.Errorf("dependency: create Perfetto: %w", err)
		}
		writeErr := writeDependencyTrace(f, raw, result)
		if err := errors.Join(writeErr, f.Close()); err != nil {
			return fmt.Errorf("dependency: write Perfetto: %w", err)
		}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}

func dependencyInventory(records timeline.Records) (dependency.Graph, []dependencyEndpoint) {
	graph := dependency.Graph{Nodes: []dependency.Node{}, Edges: []dependency.Edge{}, Missing: []string{"retained activity stream IDs do not establish an explicit submission sequence", "event handle identities and wait-to-record edges are not retained", "graph membership and replay submission links do not retain graph dependency edges", "a selected delivery start and host materialization dependencies are unverified"}}
	for i, a := range records.Activities {
		if a.End < a.Start || a.Start < 0 {
			graph.Missing = append(graph.Missing, fmt.Sprintf("invalid activity interval %d", i))
			continue
		}
		if a.Journal == "" {
			graph.Missing = append(graph.Missing, fmt.Sprintf("activity %d lacks journal clock identity", i))
			continue
		}
		record := fmt.Sprintf("%s:activity:%d", a.Journal, i)
		if a.Sequence != 0 {
			record = fmt.Sprintf("%s:record:%d", a.Journal, a.Sequence)
		}
		graph.Nodes = append(graph.Nodes, dependency.Node{ID: fmt.Sprintf("activity:%d", i), Record: record, Clock: a.Journal, Start: a.Start, End: a.End})
	}
	endpoints := []dependencyEndpoint{}
	observed := 0
	for i, m := range records.Markers {
		metadata, err := marker.ParseRange(m.Name)
		if err != nil || metadata.Event == "" {
			continue
		}
		endpoint := dependencyEndpoint{Record: fmt.Sprintf("%s:marker:%d", m.Journal, i), Journal: m.Journal, Process: m.Process, Thread: m.Thread, Time: m.Time, Metadata: metadata, Status: "observed"}
		if m.Journal == "" || m.Process == 0 || m.Thread == 0 || m.Time < 0 {
			endpoint.Status = "unsupported"
			endpoint.Reason = "marker lacks valid journal, process, thread or timestamp identity"
		} else if m.Flags != timeline.MarkerInstantaneous {
			endpoint.Status = "unsupported"
			endpoint.Reason = "delivery event is not an instantaneous marker"
		} else {
			observed++
		}
		endpoints = append(endpoints, endpoint)
	}
	if observed == 0 {
		graph.Missing = append(graph.Missing, "no scoped instantaneous delivery endpoint is retained")
	} else {
		graph.Missing = append(graph.Missing, "delivery observations do not establish token dependency completeness or a selected delivery endpoint")
	}
	return graph, endpoints
}

var dependencyHTML = template.Must(template.New("dependency").Parse(`<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Delivery dependency inventory</title>
<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccc;padding:.4rem;text-align:left;overflow-wrap:anywhere}code{overflow-wrap:anywhere}.status{padding:1rem;background:#fff2cc}</style>
<h1>Delivery dependency inventory</h1><p>Capture: <code>{{.Capture}}</code><br>SHA-256: <code>{{.SHA256}}</code><br>Collector completeness: {{.Completeness}}</p>
<p class="status">Dependency state: <strong>{{.Dependency.State}}</strong>. {{if .Dependency.CriticalPath}}Longest serial work path: {{.Dependency.WorkNS}} ns (excludes waiting gaps).{{else}}No complete critical path is established.{{end}}</p>
<p>{{.Limits}}</p><h2>Coverage and missing evidence</h2><ul>{{range .Dependency.Reasons}}<li>{{.}}</li>{{end}}</ul>
<h2>Nodes and raw sources</h2><table><tr><th>Node</th><th>Raw record</th><th>Clock identity</th><th>Start ns</th><th>End ns</th></tr>{{range .Dependency.Graph.Nodes}}<tr><td>{{.ID}}</td><td>{{.Record}}</td><td>{{.Clock}}</td><td>{{.Start}}</td><td>{{.End}}</td></tr>{{end}}</table>
<h2>Explicit dependency edges</h2><table><tr><th>From</th><th>To</th><th>Kind</th><th>Raw ordering evidence</th></tr>{{range .Dependency.Graph.Edges}}<tr><td>{{.From}}</td><td>{{.To}}</td><td>{{.Kind}}</td><td>{{.Record}}</td></tr>{{else}}<tr><td colspan="4">No explicit dependency edges are retained.</td></tr>{{end}}</table>
<h2>Delivery endpoint observations</h2><p>Endpoint observations alone do not establish dependency coverage.</p><table><tr><th>Raw marker</th><th>Event</th><th>Token</th><th>Time ns</th><th>Process / thread</th><th>Status</th></tr>{{range .Endpoints}}<tr><td>{{.Record}}</td><td>{{.Metadata.Event}}</td><td>{{.Metadata.Token}}</td><td>{{.Time}}</td><td>{{.Process}} / {{.Thread}}</td><td>{{.Status}} {{.Reason}}</td></tr>{{else}}<tr><td colspan="6">No delivery endpoint observations.</td></tr>{{end}}</table>
</html>`))

func writeDependencyHTML(w io.Writer, result dependencyResult) error {
	return dependencyHTML.Execute(w, result)
}
