package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"io"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
)

type comparisonShapeReceipt struct {
	Version int    `json:"version"`
	Domain  string `json:"domain"`
	Arms    [2]struct {
		Path         string         `json:"path"`
		SHA256       string         `json:"sha256"`
		Completeness string         `json:"completeness"`
		Stationarity iteration.Gate `json:"stationarity"`
		Markers      int            `json:"inferred_iteration_markers"`
	} `json:"arms"`
	Shape shape.Report `json:"shape_qualification"`
}

type comparisonReference struct {
	CaptureSHA256 string `json:"capture_sha256"`
	Journal       string `json:"journal"`
	Activity      int    `json:"decoded_activity_index"`
	RawSymbol     string `json:"raw_symbol"`
	OriginalStart int64  `json:"original_start_ns"`
	AlignedStart  int64  `json:"aligned_start_ns"`
	Trace         string `json:"trace"`
	Lookup        string `json:"lookup"`
}

type comparisonOperation struct {
	Identity   shape.Identity           `json:"identity"`
	Base       shape.Distribution       `json:"base_launch_distribution_ns"`
	Target     shape.Distribution       `json:"target_launch_distribution_ns"`
	SumDelta   int64                    `json:"summed_kernel_delta_ns"`
	References [2][]comparisonReference `json:"references"`
}

type comparisonSummary struct {
	ShapeReceipt   comparisonShapeReceipt `json:"shape_receipt"`
	Version        int                    `json:"version"`
	ShapeQualified bool                   `json:"shape_qualified"`
	ShapeFailures  []string               `json:"shape_failures,omitempty"`
	Arms           [2]alignmentArm        `json:"arms"`
	Alignment      alignmentReceipt       `json:"alignment"`
	Operations     []comparisonOperation  `json:"equivalent_operations"`
	Unmatched      [2][]shape.Launch      `json:"unmatched"`
	Campaign       *campaignReport        `json:"independent_campaign,omitempty"`
	Limits         string                 `json:"limits"`
}

func makeComparisonSummary(s comparisonShapeReceipt, a alignmentReceipt, c *campaignReport) (comparisonSummary, error) {
	r := comparisonSummary{Version: 1, ShapeReceipt: s, ShapeQualified: s.Shape.Pass, ShapeFailures: s.Shape.Failures, Arms: a.Arms, Alignment: a, Operations: []comparisonOperation{}, Unmatched: [2][]shape.Launch{s.Shape.BaseUnmatched, s.Shape.TargetUnmatched}, Campaign: c,
		Limits: "Launch distributions describe the full decoded captures in the shape receipt, not the alignment-selected window; their counts are not independent timing samples. Summed kernel time is not an elapsed window or delivery ratio. Fixed offsets align independent runs without clock synchronization or time stretching. Decode activity indexes are capture-local, not Perfetto slice IDs. Locate exported slices by capture journal, raw symbol and aligned timestamp. Independent block statistics come only from the supplied campaign receipt; no new sample model is constructed. Correspondence of a separately supplied campaign to these captures is unverified and its interval is not assigned to an operation."}
	if s.Version != 1 || a.Version != 1 || a.TimeScale != 1 {
		return r, errors.New("comparison-summary: unsupported receipt version or time stretching")
	}
	for i := range 2 {
		if s.Arms[i].SHA256 == "" || s.Arms[i].SHA256 != a.Arms[i].SHA256 {
			return r, fmt.Errorf("comparison-summary: arm %d capture SHA differs between receipts", i)
		}
	}
	if c != nil && c.Version != 1 {
		return r, errors.New("comparison-summary: unsupported campaign receipt version")
	}
	if s.Shape.Pass {
		if len(s.Shape.Groups) == 0 || len(s.Shape.BaseUnmatched) != 0 || len(s.Shape.TargetUnmatched) != 0 {
			return r, errors.New("comparison-summary: inconsistent shape qualification")
		}
		for _, g := range s.Shape.Groups {
			if g.Identity.Phase == "" || g.Identity.Operation == "" || g.Identity.DType == "" || g.Identity.Token < 0 || g.Identity.CacheLength < 0 {
				return r, errors.New("comparison-summary: incomplete matching identity")
			}
			for _, dims := range [][3]int{g.Identity.Grid, g.Identity.Block} {
				for _, d := range dims {
					if d <= 0 {
						return r, errors.New("comparison-summary: unavailable launch geometry")
					}
				}
			}
			if g.BaseDurations.Count != len(g.Base) || g.TargetDurations.Count != len(g.Target) {
				return r, errors.New("comparison-summary: distribution count differs from retained launches")
			}
			identity, _ := json.Marshal(g.Identity)
			op := comparisonOperation{Identity: g.Identity, Base: g.BaseDurations, Target: g.TargetDurations, SumDelta: g.TargetDurations.Sum - g.BaseDurations.Sum}
			for arm, launches := range [][]shape.Launch{g.Base, g.Target} {
				if len(launches) == 0 {
					return r, errors.New("comparison-summary: empty matched operation")
				}
				for _, l := range launches {
					if l.Identity == nil || l.MetadataStatus != "matched" || l.APIStatus != "matched" || len(l.Missing) != 0 || l.Reason != "" || l.Activity < 0 || l.Journal == "" {
						return r, errors.New("comparison-summary: unqualified launch in matched operation")
					}
					launchIdentity, _ := json.Marshal(l.Identity)
					if string(launchIdentity) != string(identity) {
						return r, errors.New("comparison-summary: launch identity differs from matched group")
					}
					op.References[arm] = append(op.References[arm], comparisonReference{CaptureSHA256: a.Arms[arm].SHA256, Journal: l.Journal, Activity: l.Activity, RawSymbol: l.RawSymbol, OriginalStart: l.Start, AlignedStart: l.Start + a.Arms[arm].Offset, Trace: a.Arms[arm].Trace, Lookup: "match capture_journal, raw_symbol and aligned_start_ns; activity index is the timeline decoder's original index"})
				}
			}
			r.Operations = append(r.Operations, op)
		}
	}
	return r, nil
}

func readComparisonReceipt(path string, v any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	// Receipt files are bounded independently of attached captures.
	data, err := io.ReadAll(io.LimitReader(f, 64<<20+1))
	if err != nil {
		return err
	}
	if len(data) > 64<<20 {
		return errors.New("comparison-summary: receipt exceeds 64 MiB")
	}
	return json.Unmarshal(data, v)
}

func runComparisonSummaryCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("comparison-summary", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: cudaprof comparison-summary -shape shape.json -alignment .alignment.json [-campaign campaign.json] -o directory")
		fs.PrintDefaults()
	}
	fs.SetOutput(stdout)
	shapePath := fs.String("shape", "", "shape-compare JSON receipt")
	alignmentPath := fs.String("alignment", "", "alignment.json from export-compare archive")
	campaignPath := fs.String("campaign", "", "optional existing campaign.json receipt")
	out := fs.String("o", "", "new output directory")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if *shapePath == "" || *alignmentPath == "" || *out == "" || fs.NArg() != 0 {
		return errors.New("usage: cudaprof comparison-summary -shape shape.json -alignment alignment.json [-campaign campaign.json] -o directory")
	}
	var s comparisonShapeReceipt
	var a alignmentReceipt
	if err := readComparisonReceipt(*shapePath, &s); err != nil {
		return err
	}
	if err := readComparisonReceipt(*alignmentPath, &a); err != nil {
		return err
	}
	var c *campaignReport
	if *campaignPath != "" {
		c = new(campaignReport)
		if err := readComparisonReceipt(*campaignPath, c); err != nil {
			return err
		}
	}
	r, err := makeComparisonSummary(s, a, c)
	if err != nil {
		return err
	}
	if err := os.Mkdir(*out, 0700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*out, "summary.json"), append(data, '\n'), 0600); err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(*out, "summary.html"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	err = comparisonSummaryHTML.Execute(f, r)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	fmt.Fprintf(stdout, "%s\n%s\n", filepath.Join(*out, "summary.json"), filepath.Join(*out, "summary.html"))
	return nil
}

var comparisonSummaryHTML = template.Must(template.New("summary").Parse(`<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Operation comparison</title>
<style>body{font:16px system-ui;max-width:1150px;margin:2rem auto;padding:0 1rem;color:#202530}table{border-collapse:collapse;width:100%}td,th{padding:.6rem;border:1px solid #ccd1da;text-align:left}code{overflow-wrap:anywhere}summary{cursor:pointer}small{color:#515966}</style>
<h1>Operation comparison</h1><p>Shape qualification: <strong>{{.ShapeQualified}}</strong></p>{{range .ShapeFailures}}<p>{{.}}</p>{{end}}
<table><tr><th>Independent capture</th><th>Completeness</th><th>Workload coverage</th><th>Selected stationarity</th><th>Elapsed selected window (ns)</th><th>Fixed offset (ns)</th></tr>
{{range .Arms}}<tr><td>{{.Label}}<br><code>{{.SHA256}}</code><br><a href="{{.Trace}}">{{.Trace}}</a></td><td>{{.Completeness}}</td><td>{{.MarkerVerdict}}; observed {{.Markers}}, expected {{.Expected}}</td><td>{{.Stationarity.Pass}}{{range .Stationarity.Failures}}<br>{{.}}{{end}}</td><td>{{.Start}} to {{.End}}</td><td>{{.Offset}}</td></tr>{{end}}</table>
<h2>Full capture stationarity</h2>{{range .ShapeReceipt.Arms}}<p><code>{{.SHA256}}</code>: completeness {{.Completeness}}; stationarity {{.Stationarity.Pass}}{{range .Stationarity.Failures}}; {{.}}{{end}}</p>{{end}}
<h2>Equivalent operations</h2><p>Full decoded capture launch distributions in nanoseconds, distinct from the alignment-selected windows above. Kernel sums can overlap; they are separate from elapsed windows and campaign delivery.</p>
{{if .Operations}}<table><tr><th>Operation</th><th>Reference launches / min / median / max / sum</th><th>Target launches / min / median / max / sum</th><th>Kernel sum delta</th><th>Raw timeline references</th></tr>
{{range .Operations}}<tr><td>{{.Identity.Phase}} token {{.Identity.Token}}: {{.Identity.Operation}} ({{.Identity.DType}}, cache {{.Identity.CacheLength}})</td><td>{{.Base.Count}} / {{.Base.Min}} / {{.Base.Median}} / {{.Base.Max}} / {{.Base.Sum}}</td><td>{{.Target.Count}} / {{.Target.Min}} / {{.Target.Median}} / {{.Target.Max}} / {{.Target.Sum}}</td><td>{{.SumDelta}}</td><td><details><summary>Locate records</summary>{{range .References}}{{range .}}<p>Trace <a href="{{.Trace}}">{{.Trace}}</a><br>SHA <code>{{.CaptureSHA256}}</code><br>journal <code>{{.Journal}}</code>; decoded activity {{.Activity}}; aligned start {{.AlignedStart}} ns<br><code>{{.RawSymbol}}</code></p>{{end}}{{end}}</details></td></tr>{{end}}</table>{{else}}<p>No qualified numerical equivalent-operation comparison is available.</p>{{end}}
<h2>Unmatched work</h2>{{range $arm,$launches := .Unmatched}}<details><summary>Arm {{$arm}}: {{len $launches}} unmatched kernels</summary>{{range $launches}}<p><code>{{.RawSymbol}}</code><br>journal {{.Journal}}; decoded activity {{.Activity}}; original start {{.Start}} ns<br>{{.Reason}}{{range .Missing}}; {{.}}{{end}}</p>{{end}}</details>{{end}}
<h2>Independent block evidence</h2>{{if .Campaign}}<p>Separately supplied campaign receipt: {{len .Campaign.Blocks}} recorded blocks. Correspondence to these captures is unverified; this interval is not an operation estimate.</p>{{if .Campaign.Interval}}<p>{{.Campaign.Interval.Blocks}} qualified independent blocks; median delivery effect {{.Campaign.Interval.Median}}%; conditional 95% interval {{.Campaign.Interval.Lower}}% to {{.Campaign.Interval.Upper}}%.</p><p>{{.Campaign.Interval.Assumption}}</p>{{else}}<p>No qualified campaign interval.</p>{{end}}{{range .Campaign.Blocks}}{{if .Excluded}}<p>Block {{.Number}} excluded: {{range .Excluded}}{{.}}; {{end}}</p>{{end}}{{end}}{{else}}<p>No independent campaign receipt was supplied. Launch counts do not provide an independent-run uncertainty estimate.</p>{{end}}
<p><small>{{.Limits}}</small></p></html>`))
