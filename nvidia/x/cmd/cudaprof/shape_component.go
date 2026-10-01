package main

import (
	"regexp"

	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

type componentBoundarySample struct {
	Journal    compactLabel `json:"journal"`
	Process    uint32       `json:"process"`
	Request    compactLabel `json:"request_id"`
	Invocation int          `json:"invocation"`
	Batch      int          `json:"batch"`
	Matches    int          `json:"regex_matches"`
}
type componentBoundaryDiagnostic struct {
	State            string                    `json:"state"`
	TotalMatches     int                       `json:"total_regex_matches"`
	BatchMatches     map[int]int               `json:"regex_matches_to_batch_count"`
	AmbiguousBatches int                       `json:"ambiguous_batches"`
	Samples          []componentBoundarySample `json:"samples"`
	OmittedSamples   int                       `json:"omitted_samples"`
}

func componentRegexDiagnostic(report shape.ComponentReport, records [2]timeline.Records, re *regexp.Regexp) [2]componentBoundaryDiagnostic {
	var out [2]componentBoundaryDiagnostic
	for arm := range out {
		d := &out[arm]
		d.State = "inferred: regex does not establish explicit workload boundaries"
		d.BatchMatches = map[int]int{}
		for _, a := range records[arm].Activities {
			if (a.Subtype == "" || a.Subtype == "kernel") && re.MatchString(a.Name) {
				d.TotalMatches++
			}
		}
		for _, b := range report.Arms[arm].Batches {
			matches := 0
			for _, i := range b.SubmissionActivities {
				if re.MatchString(records[arm].Activities[i].Name) {
					matches++
				}
			}
			d.BatchMatches[matches]++
			if matches > 1 {
				d.AmbiguousBatches++
				if len(d.Samples) < 4 {
					d.Samples = append(d.Samples, componentBoundarySample{compactText(b.Journal), b.Process, compactText(b.RequestID), b.Invocation, b.Ordinal, matches})
				}
			}
		}
		d.OmittedSamples = d.AmbiguousBatches - len(d.Samples)
		if d.AmbiguousBatches > 0 {
			d.State = "ambiguous: regex selects multiple launches per explicit batch; it cannot define component iterations"
		}
	}
	return out
}
