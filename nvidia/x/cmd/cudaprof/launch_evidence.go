package main

import "github.com/tmc/lib/nvidia/x/cudaevidence"

// classifyLaunchBlockX retains a directly recorded launch dimension only when
// the record is joined to an exact image/entry and admitted workload. Current
// metadata-only manifests have no such join and deliberately remain unread.
func classifyLaunchBlockX(r launchManifestRecord, source *cudaevidence.Source) (cudaevidence.Value, error) {
	record := cudaevidence.Record{Unit: "threads", Reason: "launch artifact, compiler or workload identity unavailable"}
	if source != nil {
		record.Source = *source
		if err := source.Identity.Validate(); err != nil {
			return cudaevidence.Value{}, err
		}
		if r.Module == source.Identity.ArtifactDigest && r.Function == source.Identity.Entry && r.ActivityCorrelation == r.Correlation && r.ActivityCorrelation != 0 && r.Block[0] > 0 {
			number := float64(r.Block[0])
			record.Number = &number
			record.Class = cudaevidence.Measured
			record.Basis = "CUPTI kernel activity block dimension joined by correlation and image identity"
			record.Reason = ""
		} else {
			record.Reason = "launch dimension or exact module/entry correlation join unavailable"
		}
	}
	return cudaevidence.New(record)
}
