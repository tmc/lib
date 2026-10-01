package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

func shiftedTimestamp(ts int64, origin, shift uint64) (uint64, bool) {
	if ts < 0 || uint64(ts) < origin || shift > math.MaxUint64-(uint64(ts)-origin) {
		return 0, false
	}
	return uint64(ts) - origin + shift, true
}

// Submission flows preserve identity, not completion-before-start dependencies.
// A call may fan out to several activities; each retained link stays distinct.
func writeSubmissionTrace(out io.Writer, records timeline.Records, domain string, origin, base, shift, clock uint64, identity traceExportIdentity) error {
	w := perfetto.NewWriter(out)
	w.TimestampClockID = clock
	next := base
	tracks := map[string]uint64{}
	track := func(key, label string) (uint64, error) {
		if id, ok := tracks[key]; ok {
			return id, nil
		}
		next++
		tracks[key] = next
		return next, w.WriteTrackDescriptor(next, label, 0)
	}
	type event struct {
		ts, track, flow uint64
		source          bool
		name            string
		args            []perfetto.DebugAnnotation
	}
	events := []event{}
	links := timeline.Correlations(records, domain)
	next++
	receiptTrack := next
	metadata := tracecontract.Track{UUID: receiptTrack, Name: "Submission provenance receipts", Role: "submission_receipt", Clock: identity.sha256 + ":" + identity.arm + ":export-relative", Arm: identity.arm, CaptureSHA256: identity.sha256, Provenance: "native domain-specific correlation results; no causal inference", Aggregation: "none"}
	if err := metadata.Write(w, shift); err != nil {
		return err
	}
	coverage := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("cudaprof_schema", "cudaprof.correlation_coverage.v1"), perfetto.NewUintAnnotation("cudaprof_schema_version", 1), perfetto.NewStringAnnotation("cudaprof_role", "submission_receipt"), perfetto.NewStringAnnotation("cudaprof_capture_sha256", identity.sha256), perfetto.NewStringAnnotation("cudaprof_arm", identity.arm), perfetto.NewStringAnnotation("domain", domain), perfetto.NewUintAnnotation("activities", uint64(len(links)))}
	if err := w.WriteFlowInstant(receiptTrack, shift, "Submission receipt coverage", "cudaprof_submission_receipt", coverage, nil, nil); err != nil {
		return err
	}
	for _, link := range links {
		activity := records.Activities[link.Activity]
		encoded, err := json.Marshal(link)
		if err != nil {
			return err
		}
		args := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("cudaprof_schema", "cudaprof.correlation.v1"), perfetto.NewUintAnnotation("cudaprof_schema_version", 1), perfetto.NewStringAnnotation("cudaprof_role", "submission_receipt"), perfetto.NewStringAnnotation("cudaprof_capture_sha256", identity.sha256), perfetto.NewStringAnnotation("cudaprof_arm", identity.arm), perfetto.NewStringAnnotation("raw_journal", activity.Journal), perfetto.NewUintAnnotation("raw_activity_sequence", activity.Sequence), perfetto.NewIntAnnotation("raw_activity_index", int64(link.Activity)), perfetto.NewStringAnnotation("domain", domain), perfetto.NewStringAnnotation("api_status", link.APIStatus), perfetto.NewStringAnnotation("range_status", link.RangeStatus), perfetto.NewStringAnnotation("metadata_status", link.MetadataStatus), perfetto.NewStringAnnotation("provenance", link.Provenance), perfetto.NewStringAnnotation("native_match_hex", hex.EncodeToString(encoded))}
		if err := w.WriteFlowInstant(receiptTrack, shift, "Native submission evidence", "cudaprof_submission_receipt", args, nil, nil); err != nil {
			return err
		}
	}
	missing := 0
	for _, link := range links {
		if link.APIStatus != "matched" {
			missing++
			continue
		}
		a, c := records.Activities[link.Activity], records.Calls[link.Call]
		apiTS, okAPI := shiftedTimestamp(c.Start, origin, shift)
		gpuTS, okGPU := shiftedTimestamp(a.Start, origin, shift)
		if !okAPI || !okGPU || c.Start > a.Start {
			missing++
			continue
		}
		host, err := track(fmt.Sprintf("host:%s:%d:%d", c.Journal, c.Process, c.Thread), fmt.Sprintf("submission identity: %s pid %d thread %d", c.Journal, c.Process, c.Thread))
		if err != nil {
			return err
		}
		device, err := track(fmt.Sprintf("device:%s:%d:%d", a.Journal, a.Device, a.Stream), fmt.Sprintf("submission identity: %s device %d stream %d", a.Journal, a.Device, a.Stream))
		if err != nil {
			return err
		}
		receipt, err := json.Marshal(link)
		if err != nil {
			return err
		}
		args := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("meaning", "exact submission identity; not a dependency edge or duration cause"), perfetto.NewStringAnnotation("correlation_receipt", string(receipt)), perfetto.NewStringAnnotation("raw_journal", a.Journal), perfetto.NewUintAnnotation("raw_activity_sequence", a.Sequence), perfetto.NewUintAnnotation("raw_api_sequence", c.Sequence), perfetto.NewIntAnnotation("raw_activity_index", int64(link.Activity)), perfetto.NewIntAnnotation("raw_api_index", int64(link.Call))}
		flow := base + uint64(link.Activity) + 1
		events = append(events, event{apiTS, host, flow, true, c.Name(), args}, event{gpuTS, device, flow, false, a.Name, args})
	}
	sort.SliceStable(events, func(i, j int) bool {
		if events[i].ts != events[j].ts {
			return events[i].ts < events[j].ts
		}
		return events[i].source && !events[j].source
	})
	for _, e := range events {
		var start, end []uint64
		if e.source {
			start = []uint64{e.flow}
		} else {
			end = []uint64{e.flow}
		}
		if err := w.WriteFlowInstant(e.track, e.ts, e.name, "cudaprof_submission_identity", e.args, start, end); err != nil {
			return err
		}
	}
	id, err := track("coverage", "Submission identity coverage")
	if err != nil {
		return err
	}
	return w.WriteFlowInstant(id, shift, "submission links retained; missing/unsupported stay unlinked", "cudaprof_submission_identity", []perfetto.DebugAnnotation{perfetto.NewIntAnnotation("linked_activities", int64(len(events)/2)), perfetto.NewIntAnnotation("unlinked_activities", int64(missing)), perfetto.NewStringAnnotation("domain", domain)}, nil, nil)
}

func lifecycleOrigin(records timeline.Records, origin uint64) uint64 {
	for _, e := range records.GraphLifecycle {
		if e.TimestampDomain == "monotonic" && e.Start >= 0 && uint64(e.Start) < origin {
			origin = uint64(e.Start)
		}
	}
	return origin
}

func writeGraphLifecycleTrace(out io.Writer, records timeline.Records, origin, base, shift, clock uint64) error {
	w := perfetto.NewWriter(out)
	w.TimestampClockID = clock
	links := timeline.GraphLifecycle(records.GraphLifecycle)
	for i, e := range records.GraphLifecycle {
		id := base + uint64(i) + 1
		if err := w.WriteTrackDescriptor(id, fmt.Sprintf("graph lifecycle: %s [%s]", e.Kind, e.Journal), 0); err != nil {
			return err
		}
		data, err := json.Marshal(struct {
			Event timeline.GraphEvent
			Link  timeline.GraphLink
		}{e, links[i]})
		if err != nil {
			return err
		}
		args := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("lifecycle_receipt", string(data)), perfetto.NewStringAnnotation("raw_journal", e.Journal), perfetto.NewUintAnnotation("raw_record_sequence", e.Sequence), perfetto.NewStringAnnotation("coverage", "observed lifecycle identity; mutation and dependency completeness must be separately established")}
		ts, valid := shiftedTimestamp(e.Start, origin, shift)
		if !valid || e.TimestampDomain != "monotonic" {
			ts = shift
			args = append(args, perfetto.NewStringAnnotation("timeline_availability", "unavailable: missing common clock or out-of-range timestamp"))
		}
		if err := w.WriteFlowInstant(id, ts, e.Kind, "cudaprof_graph_lifecycle", args, nil, nil); err != nil {
			return err
		}
	}
	id := base + uint64(len(records.GraphLifecycle)) + 1
	if err := w.WriteTrackDescriptor(id, "Graph lifecycle coverage", 0); err != nil {
		return err
	}
	return w.WriteFlowInstant(id, shift, "graph lifecycle coverage requires explicit qualification", "cudaprof_graph_lifecycle", []perfetto.DebugAnnotation{perfetto.NewIntAnnotation("retained_events", int64(len(records.GraphLifecycle))), perfetto.NewStringAnnotation("meaning", "graph membership alone does not identify construction, replay or dependencies")}, nil, nil)
}
