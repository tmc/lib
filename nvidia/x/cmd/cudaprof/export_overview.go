package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

func traceOrigin(records []traceRecord) uint64 {
	origin := uint64(math.MaxUint64)
	for _, r := range records {
		if r.rec.Start < origin {
			origin = r.rec.Start
		}
	}
	if origin == math.MaxUint64 {
		return 0
	}
	return origin
}

func traceAnnotations(r traceRecord) []perfetto.DebugAnnotation {
	out := append([]perfetto.DebugAnnotation(nil), r.annotations...)
	out = append(out, perfetto.NewStringAnnotation("raw_journal", r.journal))
	if r.sequence != 0 {
		out = append(out, perfetto.NewUintAnnotation("raw_record_sequence", r.sequence))
	}
	return out
}

// writeOverviewTrace appends aggregate tracks without changing the raw tracks.
// Only retained interval unions become slices; summed kernel work is an argument.
func writeOverviewTrace(out io.Writer, report overview.Report, origin, base uint64) error {
	return writeOverviewTraceAt(out, report, origin, base, 0, 0)
}

func writeOverviewTraceAt(out io.Writer, report overview.Report, origin, base, shift uint64, clockID uint64) error {
	w := perfetto.NewWriter(out)
	w.TimestampClockID = clockID
	next := base
	descriptor := func(name string, parent uint64) (uint64, error) {
		next++
		return next, w.WriteTrackDescriptor(next, name, parent)
	}
	var emit func(*overview.Node, uint64) error
	emit = func(n *overview.Node, parent uint64) error {
		if n == nil {
			return nil
		}
		id, err := descriptor(n.Kind+": "+n.Label, parent)
		if err != nil {
			return err
		}
		refs, err := json.Marshal(n.References)
		if err != nil {
			return err
		}
		args := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("overview_id", n.ID), perfetto.NewStringAnnotation("provenance", n.Provenance), perfetto.NewStringAnnotation("raw_references", string(refs)), perfetto.NewIntAnnotation("summed_kernel_ns", n.Metrics.KernelSumNanos), perfetto.NewIntAnnotation("kernel_count", int64(n.Metrics.KernelCount))}
		intervals := func(label string, values []overview.Interval) error {
			tracks := map[string]uint64{}
			for _, v := range values {
				if v.Start < 0 || v.End < v.Start || uint64(v.Start) < origin {
					continue
				}
				track, ok := tracks[v.Journal]
				if !ok {
					track, err = descriptor(label+" ["+v.Journal+"]", id)
					if err != nil {
						return err
					}
					tracks[v.Journal] = track
				}
				if err := w.WriteSliceBeginWithAnnotations(track, uint64(v.Start)-origin+shift, label, "cudaprof_overview", args, nil); err != nil {
					return err
				}
				if err := w.WriteSliceEnd(track, uint64(v.End)-origin+shift); err != nil {
					return err
				}
			}
			return nil
		}
		if err := intervals("host scope union", n.Metrics.HostRanges); err != nil {
			return err
		}
		devices := make([]uint32, 0, len(n.Metrics.GPUUnion))
		for device := range n.Metrics.GPUUnion {
			devices = append(devices, device)
		}
		sort.Slice(devices, func(i, j int) bool { return devices[i] < devices[j] })
		for _, device := range devices {
			if err := intervals(fmt.Sprintf("GPU %d activity union", device), n.Metrics.GPUUnion[device]); err != nil {
				return err
			}
		}
		for _, child := range n.Children {
			if err := emit(child, id); err != nil {
				return err
			}
		}
		return nil
	}
	if err := emit(report.Hierarchy, 0); err != nil {
		return fmt.Errorf("write hierarchy: %w", err)
	}
	for _, token := range report.Tokens {
		if len(token.MaterializationRanges) > 0 {
			refs, _ := json.Marshal(token.Materialization)
			for _, interval := range token.MaterializationRanges {
				if interval.Start < 0 || uint64(interval.Start) < origin {
					continue
				}
				track, err := descriptor(fmt.Sprintf("token %d observed host materialization scope [%s]", token.Index, interval.Journal), base+1)
				if err != nil {
					return err
				}
				args := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("raw_references", string(refs)), perfetto.NewStringAnnotation("provenance", "observed host scope; GPU completion is not inferred")}
				if err := w.WriteSliceBeginWithAnnotations(track, uint64(interval.Start)-origin+shift, "host materialization scope", "cudaprof_overview", args, nil); err != nil {
					return err
				}
				if err := w.WriteSliceEnd(track, uint64(interval.End)-origin+shift); err != nil {
					return err
				}
			}
		}
		if token.Delivery == nil || *token.Delivery < 0 || uint64(*token.Delivery) < origin {
			continue
		}
		track, err := descriptor(fmt.Sprintf("delivered token %d [%s]", token.Index, token.Journal), base+1)
		if err != nil {
			return err
		}
		refs, _ := json.Marshal(token.References)
		args := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("overview_id", token.ID), perfetto.NewStringAnnotation("provenance", token.Provenance), perfetto.NewStringAnnotation("raw_references", string(refs)), perfetto.NewStringAnnotation("meaning", "observed delivery; not a causal path")}
		if err := w.WriteFlowInstant(track, uint64(*token.Delivery)-origin+shift, "observed delivered", "cudaprof_overview", args, nil, nil); err != nil {
			return err
		}
	}
	for _, token := range report.Inferred {
		refs, _ := json.Marshal(token.References)
		args := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("provenance", "inferred kernel boundary; not delivered/materialized"), perfetto.NewStringAnnotation("raw_references", string(refs))}
		track, err := descriptor(fmt.Sprintf("inferred kernel interval %d [%s]", token.Index, token.Journal), 0)
		if err != nil {
			return err
		}
		if token.InferredBoundary != nil {
			iv := *token.InferredBoundary
			if iv.Start < 0 || uint64(iv.Start) < origin {
				continue
			}
			if err := w.WriteSliceBeginWithAnnotations(track, uint64(iv.Start)-origin+shift, "inferred kernel interval", "cudaprof_inferred", args, nil); err != nil {
				return err
			}
			if err := w.WriteSliceEnd(track, uint64(iv.End)-origin+shift); err != nil {
				return err
			}
		}
	}
	return nil
}
