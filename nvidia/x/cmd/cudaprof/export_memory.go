package main

import (
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// writeMemoryTrace emits native UM service intervals and policy/query events.
// These tracks never represent occupancy or present physical residency.
func writeMemoryTrace(out io.Writer, records timeline.Records, origin, base, shift, clockID uint64) error {
	writer := perfetto.NewWriter(out)
	writer.TimestampClockID = clockID
	root := base + 1
	if err := writer.WriteTrackDescriptor(root, "memory evidence (not physical residency)", 0); err != nil {
		return err
	}
	next := root
	timestamp := func(value int64) (uint64, error) {
		if value < 0 || uint64(value) < origin || uint64(value)-origin > math.MaxUint64-shift {
			return 0, fmt.Errorf("memory trace: timestamp cannot be aligned")
		}
		return uint64(value) - origin + shift, nil
	}
	for _, event := range records.UnifiedMemory {
		if !event.Valid {
			continue
		}
		start, err := timestamp(event.Start)
		if err != nil {
			return err
		}
		end, err := timestamp(event.End)
		if err != nil {
			return err
		}
		next++
		if err = writer.WriteTrackDescriptor(next, fmt.Sprintf("UM kind %d process %d [%s]", event.CounterKind, event.Process, event.Journal), root); err != nil {
			return err
		}
		annotations := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("raw_journal", event.Journal), perfetto.NewUintAnnotation("raw_record_sequence", event.Sequence), perfetto.NewUintAnnotation("native_value", event.Value), perfetto.NewStringAnnotation("value_meaning", memoryCounterMeaning(event.CounterKind)), perfetto.NewUintAnnotation("virtual_address", event.Address), perfetto.NewUintAnnotation("raw_end", event.RawEnd), perfetto.NewUintAnnotation("flags", uint64(event.Flags)), perfetto.NewStringAnnotation("limits", "UM service evidence, not occupancy/current residency; native streamId invalid")}
		if event.SourceValid {
			annotations = append(annotations, perfetto.NewUintAnnotation("source_id", uint64(event.Source)))
		}
		if event.DestinationValid {
			annotations = append(annotations, perfetto.NewUintAnnotation("destination_id", uint64(event.Destination)))
		}
		name := fmt.Sprintf("UM counter %d", event.CounterKind)
		if !event.EndValid || end == start {
			if err = writer.WriteFlowInstant(next, start, name, "cudaprof_memory", annotations, nil, nil); err != nil {
				return err
			}
		} else {
			if err = writer.WriteSliceBeginWithAnnotations(next, start, name, "cudaprof_memory", annotations, nil); err != nil {
				return err
			}
			if err = writer.WriteSliceEnd(next, end); err != nil {
				return err
			}
		}
	}
	for _, event := range records.MemoryPolicies {
		// The callback uses the agent host clock, unlike native UM timestamps.
		// A missing clock-unification observation cannot be repaired by proximity.
		if event.TimestampDomain != "monotonic" {
			continue
		}
		at, err := timestamp(event.Time)
		if err != nil {
			return err
		}
		next++
		if err = writer.WriteTrackDescriptor(next, fmt.Sprintf("memory policy %s [%s]", event.Event, event.Journal), root); err != nil {
			return err
		}
		annotations := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("raw_journal", event.Journal), perfetto.NewUintAnnotation("raw_record_sequence", event.Sequence), perfetto.NewUintAnnotation("pointer", event.Pointer), perfetto.NewUintAnnotation("range_bytes", event.Bytes), perfetto.NewIntAnnotation("attribute", int64(event.Attribute)), perfetto.NewIntAnnotation("advice", int64(event.Advice)), perfetto.NewIntAnnotation("requested_location_type", int64(event.LocationType)), perfetto.NewIntAnnotation("requested_location_id", int64(event.LocationID)), perfetto.NewStringAnnotation("limits", "query/policy/prefetch request, not current physical residency or async completion")}
		if event.ResultKnown {
			annotations = append(annotations, perfetto.NewIntAnnotation("native_result", int64(event.Result)))
		}
		if event.Value != nil {
			annotations = append(annotations, perfetto.NewIntAnnotation("native_query_value", int64(*event.Value)))
		}
		if err = writer.WriteFlowInstant(next, at, event.Event, "cudaprof_memory_policy", annotations, nil, nil); err != nil {
			return err
		}
	}
	return nil
}

func memoryTraceOrigin(records timeline.Records, origin uint64) uint64 {
	for _, event := range records.UnifiedMemory {
		if event.Valid && event.Start >= 0 && uint64(event.Start) < origin {
			origin = uint64(event.Start)
		}
	}
	for _, event := range records.MemoryPolicies {
		if event.TimestampDomain == "monotonic" && event.Time >= 0 && uint64(event.Time) < origin {
			origin = uint64(event.Time)
		}
	}
	return origin
}
