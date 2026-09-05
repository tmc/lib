package cuptiactivity

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

// WriteRecord writes a complete activity on a caller-defined Perfetto track.
// The caller must assign separate tracks to overlapping activities and define
// their descriptors. origin is a CUPTI timestamp shared by all records in the
// trace; subtracting it does not establish alignment with other clock domains.
// Use a standalone trace containing only records with that origin. This writer
// does not emit clock synchronization packets for merging with host traces.
// Incomplete records are rejected before writing any trace data.
func WriteRecord(w *perfetto.Writer, track uint64, origin uint64, r Record) error {
	if w == nil {
		return errors.New("cuptiactivity: nil trace writer")
	}
	if !r.Complete || r.Start == 0 || r.End < r.Start || r.Start < origin {
		return errors.New("cuptiactivity: invalid activity timestamps")
	}
	name := r.Name
	if name == "" {
		name = r.Kind.String()
	}
	annotations := []perfetto.DebugAnnotation{
		perfetto.NewStringAnnotation("timestamp_domain", "cupti-relative"),
		perfetto.NewUintAnnotation("timestamp_origin_ns", origin),
		perfetto.NewUintAnnotation("correlation_id", uint64(r.CorrelationID)),
		perfetto.NewUintAnnotation("device_id", uint64(r.DeviceID)),
		perfetto.NewUintAnnotation("context_id", uint64(r.ContextID)),
		perfetto.NewUintAnnotation("stream_id", uint64(r.StreamID)),
		perfetto.NewUintAnnotation("process_id", uint64(r.ProcessID)),
		perfetto.NewUintAnnotation("thread_id", uint64(r.ThreadID)),
		perfetto.NewUintAnnotation("callback_id", uint64(r.CallbackID)),
		perfetto.NewUintAnnotation("return_value", uint64(r.ReturnValue)),
		perfetto.NewUintAnnotation("bytes", r.Bytes),
	}
	if err := w.WriteSliceBeginWithAnnotations(track, r.Start-origin, name, "cupti", annotations, nil); err != nil {
		return fmt.Errorf("cuptiactivity: write activity start: %w", err)
	}
	if err := w.WriteSliceEnd(track, r.End-origin); err != nil {
		return fmt.Errorf("cuptiactivity: write activity end: %w", err)
	}
	return nil
}
