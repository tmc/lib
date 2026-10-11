package cudatrace

import (
	"fmt"
	"io"
	"math"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// ObservationClock identifies one source clock. Unmapped source clocks must be
// exported in separate files: a track name alone does not separate timelines.
// Domain and Epoch identify the clock, not a mapping to host or GPU time.
type ObservationClock struct{ Domain, Epoch string }

// ObservationReport accompanies an observation trace in a snapshot manifest.
// The trace is diagnostic: it cannot establish sensor timestamps, device
// durations, or a join to another clock. Keep the original capture alongside it.
type ObservationReport struct {
	Clock            ObservationClock `json:"clock"`
	JoinStatus       string           `json:"join_status"`
	Records          uint64           `json:"records"`
	LostRecords      uint64           `json:"lost_records"`
	IncompleteReason string           `json:"incomplete_reason,omitempty"`
}

func observationWriter(w io.Writer, clock ObservationClock, count, limit int) (*perfetto.Writer, error) {
	if w == nil || clock.Domain == "" || clock.Epoch == "" || len(clock.Domain) > 256 || len(clock.Epoch) > 256 || limit <= 0 || count > limit {
		return nil, fmt.Errorf("cudatrace: invalid observation writer, clock, or record budget")
	}
	p := perfetto.NewWriter(w)
	// Each output is an independent artifact. No clock snapshot silently maps
	// this custom source clock onto host monotonic or a CUDA event clock.
	p.TimestampClockID = 128
	return p, nil
}

// WriteHostObservations exports probe observations as host API instants. The
// decoder has no exit timestamp, so even a known API return remains an instant,
// never a host or device duration. limit bounds records and track cardinality.
// Use a separate writer/file from any uncalibrated telemetry or device trace.
func WriteHostObservations(w io.Writer, clock ObservationClock, events []cudaprobe.Event, lost uint64, limit int) (ObservationReport, error) {
	report := ObservationReport{Clock: clock, JoinStatus: "unmapped_separate_artifact", LostRecords: lost}
	p, err := observationWriter(w, clock, len(events), limit)
	if err != nil {
		return report, err
	}
	last := make(map[cudaprobe.Process]cudaprobe.Event)
	for _, e := range events {
		if prev, ok := last[e.Identity.Process]; ok && (e.Sequence <= prev.Sequence || e.Timestamp < prev.Timestamp) {
			return report, fmt.Errorf("cudatrace: decreasing host sequence or timestamp")
		}
		last[e.Identity.Process] = e
		if e.Identity.Process.PID == 0 || e.Identity.Process.Generation == 0 || e.Sequence == 0 || e.Identity.Symbol == "" || len(e.Identity.Symbol) > 256 || len(e.Reason) > 256 {
			return report, fmt.Errorf("cudatrace: invalid host observation identity or label")
		}
	}
	tracks := make(map[cudaprobe.Process]uint64)
	for _, e := range events {
		track := tracks[e.Identity.Process]
		if track == 0 {
			track = uint64(len(tracks) + 1)
			tracks[e.Identity.Process] = track
			name := fmt.Sprintf("Host API pid=%d generation=%d", e.Identity.Process.PID, e.Identity.Process.Generation)
			if err = p.WriteTrackDescriptor(track, name, 0); err != nil {
				return report, err
			}
		}
		annotations := []perfetto.DebugAnnotation{
			perfetto.NewUintAnnotation("sequence", e.Sequence), perfetto.NewBoolAnnotation("decoded", e.Decoded), perfetto.NewBoolAnnotation("return_known", e.ReturnKnown), perfetto.NewStringAnnotation("device_duration_status", "unavailable"), perfetto.NewStringAnnotation("clock_domain", clock.Domain), perfetto.NewStringAnnotation("clock_epoch", clock.Epoch),
		}
		if e.ReturnKnown {
			annotations = append(annotations, perfetto.NewIntAnnotation("api_status", int64(e.Status)))
		}
		if e.Reason != "" {
			annotations = append(annotations, perfetto.NewStringAnnotation("decode_reason", e.Reason))
		}
		if e.Copy != nil {
			annotations = append(annotations, perfetto.NewUintAnnotation("copy_bytes", e.Copy.Bytes))
		}
		if err = p.WriteInstantWithCallstack(track, e.Timestamp, e.Identity.Symbol, "host.cuda_api", annotations, 0); err != nil {
			return report, err
		}
		report.Records++
	}
	if lost > 0 {
		report.IncompleteReason = "source_loss"
	}
	return report, nil
}

// WriteTelemetryCounters exports numerical readings in their query clock.
// Query start is an observation placement, never a sensor timestamp. Unavailable
// and missing-status readings produce annotated instants rather than numerical
// counters. Values above MaxInt64 retain their exact integers in annotations. limit
// bounds the total readings and tracks, including unavailable readings.
// Preserve the returned report and original JSONL in the snapshot bundle.
// Each counter track is a diagnostic view. Viewers may extend its final value;
// that extension is not measured coverage. Tracks restart after unavailable,
// unrepresentable, omitted, or clock-discontinuous observations, preventing
// interpolation between values across those boundaries.
func WriteTelemetryCounters(w io.Writer, manifest nvtelemetry.Manifest, samples []nvtelemetry.Sample, limit int) (ObservationReport, error) {
	clock := ObservationClock{manifest.ClockMonotonicDomain, manifest.ClockMonotonicEpochID}
	report := ObservationReport{Clock: clock, JoinStatus: "unmapped_separate_artifact", LostRecords: manifest.LostRecords, IncompleteReason: manifest.IncompleteReason}
	if limit <= 0 || len(samples) > limit || len(manifest.Devices) > limit {
		return report, fmt.Errorf("cudatrace: telemetry input budget exceeded")
	}
	count := 0
	var previousStart int64
	var previousSeq uint64
	devices := make(map[string]bool)
	for _, d := range manifest.Devices {
		if d.UUID == "" || len(d.UUID) > 256 || devices[d.UUID] {
			return report, fmt.Errorf("cudatrace: invalid telemetry device identity")
		}
		devices[d.UUID] = true
	}
	for i, s := range samples {
		if i > 0 && s.QueryStartNanos < previousStart {
			return report, fmt.Errorf("cudatrace: decreasing sample timestamp")
		}
		previousStart = s.QueryStartNanos
		if s.Seq != 0 {
			if s.Seq <= previousSeq {
				return report, fmt.Errorf("cudatrace: nonincreasing sample sequence")
			}
			previousSeq = s.Seq
		}
		if len(s.Readings) > limit-count {
			return report, fmt.Errorf("cudatrace: telemetry record budget exceeded")
		}
		count += len(s.Readings)
		if s.QueryStartNanos < 0 || s.QueryEndNanos < s.QueryStartNanos {
			return report, fmt.Errorf("cudatrace: invalid telemetry sample bounds")
		}
		for _, r := range s.Readings {
			if _, ok := nvtelemetry.LookupMetricKind(r.Name); !ok {
				return report, fmt.Errorf("cudatrace: unknown metric kind %q", r.Name)
			}
			if !devices[r.DeviceUUID] || r.Name == "" || r.Unit == "" || len(r.Name) > 256 || len(r.Unit) > 256 || len(r.UnavailableReason) > 256 {
				return report, fmt.Errorf("cudatrace: invalid telemetry identity or label")
			}
			if unit, ok := nvtelemetry.LookupMetricUnit(r.Name); ok && r.Unit != unit {
				return report, fmt.Errorf("cudatrace: incorrect canonical unit for %q", r.Name)
			}
			switch r.Evidence {
			case "", nvtelemetry.EvidenceUnavailable:
			case nvtelemetry.EvidenceMeasured, nvtelemetry.EvidenceDerived:
				if r.UnavailableReason != "" {
					return report, fmt.Errorf("cudatrace: active evidence with unavailable reason")
				}
			default:
				return report, fmt.Errorf("cudatrace: unknown reading evidence")
			}
			if r.HasQueryBounds && (r.QueryStartNanos < s.QueryStartNanos || r.QueryEndNanos < r.QueryStartNanos || r.QueryEndNanos > s.QueryEndNanos) {
				return report, fmt.Errorf("cudatrace: reading outside sample query bounds")
			}
			if kind, _ := nvtelemetry.LookupMetricKind(r.Name); r.IsAvailable() && (kind == nvtelemetry.KindBitmask || kind == nvtelemetry.KindEvent) && !r.IsInt {
				return report, fmt.Errorf("cudatrace: discrete metric requires integer value")
			}
			if r.IsAvailable() && !r.IsInt && (math.IsNaN(r.Value) || math.IsInf(r.Value, 0)) {
				return report, fmt.Errorf("cudatrace: nonfinite telemetry value")
			}
		}
	}
	p, err := observationWriter(w, clock, count, limit)
	if err != nil {
		return report, err
	}
	type key struct{ device, metric, unit string }
	tracks := make(map[key]uint64)
	lastSample := make(map[key]int)
	var nextTrack uint64
	for i, s := range samples {
		if s.ClockDiscontinuity {
			clear(tracks)
			clear(lastSample)
		}
		for _, r := range s.Readings {
			k := key{r.DeviceUUID, r.Name, r.Unit}
			track := tracks[k]
			if track != 0 && lastSample[k] != i-1 && lastSample[k] != i {
				track = 0
			}
			lastSample[k] = i
			if track == 0 {
				nextTrack++
				track = nextTrack
				tracks[k] = track
				if err = p.WriteCounterDescriptor(track, k.device+" "+k.metric, k.unit, 0); err != nil {
					return report, err
				}
			}
			ts := uint64(s.QueryStartNanos)
			if r.HasQueryBounds {
				ts = uint64(r.QueryStartNanos)
			}
			annotations := []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("timestamp_semantics", "query_start_not_sensor"), perfetto.NewStringAnnotation("clock_domain", clock.Domain), perfetto.NewStringAnnotation("clock_epoch", clock.Epoch), perfetto.NewStringAnnotation("evidence", r.Evidence)}
			name := "reading_metadata"
			if r.Evidence == "" {
				name = "unknown"
				annotations = append(annotations, perfetto.NewStringAnnotation("reason", r.UnavailableReason))
			} else if !r.IsAvailable() {
				name = "unavailable"
				annotations = append(annotations, perfetto.NewStringAnnotation("reason", r.UnavailableReason))
			} else if r.IsInt && r.IntValue > math.MaxInt64 {
				name = "unsigned_counter_out_of_range"
				annotations = append(annotations, perfetto.NewUintAnnotation("exact_uint64", r.IntValue), perfetto.NewStringAnnotation("exact_decimal", strconv.FormatUint(r.IntValue, 10)))
			} else if kind, ok := nvtelemetry.LookupMetricKind(r.Name); ok && (kind == nvtelemetry.KindBitmask || kind == nvtelemetry.KindEvent) {
				name = "discrete_reading"
				annotations = append(annotations, perfetto.NewUintAnnotation("exact_uint64", r.IntValue))
			} else if r.IsInt {
				err = p.WriteCounterValue(track, ts, int64(r.IntValue))
			} else {
				err = p.WriteDoubleCounterValue(track, ts, r.Value)
			}
			if err != nil {
				return report, err
			}
			if err = p.WriteInstantWithCallstack(track, ts, name, "telemetry.status", annotations, 0); err != nil {
				return report, err
			}
			if name != "reading_metadata" {
				delete(tracks, k)
				delete(lastSample, k)
			}
			report.Records++
		}
	}
	if report.LostRecords > 0 && report.IncompleteReason == "" {
		report.IncompleteReason = "source_loss"
	}
	return report, nil
}
