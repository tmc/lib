package cudadiag

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"reflect"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// TelemetryOptions supplies independently retained admission inputs. Neither
// a stream's provenance labels nor matching raw bytes authenticate a producer.
// ExpectedSources pins producer, class, schema, transform and observer metadata
// by raw SHA256. Receipt is the independent capture receipt, not the stream's
// copy. RawArtifacts contains its retained source bytes. The zero value cannot
// admit an exact diagnosis. Callers must not mutate options during the call.
type TelemetryOptions struct {
	MaxBytes           int64
	Workload           cudaworkload.WorkloadIdentity
	OpID               cudaworkload.OpID
	Assignment         nvtelemetry.EntityAssignment
	ClockEpoch         string
	Observer           nvtelemetry.Observer
	Receipt            *tracecontract.Receipt
	RawArtifacts       map[string][]byte
	ExpectedSources    map[string]nvtelemetry.Observation
	HistogramBounds    map[string][]string
	RequireDenominator []string
	WallInterval       tracecontract.Interval
	Thresholds         Thresholds
}

// TelemetryFinding retains an exact canonical reading alongside its admission.
// Integer values remain in Reading.Number; Finding.Value is never a lossy
// float conversion of an integer. Unavailable findings retain their reading.
// Admission applies only to the retained reading and its evidence links, not
// to producer population completeness or known-zero producer loss. Diagnosis
// applies its independent population requirements separately.
type TelemetryFinding struct {
	Reading nvtelemetry.Reading   `json:"reading"`
	Finding tracecontract.Finding `json:"finding"`
}

// TelemetryDiagnosis contains a mixed stream's admitted observations and the
// existing host/device diagnosis. Synthetic or imported inputs never become
// physically observed findings merely by passing protocol controls.
type TelemetryDiagnosis struct {
	Diagnosis    DiagnosisResult    `json:"diagnosis"`
	Observations []TelemetryFinding `json:"observations"`
}

// DiagnoseTelemetry reads the canonical stream, validates independent evidence
// links, joins explicit workload correlations and calls Diagnose. A malformed
// stream returns an error; missing evidence produces retained withheld findings.
// Reads and replay are bounded and cancellation is checked between reads and
// records. Cancellation cannot interrupt a caller-owned blocking Reader.
func DiagnoseTelemetry(ctx context.Context, r io.Reader, opt TelemetryOptions) (TelemetryDiagnosis, error) {
	var out TelemetryDiagnosis
	if ctx == nil || r == nil {
		return out, fmt.Errorf("telemetry diagnosis: missing context or reader")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if opt.MaxBytes < 0 || opt.MaxBytes > 64<<20 || len(opt.RawArtifacts) > 256 || len(opt.ExpectedSources) > 256 || len(opt.HistogramBounds) > 256 || len(opt.RequireDenominator) > 256 {
		return out, fmt.Errorf("telemetry diagnosis: admission input limit")
	}
	var size int64
	for _, raw := range opt.RawArtifacts {
		size += int64(len(raw))
		if size > 64<<20 {
			return out, fmt.Errorf("telemetry diagnosis: artifact byte limit")
		}
	}
	m, _, samples, err := nvtelemetry.ReadStream(telemetryReader{ctx, r}, opt.MaxBytes)
	if err != nil {
		return out, err
	}
	in := DiagnosticInput{WallInterval: opt.WallInterval, Thresholds: opt.Thresholds}
	if m.Receipt != nil {
		in.Receipt = *m.Receipt
	}
	if in.Receipt.LostRecords == nil {
		in.Receipt.LostRecords = map[string]tracecontract.Loss{}
	} else {
		losses := map[string]tracecontract.Loss{}
		for k, v := range in.Receipt.LostRecords {
			losses[k] = v
		}
		in.Receipt.LostRecords = losses
	}
	joiner := cudaworkload.NewJoiner(cudaworkload.JoinConfig{MaxPendingRecords: 4096})
	var joins []cudaworkload.JoinResult
	var dependentReason string
	var associated bool
	for _, sample := range samples {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		for _, reading := range sample.Readings {
			if err := ctx.Err(); err != nil {
				return out, err
			}
			reason := telemetryAdmission(m, sample, reading, opt)
			f := tracecontract.Finding{ID: fmt.Sprintf("telemetry-%d-%d", sample.Seq, len(out.Observations)), Rule: reading.Name, Class: tracecontract.FindingUnavailable, Unit: reading.Unit, Admission: tracecontract.Admission{Status: "withheld", ReasonCode: tracecontract.ReasonEvidenceUnavailable, Reason: reason}}
			o := reading.Observation
			if reason == "" {
				f.Class = tracecontract.FindingObserved
				if o.SourceClass != "native" {
					f.Class = tracecontract.FindingAssociated
					associated = true
				}
				f.Admission = tracecontract.Admission{Admitted: true, Status: "admitted", ReasonCode: tracecontract.ReasonAdmitted, Reason: "retained canonical reading and independent evidence links admitted; no producer population completeness claim"}
				if reading.Number != nil {
					f.Value = reading.Number.Float
				}
				if o.Interval != nil {
					switch o.Kind {
					case "host_call":
						if o.Host == nil {
							reason = "missing host submission"
						} else {
							in.HostCalls = append(in.HostCalls, HostCallEvent{Span: *o.Interval, Name: reading.Name, HasReturn: o.HasReturn})
							joins = append(joins, joiner.SubmitHost(*o.Host)...)
						}
					case "kernel":
						if o.Device == nil {
							reason = "missing device activity"
						} else {
							d := o.Device
							in.Kernels = append(in.Kernels, KernelEvent{Span: *o.Interval, Name: reading.Name, StreamID: d.Context.Stream, ContextID: d.Context.Context, DeviceID: d.DeviceID})
							joins = append(joins, joiner.SubmitDevice(*d)...)
						}
					case "sched_running", "sched_runnable", "sched_blocked":
						in.Sched.Process = o.Workload.Process
						in.Sched.TID = o.TID
						in.Sched.ProbePresent = true
						loss := m.LostRecordsByFamily[o.Family]
						in.Sched.Loss = loss.Dropped
						in.Sched.UnknownLoss = loss.Unknown
						state := map[string]SchedState{"sched_running": SchedRunning, "sched_runnable": SchedRunnable, "sched_blocked": SchedBlocked}[o.Kind]
						in.Sched.Intervals = append(in.Sched.Intervals, SchedInterval{State: state, Span: *o.Interval})
					}
				}
			}
			if reason != "" {
				f.Class = tracecontract.FindingUnavailable
				f.Value = nil
				f.Admission = tracecontract.Admission{Status: "withheld", ReasonCode: tracecontract.ReasonEvidenceUnavailable, Reason: reason}
				if o != nil && (o.Kind == "host_call" || o.Kind == "kernel" || o.Kind == "sched_running" || o.Kind == "sched_runnable" || o.Kind == "sched_blocked") && dependentReason == "" {
					dependentReason = reason
				}
			}
			out.Observations = append(out.Observations, TelemetryFinding{Reading: reading, Finding: f})
		}
	}
	joins = append(joins, joiner.Flush()...)
	if len(joins) == 0 && dependentReason == "" {
		dependentReason = "missing explicit workload correlation"
	}
	for _, join := range joins {
		if join.Outcome != cudaworkload.OutcomeExact || join.OpID != opt.OpID || !sameWorkload(join.Workload, opt.Workload) {
			if dependentReason == "" {
				dependentReason = "ambiguous or missing explicit workload correlation"
			}
			break
		}
	}
	if dependentReason != "" {
		out.Diagnosis = withheldDiagnosis(DiagnosisResult{Workload: opt.Workload, OpID: opt.OpID}, in, dependentReason)
	} else {
		out.Diagnosis = Diagnose(opt.Workload, opt.OpID, in)
	}
	if associated && out.Diagnosis.Admitted {
		out.Diagnosis.PrimaryFinding.Class = tracecontract.FindingAssociated
		for i := range out.Diagnosis.Findings {
			if out.Diagnosis.Findings[i].Class == tracecontract.FindingObserved {
				out.Diagnosis.Findings[i].Class = tracecontract.FindingAssociated
			}
		}
	}
	return out, nil
}

type telemetryReader struct {
	ctx context.Context
	r   io.Reader
}

func (r telemetryReader) Read(b []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(b)
}

func sameWorkload(a, b cudaworkload.WorkloadIdentity) bool {
	return reflect.DeepEqual(a, b) && a.Process.PID != 0 && a.Process.Generation != 0 && a.RunID != "" && a.WorkloadName != ""
}

func sameObserver(a, b nvtelemetry.Observer) bool {
	return a.ID == b.ID && a.Version == b.Version && a.Mode == b.Mode && a.Pass == b.Pass && a.Population == b.Population && a.Filter == b.Filter && a.Replay == b.Replay && reflect.DeepEqual(a.SamplingCoverage, b.SamplingCoverage)
}

func telemetryAdmission(m *nvtelemetry.Manifest, s nvtelemetry.Sample, r nvtelemetry.Reading, opt TelemetryOptions) string {
	if m.SchemaVersion < nvtelemetry.SchemaVersionV3 || m.SchemaVersion > nvtelemetry.SchemaVersionV4 {
		return "legacy stream lacks V3 evidence links"
	}
	if r.Observation == nil {
		return "missing observation identity"
	}
	o := r.Observation
	if !r.IsAvailable() {
		return "unavailable reading: " + r.UnavailableReason
	}
	if o.Workload == nil || !sameWorkload(*o.Workload, opt.Workload) || o.OpID != opt.OpID || opt.OpID == 0 {
		return "missing or mismatched workload identity"
	}
	if o.Assignment == nil || opt.Assignment.Validate() != nil || o.Assignment.Validate() != nil || !reflect.DeepEqual(*o.Assignment, opt.Assignment) || o.Assignment.Process.BootID == "" || o.Assignment.Process.NodeID == "" || o.Assignment.Process.PIDNamespace == "" || !o.Assignment.Process.HasExecGeneration || o.Assignment.Process.Process != opt.Workload.Process || o.Assignment.Device.UUID != r.DeviceUUID || o.Assignment.State != "assigned" || !o.Assignment.Authoritative {
		return "process or MIG allocation identity mismatch"
	}
	declared := false
	for _, device := range m.Devices {
		if reflect.DeepEqual(device, o.Assignment.Device) {
			declared = true
			break
		}
	}
	if !declared {
		return "declared device scope mismatch"
	}
	if o.Interval == nil || o.ClockEpoch == "" || opt.ClockEpoch == "" || o.ClockEpoch != opt.ClockEpoch || o.ClockEpoch != m.ClockMonotonicEpochID || o.Interval.Clock == "" || o.Interval.Clock != m.ClockMonotonicDomain || o.Interval.Clock != opt.WallInterval.Clock || o.UncertaintyNS == nil {
		return "missing epoch or ambiguous clock"
	}
	if o.Assignment.ClockDomain != o.Interval.Clock || o.Assignment.ClockEpoch != o.ClockEpoch || o.Assignment.StartNanos > o.Interval.Start || o.Assignment.EndNanos <= o.Interval.End {
		return "allocation interval or epoch mismatch"
	}
	maxUncertainty := opt.Thresholds.MaxUncertaintyNS
	if maxUncertainty <= 0 {
		maxUncertainty = 50
	}
	if *o.UncertaintyNS > maxUncertainty || s.ClockDiscontinuity || o.Interval.Start < opt.WallInterval.Start || o.Interval.End > opt.WallInterval.End {
		return "clock uncertainty or capture bounds mismatch"
	}
	if opt.Receipt == nil || m.Receipt == nil || !reflect.DeepEqual(*opt.Receipt, *m.Receipt) {
		return "missing or mismatched independent trace receipt"
	}
	if _, err := opt.Receipt.JSON(); err != nil {
		return "invalid independent trace receipt"
	}
	if opt.Receipt.ClockUncertaintyNS == nil || *opt.Receipt.ClockUncertaintyNS > maxUncertainty {
		return "trace clock uncertainty is unknown or exceeded"
	}
	if opt.Receipt.ClockDomain != o.Interval.Clock || opt.Receipt.CaptureBounds == nil || opt.Receipt.CaptureBounds.Clock != o.Interval.Clock || opt.Receipt.CaptureBounds.Start > o.Interval.Start || opt.Receipt.CaptureBounds.End < o.Interval.End {
		return "trace capture clock or bounds mismatch"
	}
	for _, name := range []string{"collector", "import", "workload", "shape", "correctness", "stationarity"} {
		if opt.Receipt.Gates[name].Disposition != "passed" {
			return "independent evidence gate not passed: " + name
		}
	}
	raw, ok := opt.RawArtifacts[o.RawSHA256]
	if !ok || fmt.Sprintf("%x", sha256.Sum256(raw)) != o.RawSHA256 {
		return "missing or corrupt raw artifact"
	}
	traceRaw, ok := opt.RawArtifacts[opt.Receipt.InputSHA256]
	if !ok || fmt.Sprintf("%x", sha256.Sum256(traceRaw)) != opt.Receipt.InputSHA256 {
		return "missing or corrupt trace artifact"
	}
	expect, ok := opt.ExpectedSources[o.RawSHA256]
	if !sameObserver(o.Observer, opt.Observer) || o.Observer.ID == "" || o.Observer.Version == "" || o.Observer.Mode == "" || o.Observer.Pass == "" || o.Observer.Population == "" {
		return "observer mode or pass mismatch"
	}
	if (o.Kind == "host_call" || o.Kind == "kernel" || o.Kind == "sched_running" || o.Kind == "sched_runnable" || o.Kind == "sched_blocked") && o.Observer.Mode != "activity" {
		return "observer mode cannot establish complete activity timing"
	}
	if !ok || o.SourceClass == "" || o.Producer == "" || o.ProducerVersion == "" || o.ProducerSchema == "" || o.Transform == "" || o.TransformSHA256 == "" || expect.SourceClass != o.SourceClass || expect.Producer != o.Producer || expect.ProducerVersion != o.ProducerVersion || expect.ProducerSchema != o.ProducerSchema || expect.Transform != o.Transform || expect.TransformSHA256 != o.TransformSHA256 || !sameObserver(expect.Observer, o.Observer) {
		return "missing or mismatched independent source provenance"
	}
	transform, ok := opt.RawArtifacts[o.TransformSHA256]
	if !ok || fmt.Sprintf("%x", sha256.Sum256(transform)) != o.TransformSHA256 {
		return "missing or corrupt transformation artifact"
	}
	if o.SourceClass == "modeled" {
		return "modeled source cannot establish an observation"
	}
	loss, ok := m.LostRecordsByFamily[o.Family]
	if !ok || loss.Unknown {
		return "required family loss is unassessed"
	}
	if loss.Dropped != 0 {
		return "required family records lost"
	}
	if receiptLoss, ok := opt.Receipt.LostRecords[o.Family]; ok && (receiptLoss.Unknown || receiptLoss.Dropped != 0) {
		return "independent receipt reports required family loss"
	}
	if !m.IsComplete && m.IncompleteReason != "lost_records" {
		return "incomplete stream"
	}
	if o.CounterReset || (r.Source != nil && r.Source.CounterReset) {
		return "counter reset forbids comparable delta"
	}
	if o.Kind == "counter" && o.CounterEpoch == "" {
		return "missing counter epoch"
	}
	if o.Kind == "histogram" {
		if o.Histogram == nil || !slices.Equal(opt.HistogramBounds[r.Name], o.Histogram.Bounds) {
			return "histogram boundary mismatch or unknown boundaries"
		}
	}
	if r.Evidence == nvtelemetry.EvidenceDerived || slices.Contains(opt.RequireDenominator, r.Name) {
		if o.Denominator == nil || !positiveTelemetryNumber(o.Denominator) || o.DenominatorUnit == "" || o.Scope == "" {
			return "missing useful-work denominator or scope"
		}
	}
	if o.Host != nil && (o.Host.Process != opt.Workload.Process || !sameWorkload(o.Host.Workload, opt.Workload) || o.Host.OpID != opt.OpID || o.Host.TimestampClock != o.Interval.Clock || o.Host.TimestampEpoch != o.ClockEpoch || o.Host.TimestampNs != uint64(o.Interval.Start)) {
		return "host process, operation or clock identity mismatch"
	}
	if o.Device != nil && (o.Device.Process != opt.Workload.Process || o.Device.StartNs != uint64(o.Interval.Start) || o.Device.EndNs != uint64(o.Interval.End)) {
		return "device process or interval identity mismatch"
	}
	return ""
}

func positiveTelemetryNumber(n *nvtelemetry.Number) bool {
	if n.Float != nil {
		return *n.Float > 0
	}
	if n.Int != nil {
		return *n.Int > 0
	}
	return n.Uint != nil && *n.Uint > 0
}
