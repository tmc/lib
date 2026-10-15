package cudadiagdemo

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"reflect"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

const portableClock = "process_monotonic"
const timingFile = "submission-timing-v1.json"
const maxTimingRecords = 4096
const maxTimingBytes = 8 << 20

type timingCapture struct {
	Source   TimingSource       `json:"source"`
	Records  []SubmissionTiming `json:"records"`
	Dropped  uint64             `json:"dropped"`
	Complete bool               `json:"complete"`
}

// timingExpectation is retained by the observer before snapshotting, not read
// from the captured declaration. A digest checks bytes, not authenticity.
type timingExpectation struct {
	source         TimingSource
	digest         string
	oraclePassed   bool
	baselinePassed bool
}

func portableTimingSource(epoch string) TimingSource {
	return TimingSource{Version: 1, Producer: "gpudiagdemo-portable-service-v1", SourceClass: "synthetic", ClockEpoch: epoch, Observer: nvtelemetry.Observer{ID: "gpudiagdemo-service-timing", Version: "1", Mode: "activity", Pass: "single", Population: "retained service invocations", Filter: "none", Replay: "none"}}
}

func timingDigest(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }

func (sf *ServiceFixture) retainTiming(record SubmissionTiming) {
	record.Host.Workload = record.Host.Workload.Clone()
	sf.mu.Lock()
	defer sf.mu.Unlock()
	if len(sf.timings) >= sf.timingLimit {
		sf.timingDropped = addTimingLoss(sf.timingDropped, 1)
		return
	}
	sf.timings = append(sf.timings, record)
}

func addTimingLoss(dropped, n uint64) uint64 {
	if n > ^uint64(0)-dropped {
		return ^uint64(0)
	}
	return dropped + n
}

func (sf *ServiceFixture) attachTiming() (timingExpectation, error) {
	sf.mu.Lock()
	capture := timingCapture{Source: sf.source, Records: append([]SubmissionTiming(nil), sf.timings...), Dropped: sf.timingDropped, Complete: true}
	sf.mu.Unlock()
	data, err := json.Marshal(capture)
	if err != nil {
		return timingExpectation{}, err
	}
	if len(data) > maxTimingBytes {
		return timingExpectation{}, errors.New("cudadiagdemo: timing attachment byte limit")
	}
	sf.recorder.AttachFile(timingFile, data)
	return timingExpectation{source: capture.Source, digest: timingDigest(data)}, nil
}

func (sf *ServiceFixture) captureTiming(ctx context.Context) ([]byte, timingExpectation, error) {
	expect, err := sf.attachTiming()
	if err != nil {
		return nil, expect, err
	}
	var out bytes.Buffer
	_, err = sf.recorder.Snapshot(ctx, &out, "portable service timing")
	return out.Bytes(), expect, err
}

func decodeTiming(data []byte, expect timingExpectation, target Response, thresholds cudadiag.Thresholds) (cudadiag.DiagnosticInput, error) {
	var in cudadiag.DiagnosticInput
	if !expect.oraclePassed || !expect.baselinePassed {
		return in, errors.New("cudadiagdemo: independent output or baseline gate unavailable")
	}
	if len(data) == 0 || len(data) > 16<<20 {
		return in, errors.New("cudadiagdemo: missing or excessive capture")
	}
	bundle, err := cudaflightrecorder.ReadBundle(bytes.NewReader(data))
	if err != nil {
		return in, err
	}
	raw := bundle.Files[timingFile]
	if len(raw) == 0 || len(raw) > maxTimingBytes || timingDigest(raw) != expect.digest {
		return in, errors.New("cudadiagdemo: missing or changed captured timing bytes")
	}
	var capture timingCapture
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&capture); err != nil {
		return in, err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return in, errors.New("cudadiagdemo: trailing timing data")
	}
	if !reflect.DeepEqual(capture.Source, expect.source) || capture.Source.Version != 1 || capture.Source.SourceClass != "synthetic" || capture.Source.ClockEpoch == "" || !reflect.DeepEqual(capture.Source, portableTimingSource(capture.Source.ClockEpoch)) {
		return in, errors.New("cudadiagdemo: timing source or observer mismatch")
	}
	if !capture.Complete || capture.Dropped != 0 || len(capture.Records) == 0 || len(capture.Records) > maxTimingRecords {
		return in, errors.New("cudadiagdemo: incomplete timing population or loss")
	}
	var selected *SubmissionTiming
	seen := make(map[cudaworkload.OpID]bool)
	joiner := cudaworkload.NewJoiner(cudaworkload.JoinConfig{MaxPendingRecords: maxTimingRecords})
	var joins []cudaworkload.JoinResult
	for i := range capture.Records {
		r := &capture.Records[i]
		if r.Host.OpID == 0 || seen[r.Host.OpID] {
			return in, errors.New("cudadiagdemo: duplicate or absent timing operation")
		}
		seen[r.Host.OpID] = true
		if err := validateTimingRecord(*r, capture.Source, target.Observation.Host.Workload); err != nil {
			return in, err
		}
		if r.Host.OpID == target.OpID {
			selected = r
		}
		joins = append(joins, joiner.SubmitHost(r.Host)...)
	}
	// Submit the whole frozen population before accepting a target. A competing
	// alias or contradictory operation must not disappear by selecting one pair.
	for _, r := range capture.Records {
		joins = append(joins, joiner.SubmitDevice(r.Device)...)
	}
	joins = append(joins, joiner.Flush()...)
	if selected == nil {
		return in, errors.New("cudadiagdemo: target timing absent from capture")
	}
	if len(joins) != len(capture.Records) {
		return in, errors.New("cudadiagdemo: incomplete timing joins")
	}
	matched := 0
	for _, join := range joins {
		if join.Outcome != cudaworkload.OutcomeExact {
			return in, errors.New("cudadiagdemo: timing population join not exact")
		}
		if join.OpID == target.OpID && sameTimingWorkload(join.Workload, target.Observation.Host.Workload) {
			matched++
		}
	}
	if matched != 1 {
		return in, errors.New("cudadiagdemo: target timing join not unique")
	}
	r := *selected
	if r.RequestID != target.RequestID || r.Host.TID != target.Observation.Host.TID {
		return in, errors.New("cudadiagdemo: target request or observed API thread mismatch")
	}
	wall, pre := r.Service, r.PreEntry
	h, device := r.Host, r.Device
	// These gates describe this bounded portable replay only. No native backend,
	// scheduler coverage, unknown cross-clock bound or physical zero loss is set.
	gates := make(map[string]tracecontract.Gate)
	for _, name := range []string{"collector", "import", "workload", "shape", "stationarity", "correctness"} {
		gates[name] = tracecontract.Gate{Disposition: "passed", Reason: "bounded synthetic service replay; caller separately checks baseline and output oracle", Provenance: capture.Source.Producer}
	}
	uncertainty := int64(0) // Same process epoch, no cross-domain conversion.
	h.EvidenceRef, device.EvidenceRef = expect.digest, expect.digest
	in = cudadiag.DiagnosticInput{WallInterval: wall, HostCalls: []cudadiag.HostCallEvent{r.API}, Kernels: []cudadiag.KernelEvent{{Span: tracecontract.Interval{Start: int64(device.StartNs), End: int64(device.EndNs), Clock: portableClock}, Name: device.Name, StreamID: device.Context.Stream, ContextID: device.Context.Context, DeviceID: device.DeviceID}}, PreEntries: []cudadiag.PreEntryEvent{{Span: pre, Submission: h, Complete: true, SourceClass: capture.Source.SourceClass}}, Thresholds: thresholds, Receipt: tracecontract.Receipt{Version: 1, Command: "gpudiagdemo portable service replay", InputPath: "capture.tar#" + timingFile, InputSHA256: expect.digest, Kind: "capture", ClockDomain: portableClock, ClockProvenance: "time.Since process epoch " + capture.Source.ClockEpoch + "; device interval synthetic", ClockUncertaintyNS: &uncertainty, CaptureBounds: &tracecontract.CaptureBounds{Start: wall.Start, End: wall.End, Clock: portableClock}, Gates: gates, Backends: []tracecontract.Backend{{Name: capture.Source.Producer, Version: "1"}}, LostRecords: map[string]tracecontract.Loss{"pre_entry": {Total: uint64(len(capture.Records))}, "activity": {Total: uint64(len(capture.Records))}}, ProbeCoverage: map[string]tracecontract.ProbeCoverage{"application_timing": {Present: true, Ratio: 1}, "synthetic_activity": {Present: true, Ratio: 1}}}}
	return in, nil
}

func validateTimingRecord(r SubmissionTiming, source TimingSource, workload cudaworkload.WorkloadIdentity) error {
	wall, pre, api := r.Service, r.PreEntry, r.API.Span
	if !r.Complete || r.RequestID == "" || len(r.RequestID) > 256 || wall.Clock != portableClock || pre.Clock != wall.Clock || api.Clock != wall.Clock || wall.Start < 0 || wall.End < wall.Start || pre.Start < wall.Start || pre.End < pre.Start || pre.End != api.Start || api.End < api.Start || api.End > wall.End || !r.API.HasReturn {
		return errors.New("cudadiagdemo: incomplete or inconsistent service endpoints")
	}
	h, device := r.Host, r.Device
	if !sameTimingWorkload(h.Workload, workload) || h.Workload.Process != h.Process || h.Process.PID == 0 || h.Process.Generation == 0 || h.Workload.RunID == "" || h.Workload.WorkloadName == "" || h.TimestampNs != uint64(api.Start) || h.TimestampClock != portableClock || h.TimestampEpoch != source.ClockEpoch || h.Symbol != r.API.Name || h.Provenance != cudaworkload.ProvenanceSynthetic || h.NativeCorrelationID == 0 {
		return errors.New("cudadiagdemo: host operation or clock mismatch")
	}
	if device.Process != h.Process || device.ExternalCorrelationID != uint64(h.OpID) || device.NativeCorrelationID != h.NativeCorrelationID || device.Context != h.Context || device.Kind != "kernel" || device.Name == "" || device.StartNs < uint64(api.End) || device.EndNs < device.StartNs || device.EndNs > uint64(wall.End) {
		return errors.New("cudadiagdemo: device identity or interval mismatch")
	}
	return nil
}

func sameTimingWorkload(a, b cudaworkload.WorkloadIdentity) bool {
	return a.Process == b.Process && a.RunID == b.RunID && a.WorkloadName == b.WorkloadName && maps.Equal(a.Labels, b.Labels)
}

func diagnoseTiming(data []byte, expect timingExpectation, response Response, thresholds cudadiag.Thresholds) (cudadiag.DiagnosticInput, cudadiag.DiagnosisResult) {
	in, err := decodeTiming(data, expect, response, thresholds)
	if err != nil {
		// The existing diagnoser retains the missing evidence as a withheld result.
		in = cudadiag.DiagnosticInput{}
	}
	target := response.Observation.Host.Workload
	result := cudadiag.Diagnose(target, response.OpID, in)
	if err != nil {
		result.PrimaryFinding.Admission.Reason = err.Error()
		for i := range result.Findings {
			if result.Findings[i].ID == result.PrimaryFinding.ID {
				result.Findings[i] = result.PrimaryFinding
			}
		}
	}
	return in, result
}

func baselineThresholds(records []cudadiag.DiagnosticInput) cudadiag.Thresholds {
	t := cudadiag.DefaultThresholds()
	t.BaselinePreEntryDuration, t.BaselineHostAPIDuration, t.BaselineKernelDuration = 1, 1, 1
	for _, in := range records {
		t.BaselinePreEntryDuration = max(t.BaselinePreEntryDuration, time.Duration(in.PreEntries[0].Span.Duration()))
		t.BaselineHostAPIDuration = max(t.BaselineHostAPIDuration, time.Duration(in.HostCalls[0].Span.Duration()))
		t.BaselineKernelDuration = max(t.BaselineKernelDuration, time.Duration(in.Kernels[0].Span.Duration()))
	}
	return t
}
