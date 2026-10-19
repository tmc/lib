package nvtelemetry

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

// Raw cumulative NVLink error-count metrics. These are not byte traffic,
// throughput or congestion measurements. Reset coverage remains unknown.
const (
	MetricNVLinkDLReplayErrors   = "nvlink_dl_replay_errors"
	MetricNVLinkDLRecoveryErrors = "nvlink_dl_recovery_errors"
	MetricNVLinkDLCRCFlitErrors  = "nvlink_dl_crc_flit_errors"
	MetricNVLinkDLCRCDataErrors  = "nvlink_dl_crc_data_errors"
	MetricNVLinkDLECCDataErrors  = "nvlink_dl_ecc_data_errors"
	NVLinkErrorCounterAPI        = "nvmlDeviceGetNvLinkErrorCounter"
)

// NVLinkErrorCounterDevice is an optional read-only addition to NVLinkDevice.
// nvml.Device implements it. The caller owns NVML initialization and the handle;
// the reader never resets counters or configures/enables utilization collection.
type NVLinkErrorCounterDevice interface {
	NvLinkErrorCounter(uint32, nvml.NvLinkErrorCounter) (uint64, error)
}

// NVLinkCounterConfig selects documented raw counters and existing topology
// configuration. Counters must be distinct enum values 0–4; none are implicit.
type NVLinkCounterConfig struct {
	Topology NVLinkConfig
	Counters []nvml.NvLinkErrorCounter
}

// NVLinkCounterObservation preserves an exact raw API result or unavailable
// status. Available zero has a nonnil Value; failures never fabricate zero.
// JSON requires explicit link and counter fields, including when they are zero.
type NVLinkCounterObservation struct {
	Link    uint32                  `json:"link"`
	Counter nvml.NvLinkErrorCounter `json:"counter"`
	Query   NVLinkQuery             `json:"query"`
	Value   *uint64                 `json:"value,omitempty"`
}

// UnmarshalJSON requires the observed link and counter identities. An omitted
// identity is not evidence that the API queried link or counter zero.
func (o *NVLinkCounterObservation) UnmarshalJSON(data []byte) error {
	if o == nil {
		return fmt.Errorf("nil NVLink counter observation")
	}
	var wire struct {
		Link    *uint32                  `json:"link"`
		Counter *nvml.NvLinkErrorCounter `json:"counter"`
		Query   NVLinkQuery              `json:"query"`
		Value   *uint64                  `json:"value,omitempty"`
	}
	if err := decodeOperationsJSON(data, &wire); err != nil {
		return err
	}
	if wire.Link == nil || wire.Counter == nil {
		return fmt.Errorf("missing NVLink counter observation identity")
	}
	*o = NVLinkCounterObservation{Link: *wire.Link, Counter: *wire.Counter, Query: wire.Query, Value: wire.Value}
	return nil
}

// NVLinkCounterSnapshot is version 1 raw-counter evidence bracketed by existing
// topology observations. Versions/API/source IDs are retained, not authenticated.
// Query intervals are host bounds, not sensor timestamps. ResetCoverage is
// always unknown: this API does not establish a reset/wrap/saturation epoch.
type NVLinkCounterSnapshot struct {
	Version       int                        `json:"version"`
	API           string                     `json:"api"`
	ResetCoverage string                     `json:"reset_coverage"`
	Before        NVLinkSnapshot             `json:"before"`
	After         NVLinkSnapshot             `json:"after"`
	Counters      []nvml.NvLinkErrorCounter  `json:"counters"`
	Observations  []NVLinkCounterObservation `json:"observations"`
}

// NVLinkCounterReader reads at most 32 links and five counters per batch. It
// allocates no native resources and needs no Close. Synchronous NVML calls cannot
// be interrupted; cancellation is checked between calls. It is not concurrent
// safe and its zero value is unusable. Existing NVLinkDevice implementations do
// not need the optional counter interface unless this reader is constructed.
type NVLinkCounterReader struct {
	topology *NVLinkReader
	device   NVLinkErrorCounterDevice
	counters []nvml.NvLinkErrorCounter
}

func nvlinkCounterMetric(counter nvml.NvLinkErrorCounter) (string, bool) {
	switch counter {
	case nvml.NvlinkErrorDlReplay:
		return MetricNVLinkDLReplayErrors, true
	case nvml.NvlinkErrorDlRecovery:
		return MetricNVLinkDLRecoveryErrors, true
	case nvml.NvlinkErrorDlCrcFlit:
		return MetricNVLinkDLCRCFlitErrors, true
	case nvml.NvlinkErrorDlCrcData:
		return MetricNVLinkDLCRCDataErrors, true
	case nvml.NvlinkErrorDlEccData:
		return MetricNVLinkDLECCDataErrors, true
	}
	return "", false
}

func validateNVLinkCounters(counters []nvml.NvLinkErrorCounter) error {
	if len(counters) == 0 || len(counters) > 5 {
		return fmt.Errorf("invalid NVLink counter selection")
	}
	var seen [5]bool
	for _, counter := range counters {
		if _, ok := nvlinkCounterMetric(counter); !ok {
			return fmt.Errorf("unsupported NVLink counter %d", counter)
		}
		if seen[int(counter)] {
			return fmt.Errorf("duplicate NVLink counter")
		}
		seen[int(counter)] = true
	}
	return nil
}

// NewNVLinkCounterReader binds the optional counter API to the same physical
// device as the existing topology reader. Invalid selections fail before any
// device calls. No library/version is presumed to support a selected field;
// unsupported/denied calls remain per-query unavailable records.
func NewNVLinkCounterReader(device NVLinkDevice, clock Clock, config NVLinkCounterConfig) (*NVLinkCounterReader, error) {
	if err := validateNVLinkCounters(config.Counters); err != nil {
		return nil, err
	}
	counters, ok := device.(NVLinkErrorCounterDevice)
	if !ok || nvlinkNil(counters) {
		return nil, fmt.Errorf("NVLink counter interface unavailable")
	}
	topology, err := NewNVLinkReader(device, clock, config.Topology)
	if err != nil {
		return nil, err
	}
	return &NVLinkCounterReader{topology: topology, device: counters, counters: append([]nvml.NvLinkErrorCounter(nil), config.Counters...)}, nil
}

// Read preserves raw counts between topology snapshots. UUID, peer, source,
// clock or topology-epoch changes refuse the trusted batch. Stable unknown
// endpoints remain unknown; neither a nominal version nor a positive count
// establishes an active transfer route or a transport cause.
func (r *NVLinkCounterReader) Read(ctx context.Context) (*NVLinkCounterSnapshot, error) {
	if r == nil || r.topology == nil || nvlinkNil(r.device) || nvlinkNil(ctx) {
		return nil, fmt.Errorf("invalid NVLink counter reader or context")
	}
	before, err := r.topology.Read(ctx)
	if err != nil {
		return nil, err
	}
	s := &NVLinkCounterSnapshot{Version: 1, API: NVLinkErrorCounterAPI, ResetCoverage: "unknown", Before: *before, Counters: append([]nvml.NvLinkErrorCounter(nil), r.counters...)}
	s.Observations = make([]NVLinkCounterObservation, 0, len(before.Links)*len(r.counters))
	for _, link := range before.Links {
		for _, counter := range r.counters {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			var value uint64
			var callErr error
			q := nvlinkQuery(r.topology.clock, func() error { value, callErr = r.device.NvLinkErrorCounter(link.Link, counter); return callErr })
			// The Go binding distinguishes an absent library symbol (nil Err)
			// from a present symbol that failed to bind. Neither is native
			// NVML result 13, so do not report either as a native result code.
			var missing *nvml.SymbolError
			if errors.As(callErr, &missing) && missing != nil {
				q.State = "failed"
				q.NativeCode = ""
				if missing.Symbol == NVLinkErrorCounterAPI && missing.Err == nil {
					q.State = "unsupported"
				}
			}
			if q.NativeCode == strconv.FormatInt(int64(nvml.ErrGpuIsLost), 10) {
				return nil, fmt.Errorf("NVLink counter query: %w", nvml.ErrGpuIsLost)
			}
			o := NVLinkCounterObservation{Link: link.Link, Counter: counter, Query: q}
			if q.State == "available" {
				o.Value = &value
			}
			s.Observations = append(s.Observations, o)
		}
	}
	after, err := r.topology.Read(ctx)
	if err != nil {
		return nil, err
	}
	s.After = *after
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return s, nil
}

func sameNVLinkFacts(a, b NVLinkObservation) bool {
	for _, q := range []*NVLinkQuery{&a.StateQuery, &a.PCIQuery, &a.TypeQuery, &a.VersionQuery, &b.StateQuery, &b.PCIQuery, &b.TypeQuery, &b.VersionQuery} {
		q.StartNanos = 0
		q.EndNanos = 0
	}
	return a == b
}

func validateNVLinkCounterQuery(q NVLinkQuery, last, at int64) error {
	if q.StartNanos < last || q.EndNanos < q.StartNanos || q.EndNanos > at {
		return fmt.Errorf("invalid NVLink counter query bounds")
	}
	switch q.State {
	case "available", "unsupported", "denied", "failed":
	default:
		return fmt.Errorf("invalid NVLink counter query state")
	}
	if q.NativeCode == "" {
		return nil
	}
	code, err := strconv.ParseInt(q.NativeCode, 10, 32)
	if err != nil || code == 0 || q.NativeCode != strconv.FormatInt(code, 10) || q.State == "available" {
		return fmt.Errorf("invalid NVLink counter native code")
	}
	expect := "failed"
	switch nvml.Result(code) {
	case nvml.ErrNotSupported, nvml.ErrFunctionNotFound:
		expect = "unsupported"
	case nvml.ErrNoPermission:
		expect = "denied"
	case nvml.ErrGpuIsLost:
		return fmt.Errorf("NVLink counter device lost")
	}
	if q.State != expect {
		return fmt.Errorf("NVLink counter native code contradicts state")
	}
	return nil
}

// Validate checks version 1, selected fields, exact topology/peer identity,
// presence and ordering. Unknown counter epochs cannot become rate evidence.
func (s NVLinkCounterSnapshot) Validate() error {
	if s.Version != 1 || s.API != NVLinkErrorCounterAPI || s.ResetCoverage != "unknown" {
		return fmt.Errorf("unsupported NVLink counter schema or semantics")
	}
	if err := validateNVLinkCounters(s.Counters); err != nil {
		return err
	}
	a, b := s.Before, s.After
	if err := a.Validate(); err != nil {
		return err
	}
	if err := b.Validate(); err != nil {
		return err
	}
	if a.NodeID != b.NodeID || a.SourceID != b.SourceID || a.NVMLVersion != b.NVMLVersion || a.DeviceUUID != b.DeviceUUID || a.TopologyEpoch != b.TopologyEpoch || a.ClockDomain != b.ClockDomain || a.ClockEpoch != b.ClockEpoch || a.ValidFromNanos != b.ValidFromNanos || a.ValidUntilNanos != b.ValidUntilNanos || a.AtNanos > b.AtNanos || !slices.Equal(a.PeerAssignments, b.PeerAssignments) || len(a.Links) != len(b.Links) {
		return fmt.Errorf("NVLink counter topology identity changed")
	}
	for i := range a.Links {
		if !sameNVLinkFacts(a.Links[i], b.Links[i]) {
			return fmt.Errorf("NVLink counter peer or link facts changed")
		}
	}
	if len(s.Observations) != len(a.Links)*len(s.Counters) {
		return fmt.Errorf("incomplete NVLink counter selection")
	}
	last := a.AtNanos
	for i, o := range s.Observations {
		if o.Link != a.Links[i/len(s.Counters)].Link || o.Counter != s.Counters[i%len(s.Counters)] {
			return fmt.Errorf("NVLink counter identity or ordering mismatch")
		}
		if err := validateNVLinkCounterQuery(o.Query, last, b.AtNanos); err != nil {
			return err
		}
		last = o.Query.EndNanos
		if (o.Query.State == "available") != (o.Value != nil) {
			return fmt.Errorf("NVLink counter value presence mismatch")
		}
	}
	if b.Links[0].StateQuery.StartNanos < last {
		return fmt.Errorf("NVLink counter final topology precedes queries")
	}
	return nil
}

// ReadNVLinkCounterSnapshot decodes one strict bounded raw report (at most
// 256 KiB). Unknown/duplicate fields, future versions and inconsistent endpoint
// identities refuse rather than create a new route from counter values.
func ReadNVLinkCounterSnapshot(r io.Reader, maxBytes int64) (*NVLinkCounterSnapshot, error) {
	if nvlinkNil(r) || maxBytes <= 0 || maxBytes > 256<<10 {
		return nil, fmt.Errorf("invalid NVLink counter reader limit")
	}
	raw, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > maxBytes {
		return nil, fmt.Errorf("NVLink counter report exceeds byte limit")
	}
	var s NVLinkCounterSnapshot
	if err := decodeOperationsJSON(raw, &s); err != nil {
		return nil, err
	}
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return &s, nil
}

// NVLinkCounterReadings converts a bounded raw report into existing canonical
// V3+ exact integer readings. sourceClass is the caller's native, imported or
// synthetic assertion, never an authenticity check. Scope retains endpoint,
// topology, selected API and raw error status; query times are not sensor times.
// The caller supplies independently held observer/assignment/workload and
// transformation artifact links if needed. CounterEpoch stays empty because
// reset coverage is unknown; DiagnoseTelemetry must withhold epoch-dependent
// findings. No gauge relabeling, rates or throughput estimates are performed.
func NVLinkCounterReadings(raw []byte, sourceClass string) ([]Reading, error) {
	if sourceClass != "native" && sourceClass != "imported" && sourceClass != "synthetic" {
		return nil, fmt.Errorf("NVLink counter source class unavailable")
	}
	s, err := ReadNVLinkCounterSnapshot(bytes.NewReader(raw), 256<<10)
	if err != nil {
		return nil, err
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(raw))
	out := make([]Reading, 0, len(s.Observations))
	for i, counter := range s.Observations {
		link := s.Before.Links[i/len(s.Counters)]
		name, _ := nvlinkCounterMetric(counter.Counter)
		scope, err := json.Marshal(struct {
			Node          string `json:"node"`
			Source        string `json:"source"`
			API           string `json:"api"`
			UUID          string `json:"uuid"`
			TopologyEpoch string `json:"topology_epoch"`
			PeerState     string `json:"peer_state"`
			PeerUUID      string `json:"peer_uuid"`
			RemotePCI     string `json:"remote_pci"`
			ResetCoverage string `json:"reset_coverage"`
			QueryState    string `json:"query_state"`
			NativeCode    string `json:"native_code"`
			Link          uint32 `json:"link"`
		}{s.Before.NodeID, s.Before.SourceID, s.API, s.Before.DeviceUUID, s.Before.TopologyEpoch, link.PeerState, link.PeerUUID, link.RemotePCI, s.ResetCoverage, counter.Query.State, counter.Query.NativeCode, counter.Link})
		if err != nil {
			return nil, err
		}
		o := &Observation{Family: "nvml", SourceClass: sourceClass, Producer: s.API, ProducerVersion: s.Before.NVMLVersion, ProducerSchema: "nvml.nvlink.error-counters.v1", RawSHA256: digest, Transform: "nvlink-raw-error-counts-v1", Kind: KindCounter, Scope: string(scope), ClockEpoch: s.Before.ClockEpoch, Interval: &tracecontract.Interval{Start: counter.Query.StartNanos, End: counter.Query.EndNanos, Clock: s.Before.ClockDomain}}
		r := Reading{Name: name, Unit: "count", Backend: s.Before.SourceID, DeviceUUID: s.Before.DeviceUUID, Observation: o, TimestampMonotonic: s.After.AtNanos, HasQueryBounds: true, QueryStartNanos: counter.Query.StartNanos, QueryEndNanos: counter.Query.EndNanos, Evidence: EvidenceUnavailable}
		switch counter.Query.State {
		case "available":
			v := *counter.Value
			r.Number = &Number{Uint: &v}
			r.Evidence = EvidenceMeasured
		case "unsupported":
			r.UnavailableReason = ReasonUnsupported
		case "denied":
			r.UnavailableReason = ReasonPermissionDenied
		default:
			r.UnavailableReason = ReasonQueryFailure
		}
		out = append(out, r)
	}
	return out, nil
}
