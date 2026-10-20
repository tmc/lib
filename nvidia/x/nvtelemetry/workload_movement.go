package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// MovementEndpoint is one caller-supplied side of a transfer. Entity is the
// supplied identity (a GPU UUID, host NUMA node, storage volume or NIC port);
// Kind is device, host_pinned, host_pageable, managed, storage, network or
// unknown. A GPU UUID does not identify a NIC rail, file system or switch port.
type MovementEndpoint struct {
	Entity string `json:"entity"`
	Kind   string `json:"kind"`
}

// MovementPath is the supplied transfer mechanism and the digest of the raw
// artifact documenting it. Mechanism is unknown, pcie, nvlink, nvswitch,
// host_staged, gds_direct, gds_fallback or network_rdma. Basis is
// producer_reported or independent_artifact. A declared path is a claim to be
// checked against topology and artifacts, not an observed route; a consumer
// may support topology checks for only some mechanisms.
type MovementPath struct {
	Mechanism string `json:"mechanism"`
	Basis     string `json:"basis,omitempty"`
	Evidence  string `json:"evidence,omitempty"`
}

// MovementTransfer is one caller-supplied data movement record. ID is the
// supplied transfer identity; RequestID, Attempt and OpID are retained only
// when supplied and are never synthesized. Bytes is exact with Unit "bytes";
// nil means unreported, not zero. EnqueueNanos is submission or async API
// return, StartNanos and ProgressNanos are producer-reported, and only a
// completed State with CompleteNanos records completion. All times use the
// transfer's clock domain and epoch. RawSHA256 identifies the producer record;
// Family names its loss family in the enclosing context.
type MovementTransfer struct {
	ID            string            `json:"id"`
	Family        string            `json:"family"`
	RawSHA256     string            `json:"raw_sha256"`
	RequestID     string            `json:"request_id,omitempty"`
	Attempt       string            `json:"attempt,omitempty"`
	OpID          cudaworkload.OpID `json:"op_id,omitempty"`
	Source        MovementEndpoint  `json:"source"`
	Destination   MovementEndpoint  `json:"destination"`
	Direction     string            `json:"direction"` // host_to_device, device_to_host, device_to_device, peer_to_peer, storage_to_device, device_to_storage, network_send, network_receive, unknown
	Bytes         *uint64           `json:"bytes,omitempty"`
	Unit          string            `json:"unit,omitempty"`
	State         string            `json:"state"` // enqueued, in_progress, completed, failed, cancelled, unknown
	EnqueueNanos  *int64            `json:"enqueue_nanos,omitempty"`
	StartNanos    *int64            `json:"start_nanos,omitempty"`
	ProgressNanos *int64            `json:"progress_nanos,omitempty"`
	ProgressBytes *uint64           `json:"progress_bytes,omitempty"`
	CompleteNanos *int64            `json:"complete_nanos,omitempty"`
	ClockDomain   string            `json:"clock_domain,omitempty"`
	ClockEpoch    string            `json:"clock_epoch,omitempty"`
	UncertaintyNS *int64            `json:"uncertainty_ns,omitempty"`
	TopologyEpoch string            `json:"topology_epoch,omitempty"`
	Path          MovementPath      `json:"path"`
}

// MovementContext is a version 1 caller-supplied movement interchange. It is
// not a NIXL, cuFile, NCCL or fabric exporter mapping. LossByFamily reports
// producer loss per transfer family; a missing family is unassessed loss.
type MovementContext struct {
	Version         int                           `json:"version"`
	RunID           string                        `json:"run_id"`
	Producer        string                        `json:"producer"`
	ProducerVersion string                        `json:"producer_version"`
	ProducerSchema  string                        `json:"producer_schema"`
	SourceClass     string                        `json:"source_class"` // native, managed, imported, synthetic, modeled
	LossByFamily    map[string]tracecontract.Loss `json:"loss_by_family"`
	Transfers       []MovementTransfer            `json:"transfers"`
}

// ReadMovementContext reads a bounded version 1 movement context with its
// original bytes and digest. Source must identify application
// movement-context-v1. Duplicate keys, unknown fields, unsupported versions
// and enumerations, non-byte units, duplicate transfer IDs, contradictory
// completion state and unordered times are refused. Supplied identities are
// retained as supplied; the reader verifies no path, topology or join.
func ReadMovementContext(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, MovementContext, error) {
	if s.Provider != "application" || s.Version != "movement-context-v1" {
		return WorkloadExport{}, MovementContext{}, errors.New("movement context requires application movement-context-v1 source")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, MovementContext{}, err
	}
	var c MovementContext
	if err := decodeWorkloadJSON(e.Raw, &c); err != nil {
		return e, c, err
	}
	d := json.NewDecoder(bytes.NewReader(e.Raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return e, c, fmt.Errorf("decode movement context: %w", err)
	}
	if err := validateMovementContext(c, s.RunID, l); err != nil {
		return e, c, err
	}
	return e, c, nil
}

// WriteMovementContext validates and writes the interchange, checking the
// encoded byte budget one transfer at a time. It generates no identities.
func WriteMovementContext(w io.Writer, c MovementContext, l WorkloadLimits) error {
	l, err := workloadLimits(l)
	if err != nil {
		return err
	}
	if w == nil {
		return errors.New("missing movement context writer")
	}
	if err := validateMovementContext(c, c.RunID, l); err != nil {
		return err
	}
	head := c
	head.Transfers = []MovementTransfer{}
	raw, err := json.Marshal(head)
	if err != nil {
		return err
	}
	raw = raw[:len(raw)-len(`]}`)]
	for i, t := range c.Transfers {
		encoded, err := json.Marshal(t)
		if err != nil {
			return err
		}
		if i > 0 {
			raw = append(raw, ',')
		}
		if int64(len(raw))+int64(len(encoded))+2 > l.MaxBytes {
			return errors.New("movement context byte limit exceeded")
		}
		raw = append(raw, encoded...)
	}
	raw = append(raw, ']', '}')
	if int64(len(raw)) > l.MaxBytes {
		return errors.New("movement context byte limit exceeded")
	}
	n, err := w.Write(raw)
	if err == nil && n != len(raw) {
		return io.ErrShortWrite
	}
	return err
}

func validateMovementContext(c MovementContext, run string, l WorkloadLimits) error {
	if c.Version != 1 || c.RunID == "" || c.RunID != run || c.Producer == "" || c.ProducerVersion == "" || c.ProducerSchema == "" || c.Transfers == nil || len(c.Transfers) > l.MaxMetrics || len(c.LossByFamily) > l.MaxMetrics {
		return errors.New("invalid movement context version, run, producer or transfer count")
	}
	switch c.SourceClass {
	case "native", "managed", "imported", "synthetic", "modeled":
	default:
		return errors.New("unsupported movement source class")
	}
	for _, s := range []string{c.RunID, c.Producer, c.ProducerVersion, c.ProducerSchema} {
		if len(s) > l.MaxStringBytes || !utf8.ValidString(s) {
			return errors.New("movement context string limit exceeded")
		}
	}
	for family, loss := range c.LossByFamily {
		if family == "" || len(family) > l.MaxStringBytes || !utf8.ValidString(family) || loss.Dropped > loss.Total {
			return errors.New("invalid movement family loss")
		}
	}
	seen := make(map[string]bool)
	families := make(map[string]uint64)
	for _, t := range c.Transfers {
		if err := validateMovementTransfer(t, l); err != nil {
			return fmt.Errorf("movement transfer %q: %w", t.ID, err)
		}
		if seen[t.ID] {
			return errors.New("duplicate movement transfer id")
		}
		seen[t.ID] = true
		families[t.Family]++
	}
	for family, loss := range c.LossByFamily {
		if loss.Total < families[family] {
			return errors.New("movement family loss total below supplied transfers")
		}
	}
	return nil
}

func validateMovementTransfer(t MovementTransfer, l WorkloadLimits) error {
	for _, s := range []string{t.ID, t.Family, t.RequestID, t.Attempt, t.Source.Entity, t.Destination.Entity, t.ClockDomain, t.ClockEpoch, t.TopologyEpoch} {
		if len(s) > l.MaxStringBytes || !utf8.ValidString(s) {
			return errors.New("string limit exceeded")
		}
	}
	if t.ID == "" || t.Family == "" || !streamHash(t.RawSHA256) || t.Source.Entity == "" || t.Destination.Entity == "" {
		return errors.New("missing identity, family or raw lineage")
	}
	if t.Attempt != "" && t.RequestID == "" {
		return errors.New("attempt without request")
	}
	for _, kind := range []string{t.Source.Kind, t.Destination.Kind} {
		switch kind {
		case "device", "host_pinned", "host_pageable", "managed", "storage", "network", "unknown":
		default:
			return errors.New("unsupported endpoint kind")
		}
	}
	switch t.Direction {
	case "host_to_device", "device_to_host", "device_to_device", "peer_to_peer", "storage_to_device", "device_to_storage", "network_send", "network_receive", "unknown":
	default:
		return errors.New("unsupported direction")
	}
	if (t.Bytes == nil) != (t.Unit == "") || t.Bytes != nil && t.Unit != "bytes" {
		return errors.New("bytes require exact unit bytes")
	}
	if (t.ProgressBytes == nil) != (t.ProgressNanos == nil) || t.ProgressBytes != nil && t.Bytes != nil && *t.ProgressBytes > *t.Bytes {
		return errors.New("invalid progress")
	}
	switch t.State {
	case "completed":
		if t.CompleteNanos == nil {
			return errors.New("completed state lacks completion time")
		}
	case "enqueued", "in_progress", "failed", "cancelled", "unknown":
		if t.CompleteNanos != nil {
			return errors.New("completion time without completed state")
		}
	default:
		return errors.New("unsupported state")
	}
	var last int64
	timed := false
	for _, p := range []*int64{t.EnqueueNanos, t.StartNanos, t.ProgressNanos, t.CompleteNanos} {
		if p == nil {
			continue
		}
		if *p < 0 || timed && *p < last {
			return errors.New("negative or unordered time")
		}
		last, timed = *p, true
	}
	if timed && (t.ClockDomain == "" || t.ClockEpoch == "") {
		return errors.New("times lack clock domain or epoch")
	}
	if t.UncertaintyNS != nil && *t.UncertaintyNS < 0 {
		return errors.New("negative clock uncertainty")
	}
	switch t.Path.Mechanism {
	case "unknown":
		if t.Path.Basis != "" || t.Path.Evidence != "" {
			return errors.New("unknown path has evidence")
		}
	case "pcie", "nvlink", "nvswitch", "host_staged", "gds_direct", "gds_fallback", "network_rdma":
		if t.Path.Basis != "producer_reported" && t.Path.Basis != "independent_artifact" || !streamHash(t.Path.Evidence) {
			return errors.New("path lacks basis or evidence digest")
		}
	default:
		return errors.New("unsupported path mechanism")
	}
	return nil
}
