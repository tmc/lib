package nvtelemetry

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"strconv"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/dcgm"
)

// DCGMClient owns one exclusive connection and all objects it creates. Close
// must remove only its own watches/groups and retain externally owned engines.
// Native calls need not support interruption; contexts are checked between calls.
type DCGMClient interface {
	NewGroup(string, dcgm.GroupType) (dcgm.GpuGrp, error)
	AddEntity(dcgm.GpuGrp, dcgm.FieldEntityGroup, dcgm.FieldEid) error
	NewFieldGroup(string, []dcgm.FieldID) (dcgm.FieldGrp, error)
	Watch(dcgm.GpuGrp, dcgm.FieldGrp, time.Duration, time.Duration, int32) error
	LatestValues(uint32, []dcgm.FieldID) ([]dcgm.FieldValueV1, error)
	Close() error
}

type dcgmConnection struct{ *dcgm.Session }

func (c *dcgmConnection) LatestValues(id uint32, fields []dcgm.FieldID) ([]dcgm.FieldValueV1, error) {
	return c.Handle.LatestValues(id, fields)
}

// OpenDCGMClient explicitly connects to an existing host engine. The caller
// initializes the DCGM library and owns process-wide shutdown. Persisted watches
// are refused. No embedded engine, automatic Init or global Shutdown is used.
func OpenDCGMClient(ctx context.Context, address string, opts dcgm.ConnectOptions) (DCGMClient, error) {
	if ctx == nil {
		return nil, fmt.Errorf("dcgm sampler: nil context")
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	if opts.Timeout <= 0 || opts.Timeout > time.Minute || opts.PersistAfterDisconnect {
		return nil, fmt.Errorf("dcgm sampler: bounded nonpersistent connection required")
	}
	s, e := dcgm.Connect(address, opts)
	if e != nil {
		return nil, e
	}
	if e := ctx.Err(); e != nil {
		return nil, errors.Join(e, s.Close())
	}
	return &dcgmConnection{s}, nil
}

// DCGMDevice binds an explicitly selected physical engine ID to its UUID.
// The caller must verify this association against the admitted host inventory.
// MIG, switch and link entities require separate entity-aware adapters.
type DCGMDevice struct {
	ID   uint32
	UUID string
}

// DCGMField binds a generated DCGM field ID to a replayable scalar mapping.
type DCGMField struct {
	ID      dcgm.FieldID
	Mapping ManagedField
}

// DCGMConfig bounds the watch and records actual version and origin provenance.
type DCGMConfig struct {
	SourceID, ProducerVersion, NativeOrigin string
	Devices                                 []DCGMDevice
	Fields                                  []DCGMField
	Interval, MaxKeepAge                    time.Duration
	MaxKeepSamples                          int32
	MaxSampleAge, MaxFutureSkew             time.Duration
}

// DCGMReadingSource retains native scalar bits, status and DCGM cache timestamp.
// TimestampMicroseconds is a DCGM field sampling timestamp (zero means unknown),
// not independent proof of when the underlying sensor acquired a value.
type DCGMReadingSource struct {
	Version               uint32 `json:"version"`
	FieldID               uint16 `json:"field_id"`
	FieldName             string `json:"field_name"`
	FieldType             uint16 `json:"field_type"`
	Status                int32  `json:"status"`
	RawBits               uint64 `json:"raw_bits"`
	TimestampMicroseconds int64  `json:"timestamp_microseconds"`
	RawText               string `json:"raw_text,omitempty"`
}

// DCGMSampler implements Sampler with an owned, bounded DCGM watch. Its zero
// value is closed. Close waits for an active native query; no goroutine is left
// behind to pretend a blocked native call has been cancelled.
type DCGMSampler struct {
	mu       sync.Mutex
	client   DCGMClient
	clock    Clock
	config   DCGMConfig
	closed   bool
	closeErr error
}

// NewDCGMSampler takes ownership of a fresh exclusive client after validation.
// Setup failures close that client and preserve cleanup errors. Invalid config
// leaves ownership with the caller. Another agent's watches are never borrowed.
func NewDCGMSampler(ctx context.Context, client DCGMClient, clock Clock, c DCGMConfig) (*DCGMSampler, error) {
	if ctx == nil || dcgmNil(client) || dcgmNil(clock) {
		return nil, fmt.Errorf("dcgm sampler: nil input")
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	if len(c.Devices) == 0 || len(c.Devices) > 64 || len(c.Fields) == 0 || len(c.Fields) > 128 || c.Interval < time.Millisecond || c.Interval > time.Hour || c.MaxKeepAge <= 0 || c.MaxKeepAge > time.Hour || c.MaxKeepSamples < 1 || c.MaxKeepSamples > 1024 {
		return nil, fmt.Errorf("dcgm sampler: invalid watch bounds")
	}
	if c.MaxSampleAge < 0 || c.MaxSampleAge > time.Hour || c.MaxFutureSkew < 0 || c.MaxFutureSkew > time.Hour {
		return nil, fmt.Errorf("dcgm sampler: invalid freshness policy")
	}
	for _, v := range []string{c.SourceID, c.ProducerVersion, c.NativeOrigin} {
		if len(v) == 0 || len(v) > 256 {
			return nil, fmt.Errorf("dcgm sampler: invalid provenance")
		}
	}
	ids, uuids := map[uint32]bool{}, map[string]bool{}
	for _, d := range c.Devices {
		if d.ID > math.MaxInt32 || d.UUID == "" || len(d.UUID) > 256 || ids[d.ID] || uuids[d.UUID] {
			return nil, fmt.Errorf("dcgm sampler: invalid or duplicate device")
		}
		ids[d.ID] = true
		uuids[d.UUID] = true
	}
	fields := make([]dcgm.FieldID, len(c.Fields))
	seen := map[dcgm.FieldID]bool{}
	names := map[string]bool{}
	for i, f := range c.Fields {
		for _, v := range []string{f.Mapping.Metric, f.Mapping.Name, f.Mapping.Type, f.Mapping.SourceUnit} {
			if len(v) > 256 {
				return nil, fmt.Errorf("dcgm sampler: mapping extent")
			}
		}
		if seen[f.ID] || names[f.Mapping.Name] || f.Mapping.Numerator <= 0 || f.Mapping.Denominator <= 0 || !metricName(f.Mapping.Metric) {
			return nil, fmt.Errorf("dcgm sampler: invalid or duplicate field")
		}
		if e := validateDCGMMapping(f.Mapping); e != nil {
			return nil, e
		}
		seen[f.ID] = true
		names[f.Mapping.Name] = true
		fields[i] = f.ID
	}
	c.Devices = append([]DCGMDevice(nil), c.Devices...)
	c.Fields = append([]DCGMField(nil), c.Fields...)
	fail := func(e error) (*DCGMSampler, error) { return nil, errors.Join(e, client.Close()) }
	var nonce [16]byte
	if _, e := rand.Read(nonce[:]); e != nil {
		return fail(e)
	}
	name := fmt.Sprintf("nvtelemetry-%x", nonce)
	group, e := client.NewGroup(name, dcgm.GroupEmpty)
	if e != nil {
		return fail(e)
	}
	for _, d := range c.Devices {
		if e := ctx.Err(); e != nil {
			return fail(e)
		}
		if e := client.AddEntity(group, dcgm.FeGpu, dcgm.FieldEid(d.ID)); e != nil {
			return fail(e)
		}
	}
	fieldGroup, e := client.NewFieldGroup(name, fields)
	if e != nil {
		return fail(e)
	}
	if e := ctx.Err(); e != nil {
		return fail(e)
	}
	if e := client.Watch(group, fieldGroup, c.Interval, c.MaxKeepAge, c.MaxKeepSamples); e != nil {
		return fail(e)
	}
	if e := ctx.Err(); e != nil {
		return fail(e)
	}
	return &DCGMSampler{client: client, clock: clock, config: c}, nil
}

func (s *DCGMSampler) Devices() []DeviceMetadata {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]DeviceMetadata, len(s.config.Devices))
	for i, d := range s.config.Devices {
		out[i] = DeviceMetadata{UUID: d.UUID, Index: int(d.ID), LibraryVersion: s.config.ProducerVersion, Status: "partial", Warnings: []string{"model and architecture not queried"}}
	}
	return out
}

func (s *DCGMSampler) Sample(ctx context.Context) ([]Reading, error) {
	if s == nil || ctx == nil {
		return nil, fmt.Errorf("dcgm sampler: nil input")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.client == nil {
		return nil, dcgm.ErrClosed
	}
	fields := make([]dcgm.FieldID, len(s.config.Fields))
	for i, f := range s.config.Fields {
		fields[i] = f.ID
	}
	out := make([]Reading, 0, len(s.config.Devices)*len(fields))
	for _, d := range s.config.Devices {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		start := s.clock.NowMonotonic()
		values, e := s.client.LatestValues(d.ID, fields)
		end := s.clock.NowMonotonic()
		if e != nil {
			return nil, e
		}
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		if start < 0 || end < start || len(values) != len(fields) {
			return nil, fmt.Errorf("dcgm sampler: invalid query bounds or result count")
		}
		seen := map[dcgm.FieldID]bool{}
		byID := map[dcgm.FieldID]dcgm.FieldValueV1{}
		for _, v := range values {
			if v.Version != dcgm.FieldValueVersion1 {
				return nil, fmt.Errorf("dcgm sampler: field layout version mismatch")
			}
			if !containsDCGMField(fields, v.ID()) || seen[v.ID()] {
				return nil, fmt.Errorf("dcgm sampler: unexpected or duplicate result field")
			}
			seen[v.ID()] = true
			byID[v.ID()] = v
		}
		for _, f := range s.config.Fields {
			v := byID[f.ID]
			r, e := s.reading(d, f, v, start, end)
			if e != nil {
				return nil, e
			}
			out = append(out, r)
		}
	}
	return out, nil
}

func containsDCGMField(fields []dcgm.FieldID, id dcgm.FieldID) bool {
	for _, f := range fields {
		if f == id {
			return true
		}
	}
	return false
}

func dcgmNil(v any) bool {
	if v == nil {
		return true
	}
	r := reflect.ValueOf(v)
	switch r.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return r.IsNil()
	}
	return false
}

func (s *DCGMSampler) reading(d DCGMDevice, f DCGMField, v dcgm.FieldValueV1, start, end int64) (Reading, error) {
	m := f.Mapping
	m.Metric = fmt.Sprintf("dcgm_field_%d", f.ID)
	unit, _ := LookupMetricUnit(m.Name)
	raw := strconv.FormatUint(v.Value[0], 10)
	if v.Type() == dcgm.FieldTypeInt64 || v.Type() == dcgm.FieldTypeTimestamp {
		raw = strconv.FormatInt(int64(v.Value[0]), 10)
	} else if v.Type() == dcgm.FieldTypeDouble {
		raw = strconv.FormatFloat(math.Float64frombits(v.Value[0]), 'g', -1, 64)
	}
	source := &ReadingSource{Schema: "nvtelemetry.managed.v1", SourceID: s.config.SourceID, Producer: "dcgm", ProducerVersion: s.config.ProducerVersion, NativeOrigin: s.config.NativeOrigin, FieldID: m.Metric, RawValue: raw, Mapping: m, MappingDigest: managedMappingDigest(m), Labels: map[string]string{"UUID": d.UUID}, Native: &DCGMReadingSource{Version: v.Version, FieldID: uint16(v.ID()), FieldName: f.Mapping.Metric, FieldType: v.Type(), Status: v.Status, RawBits: v.Value[0], TimestampMicroseconds: v.Ts}}
	if v.Type() == dcgm.FieldTypeString {
		if text, ok := v.Text(); ok {
			if len(text) > 256 {
				return Reading{}, fmt.Errorf("dcgm sampler: raw text extent")
			}
			source.Native.RawText = text
		}
	}
	r := Reading{Name: m.Name, Unit: unit, Backend: "dcgm", DeviceUUID: d.UUID, Evidence: EvidenceMeasured, Source: source, QueryStartNanos: start, QueryEndNanos: end, HasQueryBounds: true, TimestampMonotonic: end, TimestampUnix: s.clock.NowUnix(), RefreshCadenceNanos: int64(s.config.Interval)}
	if v.Status != 0 {
		r.Evidence = EvidenceUnavailable
		r.UnavailableReason = ReasonQueryFailure
		if dcgm.Result(v.Status) == dcgm.ErrNotSupported {
			r.UnavailableReason = ReasonUnsupported
		}
		if dcgm.Result(v.Status) == dcgm.ErrNoPermission {
			r.UnavailableReason = ReasonPermissionDenied
		}
		return r, nil
	}
	if v.Ts < 0 {
		r.Evidence = EvidenceUnavailable
		r.UnavailableReason = ReasonMisalignedClock
		return r, nil
	}
	if s.config.MaxSampleAge > 0 || s.config.MaxFutureSkew > 0 {
		if v.Ts == 0 || v.Ts > math.MaxInt64/1000 || r.TimestampUnix <= 0 {
			r.Evidence = EvidenceUnavailable
			r.UnavailableReason = "unknown_freshness"
			return r, nil
		}
		ts := v.Ts * 1000
		if (s.config.MaxSampleAge > 0 && ts < r.TimestampUnix && r.TimestampUnix-ts > int64(s.config.MaxSampleAge)) || (s.config.MaxFutureSkew > 0 && ts > r.TimestampUnix && ts-r.TimestampUnix > int64(s.config.MaxFutureSkew)) {
			r.Evidence = EvidenceUnavailable
			r.UnavailableReason = ReasonStale
			return r, nil
		}
	}
	if v.Type() != dcgm.FieldTypeInt64 && v.Type() != dcgm.FieldTypeDouble && v.Type() != dcgm.FieldTypeTimestamp {
		r.Evidence = EvidenceUnavailable
		r.UnavailableReason = "unsupported_field_type"
		return r, nil
	}
	if v.Blank() {
		r.Evidence = EvidenceUnavailable
		r.UnavailableReason = ReasonBlank
		if v.Type() == dcgm.FieldTypeInt64 || v.Type() == dcgm.FieldTypeTimestamp {
			r.UnavailableReason = dcgmBlankReason(uint64(int64(v.Value[0]) - dcgm.Int64Blank))
		} else if x := math.Float64frombits(v.Value[0]) - dcgm.Fp64Blank; x >= 0 && x <= 15 {
			r.UnavailableReason = dcgmBlankReason(uint64(x))
		}
		return r, nil
	}
	x, ok := boundedDecimal(raw)
	if !ok {
		r.Evidence = EvidenceUnavailable
		r.UnavailableReason = "non_finite"
		return r, nil
	}
	x.Mul(x, big.NewRat(m.Numerator, m.Denominator))
	if m.Integer {
		if !x.IsInt() || !x.Num().IsUint64() {
			return Reading{}, fmt.Errorf("dcgm sampler: integer conversion overflow or precision loss")
		}
		r.IsInt = true
		r.IntValue = x.Num().Uint64()
		r.Value = float64(r.IntValue)
	} else {
		r.Value, _ = x.Float64()
		if math.IsInf(r.Value, 0) {
			return Reading{}, fmt.Errorf("dcgm sampler: conversion overflow")
		}
	}
	return r, nil
}

// Close closes the owned client once and consistently returns its cleanup error.
func (s *DCGMSampler) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return s.closeErr
	}
	s.closed = true
	if s.client != nil {
		s.closeErr = s.client.Close()
	}
	return s.closeErr
}

func validateDCGMNative(r Reading) error {
	n := r.Source.Native
	if n == nil {
		return nil
	}
	if r.Source.Producer != "dcgm" || n.Version != dcgm.FieldValueVersion1 || r.Source.FieldID != fmt.Sprintf("dcgm_field_%d", n.FieldID) || len(n.FieldName) > 256 || len(n.RawText) > 256 || !metricName(n.FieldName) {
		return fmt.Errorf("invalid native source provider or layout")
	}
	raw := strconv.FormatUint(n.RawBits, 10)
	switch n.FieldType {
	case dcgm.FieldTypeInt64, dcgm.FieldTypeTimestamp:
		raw = strconv.FormatInt(int64(n.RawBits), 10)
	case dcgm.FieldTypeDouble:
		raw = strconv.FormatFloat(math.Float64frombits(n.RawBits), 'g', -1, 64)
	}
	if r.Source.RawValue != raw {
		return fmt.Errorf("native raw bits contradiction")
	}
	if n.RawText != "" && n.FieldType != dcgm.FieldTypeString {
		return fmt.Errorf("native text type contradiction")
	}
	if r.IsAvailable() {
		if n.Status != 0 || n.TimestampMicroseconds < 0 {
			return fmt.Errorf("native status or timestamp contradiction")
		}
		v := dcgm.FieldValueV1{FieldType: n.FieldType}
		v.Value[0] = n.RawBits
		if n.FieldType != dcgm.FieldTypeInt64 && n.FieldType != dcgm.FieldTypeDouble && n.FieldType != dcgm.FieldTypeTimestamp || v.Blank() {
			return fmt.Errorf("unavailable native value claimed measured")
		}
	}
	return nil
}

func validateDCGMMapping(m ManagedField) error {
	kind, ok := LookupMetricKind(m.Name)
	if !ok {
		return fmt.Errorf("unknown dcgm metric")
	}
	if kind == KindBitmask || kind == KindEvent {
		if !m.Integer || m.Numerator != 1 || m.Denominator != 1 || (m.Type != "" && m.Type != kind) {
			return fmt.Errorf("invalid discrete dcgm mapping")
		}
		return nil
	}
	return validateManagedMapping(m)
}

// DCGMIMEXFields selects raw status fields from the existing generated IDs.
// No enum-to-health judgment mapping or IMEX action is performed. String-typed
// results remain unavailable to the numeric model, with RawText retained.
func DCGMIMEXFields() []DCGMField {
	return []DCGMField{
		{dcgm.FieldImexDomainStatus, ManagedField{Metric: "DCGM_FI_IMEX_DOMAIN_STATUS", Name: "imex_domain_status", Type: KindEvent, Numerator: 1, Denominator: 1, Integer: true, SourceUnit: "code"}},
		{dcgm.FieldImexDaemonStatus, ManagedField{Metric: "DCGM_FI_IMEX_DAEMON_STATUS", Name: "imex_daemon_status", Type: KindEvent, Numerator: 1, Denominator: 1, Integer: true, SourceUnit: "code"}},
	}
}
