package nvtelemetry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/dcgm"
)

// CapturePolicy is an opt-in, bounded NVIDIA capture profile. It authorizes
// only the listed device UUIDs. The source must already be configured to that
// exact set; filtering after a broad hardware query is not target isolation.
// No environment, command line, model input or credential data is collected.
type CapturePolicy struct {
	Version             int             `json:"version"`
	SourceID            string          `json:"source_id"`
	AllowedDevices      []string        `json:"allowed_devices"`
	MaxReadings         int             `json:"max_readings"`
	MaxSampleBytes      int             `json:"max_sample_bytes"`
	AllowedSourceLabels []string        `json:"allowed_source_labels,omitempty"`
	Collector           CollectorConfig `json:"collector"`
}

// Validate checks a capture policy before the sampler or collector starts.
func (p CapturePolicy) Validate() error {
	if p.Version != 1 || p.SourceID == "" || len(p.SourceID) > 128 || len(p.AllowedDevices) == 0 || len(p.AllowedDevices) > 1024 || p.MaxReadings <= 0 || p.MaxReadings > 1<<20 || p.MaxSampleBytes < 2 || p.MaxSampleBytes > 64<<20 {
		return fmt.Errorf("invalid capture policy identity or limits")
	}
	seen := make(map[string]bool)
	for _, id := range p.AllowedDevices {
		if id == "" || len(id) > 256 || seen[id] {
			return fmt.Errorf("invalid or duplicate allowed device")
		}
		seen[id] = true
	}
	if _, err := SafeCaptureLabels(nil, p.AllowedSourceLabels); err != nil {
		return err
	}
	c := p.Collector
	if c.IntervalNanos <= 0 || c.MaxDurationNanos < 0 || c.MaxRecordBudget <= 0 || c.MaxRecordBudget > 1<<20 || c.MaxByteBudget <= 0 || c.MaxByteBudget > 256<<20 || c.MaxBufferedRecords <= 0 || c.MaxBufferedRecords > c.MaxRecordBudget || (c.BudgetPolicy != BudgetPolicyFail && c.BudgetPolicy != BudgetPolicyDrop) {
		return fmt.Errorf("capture policy requires explicit bounded collector configuration")
	}
	return nil
}

// ReadCapturePolicy reads one strict version 1 profile with a bounded size.
func ReadCapturePolicy(r io.Reader, maxBytes int64) (*CapturePolicy, error) {
	if r == nil {
		return nil, fmt.Errorf("capture policy reader is nil")
	}
	if maxBytes <= 0 || maxBytes > 1<<20 {
		return nil, fmt.Errorf("invalid capture policy byte limit")
	}
	b, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read capture policy: %w", err)
	}
	if int64(len(b)) > maxBytes {
		return nil, fmt.Errorf("capture policy exceeds byte limit")
	}
	var p CapturePolicy
	if err := decodeOperationsJSON(b, &p); err != nil {
		return nil, fmt.Errorf("decode capture policy: %w", err)
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return &p, nil
}

// CaptureRegistry prevents duplicate governed capture of a GPU in one
// process. Its zero value is usable. It is not a cross-process lock, tenant
// authorization system or replacement for a native GPU lease.
type CaptureRegistry struct {
	mu      sync.Mutex
	devices map[string]*GovernedSampler
}

// GovernedSampler wraps an already target-scoped Sampler for use by Collector.
// It bounds each returned batch, rejects identity changes, and serializes
// sampling without queueing. Close cancels in-flight work and retains registry
// ownership until Sample returns. Sources must honor context cancellation;
// cancellation does not establish device completion.
// Use CaptureRegistry.NewSampler; the zero value is not a usable sampler.
// GovernedSampler is safe for concurrent use.
type GovernedSampler struct {
	mu       sync.Mutex
	source   Sampler
	clock    Clock
	policy   CapturePolicy
	registry *CaptureRegistry
	devices  []DeviceMetadata
	busy     bool
	closed   bool
	cancel   context.CancelFunc
	done     chan struct{}
	stats    CaptureStatus
}

// CaptureStatus contains collector self-observation, not device health.
type CaptureStatus struct {
	Closed                 bool
	InFlight               bool
	SuccessfulQueries      uint64
	FailedQueries          uint64
	LastSuccessNanos       int64
	HasLastSuccess         bool
	LastQueryDurationNanos int64
	SuppressedDiagnostics  uint64
	LastQueryFailed        bool
	SuppressedSourceLabels uint64
}

// NewSampler acquires process-local ownership of the policy's devices.
// Source metadata must describe precisely the allowed set before any sample.
// The caller still owns the underlying sampler and any NVML initialization.
func (r *CaptureRegistry) NewSampler(source Sampler, clock Clock, p CapturePolicy) (*GovernedSampler, error) {
	if r == nil || source == nil || clock == nil {
		return nil, fmt.Errorf("capture registry, source and clock are required")
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	p.AllowedDevices = append([]string(nil), p.AllowedDevices...)
	p.AllowedSourceLabels = append([]string(nil), p.AllowedSourceLabels...)
	devices := source.Devices()
	if err := checkCaptureDevices(devices, p.AllowedDevices); err != nil {
		return nil, err
	}
	s := &GovernedSampler{source: source, clock: clock, policy: p, registry: r, devices: cloneCaptureDevices(devices)}
	for i := range s.devices {
		s.stats.SuppressedDiagnostics += uint64(len(s.devices[i].Warnings))
		s.devices[i].Warnings = nil
		for j := range s.devices[i].MetadataFailures {
			if s.devices[i].MetadataFailures[j].Error != "" {
				s.stats.SuppressedDiagnostics++
			}
			s.devices[i].MetadataFailures[j].Error = ""
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.devices == nil {
		r.devices = make(map[string]*GovernedSampler)
	}
	for _, id := range p.AllowedDevices {
		if r.devices[id] != nil {
			return nil, fmt.Errorf("device %q already has a governed collector", id)
		}
	}
	for _, id := range p.AllowedDevices {
		r.devices[id] = s
	}
	return s, nil
}

func checkCaptureDevices(devices []DeviceMetadata, allowed []string) error {
	if len(devices) != len(allowed) {
		return fmt.Errorf("sampler is not scoped to allowed devices")
	}
	seen := make(map[string]bool)
	for _, d := range devices {
		if d.UUID == "" || seen[d.UUID] {
			return fmt.Errorf("invalid sampler identity")
		}
		seen[d.UUID] = true
	}
	for _, id := range allowed {
		if !seen[id] {
			return fmt.Errorf("sampler device is outside target allowlist")
		}
	}
	return nil
}

// Devices returns the target-scoped metadata with independent slice/map storage.
func (s *GovernedSampler) Devices() []DeviceMetadata {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return cloneCaptureDevices(s.devices)
}

func cloneCaptureDevices(devices []DeviceMetadata) []DeviceMetadata {
	out := append([]DeviceMetadata(nil), devices...)
	for i := range out {
		out[i].Warnings = append([]string(nil), out[i].Warnings...)
		out[i].MetadataFailures = append([]MetadataFailure(nil), out[i].MetadataFailures...)
		out[i].FieldCapabilities = append([]string(nil), out[i].FieldCapabilities...)
		if out[i].FieldStatuses != nil {
			m := make(map[string]string, len(out[i].FieldStatuses))
			for k, v := range out[i].FieldStatuses {
				m[k] = v
			}
			out[i].FieldStatuses = m
		}
	}
	return out
}

// Sample implements Sampler with no queue or background capture. Rejected
// batches are never returned to Collector's writer or sample hook.
// Opaque native string/binary diagnostics are refused because their raw bytes
// may contain secrets. A caller choosing the raw sampler path must separately
// authorize private raw capture; this wrapper never fabricates redacted bits.
func (s *GovernedSampler) Sample(ctx context.Context) (readings []Reading, err error) {
	if s == nil || s.source == nil {
		return nil, fmt.Errorf("governed sampler is uninitialized")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	if s.closed || s.busy {
		s.mu.Unlock()
		return nil, fmt.Errorf("governed sampler is closed or busy")
	}
	ctx, cancel := context.WithCancel(ctx)
	s.busy = true
	s.cancel = cancel
	s.done = make(chan struct{})
	start := s.clock.NowMonotonic()
	s.mu.Unlock()
	defer func() {
		cancel()
		end := s.clock.NowMonotonic()
		if start < 0 || end < start || end < 0 {
			readings = nil
			if err == nil {
				err = fmt.Errorf("collector clock moved backwards")
			}
		}
		s.mu.Lock()
		s.busy = false
		s.cancel = nil
		if start >= 0 && end >= start {
			s.stats.LastQueryDurationNanos = end - start
		}
		if err != nil {
			s.stats.FailedQueries++
		} else {
			s.stats.SuccessfulQueries++
			s.stats.LastSuccessNanos = end
			s.stats.HasLastSuccess = true
		}
		s.stats.LastQueryFailed = err != nil
		closed := s.closed
		close(s.done)
		s.mu.Unlock()
		if closed {
			s.release()
		}
	}()
	if err := checkCaptureDevices(s.source.Devices(), s.policy.AllowedDevices); err != nil {
		return nil, err
	}
	readings, err = s.source.Sample(ctx)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := checkCaptureDevices(s.source.Devices(), s.policy.AllowedDevices); err != nil {
		return nil, err
	}
	if len(readings) > s.policy.MaxReadings {
		return nil, fmt.Errorf("sample exceeds reading limit")
	}
	allowed := make(map[string]bool)
	for _, id := range s.policy.AllowedDevices {
		allowed[id] = true
	}
	for _, reading := range readings {
		if !allowed[reading.DeviceUUID] {
			return nil, fmt.Errorf("sample device is outside target allowlist")
		}
		if reading.UnavailableReason != "" && !safeUnavailableReason(reading.UnavailableReason) {
			return nil, fmt.Errorf("sample has an unsupported unavailable reason")
		}
	}
	// Check variable extents before copying maps or allocating JSON. Encode
	// one bounded record at a time rather than materializing an oversized batch.
	owned := make([]Reading, 0)
	encodedBytes := 2
	var suppressed uint64
	for _, reading := range readings {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := checkGovernedStrings(reading, s.policy.MaxSampleBytes); err != nil {
			return nil, err
		}
		if err := ValidateReadingSource(reading.Source); err != nil {
			return nil, fmt.Errorf("validate governed source: %w", err)
		}
		if err := validateManagedValue(reading); err != nil {
			return nil, fmt.Errorf("validate governed mapped value: %w", err)
		}
		if reading.Source != nil && (reading.Source.SourceID != s.policy.SourceID || reading.Source.Mapping.Name != reading.Name || reading.Source.Producer != reading.Backend) {
			return nil, fmt.Errorf("sample source identity or mapping mismatch")
		}
		copy := cloneReadings([]Reading{reading})[0]
		if copy.Source != nil {
			labels, err := SafeCaptureLabels(copy.Source.Labels, s.policy.AllowedSourceLabels)
			if err != nil {
				return nil, err
			}
			for key, value := range copy.Source.Labels {
				switch key {
				case "UUID":
					if value != reading.DeviceUUID {
						return nil, fmt.Errorf("sample source device label mismatch")
					}
					labels[key] = value
				case "gpu", "device", "GPU_I_ID", "GPU_CI_ID", "modelName":
					labels[key] = value
				default:
					if _, allowed := labels[key]; !allowed {
						suppressed++
					}
				}
			}
			copy.Source.Labels = labels
		}
		b, err := json.Marshal(copy)
		if err != nil {
			return nil, fmt.Errorf("encode governed sample: %w", err)
		}
		separator := 0
		if len(owned) > 0 {
			separator = 1
		}
		if len(b) > s.policy.MaxSampleBytes-encodedBytes-separator {
			return nil, fmt.Errorf("sample exceeds byte limit")
		}
		encodedBytes += len(b) + separator
		owned = append(owned, copy)
	}
	s.mu.Lock()
	s.stats.SuppressedSourceLabels += suppressed
	s.mu.Unlock()
	return owned, nil
}

func checkGovernedStrings(r Reading, limit int) error {
	values := []string{r.Name, r.Unit, r.Backend, r.DeviceUUID, r.Evidence, r.UnavailableReason, r.SourceTimestampDomain}
	if source := r.Source; source != nil {
		if len(source.Labels) > 16 {
			return fmt.Errorf("sample source label limit")
		}
		values = append(values, source.Schema, source.SourceID, source.Producer, source.ProducerVersion, source.NativeOrigin, source.FieldID, source.RawValue, source.RawTimestamp, source.TimestampUnit, source.MappingDigest, source.Mapping.Metric, source.Mapping.Name, source.Mapping.Type, source.Mapping.SourceUnit)
		if source.Native != nil {
			if len(source.Native.FieldName) > 256 || len(source.Native.RawText) > 256 {
				return fmt.Errorf("sample native source string extent")
			}
			values = append(values, source.Native.FieldName, source.Native.RawText)
			if source.Native.FieldType != dcgm.FieldTypeInt64 && source.Native.FieldType != dcgm.FieldTypeDouble && source.Native.FieldType != dcgm.FieldTypeTimestamp {
				return fmt.Errorf("opaque native diagnostics require explicit raw capture")
			}
		}
		for key, value := range source.Labels {
			values = append(values, key, value)
		}
	}
	remaining := limit
	for _, value := range values {
		size, ok := governedStringSize(value, remaining)
		if !ok {
			return fmt.Errorf("sample string extents exceed byte limit")
		}
		remaining -= size
	}
	return nil
}

// governedStringSize counts encoding/json's default string escaping without
// allocating an escaped copy. The per-record encoder checks fixed overhead.
func governedStringSize(value string, limit int) (int, bool) {
	size := 2
	if size > limit {
		return 0, false
	}
	for i := 0; i < len(value); {
		cost, width := 1, 1
		c := value[i]
		if c < utf8.RuneSelf {
			switch c {
			case '\\', '"', '\b', '\f', '\n', '\r', '\t':
				cost = 2
			case '<', '>', '&':
				cost = 6
			default:
				if c < 0x20 {
					cost = 6
				}
			}
		} else {
			r, w := utf8.DecodeRuneInString(value[i:])
			width = w
			cost = w
			if r == utf8.RuneError && w == 1 {
				cost = 3
			}
			if r == '\u2028' || r == '\u2029' {
				cost = 6
			}
		}
		if cost > limit-size {
			return 0, false
		}
		size += cost
		i += width
	}
	return size, true
}

func safeUnavailableReason(reason string) bool {
	switch reason {
	case ReasonUnsupported, ReasonPermissionDenied, ReasonGPULost, ReasonBlank, ReasonStale, ReasonQueryFailure, ReasonGapExceeded, ReasonCounterReset, ReasonZeroTokens, ReasonMisalignedClock, "unknown_freshness", "non_finite", "unsupported_field_type":
		return true
	}
	return false
}

// Status returns a coherent snapshot of bounded collector self-observation.
func (s *GovernedSampler) Status() CaptureStatus {
	if s == nil {
		return CaptureStatus{Closed: true}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	status := s.stats
	status.Closed = s.closed
	status.InFlight = s.busy
	return status
}

// Close prevents new queries, cancels the current query and waits for its
// return, bounded by ctx. A timeout retains ownership until the query returns;
// it never permits a second collector to race an unfinished source.
func (s *GovernedSampler) Close(ctx context.Context) error {
	if s == nil || s.registry == nil {
		return fmt.Errorf("governed sampler is uninitialized")
	}
	s.mu.Lock()
	s.closed = true
	busy, done := s.busy, s.done
	if s.cancel != nil {
		s.cancel()
	}
	s.mu.Unlock()
	if busy {
		select {
		case <-done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	s.release()
	return nil
}

func (s *GovernedSampler) release() {
	r := s.registry
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, id := range s.policy.AllowedDevices {
		if r.devices[id] == s {
			delete(r.devices, id)
		}
	}
}

// SafeCaptureLabels selects explicit non-sensitive identity labels before
// storage/export. It accepts only job_id, namespace, executable and stack_symbol
// values with a conservative character set. Arbitrary command/environment and
// diagnostic text are unsupported; this is not general secret detection.
func SafeCaptureLabels(input map[string]string, allowed []string) (map[string]string, error) {
	if len(input) > 64 || len(allowed) > 4 {
		return nil, fmt.Errorf("capture labels exceed limits")
	}
	out := make(map[string]string)
	seen := make(map[string]bool)
	for _, key := range allowed {
		if seen[key] {
			return nil, fmt.Errorf("duplicate capture label %q", key)
		}
		seen[key] = true
		if key != "job_id" && key != "namespace" && key != "executable" && key != "stack_symbol" {
			return nil, fmt.Errorf("unsupported capture label %q", key)
		}
		value, ok := input[key]
		if !ok {
			continue
		}
		if len(value) > 256 || strings.Contains(strings.ToLower(value), "token") || strings.Contains(strings.ToLower(value), "password") || strings.Contains(strings.ToLower(value), "secret") {
			return nil, fmt.Errorf("unsafe capture label %q", key)
		}
		for _, c := range value {
			if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || strings.ContainsRune("._:/-", c)) {
				return nil, fmt.Errorf("unsafe capture label %q", key)
			}
		}
		out[key] = value
	}
	return out, nil
}

// Ready reports query readiness against this sampler's monotonic clock.
// Unavailable optional fields do not make a successful query fail readiness.
func (s *GovernedSampler) Ready(nowNanos, maxAgeNanos int64) error {
	status := s.Status()
	if status.Closed {
		return errors.New("governed sampler is closed")
	}
	if status.LastQueryFailed {
		return errors.New("governed sampler last query failed")
	}
	if !status.HasLastSuccess || nowNanos < status.LastSuccessNanos || maxAgeNanos < 0 || nowNanos-status.LastSuccessNanos > maxAgeNanos {
		return errors.New("governed sampler has no fresh successful query")
	}
	return nil
}
