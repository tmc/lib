package cudashadow

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// Domain classifies the GPU policy domain under evaluation.
type Domain string

const (
	// DomainEviction represents UVM physical memory page/chunk eviction policy.
	DomainEviction Domain = "eviction"
	// DomainPrefetch represents UVM page prefetch policy.
	DomainPrefetch Domain = "prefetch"
	// DomainScheduling represents GPU compute channel and timeslice scheduling policy.
	DomainScheduling Domain = "scheduling"
	// DomainPTXInstrument represents dynamic PTX instrumentation policy.
	DomainPTXInstrument Domain = "ptx_instrument"
)

// Hook identifies the driver struct_ops or runtime entrypoint.
type Hook string

const (
	// HookBlockActivate corresponds to UVM gpu_block_activate.
	HookBlockActivate Hook = "gpu_block_activate"
	// HookBlockAccess corresponds to UVM gpu_block_access.
	HookBlockAccess Hook = "gpu_block_access"
	// HookEvictPrepare corresponds to UVM gpu_evict_prepare.
	HookEvictPrepare Hook = "gpu_evict_prepare"
	// HookPagePrefetch corresponds to UVM gpu_page_prefetch.
	HookPagePrefetch Hook = "gpu_page_prefetch"
	// HookPagePrefetchIter corresponds to UVM gpu_page_prefetch_iter.
	HookPagePrefetchIter Hook = "gpu_page_prefetch_iter"
	// HookSchedTimeslice corresponds to GPU scheduler timeslice assignment.
	HookSchedTimeslice Hook = "gpu_sched_timeslice"
	// HookPTXInstrument corresponds to bpftime PTX attach interception.
	HookPTXInstrument Hook = "bpftime_ptx_attach"
)

// InputStatus indicates whether an observed input was valid, stale, or unsupported.
type InputStatus string

const (
	// StatusValid represents an input that passed freshness and capability validation.
	StatusValid InputStatus = "valid"
	// StatusStale represents an input whose timestamp expired or whose sequence arrived out-of-order.
	StatusStale InputStatus = "stale"
	// StatusUnsupported represents an input targeting an unrecognized hook, domain, or unsupported device.
	StatusUnsupported InputStatus = "unsupported"
)

// DecisionType specifies the proposed policy action.
type DecisionType string

const (
	// DecisionBypass skips custom policy and uses default driver behavior.
	DecisionBypass DecisionType = "bypass"
	// DecisionProtect moves chunk to tail to protect it from eviction.
	DecisionProtect DecisionType = "protect"
	// DecisionEvict marks chunk for early eviction (e.g. move to head).
	DecisionEvict DecisionType = "evict"
	// DecisionPrefetch triggers asynchronous page prefetch for a target range.
	DecisionPrefetch DecisionType = "prefetch"
	// DecisionThrottle suppresses prefetching or evictions under contention.
	DecisionThrottle DecisionType = "throttle"
	// DecisionTimeslice sets the timeslice duration for a scheduling channel.
	DecisionTimeslice DecisionType = "timeslice"
	// DecisionSkip executes no action.
	DecisionSkip DecisionType = "skip"
	// DecisionReject explicitly rejects an unsupported or invalid input.
	DecisionReject DecisionType = "reject"
)

// Provenance captures the provenance and observation context of a policy input.
type Provenance struct {
	DeviceID      int           `json:"device_id"`
	DriverVersion string        `json:"driver_version"`
	KernelModule  string        `json:"kernel_module"`
	Hook          Hook          `json:"hook"`
	Sequence      uint64        `json:"sequence"`
	TimestampNano int64         `json:"timestamp_nano"`
	TTL           time.Duration `json:"ttl,omitempty"`
}

// Validate verifies that mandatory provenance fields are present and well-formed.
func (p Provenance) Validate() error {
	if len(p.DriverVersion) > 256 || len(p.KernelModule) > 256 || len(p.Hook) > 256 {
		return errors.New("cudashadow: provenance field too large")
	}
	if strings.TrimSpace(p.DriverVersion) == "" {
		return errors.New("cudashadow: provenance missing driver version")
	}
	if strings.TrimSpace(p.KernelModule) == "" {
		return errors.New("cudashadow: provenance missing kernel module")
	}
	if strings.TrimSpace(string(p.Hook)) == "" {
		return errors.New("cudashadow: provenance missing hook")
	}
	if p.DeviceID < 0 || p.TTL < 0 {
		return errors.New("cudashadow: negative device ID or ttl")
	}
	if p.TimestampNano <= 0 {
		return errors.New("cudashadow: provenance timestamp must be positive")
	}
	return nil
}

// PolicyInfo identifies the policy implementation, version, and configuration.
type PolicyInfo struct {
	Name       string            `json:"name"`
	Version    string            `json:"version"`
	Domain     Domain            `json:"domain"`
	Parameters map[string]string `json:"parameters,omitempty"`
	Seed       int64             `json:"seed"`
}

// Validate ensures policy metadata is complete.
func (pi PolicyInfo) Validate() error {
	if len(pi.Name) > 256 || len(pi.Version) > 256 || len(pi.Domain) > 256 {
		return errors.New("cudashadow: policy field too large")
	}
	if strings.TrimSpace(pi.Name) == "" {
		return errors.New("cudashadow: policy missing name")
	}
	if strings.TrimSpace(pi.Version) == "" {
		return errors.New("cudashadow: policy missing version")
	}
	if strings.TrimSpace(string(pi.Domain)) == "" {
		return errors.New("cudashadow: policy missing domain")
	}
	if len(pi.Parameters) > 64 {
		return errors.New("cudashadow: too many policy parameters")
	}
	for k, v := range pi.Parameters {
		if len(k) > 256 || len(v) > 4096 {
			return errors.New("cudashadow: policy parameter too large")
		}
	}
	return nil
}

// EvictionData holds UVM chunk access/eviction input parameters.
type EvictionData struct {
	PID         int    `json:"pid"`
	ChunkAddr   uint64 `json:"chunk_addr"`
	ChunkSize   uint64 `json:"chunk_size"`
	VABlockAddr uint64 `json:"va_block_addr"`
	AccessCount uint64 `json:"access_count"`
}

// PrefetchData holds fault and address range prefetch input parameters.
type PrefetchData struct {
	PID          int    `json:"pid"`
	FaultAddr    uint64 `json:"fault_addr"`
	FirstPage    uint16 `json:"first_page"`
	OuterPage    uint16 `json:"outer_page"`
	AccessBitmap uint64 `json:"access_bitmap"`
	StrideBytes  int64  `json:"stride_bytes"`
}

// SchedulingData holds channel and context scheduling input parameters.
type SchedulingData struct {
	PID          int    `json:"pid"`
	ChannelID    uint32 `json:"channel_id"`
	CurrentSlice uint64 `json:"current_slice_us"`
	QueueLength  uint32 `json:"queue_length"`
	Priority     int    `json:"priority"`
}

// PTXData holds kernel instrumentation input parameters.
type PTXData struct {
	KernelName     string `json:"kernel_name"`
	TargetArch     string `json:"target_arch"`
	FunctionOffset uint64 `json:"function_offset"`
	CodeSize       uint64 `json:"code_size"`
}

// Input represents an observed policy input payload.
type Input struct {
	Provenance Provenance      `json:"provenance"`
	Domain     Domain          `json:"domain"`
	Eviction   *EvictionData   `json:"eviction,omitempty"`
	Prefetch   *PrefetchData   `json:"prefetch,omitempty"`
	Scheduling *SchedulingData `json:"scheduling,omitempty"`
	PTX        *PTXData        `json:"ptx,omitempty"`
}

// Validate verifies consistency between domain and payload.
func (in Input) Validate() error {
	if err := in.validateBounds(); err != nil {
		return err
	}
	if err := in.Provenance.Validate(); err != nil {
		return err
	}
	count := 0
	for _, present := range []bool{in.Eviction != nil, in.Prefetch != nil, in.Scheduling != nil, in.PTX != nil} {
		if present {
			count++
		}
	}
	if count != 1 {
		return errors.New("cudashadow: expected exactly one domain payload")
	}
	switch in.Provenance.Hook {
	case HookBlockActivate, HookBlockAccess, HookEvictPrepare:
		if in.Domain != DomainEviction {
			return errors.New("cudashadow: hook/domain mismatch")
		}
	case HookPagePrefetch, HookPagePrefetchIter:
		if in.Domain != DomainPrefetch {
			return errors.New("cudashadow: hook/domain mismatch")
		}
	case HookSchedTimeslice:
		if in.Domain != DomainScheduling {
			return errors.New("cudashadow: hook/domain mismatch")
		}
	case HookPTXInstrument:
		if in.Domain != DomainPTXInstrument {
			return errors.New("cudashadow: hook/domain mismatch")
		}
	default:
		return fmt.Errorf("cudashadow: %w: unknown hook %q", cudaerrors.ErrUnsupported, in.Provenance.Hook)
	}
	switch in.Domain {
	case DomainEviction:
		if in.Eviction == nil {
			return errors.New("cudashadow: eviction domain missing eviction payload")
		}
		if in.Eviction.ChunkAddr > math.MaxInt64 {
			return errors.New("cudashadow: chunk address cannot be represented in decision")
		}
	case DomainPrefetch:
		if in.Prefetch == nil {
			return errors.New("cudashadow: prefetch domain missing prefetch payload")
		}
		if in.Prefetch.PID <= 0 || in.Prefetch.FaultAddr > math.MaxInt64 || in.Prefetch.FirstPage > in.Prefetch.OuterPage || in.Prefetch.OuterPage > 512 {
			return errors.New("cudashadow: invalid prefetch range or identity")
		}
	case DomainScheduling:
		if in.Scheduling == nil {
			return errors.New("cudashadow: scheduling domain missing scheduling payload")
		}
	case DomainPTXInstrument:
		if in.PTX == nil {
			return errors.New("cudashadow: ptx_instrument domain missing ptx payload")
		}
		if in.PTX.CodeSize > math.MaxInt64 || in.PTX.FunctionOffset > in.PTX.CodeSize {
			return errors.New("cudashadow: invalid ptx bounds")
		}
		if len(in.PTX.KernelName) > 4096 || len(in.PTX.TargetArch) > 256 {
			return errors.New("cudashadow: ptx metadata too large")
		}
	default:
		return fmt.Errorf("cudashadow: %w: unknown domain %q", cudaerrors.ErrUnsupported, in.Domain)
	}
	return nil
}

func (in Input) validateBounds() error {
	if len(in.Domain) > 256 || len(in.Provenance.DriverVersion) > 256 || len(in.Provenance.KernelModule) > 256 || len(in.Provenance.Hook) > 256 {
		return errors.New("cudashadow: input field too large")
	}
	if in.PTX != nil && (len(in.PTX.KernelName) > 4096 || len(in.PTX.TargetArch) > 256) {
		return errors.New("cudashadow: ptx field too large")
	}
	return nil
}

// Decision represents the proposed action, parameters, and rationale.
type Decision struct {
	Type       DecisionType     `json:"type"`
	Parameters map[string]int64 `json:"parameters,omitempty"`
	Reason     string           `json:"reason"`
}

func (d Decision) validate(domain Domain) error {
	if d.Reason == "" || len(d.Reason) > 4096 || len(d.Parameters) > 64 {
		return errors.New("cudashadow: invalid decision fields")
	}
	for k := range d.Parameters {
		if len(k) > 256 {
			return errors.New("cudashadow: decision parameter too large")
		}
	}
	switch d.Type {
	case DecisionBypass, DecisionSkip, DecisionReject:
		return nil
	case DecisionEvict, DecisionProtect:
		if domain == DomainEviction {
			return nil
		}
	case DecisionPrefetch, DecisionThrottle:
		if domain == DomainPrefetch {
			return nil
		}
	case DecisionTimeslice:
		if domain == DomainScheduling {
			return nil
		}
	}
	return fmt.Errorf("cudashadow: %w: decision %q for domain %q", cudaerrors.ErrUnsupported, d.Type, domain)
}

// ObservedOutcome records the baseline outcome observed without policy mutation.
type ObservedOutcome struct {
	TimestampNano int64             `json:"timestamp_nano"`
	Action        string            `json:"action"`
	LatencyNano   int64             `json:"latency_nano"`
	Details       map[string]string `json:"details,omitempty"`
}

func (outcome ObservedOutcome) validate() error {
	if outcome.TimestampNano <= 0 || outcome.LatencyNano < 0 || outcome.Action == "" || len(outcome.Action) > 4096 || len(outcome.Details) > 64 {
		return errors.New("cudashadow: invalid observed outcome")
	}
	for k, v := range outcome.Details {
		if len(k) > 256 || len(v) > 4096 {
			return errors.New("cudashadow: outcome detail too large")
		}
	}
	return nil
}

// Snapshot is a bounded, serializable record of a shadow policy decision.
type Snapshot struct {
	ID              string           `json:"id"`
	Provenance      Provenance       `json:"provenance"`
	Policy          PolicyInfo       `json:"policy"`
	Input           Input            `json:"input"`
	Decision        Decision         `json:"decision"`
	Status          InputStatus      `json:"status"`
	ObservedOutcome *ObservedOutcome `json:"observed_outcome,omitempty"`
}

// Validate verifies that a snapshot contains required fields.
func (s Snapshot) Validate() error {
	if err := s.Input.validateBounds(); err != nil {
		return err
	}
	if strings.TrimSpace(s.ID) == "" || len(s.ID) > 256 {
		return errors.New("cudashadow: snapshot missing id")
	}
	if s.Status != StatusValid && s.Status != StatusStale && s.Status != StatusUnsupported {
		return errors.New("cudashadow: invalid snapshot status")
	}
	if err := s.Provenance.Validate(); err != nil && s.Status != StatusUnsupported {
		return fmt.Errorf("cudashadow snapshot provenance: %w", err)
	}
	if err := s.Policy.Validate(); err != nil {
		return fmt.Errorf("cudashadow snapshot policy: %w", err)
	}
	if err := s.Input.Validate(); err != nil && s.Status != StatusUnsupported {
		return fmt.Errorf("cudashadow snapshot input: %w", err)
	}
	if err := s.Decision.validate(s.Input.Domain); err != nil {
		return err
	}
	if strings.TrimSpace(s.Decision.Reason) == "" {
		return errors.New("cudashadow: snapshot missing decision reason")
	}
	if s.Status != StatusValid && s.Status != StatusStale && s.Status != StatusUnsupported {
		return errors.New("cudashadow: invalid snapshot status")
	}
	if s.Provenance != s.Input.Provenance {
		return errors.New("cudashadow: contradictory snapshot provenance")
	}
	if s.Status == StatusValid && s.Policy.Domain != s.Input.Domain {
		return errors.New("cudashadow: contradictory policy domain")
	}
	if s.Status == StatusStale && s.Decision.Type != DecisionSkip || s.Status == StatusUnsupported && s.Decision.Type != DecisionReject {
		return errors.New("cudashadow: status contradicts decision")
	}
	if s.ObservedOutcome != nil {
		if err := s.ObservedOutcome.validate(); err != nil {
			return err
		}
		if s.Status != StatusValid || s.ObservedOutcome.TimestampNano < s.Provenance.TimestampNano || s.ObservedOutcome.LatencyNano < 0 || s.ObservedOutcome.Action == "" {
			return errors.New("cudashadow: invalid observed outcome")
		}
	}
	return nil
}

// MarshalJSON serializes a snapshot to indented or standard JSON.
func (s Snapshot) MarshalJSON() ([]byte, error) {
	type Alias Snapshot
	return json.Marshal((Alias)(s))
}

// DecodeSnapshot parses a single JSON Snapshot, ensuring all required fields are present
// and rejecting unknown fields or trailing input.
func DecodeSnapshot(data []byte) (Snapshot, error) {
	if len(data) > 64<<10 {
		return Snapshot{}, errors.New("cudashadow: snapshot exceeds 64 KiB")
	}
	if err := uniqueJSON(data); err != nil {
		return Snapshot{}, err
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	var s Snapshot
	if err := d.Decode(&s); err != nil {
		return Snapshot{}, fmt.Errorf("cudashadow decode snapshot: %w", err)
	}
	var trailing any
	if err := d.Decode(&trailing); err != io.EOF {
		return Snapshot{}, errors.New("cudashadow decode snapshot: trailing data")
	}
	if err := s.Validate(); err != nil {
		return Snapshot{}, err
	}
	return s, nil
}
