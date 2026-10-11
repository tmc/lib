package nvtelemetry

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"github.com/tmc/lib/nvidia/nvml"
)

type healthNVMLDevice interface {
	UUID() (string, error)
	TotalEccErrors(nvml.MemoryErrorType, nvml.EccCounterType) (uint64, error)
	RemappedRowsV1() (uint32, uint32, uint32, uint32, error)
	Temperature(nvml.TemperatureSensors) (uint32, error)
	PowerUsage() (uint32, error)
	CurrentClocksEventReasons() (uint64, error)
	CurrPcieLinkGeneration() (uint32, error)
	CurrPcieLinkWidth() (uint32, error)
	NvLinkState(uint32) (nvml.EnableState, error)
}

// NVMLHealthReader reads selected, existing NVML health APIs for one device.
// It never enumerates other devices, registers events, changes link settings
// or invokes repair/reset. The caller owns NVML initialization and the handle.
// It is not safe for concurrent use; the zero value is not usable.
type NVMLHealthReader struct {
	device        healthNVMLDevice
	clock         Clock
	uuid          string
	version       string
	nodeID        string
	topologyEpoch string
	scope         string
	links         []uint32
}

// NewNVMLHealthReader binds a device handle to its queried UUID. scope must be
// a MIGScope constant: a caller without independently established scope must
// use MIGScopeUnknown. links explicitly selects at most 32 link indices.
// The reader rechecks UUID before every batch and refuses identity changes.
func NewNVMLHealthReader(device nvml.Device, clock Clock, nodeID, topologyEpoch, scope string, links []uint32) (*NVMLHealthReader, error) {
	if device == 0 {
		return nil, fmt.Errorf("NVML health reader requires a device handle")
	}
	version, err := nvml.SystemGetNVMLVersion()
	if err != nil {
		return nil, fmt.Errorf("query health NVML version: %w", err)
	}
	return newNVMLHealthReader(device, clock, nodeID, topologyEpoch, scope, links, version)
}

func newNVMLHealthReader(device healthNVMLDevice, clock Clock, nodeID, topologyEpoch, scope string, links []uint32, version string) (*NVMLHealthReader, error) {
	if device == nil || clock == nil || nodeID == "" || topologyEpoch == "" || version == "" || len(links) > 32 {
		return nil, fmt.Errorf("invalid NVML health reader configuration")
	}
	if scope != MIGScopeDisabled && scope != MIGScopePhysicalParent && scope != MIGScopeMIGInstance && scope != MIGScopeUnknown {
		return nil, fmt.Errorf("invalid NVML health scope")
	}
	seen := make(map[uint32]bool)
	for _, link := range links {
		if link >= 32 || seen[link] {
			return nil, fmt.Errorf("invalid or duplicate NVLink index")
		}
		seen[link] = true
	}
	uuid, err := device.UUID()
	if err != nil {
		return nil, fmt.Errorf("query health device identity: %w", err)
	}
	if uuid == "" {
		return nil, fmt.Errorf("health device UUID is empty")
	}
	return &NVMLHealthReader{device: device, clock: clock, uuid: uuid, version: version, nodeID: nodeID, topologyEpoch: topologyEpoch, scope: scope, links: append([]uint32(nil), links...)}, nil
}

// Read returns a bounded read-only health bundle with one query interval per
// field. ECC counters are volatile (driver-lifetime), not lifetime totals.
// Link state is a boolean, not traffic or a peer-identity observation. Sensor
// timestamps and integration intervals remain unknown. Field failures retain
// typed unavailable reasons and native error text; sanitize diagnostic text
// before external publication. No measurements are invented for missing APIs.
// Cancellation is checked between NVML calls and cannot interrupt a native call.
func (r *NVMLHealthReader) Read(ctx context.Context) (*HealthSnapshot, error) {
	if r == nil || r.device == nil {
		return nil, fmt.Errorf("NVML health reader is uninitialized")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	uuid, err := r.device.UUID()
	if err != nil {
		return nil, fmt.Errorf("query health device identity: %w", err)
	}
	if uuid != r.uuid {
		return nil, fmt.Errorf("NVML health device identity changed")
	}
	domain, epoch := clockMetadata(r.clock)
	if domain == "" || epoch == "" {
		return nil, fmt.Errorf("health clock identity is unavailable")
	}
	s := &HealthSnapshot{Version: 1, NodeID: r.nodeID, TopologyEpoch: r.topologyEpoch, ClockDomain: domain, ClockEpoch: epoch, DeviceStates: map[string]string{uuid: "present"}, Complete: true}
	source := HealthSource{ID: "nvml", Kind: "nvml", Version: r.version, State: "available"}
	add := func(name, unit, scope string, query func() (uint64, error)) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		start := r.clock.NowMonotonic()
		value, err := query()
		end := r.clock.NowMonotonic()
		if start < 0 || end < start {
			return fmt.Errorf("health clock moved backwards")
		}
		reading := Reading{Name: name, Unit: unit, Backend: "nvml", DeviceUUID: uuid, Evidence: EvidenceMeasured, IsInt: true, IntValue: value, Value: float64(value), TimestampMonotonic: end, TimestampUnix: r.clock.NowUnix(), HasQueryBounds: true, QueryStartNanos: start, QueryEndNanos: end}
		o := HealthObservation{SourceID: "nvml", Class: "raw", Reading: reading, Scope: scope}
		if err != nil {
			o.Reading.Evidence = EvidenceUnavailable
			o.Reading.UnavailableReason = ClassifyNVMLError(err)
			o.Reading.IntValue = 0
			o.Reading.Value = 0
			o.NativeError = err.Error()
			if o.Reading.UnavailableReason == ReasonGPULost {
				s.DeviceStates[uuid] = "lost"
			}
		} else {
			source.HasLastSuccess = true
			source.LastSuccessNanos = end
		}
		s.Observations = append(s.Observations, o)
		return nil
	}
	queries := []struct {
		name, unit string
		query      func() (uint64, error)
	}{
		{"ecc_corrected_errors", "count", func() (uint64, error) {
			return r.device.TotalEccErrors(nvml.MemoryErrorTypeCorrected, nvml.VolatileEcc)
		}},
		{"ecc_uncorrected_errors", "count", func() (uint64, error) {
			return r.device.TotalEccErrors(nvml.MemoryErrorTypeUncorrected, nvml.VolatileEcc)
		}},
		{MetricTemperatureC, UnitCelsius, func() (uint64, error) { v, e := r.device.Temperature(nvml.TemperatureGpu); return uint64(v), e }},
		{MetricPowerUsageMW, UnitMilliwatts, func() (uint64, error) { v, e := r.device.PowerUsage(); return uint64(v), e }},
		{MetricClocksEventReasons, UnitBitmask, r.device.CurrentClocksEventReasons},
		{"pcie_link_generation", "generation", func() (uint64, error) { v, e := r.device.CurrPcieLinkGeneration(); return uint64(v), e }},
		{"pcie_link_width", "lanes", func() (uint64, error) { v, e := r.device.CurrPcieLinkWidth(); return uint64(v), e }},
	}
	for _, q := range queries {
		if err := add(q.name, q.unit, r.scope, q.query); err != nil {
			return nil, err
		}
	}
	// RemappedRowsV1 is the existing scalar-output API. Avoid introducing a
	// versioned struct layout merely to read the same available facts.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	start := r.clock.NowMonotonic()
	corrected, uncorrected, pending, failure, remapErr := r.device.RemappedRowsV1()
	end := r.clock.NowMonotonic()
	if start < 0 || end < start {
		return nil, fmt.Errorf("health clock moved backwards")
	}
	for _, q := range []struct {
		name, unit string
		value      uint32
	}{{"corrected_remapped_rows", "count", corrected}, {"uncorrected_remapped_rows", "count", uncorrected}, {"repair_pending", "boolean", pending}, {"remap_failure", "boolean", failure}} {
		o := HealthObservation{SourceID: "nvml", Class: "raw", Scope: r.scope, Reading: Reading{Name: q.name, Unit: q.unit, Backend: "nvml", DeviceUUID: uuid, Evidence: EvidenceMeasured, IsInt: true, IntValue: uint64(q.value), Value: float64(q.value), TimestampMonotonic: end, TimestampUnix: r.clock.NowUnix(), HasQueryBounds: true, QueryStartNanos: start, QueryEndNanos: end}}
		if remapErr != nil {
			o.Reading.Evidence = EvidenceUnavailable
			o.Reading.UnavailableReason = ClassifyNVMLError(remapErr)
			o.Reading.Value = 0
			o.Reading.IntValue = 0
			o.NativeError = remapErr.Error()
			if o.Reading.UnavailableReason == ReasonGPULost {
				s.DeviceStates[uuid] = "lost"
			}
		} else {
			source.HasLastSuccess = true
			source.LastSuccessNanos = end
		}
		s.Observations = append(s.Observations, o)
	}
	for _, link := range r.links {
		if err := add("nvlink_state", "boolean", r.scope+"/link/"+strconv.FormatUint(uint64(link), 10), func() (uint64, error) {
			v, e := r.device.NvLinkState(link)
			if e == nil && v != nvml.FeatureDisabled && v != nvml.FeatureEnabled {
				return 0, fmt.Errorf("invalid NVLink enable state")
			}
			return uint64(v), e
		}); err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.AtNanos = r.clock.NowMonotonic()
	finalUUID, identityErr := r.device.UUID()
	if identityErr != nil {
		if ClassifyNVMLError(identityErr) == ReasonGPULost {
			s.DeviceStates[uuid] = "lost"
		} else {
			return nil, fmt.Errorf("recheck health device identity: %w", identityErr)
		}
	} else if finalUUID != r.uuid {
		return nil, fmt.Errorf("NVML health device identity changed during query")
	}
	s.AtNanos = r.clock.NowMonotonic()
	if s.DeviceStates[uuid] == "lost" {
		for i := range s.Observations {
			o := &s.Observations[i]
			o.Reading.Evidence = EvidenceUnavailable
			o.Reading.UnavailableReason = ReasonGPULost
			o.Reading.Value = 0
			o.Reading.IntValue = 0
		}
	}
	allFailed, allDenied := len(s.Observations) > 0, len(s.Observations) > 0
	for _, o := range s.Observations {
		allFailed = allFailed && o.Reading.UnavailableReason == ReasonQueryFailure
		allDenied = allDenied && o.Reading.UnavailableReason == ReasonPermissionDenied
	}
	if allFailed {
		source.State = "failed"
	}
	if allDenied {
		source.State = "denied"
	}
	s.Sources = []HealthSource{source}
	limits := HealthLimits{MaxBytes: 64 << 10, MaxDevices: 1, MaxSources: 1, MaxObservations: 64, MaxEdges: 1}
	if err := s.Validate(limits); err != nil {
		return nil, err
	}
	return s, nil
}

func clockMetadata(clock Clock) (string, string) {
	if c, ok := clock.(ClockMetadata); ok {
		return c.MonotonicDomain(), c.MonotonicEpochID()
	}
	return "", ""
}

// NVMLXIDReader owns an event set registered solely for a queried device UUID.
// It reports native XID codes without executing catalog recommendations.
// Calls must be serialized. Close errors retain event-set ownership for an
// explicit caller retry. The zero value is not usable.
type NVMLXIDReader struct {
	clock      Clock
	uuid       string
	wait       func(*nvml.EventData, uint32) error
	identity   func(nvml.Device) (string, error)
	closeSet   func() error
	closed     bool
	registered bool
}

// NewNVMLXIDReader registers critical XID events on one independently queried
// UUID. It does not claim per-MIG-instance reporting: raw native instance IDs
// remain in observation Scope. The caller owns NVML initialization.
// If registration and cleanup both fail, a nonnil reader is returned with the
// error so the caller can retain and explicitly close the event set.
func NewNVMLXIDReader(device nvml.Device, clock Clock) (*NVMLXIDReader, error) {
	if device == 0 || clock == nil {
		return nil, fmt.Errorf("XID reader requires a device and clock")
	}
	uuid, err := device.UUID()
	if err != nil {
		return nil, fmt.Errorf("query XID device identity: %w", err)
	}
	if uuid == "" {
		return nil, fmt.Errorf("XID device UUID is empty")
	}
	set, err := nvml.NewEventSet()
	if err != nil {
		return nil, fmt.Errorf("create XID event set: %w", err)
	}
	r := &NVMLXIDReader{clock: clock, uuid: uuid, wait: set.Wait, identity: func(d nvml.Device) (string, error) { return d.UUID() }, closeSet: set.Close}
	if err := device.RegisterEvents(nvml.EventTypeXidCriticalError, set); err != nil {
		if cleanupErr := r.Close(); cleanupErr != nil {
			return r, errors.Join(fmt.Errorf("register XID events: %w", err), cleanupErr)
		}
		return nil, fmt.Errorf("register XID events: %w", err)
	}
	r.registered = true
	return r, nil
}

// Poll performs at most maxEvents zero-timeout native waits. A timeout means
// no event was observed, not that the GPU is healthy. Events have query bounds
// only; NVML does not provide a sensor timestamp here. Saturation returns the
// bounded observations plus an error rather than claiming a complete drain.
// An identity contradiction discards the batch; other failures retain its
// partial observations and return an error. Callers must record that loss.
func (r *NVMLXIDReader) Poll(ctx context.Context, maxEvents int) ([]HealthObservation, error) {
	if r == nil || r.wait == nil || !r.registered || r.closed || maxEvents <= 0 || maxEvents > 1024 {
		return nil, fmt.Errorf("invalid XID reader or event limit")
	}
	if domain, epoch := clockMetadata(r.clock); domain == "" || epoch == "" {
		return nil, fmt.Errorf("XID clock identity is unavailable")
	}
	out := make([]HealthObservation, 0, maxEvents)
	for i := 0; i < maxEvents; i++ {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		var event nvml.EventData
		start := r.clock.NowMonotonic()
		err := r.wait(&event, 0)
		end := r.clock.NowMonotonic()
		if start < 0 || end < start {
			return out, fmt.Errorf("XID query clock moved backwards")
		}
		if errors.Is(err, nvml.ErrTimeout) {
			return out, nil
		}
		if err != nil {
			return out, fmt.Errorf("query XID event: %w", err)
		}
		uuid, err := r.identity(event.Device)
		if err != nil {
			return out, fmt.Errorf("query XID event identity: %w", err)
		}
		if uuid != r.uuid || event.EventType != nvml.EventTypeXidCriticalError {
			return nil, fmt.Errorf("XID event identity or type mismatch")
		}
		out = append(out, HealthObservation{SourceID: "nvml", Class: "raw", Scope: fmt.Sprintf("native_instance_ids/%d/%d", event.GpuInstanceId, event.ComputeInstanceId), Reading: Reading{Name: MetricXIDEvent, Unit: "code", Backend: "nvml", DeviceUUID: uuid, Evidence: EvidenceMeasured, IsInt: true, IntValue: event.EventData, Value: float64(event.EventData), TimestampMonotonic: end, TimestampUnix: r.clock.NowUnix(), HasQueryBounds: true, QueryStartNanos: start, QueryEndNanos: end}})
	}
	return out, fmt.Errorf("XID event drain reached record limit")
}

// Close releases only this reader's event set. A failure leaves it owned and
// usable for an explicit close retry; no automatic retry is performed.
func (r *NVMLXIDReader) Close() error {
	if r == nil || r.closed {
		return nil
	}
	if r.closeSet == nil {
		return fmt.Errorf("XID reader is uninitialized")
	}
	if err := r.closeSet(); err != nil {
		return fmt.Errorf("close XID event set: %w", err)
	}
	r.closed = true
	return nil
}
