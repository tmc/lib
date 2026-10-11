package nvtelemetry

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/nvml"
)

// NVMLSampler acquires live hardware telemetry from NVIDIA GPUs through NVML.
type NVMLSampler struct {
	devices  []nvml.Device
	metadata []DeviceMetadata
	clock    Clock
}

// NewNVMLSampler creates a sampler for the given devices. If devices is empty,
// it queries all available NVML devices.
func NewNVMLSampler(clock Clock, devices ...nvml.Device) (*NVMLSampler, error) {
	if clock == nil {
		clock = RealClock{}
	}
	s := &NVMLSampler{
		clock: clock,
	}

	if len(devices) == 0 {
		count, err := nvml.DeviceGetCount()
		if err != nil {
			return nil, fmt.Errorf("device get count: %w", err)
		}
		for i := uint32(0); i < count; i++ {
			d, err := nvml.NewDeviceByIndex(i)
			if err != nil {
				return nil, fmt.Errorf("device handle by index %d: %w", i, err)
			}
			devices = append(devices, d)
		}
	}

	driverVersion, _ := nvml.SystemGetDriverVersion()
	libVersion, _ := nvml.SystemGetNVMLVersion()

	s.devices = devices
	s.metadata = make([]DeviceMetadata, len(devices))
	for i, d := range devices {
		var (
			failures []MetadataFailure
			warnings []string
		)

		uuid, uuidErr := d.UUID()
		if uuidErr != nil || uuid == "" {
			reason := ReasonQueryFailure
			if uuidErr != nil {
				reason = ClassifyNVMLError(uuidErr)
			}
			failures = append(failures, MetadataFailure{
				Property: "uuid",
				Reason:   reason,
				Error:    fmt.Sprintf("failed to query device UUID: %v", uuidErr),
			})
			warnings = append(warnings, "failed to query device UUID")
		}

		name, nameErr := d.Name()
		if nameErr != nil {
			failures = append(failures, MetadataFailure{
				Property: "name",
				Reason:   ClassifyNVMLError(nameErr),
				Error:    nameErr.Error(),
			})
		}

		idx, idxErr := d.Index()
		if idxErr != nil {
			failures = append(failures, MetadataFailure{
				Property: "index",
				Reason:   ClassifyNVMLError(idxErr),
				Error:    idxErr.Error(),
			})
		}

		var archStr string
		if arch, err := d.Architecture(); err == nil {
			archStr = fmt.Sprintf("0x%x", uint32(arch))
		} else {
			failures = append(failures, MetadataFailure{
				Property: "architecture",
				Reason:   ClassifyNVMLError(err),
				Error:    err.Error(),
			})
		}

		var powerLimit uint32
		if pl, err := d.EnforcedPowerLimit(); err == nil {
			powerLimit = pl
		} else {
			failures = append(failures, MetadataFailure{
				Property: "power_limit_mw",
				Reason:   ClassifyNVMLError(err),
				Error:    err.Error(),
			})
		}

		var maxSM uint32
		if sm, err := d.MaxCustomerBoostClock(nvml.ClockSm); err == nil {
			maxSM = sm
		} else {
			failures = append(failures, MetadataFailure{
				Property: "max_sm_clock_mhz",
				Reason:   ClassifyNVMLError(err),
				Error:    err.Error(),
			})
		}

		var maxMem uint32
		if mem, err := d.MaxCustomerBoostClock(nvml.ClockMem); err == nil {
			maxMem = mem
		} else {
			failures = append(failures, MetadataFailure{
				Property: "max_memory_clock_mhz",
				Reason:   ClassifyNVMLError(err),
				Error:    err.Error(),
			})
		}

		// MIG discovery
		migScope := MIGScopeDisabled
		migEnabled := false
		var migParentUUID string
		var migInstanceID int
		var hasMIGInstanceID bool
		var migComputeInstanceID int
		var hasMIGComputeInstanceID bool

		if isMig, err := d.IsMigDeviceHandle(); err == nil && isMig == 1 {
			migScope = MIGScopeMIGInstance
			migEnabled = true
			if parent, err := d.HandleFromMigDeviceHandle(); err == nil {
				if puuid, err := parent.UUID(); err == nil {
					migParentUUID = puuid
				}
			}
			if gi, err := d.GpuInstanceId(); err == nil {
				migInstanceID = int(gi)
				hasMIGInstanceID = true
			} else {
				failures = append(failures, MetadataFailure{
					Property: "mig_instance_id",
					Reason:   ClassifyNVMLError(err),
					Error:    err.Error(),
				})
				warnings = append(warnings, fmt.Sprintf("gpu instance id query failed: %v", err))
			}
			if ci, err := d.ComputeInstanceId(); err == nil {
				migComputeInstanceID = int(ci)
				hasMIGComputeInstanceID = true
			} else {
				failures = append(failures, MetadataFailure{
					Property: "mig_compute_instance_id",
					Reason:   ClassifyNVMLError(err),
					Error:    err.Error(),
				})
				warnings = append(warnings, fmt.Sprintf("compute instance id query failed: %v", err))
			}
		} else {
			currentMig, _, migErr := d.MigMode()
			if migErr != nil {
				if errors.Is(migErr, nvml.ErrNotSupported) || errors.Is(migErr, nvml.ErrFunctionNotFound) {
					migScope = MIGScopeUnsupported
				} else if errors.Is(migErr, nvml.ErrNoPermission) {
					migScope = MIGScopePermissionDenied
					warnings = append(warnings, fmt.Sprintf("mig mode query permission denied: %v", migErr))
				} else {
					migScope = MIGScopeUnknown
					warnings = append(warnings, fmt.Sprintf("mig mode query failed: %v", migErr))
				}
				failures = append(failures, MetadataFailure{
					Property: "mig_mode",
					Reason:   ClassifyNVMLError(migErr),
					Error:    migErr.Error(),
				})
			} else if currentMig == 1 {
				migScope = MIGScopePhysicalParent
				migEnabled = true
			} else {
				migScope = MIGScopeDisabled
				migEnabled = false
			}
		}

		status := DeviceStatusOk
		if uuid == "" {
			status = DeviceStatusUnidentifiable
		}

		// Dynamic per-field capability assessment
		fieldStatuses := make(map[string]string)
		var fieldCapabilities []string

		assessField := func(metric string, probeErr error) {
			if probeErr == nil {
				fieldStatuses[metric] = CapabilityQueryable
				fieldCapabilities = append(fieldCapabilities, metric)
			} else if errors.Is(probeErr, nvml.ErrNotSupported) || errors.Is(probeErr, nvml.ErrFunctionNotFound) {
				fieldStatuses[metric] = CapabilityUnsupported
			} else if errors.Is(probeErr, nvml.ErrNoPermission) {
				fieldStatuses[metric] = CapabilityPermissionDenied
			} else {
				fieldStatuses[metric] = CapabilityUnknown
			}
		}

		_, utilErr := d.UtilizationRates()
		assessField(MetricGPUUtilization, utilErr)
		assessField(MetricMemoryUtilization, utilErr)

		_, memErr := d.MemoryInfo()
		assessField(MetricMemoryUsedBytes, memErr)
		assessField(MetricMemoryTotalBytes, memErr)

		_, powerErr := d.PowerUsage()
		assessField(MetricPowerUsageMW, powerErr)

		_, energyErr := d.TotalEnergyConsumption()
		assessField(MetricTotalEnergyMJ, energyErr)

		_, tempErr := d.Temperature(nvml.TemperatureGpu)
		assessField(MetricTemperatureC, tempErr)

		_, smClockErr := d.ClockInfo(nvml.ClockSm)
		assessField(MetricSMClockMHz, smClockErr)

		_, memClockErr := d.ClockInfo(nvml.ClockMem)
		assessField(MetricMemoryClockMHz, memClockErr)

		_, eventErr := d.CurrentClocksEventReasons()
		assessField(MetricClocksEventReasons, eventErr)

		s.metadata[i] = DeviceMetadata{
			UUID:                    uuid,
			Index:                   int(idx),
			Model:                   name,
			Architecture:            archStr,
			DriverVersion:           driverVersion,
			LibraryVersion:          libVersion,
			PowerLimitMW:            powerLimit,
			MaxSMClockMHz:           maxSM,
			MaxMemoryClockMHz:       maxMem,
			MIGEnabled:              migEnabled,
			MIGParentUUID:           migParentUUID,
			MIGInstanceID:           migInstanceID,
			HasMIGInstanceID:        hasMIGInstanceID,
			MIGComputeInstanceID:    migComputeInstanceID,
			HasMIGComputeInstanceID: hasMIGComputeInstanceID,
			MIGScope:                migScope,
			Status:                  status,
			Warnings:                warnings,
			MetadataFailures:        failures,
			FieldCapabilities:       fieldCapabilities,
			FieldStatuses:           fieldStatuses,
		}
	}

	return s, nil
}

// Devices returns metadata for all sampled devices.
func (s *NVMLSampler) Devices() []DeviceMetadata {
	out := make([]DeviceMetadata, len(s.metadata))
	copy(out, s.metadata)
	return out
}

// Sample polls telemetry readings across all devices.
func (s *NVMLSampler) Sample(ctx context.Context) ([]Reading, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	mono := s.clock.NowMonotonic()
	wall := s.clock.NowUnix()

	var readings []Reading
	for i, d := range s.devices {
		meta := s.metadata[i]
		uuid := meta.UUID

		addReadingWithBounds := func(name, unit string, val float64, intVal uint64, isInt bool, err error, qStart, qEnd int64) {
			r := Reading{
				Name:               name,
				Unit:               unit,
				Backend:            "nvml",
				DeviceUUID:         uuid,
				TimestampMonotonic: mono,
				TimestampUnix:      wall,
				QueryStartNanos:    qStart,
				QueryEndNanos:      qEnd,
				HasQueryBounds:     true,
				HasSourceTimestamp: false,
			}
			if err != nil {
				r.Evidence = EvidenceUnavailable
				r.UnavailableReason = ClassifyNVMLError(err)
			} else {
				r.Evidence = EvidenceMeasured
				r.Value = val
				r.IntValue = intVal
				r.IsInt = isInt
			}
			readings = append(readings, r)
		}

		// GPU & Memory utilization (shared NVML query)
		qStart := s.clock.NowMonotonic()
		util, utilErr := d.UtilizationRates()
		qEnd := s.clock.NowMonotonic()
		addReadingWithBounds(MetricGPUUtilization, UnitPercent, float64(util.Gpu), 0, false, utilErr, qStart, qEnd)
		addReadingWithBounds(MetricMemoryUtilization, UnitPercent, float64(util.Memory), 0, false, utilErr, qStart, qEnd)

		// Memory info (shared NVML query)
		qStart = s.clock.NowMonotonic()
		mem, memErr := d.MemoryInfo()
		qEnd = s.clock.NowMonotonic()
		addReadingWithBounds(MetricMemoryUsedBytes, UnitBytes, float64(mem.Used), mem.Used, true, memErr, qStart, qEnd)
		addReadingWithBounds(MetricMemoryTotalBytes, UnitBytes, float64(mem.Total), mem.Total, true, memErr, qStart, qEnd)

		// Power
		qStart = s.clock.NowMonotonic()
		power, powerErr := d.PowerUsage()
		qEnd = s.clock.NowMonotonic()
		addReadingWithBounds(MetricPowerUsageMW, UnitMilliwatts, float64(power), uint64(power), true, powerErr, qStart, qEnd)

		// Energy
		qStart = s.clock.NowMonotonic()
		energy, energyErr := d.TotalEnergyConsumption()
		qEnd = s.clock.NowMonotonic()
		addReadingWithBounds(MetricTotalEnergyMJ, UnitMillijoules, float64(energy), energy, true, energyErr, qStart, qEnd)

		// Temperature
		qStart = s.clock.NowMonotonic()
		temp, tempErr := d.Temperature(nvml.TemperatureGpu)
		qEnd = s.clock.NowMonotonic()
		addReadingWithBounds(MetricTemperatureC, UnitCelsius, float64(temp), uint64(temp), true, tempErr, qStart, qEnd)

		// Clocks
		qStart = s.clock.NowMonotonic()
		smClock, smErr := d.ClockInfo(nvml.ClockSm)
		qEnd = s.clock.NowMonotonic()
		addReadingWithBounds(MetricSMClockMHz, UnitMHz, float64(smClock), uint64(smClock), true, smErr, qStart, qEnd)

		qStart = s.clock.NowMonotonic()
		memClock, memClockErr := d.ClockInfo(nvml.ClockMem)
		qEnd = s.clock.NowMonotonic()
		addReadingWithBounds(MetricMemoryClockMHz, UnitMHz, float64(memClock), uint64(memClock), true, memClockErr, qStart, qEnd)

		// Clock event reasons
		qStart = s.clock.NowMonotonic()
		reasons, reasonsErr := d.CurrentClocksEventReasons()
		qEnd = s.clock.NowMonotonic()
		addReadingWithBounds(MetricClocksEventReasons, UnitBitmask, float64(reasons), reasons, true, reasonsErr, qStart, qEnd)
	}

	return readings, nil
}

// ClassifyNVMLError translates NVML errors into canonical unavailable reasons.
func ClassifyNVMLError(err error) string {
	if err == nil {
		return ""
	}
	if errors.Is(err, nvml.ErrNotSupported) || errors.Is(err, nvml.ErrFunctionNotFound) {
		return ReasonUnsupported
	}
	if errors.Is(err, nvml.ErrNoPermission) {
		return ReasonPermissionDenied
	}
	if errors.Is(err, nvml.ErrGpuIsLost) {
		return ReasonGPULost
	}
	return ReasonQueryFailure
}
