package nvml

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// Helpers bound allocations from driver-supplied counts and repeated sizing
// requests. Call the raw methods when a larger buffer is required.
const helperMaxEntries = 65536
const helperSizeAttempts = 4

// IsUnsupported reports an unsupported call or a symbol absent from the loaded
// library. It examines wrapped and joined errors. Permission, lost-device and
// symbol-binding failures alone are not unsupported features.
func IsUnsupported(err error) bool {
	if errors.Is(err, ErrNotSupported) {
		return true
	}
	return missingSymbol(err)
}

func missingSymbol(err error) bool {
	if e, ok := err.(*SymbolError); ok && e != nil && e.Err == nil {
		return true
	}
	switch e := err.(type) {
	case interface{ Unwrap() []error }:
		for _, child := range e.Unwrap() {
			if missingSymbol(child) {
				return true
			}
		}
	case interface{ Unwrap() error }:
		return missingSymbol(e.Unwrap())
	}
	return false
}

// GpmSupported queries GPM support using the required structure version.
func (d Device) GpmSupported() (bool, error) {
	return gpmSupported(d.GpmQueryDeviceSupport)
}

func gpmSupported(call func(*GpmSupport) error) (bool, error) {
	info := GpmSupport{Version: GpmSupportVersion}
	if err := call(&info); err != nil {
		return false, fmt.Errorf("query gpm support: %w", err)
	}
	return info.IsSupportedDevice != 0, nil
}

// FabricInfo returns version 3 fabric information. It preserves the fabric
// status in the result, separately from the error returned by the query.
func (d Device) FabricInfo() (GpuFabricInfoV, error) {
	return fabricInfo(d.GpuFabricInfoV)
}

func fabricInfo(call func(*GpuFabricInfoV) error) (GpuFabricInfoV, error) {
	info := GpuFabricInfoV{Version: GpuFabricInfoV3Version}
	if err := call(&info); err != nil {
		return GpuFabricInfoV{}, fmt.Errorf("query fabric info: %w", err)
	}
	return info, nil
}

// ProcessesUtilization returns process samples newer than lastSeen, in CPU
// timestamp microseconds. It retries a growing buffer at most four times and
// rejects counts above 65536. ErrNotFound and other driver errors are preserved.
// NVML does not support this query on MIG-enabled devices.
func (d Device) ProcessesUtilization(lastSeen uint64) ([]ProcessUtilizationInfoV1, error) {
	return processesUtilization(lastSeen, func(info *ProcessesUtilizationInfo, _ []ProcessUtilizationInfoV1) error {
		return d.ProcessesUtilizationInfo(info)
	})
}

func processesUtilization(lastSeen uint64, call func(*ProcessesUtilizationInfo, []ProcessUtilizationInfoV1) error) ([]ProcessUtilizationInfoV1, error) {
	var samples []ProcessUtilizationInfoV1
	for attempt := 0; attempt < helperSizeAttempts; attempt++ {
		info := ProcessesUtilizationInfo{
			Version:             ProcessesUtilizationInfoV1Version,
			LastSeenTimeStamp:   lastSeen,
			ProcessSamplesCount: uint32(len(samples)),
		}
		if len(samples) != 0 {
			info.ProcUtilArray = uintptr(unsafe.Pointer(&samples[0]))
		}
		err := call(&info, samples)
		runtime.KeepAlive(samples)
		if err != nil && !errors.Is(err, ErrInsufficientSize) {
			return nil, fmt.Errorf("query process utilization: %w", err)
		}
		n := info.ProcessSamplesCount
		if n > helperMaxEntries {
			return nil, fmt.Errorf("query process utilization: count %d exceeds %d", n, helperMaxEntries)
		}
		if err == nil {
			if int(n) > len(samples) {
				return nil, fmt.Errorf("query process utilization: count %d exceeds buffer %d", n, len(samples))
			}
			return samples[:n], nil
		}
		if n == 0 || int(n) <= len(samples) {
			return nil, fmt.Errorf("query process utilization: insufficient size without a larger count: %w", err)
		}
		samples = make([]ProcessUtilizationInfoV1, n)
	}
	return nil, fmt.Errorf("query process utilization: sizing did not stabilize: %w", ErrInsufficientSize)
}

// FieldValuesFor queries fields in the given order and preserves each entry's
// NvmlReturn. An empty list makes no call. At most 65536 fields may be requested.
func (d Device) FieldValuesFor(ids ...uint32) ([]FieldValue, error) {
	return fieldValuesFor(ids, d.FieldValues)
}

func fieldValuesFor(ids []uint32, call func(int32, *FieldValue) error) ([]FieldValue, error) {
	if len(ids) > helperMaxEntries {
		return nil, fmt.Errorf("query field values: count %d exceeds %d", len(ids), helperMaxEntries)
	}
	if len(ids) == 0 {
		return nil, nil
	}
	values := make([]FieldValue, len(ids))
	for i, id := range ids {
		values[i].FieldId = id
	}
	err := call(int32(len(values)), &values[0])
	runtime.KeepAlive(values)
	if err != nil {
		return nil, fmt.Errorf("query field values: %w", err)
	}
	return values, nil
}

// GpmMetrics calculates the requested metrics from two caller-owned samples.
// The caller must collect samples more than 100 ms apart as required by NVML.
// Each metric retains its NvmlReturn; a nil query error does not mean every
// metric succeeded. An empty list makes no call. At most 333 IDs are accepted.
func GpmMetrics(sample1, sample2 GpmSample, ids ...GpmMetricId) ([]GpmMetric, error) {
	return gpmMetrics(sample1, sample2, ids, GpmMetricsGet)
}

func gpmMetrics(sample1, sample2 GpmSample, ids []GpmMetricId, call func(*GpmMetricsGetParams) error) ([]GpmMetric, error) {
	var info GpmMetricsGetParams
	if len(ids) > len(info.Metrics) {
		return nil, fmt.Errorf("query gpm metrics: count %d exceeds %d", len(ids), len(info.Metrics))
	}
	if len(ids) == 0 {
		return nil, nil
	}
	info.Version = GpmMetricsGetVersion
	info.NumMetrics = uint32(len(ids))
	info.Sample1, info.Sample2 = sample1, sample2
	for i, id := range ids {
		if id <= 0 {
			return nil, fmt.Errorf("query gpm metrics: invalid metric id %d", id)
		}
		info.Metrics[i].MetricId = uint32(id)
	}
	if err := call(&info); err != nil {
		return nil, fmt.Errorf("query gpm metrics: %w", err)
	}
	if info.NumMetrics != uint32(len(ids)) {
		return nil, fmt.Errorf("query gpm metrics: returned count %d differs from request %d", info.NumMetrics, len(ids))
	}
	return append([]GpmMetric(nil), info.Metrics[:len(ids)]...), nil
}

// GpuInstanceList returns the instances of profileID, using the profile's
// maximum instance count to allocate the required buffer. At most 65536
// instances are accepted. Returned handles remain owned by the caller.
func (d Device) GpuInstanceList(profileID uint32) ([]GpuInstance, error) {
	return gpuInstanceList(profileID, d.GpuInstanceProfileInfoByIdV, d.GpuInstances)
}

func gpuInstanceList(profileID uint32, profile func(uint32) (GpuInstanceProfileInfoV2, error), call func(uint32, *GpuInstance, *uint32) error) ([]GpuInstance, error) {
	info, err := profile(profileID)
	if err != nil {
		return nil, fmt.Errorf("query gpu instance profile: %w", err)
	}
	n := info.InstanceCount
	if n > helperMaxEntries {
		return nil, fmt.Errorf("query gpu instances: count %d exceeds %d", n, helperMaxEntries)
	}
	if n == 0 {
		return nil, nil
	}
	instances := make([]GpuInstance, n)
	err = call(profileID, &instances[0], &n)
	runtime.KeepAlive(instances)
	if err != nil {
		return nil, fmt.Errorf("query gpu instances: %w", err)
	}
	if int(n) > len(instances) {
		return nil, fmt.Errorf("query gpu instances: count %d exceeds buffer %d", n, len(instances))
	}
	return instances[:n], nil
}
