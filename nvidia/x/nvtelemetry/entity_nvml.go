package nvtelemetry

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nvml"
)

// NVMLProcessFact preserves an NVML process observation with explicit unknown
// memory and MIG identifiers. A PID is not a verified process generation or an
// MPS server/client mapping. ObservationSource names the actual queried API.
type NVMLProcessFact struct {
	PID                  uint32
	Device               DeviceMetadata
	UsedMemoryBytes      uint64
	HasUsedMemory        bool
	GPUInstanceID        uint32
	HasGPUInstanceID     bool
	ComputeInstanceID    uint32
	HasComputeInstanceID bool
	ObservationSource    string
	Snapshot             EntitySnapshot
}

// AdaptNVMLProcesses preserves bounded nvml.ProcessInfo v3 observations.
// It accepts nvml_compute_v3, nvml_graphics_v3 or nvml_mps_compute_v3 as the
// observation source. It does not perform native calls or infer process birth,
// exec, allocation recipient or device-wide consumption from these facts.
func AdaptNVMLProcesses(device DeviceMetadata, processes []nvml.ProcessInfo, source string, snapshot EntitySnapshot) ([]NVMLProcessFact, error) {
	if device.UUID == "" || len(processes) > entityMaxRecords {
		return nil, fmt.Errorf("nvml entity: missing uuid or too many processes")
	}
	switch source {
	case "nvml_compute_v3", "nvml_graphics_v3", "nvml_mps_compute_v3":
	default:
		return nil, fmt.Errorf("nvml entity: unsupported source")
	}
	if err := snapshot.assignment(source).Validate(); err != nil {
		return nil, err
	}
	var out []NVMLProcessFact
	type processInstance struct{ pid, gpu, compute uint32 }
	seen := make(map[processInstance]bool)
	for _, p := range processes {
		key := processInstance{p.Pid, p.GpuInstanceId, p.ComputeInstanceId}
		if p.Pid == 0 || seen[key] {
			return nil, fmt.Errorf("nvml entity: missing or duplicate pid")
		}
		seen[key] = true
		f := NVMLProcessFact{PID: p.Pid, Device: cloneAssignment(EntityAssignment{Device: device}).Device, ObservationSource: source, Snapshot: snapshot}
		if p.UsedGpuMemory != ^uint64(0) {
			f.UsedMemoryBytes = p.UsedGpuMemory
			f.HasUsedMemory = true
		}
		if p.GpuInstanceId != ^uint32(0) {
			f.GPUInstanceID = p.GpuInstanceId
			f.HasGPUInstanceID = true
		}
		if p.ComputeInstanceId != ^uint32(0) {
			f.ComputeInstanceID = p.ComputeInstanceId
			f.HasComputeInstanceID = true
		}
		out = append(out, f)
	}
	return out, nil
}
