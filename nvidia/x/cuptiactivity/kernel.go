package cuptiactivity

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// KernelRecord is a unified internal representation of a CUPTI activity kernel record
// across CUDA toolkit versions (Kernel10 through Kernel13).
//
// Fields common to all versions are decoded into top-level struct fields.
// Version-specific fields (such as Priority, ExecutionModel, and SourceGraphID)
// are retained when present in the native layout.
type KernelRecord struct {
	Name                string
	CorrelationID       uint32
	Start               uint64
	End                 uint64
	DeviceID            uint32
	StreamID            uint32
	RegistersPerThread  uint16
	StaticSharedMemory  int32
	DynamicSharedMemory int32
	GridX               int32
	GridY               int32
	GridZ               int32
	BlockX              int32
	BlockY              int32
	BlockZ              int32

	// Common activity and launch attributes:
	Kind                 cupti.ActivityKind
	ContextID            uint32
	Completed            uint64
	Queued               uint64
	Submitted            uint64
	GridID               int64
	LocalMemoryPerThread uint32
	LocalMemoryTotal     uint64
	LaunchType           cupti.ActivityLaunchType
	GraphID              uint32
	GraphNodeID          uint64
	DeviceLaunched       bool

	// Version-specific extensions:
	Priority          int32  // Kernel11+ (CUDA 13.1+)
	ExecutionModel    uint32 // Kernel12+ (CUDA 13.4)
	SourceGraphID     uint32 // Kernel13 (CUDA 13.4)
	SourceGraphNodeID uint64 // Kernel13 (CUDA 13.4)
}

// Record converts the KernelRecord into a general cuptiactivity.Record.
func (k KernelRecord) Record() Record {
	return Record{
		Kind:                 k.Kind,
		Name:                 k.Name,
		Start:                k.Start,
		End:                  k.End,
		Complete:             k.Start != 0 && k.End != 0 && k.End >= k.Start,
		CorrelationID:        k.CorrelationID,
		DeviceID:             k.DeviceID,
		ContextID:            k.ContextID,
		StreamID:             k.StreamID,
		Queued:               k.Queued,
		Submitted:            k.Submitted,
		Grid:                 [3]int32{k.GridX, k.GridY, k.GridZ},
		Block:                [3]int32{k.BlockX, k.BlockY, k.BlockZ},
		GridID:               k.GridID,
		RegistersPerThread:   k.RegistersPerThread,
		StaticSharedMemory:   k.StaticSharedMemory,
		DynamicSharedMemory:  k.DynamicSharedMemory,
		LocalMemoryPerThread: k.LocalMemoryPerThread,
		LocalMemoryTotal:     k.LocalMemoryTotal,
		LaunchType:           k.LaunchType,
		GraphID:              k.GraphID,
		GraphNodeID:          k.GraphNodeID,
		DeviceLaunched:       k.DeviceLaunched,
	}
}

// Kernel returns the unified KernelRecord representation for a kernel record.
// For non-kernel records, it returns the zero value.
func (r Record) Kernel() KernelRecord {
	if r.Kind != cupti.ActivityKindConcurrentKernel && r.Kind != cupti.ActivityKindKernel {
		return KernelRecord{}
	}
	return KernelRecord{
		Kind:                 r.Kind,
		Name:                 r.Name,
		Start:                r.Start,
		End:                  r.End,
		CorrelationID:        r.CorrelationID,
		DeviceID:             r.DeviceID,
		ContextID:            r.ContextID,
		StreamID:             r.StreamID,
		Queued:               r.Queued,
		Submitted:            r.Submitted,
		GridX:                r.Grid[0],
		GridY:                r.Grid[1],
		GridZ:                r.Grid[2],
		BlockX:               r.Block[0],
		BlockY:               r.Block[1],
		BlockZ:               r.Block[2],
		GridID:               r.GridID,
		RegistersPerThread:   r.RegistersPerThread,
		StaticSharedMemory:   r.StaticSharedMemory,
		DynamicSharedMemory:  r.DynamicSharedMemory,
		LocalMemoryPerThread: r.LocalMemoryPerThread,
		LocalMemoryTotal:     r.LocalMemoryTotal,
		LaunchType:           r.LaunchType,
		GraphID:              r.GraphID,
		GraphNodeID:          r.GraphNodeID,
		DeviceLaunched:       r.DeviceLaunched,
	}
}

// DecodeKernel10 decodes a CUDA 13.0–13.4 ActivityKernel10 record (216 bytes).
func DecodeKernel10(data []byte, name func(uintptr) (string, error)) (KernelRecord, error) {
	if len(data) < cupti.ActivityKernel10Size {
		return KernelRecord{}, fmt.Errorf("cuptiactivity: buffer too small for ActivityKernel10: %d bytes (want %d)", len(data), cupti.ActivityKernel10Size)
	}
	a := *(*cupti.ActivityKernel10)(unsafe.Pointer(&data[0]))
	if a.Kind != cupti.ActivityKindConcurrentKernel && a.Kind != cupti.ActivityKindKernel {
		return KernelRecord{}, fmt.Errorf("cuptiactivity: unexpected activity kind %s for ActivityKernel10", a.Kind)
	}
	k := KernelRecord{
		Kind:                 a.Kind,
		CorrelationID:        a.CorrelationId,
		Start:                a.Start,
		End:                  a.End,
		Completed:            a.Completed,
		DeviceID:             a.DeviceId,
		ContextID:            a.ContextId,
		StreamID:             a.StreamId,
		GridX:                a.GridX,
		GridY:                a.GridY,
		GridZ:                a.GridZ,
		BlockX:               a.BlockX,
		BlockY:               a.BlockY,
		BlockZ:               a.BlockZ,
		StaticSharedMemory:   a.StaticSharedMemory,
		DynamicSharedMemory:  a.DynamicSharedMemory,
		LocalMemoryPerThread: a.LocalMemoryPerThread,
		LocalMemoryTotal:     a.LocalMemoryTotalV2,
		GridID:               a.GridId,
		RegistersPerThread:   a.RegistersPerThread,
		Queued:               a.Queued,
		Submitted:            a.Submitted,
		LaunchType:           cupti.ActivityLaunchType(a.LaunchType),
		GraphID:              a.GraphId,
		GraphNodeID:          a.GraphNodeId,
		DeviceLaunched:       a.IsDeviceLaunched != 0,
	}
	if a.Name != 0 {
		if name == nil {
			return KernelRecord{}, errors.New("cuptiactivity: missing kernel name copier")
		}
		n, err := name(a.Name)
		if err != nil {
			return KernelRecord{}, fmt.Errorf("cuptiactivity: copy kernel name: %w", err)
		}
		k.Name = strings.Clone(n)
	}
	return k, nil
}

// DecodeKernel11 decodes a CUDA 13.1–13.4 ActivityKernel11 record (224 bytes).
func DecodeKernel11(data []byte, name func(uintptr) (string, error)) (KernelRecord, error) {
	if len(data) < cupti.ActivityKernel11Size {
		return KernelRecord{}, fmt.Errorf("cuptiactivity: buffer too small for ActivityKernel11: %d bytes (want %d)", len(data), cupti.ActivityKernel11Size)
	}
	a := *(*cupti.ActivityKernel11)(unsafe.Pointer(&data[0]))
	if a.Kind != cupti.ActivityKindConcurrentKernel && a.Kind != cupti.ActivityKindKernel {
		return KernelRecord{}, fmt.Errorf("cuptiactivity: unexpected activity kind %s for ActivityKernel11", a.Kind)
	}
	k := KernelRecord{
		Kind:                 a.Kind,
		CorrelationID:        a.CorrelationId,
		Start:                a.Start,
		End:                  a.End,
		Completed:            a.Completed,
		DeviceID:             a.DeviceId,
		ContextID:            a.ContextId,
		StreamID:             a.StreamId,
		GridX:                a.GridX,
		GridY:                a.GridY,
		GridZ:                a.GridZ,
		BlockX:               a.BlockX,
		BlockY:               a.BlockY,
		BlockZ:               a.BlockZ,
		StaticSharedMemory:   a.StaticSharedMemory,
		DynamicSharedMemory:  a.DynamicSharedMemory,
		LocalMemoryPerThread: a.LocalMemoryPerThread,
		LocalMemoryTotal:     a.LocalMemoryTotalV2,
		GridID:               a.GridId,
		RegistersPerThread:   a.RegistersPerThread,
		Queued:               a.Queued,
		Submitted:            a.Submitted,
		LaunchType:           cupti.ActivityLaunchType(a.LaunchType),
		GraphID:              a.GraphId,
		GraphNodeID:          a.GraphNodeId,
		DeviceLaunched:       a.IsDeviceLaunched != 0,
		Priority:             a.Priority,
	}
	if a.Name != 0 {
		if name == nil {
			return KernelRecord{}, errors.New("cuptiactivity: missing kernel name copier")
		}
		n, err := name(a.Name)
		if err != nil {
			return KernelRecord{}, fmt.Errorf("cuptiactivity: copy kernel name: %w", err)
		}
		k.Name = strings.Clone(n)
	}
	return k, nil
}

// DecodeKernel12 decodes a CUDA 13.4 ActivityKernel12 record (232 bytes).
func DecodeKernel12(data []byte, name func(uintptr) (string, error)) (KernelRecord, error) {
	if len(data) < cupti.ActivityKernel12Size {
		return KernelRecord{}, fmt.Errorf("cuptiactivity: buffer too small for ActivityKernel12: %d bytes (want %d)", len(data), cupti.ActivityKernel12Size)
	}
	a := *(*cupti.ActivityKernel12)(unsafe.Pointer(&data[0]))
	if a.Kind != cupti.ActivityKindConcurrentKernel && a.Kind != cupti.ActivityKindKernel {
		return KernelRecord{}, fmt.Errorf("cuptiactivity: unexpected activity kind %s for ActivityKernel12", a.Kind)
	}
	k := KernelRecord{
		Kind:                 a.Kind,
		CorrelationID:        a.CorrelationId,
		Start:                a.Start,
		End:                  a.End,
		Completed:            a.Completed,
		DeviceID:             a.DeviceId,
		ContextID:            a.ContextId,
		StreamID:             a.StreamId,
		GridX:                a.GridX,
		GridY:                a.GridY,
		GridZ:                a.GridZ,
		BlockX:               a.BlockX,
		BlockY:               a.BlockY,
		BlockZ:               a.BlockZ,
		StaticSharedMemory:   a.StaticSharedMemory,
		DynamicSharedMemory:  a.DynamicSharedMemory,
		LocalMemoryPerThread: a.LocalMemoryPerThread,
		LocalMemoryTotal:     a.LocalMemoryTotalV2,
		GridID:               a.GridId,
		RegistersPerThread:   a.RegistersPerThread,
		Queued:               a.Queued,
		Submitted:            a.Submitted,
		LaunchType:           cupti.ActivityLaunchType(a.LaunchType),
		GraphID:              a.GraphId,
		GraphNodeID:          a.GraphNodeId,
		DeviceLaunched:       a.IsDeviceLaunched != 0,
		Priority:             a.Priority,
		ExecutionModel:       a.ExecutionModel,
	}
	if a.Name != 0 {
		if name == nil {
			return KernelRecord{}, errors.New("cuptiactivity: missing kernel name copier")
		}
		n, err := name(a.Name)
		if err != nil {
			return KernelRecord{}, fmt.Errorf("cuptiactivity: copy kernel name: %w", err)
		}
		k.Name = strings.Clone(n)
	}
	return k, nil
}

// DecodeKernel13 decodes a CUDA 13.4 ActivityKernel13 record (248 bytes).
func DecodeKernel13(data []byte, name func(uintptr) (string, error)) (KernelRecord, error) {
	if len(data) < cupti.ActivityKernel13Size {
		return KernelRecord{}, fmt.Errorf("cuptiactivity: buffer too small for ActivityKernel13: %d bytes (want %d)", len(data), cupti.ActivityKernel13Size)
	}
	a := *(*cupti.ActivityKernel13)(unsafe.Pointer(&data[0]))
	if a.Kind != cupti.ActivityKindConcurrentKernel && a.Kind != cupti.ActivityKindKernel {
		return KernelRecord{}, fmt.Errorf("cuptiactivity: unexpected activity kind %s for ActivityKernel13", a.Kind)
	}
	k := KernelRecord{
		Kind:                 a.Kind,
		CorrelationID:        a.CorrelationId,
		Start:                a.Start,
		End:                  a.End,
		Completed:            a.Completed,
		DeviceID:             a.DeviceId,
		ContextID:            a.ContextId,
		StreamID:             a.StreamId,
		GridX:                a.GridX,
		GridY:                a.GridY,
		GridZ:                a.GridZ,
		BlockX:               a.BlockX,
		BlockY:               a.BlockY,
		BlockZ:               a.BlockZ,
		StaticSharedMemory:   a.StaticSharedMemory,
		DynamicSharedMemory:  a.DynamicSharedMemory,
		LocalMemoryPerThread: a.LocalMemoryPerThread,
		LocalMemoryTotal:     a.LocalMemoryTotalV2,
		GridID:               a.GridId,
		RegistersPerThread:   a.RegistersPerThread,
		Queued:               a.Queued,
		Submitted:            a.Submitted,
		LaunchType:           cupti.ActivityLaunchType(a.LaunchType),
		GraphID:              a.GraphId,
		GraphNodeID:          a.GraphNodeId,
		DeviceLaunched:       a.IsDeviceLaunched != 0,
		Priority:             a.Priority,
		ExecutionModel:       a.ExecutionModel,
		SourceGraphID:        a.SourceGraphId,
		SourceGraphNodeID:    a.SourceGraphNodeId,
	}
	if a.Name != 0 {
		if name == nil {
			return KernelRecord{}, errors.New("cuptiactivity: missing kernel name copier")
		}
		n, err := name(a.Name)
		if err != nil {
			return KernelRecord{}, fmt.Errorf("cuptiactivity: copy kernel name: %w", err)
		}
		k.Name = strings.Clone(n)
	}
	return k, nil
}

// DecodeKernel decodes a kernel activity record from a byte slice, selecting the
// appropriate decoder based on the record buffer size. It supports ActivityKernel10
// through ActivityKernel13.
func DecodeKernel(data []byte, name func(uintptr) (string, error)) (KernelRecord, error) {
	switch {
	case len(data) >= cupti.ActivityKernel13Size:
		return DecodeKernel13(data, name)
	case len(data) >= cupti.ActivityKernel12Size:
		return DecodeKernel12(data, name)
	case len(data) >= cupti.ActivityKernel11Size:
		return DecodeKernel11(data, name)
	case len(data) >= cupti.ActivityKernel10Size:
		return DecodeKernel10(data, name)
	default:
		return KernelRecord{}, fmt.Errorf("cuptiactivity: buffer too small for kernel record: %d bytes (minimum %d)", len(data), cupti.ActivityKernel10Size)
	}
}
