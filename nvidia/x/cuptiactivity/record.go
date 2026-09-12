package cuptiactivity

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// Record is an owned snapshot of an API, kernel, copy, memset, memory operation,
// synchronization, or external correlation activity.
// Start and End are nanoseconds in CUPTI's timestamp domain, not Unix time.
// Complete is false for missing or reversed timestamps; such records must not
// be used to compute durations. CorrelationID connects API and device records;
// it is not a globally unique identifier across sessions or processes.
type Record struct {
	Kind                                         cupti.ActivityKind
	Name                                         string
	Start, End                                   uint64
	Complete                                     bool
	CorrelationID                                uint32
	DeviceID, ContextID, StreamID                uint32
	ProcessID, ThreadID, CallbackID, ReturnValue uint32
	Bytes                                        uint64
	// Queued and Submitted are native kernel timestamps in nanoseconds. Zero
	// means unavailable. Their availability does not affect Complete.
	Queued, Submitted uint64
	// Kernel launch configuration is retained without computing occupancy.
	// Shared/local memory fields are bytes allocated or reserved, not traffic.
	// Signed native fields are preserved even when invalid or unavailable.
	Grid, Block                             [3]int32
	GridID                                  int64
	RegistersPerThread                      uint16
	StaticSharedMemory, DynamicSharedMemory int32
	LocalMemoryPerThread                    uint32
	LocalMemoryTotal                        uint64
	LaunchType                              cupti.ActivityLaunchType
	// Graph identities belong to this process/context, not a global namespace.
	// Zero graph/node IDs mean a non-graph launch in the qualified layout.
	GraphID        uint32
	GraphNodeID    uint64
	DeviceLaunched bool
	// CopyKind is the native transfer direction for memcpy records, not a
	// measurement of physical memory traffic.
	CopyKind cupti.ActivityMemcpyKind
	// Memcpy retains separate driver (CorrelationID) and runtime API links.
	// Zero IDs are unavailable; neither may be inferred from the other.
	RuntimeCorrelationID uint32
	// CopyCount is the native number of operations represented by a memcpy
	// record, including batched copies. It is not the number of records.
	// SourceMemoryKind and DestinationMemoryKind describe storage, not traffic.
	CopyCount                               uint64
	SourceMemoryKind, DestinationMemoryKind cupti.ActivityMemoryKind
	// ExternalKind and ExternalID identify the external scope associated with
	// CorrelationID in an external-correlation record. These records have no
	// execution timestamps and Complete is false.
	ExternalKind cupti.ExternalCorrelationKind
	ExternalID   uint64
	// Memory operations are timestamped instants, not execution intervals.
	// Bytes is allocation size, never memory traffic. Address and MemoryPC
	// are native allocation/operation coordinates, not sampled kernel PCs.
	MemoryOperation              cupti.ActivityMemoryOperationType
	MemoryKind                   cupti.ActivityMemoryKind
	Timestamp, Address, MemoryPC uint64
	IsAsync                      bool
	Source                       string
	// Synchronization fields retain native API/event identities. Start and End
	// describe the synchronization API, not GPU idle time. Native invalid-ID
	// sentinels are preserved; CudaEventSyncID distinguishes event reuse.
	Synchronization cupti.ActivitySynchronizationType
	CudaEventID     uint32
	CudaEventSyncID uint64
}

// QueuedToSubmitted returns the interval between queueing the kernel in a
// command buffer and submitting that buffer. It returns false for non-kernel
// records, unavailable timestamps, or reversed timestamps.
func (r Record) QueuedToSubmitted() (uint64, bool) {
	return kernelInterval(r.Kind, r.Queued, r.Submitted)
}

// SubmittedToStart returns the interval between command buffer submission and
// kernel execution start. It does not establish the cause of the interval.
// It returns false for non-kernel records, unavailable or reversed timestamps.
func (r Record) SubmittedToStart() (uint64, bool) {
	return kernelInterval(r.Kind, r.Submitted, r.Start)
}

func kernelInterval(kind cupti.ActivityKind, start, end uint64) (uint64, bool) {
	if kind != cupti.ActivityKindConcurrentKernel || start == 0 || end == 0 || end < start {
		return 0, false
	}
	return end - start, true
}

type recordIterator func(*uint8, uint, **cupti.Activity) error

// decodeBuffer consumes only records returned by CUPTI's iterator. It never
// guesses a next-record offset, since records can have different sizes.
func decodeBuffer(buffer []byte, layouts map[cupti.ActivityKind]uint, next recordIterator, name func(uintptr) (string, error), emit func(Record) error) error {
	if len(buffer) == 0 {
		return nil
	}
	var record *cupti.Activity
	seen := make(map[*cupti.Activity]bool)
	for {
		err := next(&buffer[0], uint(len(buffer)), &record)
		if errors.Is(err, cupti.ErrMaxLimitReached) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("cuptiactivity: iterate: %w", err)
		}
		if _, err := recordBytes(buffer, record, uint(unsafe.Sizeof(cupti.Activity{}))); err != nil {
			return err
		}
		if seen[record] {
			return errors.New("cuptiactivity: activity iterator repeated a record")
		}
		seen[record] = true
		kind := record.Kind
		size, ok := layouts[kind]
		if !ok {
			return fmt.Errorf("cuptiactivity: unsupported activity kind %s", kind)
		}
		bytes, err := recordBytes(buffer, record, size)
		if err != nil {
			return err
		}
		r, err := decodeRecord(bytes, kind, name)
		if err != nil {
			return err
		}
		if err := emit(r); err != nil {
			return err
		}
	}
}

func decodeRecord(data []byte, kind cupti.ActivityKind, name func(uintptr) (string, error)) (Record, error) {
	r := Record{Kind: kind}
	switch kind {
	case cupti.ActivityKindDriver, cupti.ActivityKindRuntime:
		a := *(*cupti.ActivityAPI)(unsafe.Pointer(&data[0]))
		r.Start, r.End, r.CorrelationID = a.Start, a.End, a.CorrelationId
		r.ProcessID, r.ThreadID, r.CallbackID, r.ReturnValue = a.ProcessId, a.ThreadId, a.Cbid, a.ReturnValue
	case cupti.ActivityKindConcurrentKernel:
		a := *(*cupti.ActivityKernel11)(unsafe.Pointer(&data[0]))
		r.Start, r.End, r.CorrelationID = a.Start, a.End, a.CorrelationId
		r.DeviceID, r.ContextID, r.StreamID = a.DeviceId, a.ContextId, a.StreamId
		r.Queued, r.Submitted = a.Queued, a.Submitted
		r.Grid, r.Block = [3]int32{a.GridX, a.GridY, a.GridZ}, [3]int32{a.BlockX, a.BlockY, a.BlockZ}
		r.GridID, r.RegistersPerThread = a.GridId, a.RegistersPerThread
		r.StaticSharedMemory, r.DynamicSharedMemory = a.StaticSharedMemory, a.DynamicSharedMemory
		r.LocalMemoryPerThread, r.LocalMemoryTotal = a.LocalMemoryPerThread, a.LocalMemoryTotalV2
		r.LaunchType = cupti.ActivityLaunchType(a.LaunchType)
		r.GraphID, r.GraphNodeID, r.DeviceLaunched = a.GraphId, a.GraphNodeId, a.IsDeviceLaunched != 0
		if a.Name != 0 {
			if name == nil {
				return Record{}, errors.New("cuptiactivity: missing kernel name copier")
			}
			var err error
			r.Name, err = name(a.Name)
			if err != nil {
				return Record{}, fmt.Errorf("cuptiactivity: copy kernel name: %w", err)
			}
			r.Name = strings.Clone(r.Name)
		}
	case cupti.ActivityKindMemcpy:
		a := *(*cupti.ActivityMemcpy6)(unsafe.Pointer(&data[0]))
		r.Start, r.End, r.CorrelationID = a.Start, a.End, a.CorrelationId
		r.DeviceID, r.ContextID, r.StreamID, r.Bytes = a.DeviceId, a.ContextId, a.StreamId, a.Bytes
		r.CopyKind = cupti.ActivityMemcpyKind(a.CopyKind)
		r.RuntimeCorrelationID, r.CopyCount = a.RuntimeCorrelationId, a.CopyCount
		r.SourceMemoryKind, r.DestinationMemoryKind = cupti.ActivityMemoryKind(a.SrcKind), cupti.ActivityMemoryKind(a.DstKind)
		r.GraphID, r.GraphNodeID, r.DeviceLaunched = a.GraphId, a.GraphNodeId, a.IsDeviceLaunched != 0
	case cupti.ActivityKindMemset:
		a := *(*cupti.ActivityMemset4)(unsafe.Pointer(&data[0]))
		r.Start, r.End, r.CorrelationID = a.Start, a.End, a.CorrelationId
		r.DeviceID, r.ContextID, r.StreamID, r.Bytes = a.DeviceId, a.ContextId, a.StreamId, a.Bytes
		r.GraphID, r.GraphNodeID, r.DeviceLaunched = a.GraphId, a.GraphNodeId, a.IsDeviceLaunched != 0
	case cupti.ActivityKindExternalCorrelation:
		a := *(*cupti.ActivityExternalCorrelation)(unsafe.Pointer(&data[0]))
		r.ExternalKind, r.ExternalID, r.CorrelationID = a.ExternalKind, a.ExternalId, a.CorrelationId
	case cupti.ActivityKindSynchronization:
		a := *(*cupti.ActivitySynchronization2)(unsafe.Pointer(&data[0]))
		r.Start, r.End, r.CorrelationID = a.Start, a.End, a.CorrelationId
		r.ContextID, r.StreamID = a.ContextId, a.StreamId
		r.Synchronization, r.CudaEventID, r.CudaEventSyncID = a.Type, a.CudaEventId, a.CudaEventSyncId
		r.ReturnValue = a.ReturnValue
	case cupti.ActivityKindMemory2:
		a := *(*cupti.ActivityMemory4)(unsafe.Pointer(&data[0]))
		r.MemoryOperation, r.MemoryKind = a.MemoryOperationType, a.MemoryKind
		r.Timestamp, r.Address, r.Bytes, r.MemoryPC = a.Timestamp, a.Address, a.Bytes, a.PC
		r.ProcessID, r.DeviceID, r.ContextID, r.StreamID, r.CorrelationID = a.ProcessId, a.DeviceId, a.ContextId, a.StreamId, a.CorrelationId
		r.IsAsync = a.IsAsync != 0
		for _, field := range []struct {
			address uintptr
			value   *string
		}{{a.Name, &r.Name}, {a.Source, &r.Source}} {
			if field.address == 0 {
				continue
			}
			if name == nil {
				return Record{}, errors.New("cuptiactivity: missing memory string copier")
			}
			value, err := name(field.address)
			if err != nil {
				return Record{}, fmt.Errorf("cuptiactivity: copy memory string: %w", err)
			}
			*field.value = strings.Clone(value)
		}
	default:
		return Record{}, fmt.Errorf("cuptiactivity: unsupported activity kind %s", kind)
	}
	r.Complete = r.Start != 0 && r.End != 0 && r.End >= r.Start
	return r, nil
}
