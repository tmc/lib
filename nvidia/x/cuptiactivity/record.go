package cuptiactivity

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// Record is an owned snapshot of an API, kernel, copy, or memset activity.
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
	case cupti.ActivityKindMemset:
		a := *(*cupti.ActivityMemset4)(unsafe.Pointer(&data[0]))
		r.Start, r.End, r.CorrelationID = a.Start, a.End, a.CorrelationId
		r.DeviceID, r.ContextID, r.StreamID, r.Bytes = a.DeviceId, a.ContextId, a.StreamId, a.Bytes
	default:
		return Record{}, fmt.Errorf("cuptiactivity: unsupported activity kind %s", kind)
	}
	r.Complete = r.Start != 0 && r.End != 0 && r.End >= r.Start
	return r, nil
}
