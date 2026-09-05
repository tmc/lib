package cuptiactivity

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestDecodeAPIRecords(t *testing.T) {
	for _, times := range [][2]uint64{{10, 20}, {0, 20}, {20, 0}, {20, 10}, {10, 10}} {
		a := cupti.ActivityAPI{Kind: cupti.ActivityKindDriver, Start: times[0], End: times[1], CorrelationId: 7, ThreadId: 8, ProcessId: 9, Cbid: 10, ReturnValue: 11}
		data := unsafe.Slice((*byte)(unsafe.Pointer(&a)), int(unsafe.Sizeof(a)))
		calls := 0
		next := func(_ *uint8, _ uint, p **cupti.Activity) error {
			calls++
			if calls > 1 {
				return cupti.ErrMaxLimitReached
			}
			*p = (*cupti.Activity)(unsafe.Pointer(&a))
			return nil
		}
		var got []Record
		err := decodeBuffer(data, map[cupti.ActivityKind]uint{a.Kind: uint(len(data))}, next, nil, func(r Record) error { got = append(got, r); return nil })
		if err != nil || len(got) != 1 {
			t.Fatalf("records=%v error=%v", got, err)
		}
		r := got[0]
		if r.CorrelationID != 7 || r.ThreadID != 8 || r.ProcessID != 9 || r.CallbackID != 10 || r.ReturnValue != 11 {
			t.Fatalf("lost correlation/API fields: %+v", r)
		}
		wantComplete := times[0] != 0 && times[1] >= times[0]
		if r.Complete != wantComplete {
			t.Fatalf("times=%v complete=%v", times, r.Complete)
		}
		a.Start = 999
		if r.Start != times[0] {
			t.Fatal("record aliases native storage")
		}
	}
}

func TestDecodeKernelName(t *testing.T) {
	a := cupti.ActivityKernel11{Kind: cupti.ActivityKindConcurrentKernel, Start: 1, End: 2, Name: 123, DeviceId: 3, ContextId: 4, StreamId: 5, CorrelationId: 6}
	data := unsafe.Slice((*byte)(unsafe.Pointer(&a)), int(unsafe.Sizeof(a)))
	r, err := decodeRecord(data, a.Kind, func(p uintptr) (string, error) {
		if p != 123 {
			t.Fatal("wrong name pointer")
		}
		return "kernel", nil
	})
	if err != nil || r.Name != "kernel" || r.DeviceID != 3 || r.ContextID != 4 || r.StreamID != 5 || r.CorrelationID != 6 {
		t.Fatalf("record=%+v error=%v", r, err)
	}
	if _, err := decodeRecord(data, a.Kind, nil); err == nil {
		t.Fatal("dropped kernel name silently")
	}
}

func TestDecodeMemoryRecords(t *testing.T) {
	copyRecord := cupti.ActivityMemcpy6{Kind: cupti.ActivityKindMemcpy, Start: 10, End: 20, Bytes: 4096, DeviceId: 1, ContextId: 2, StreamId: 3, CorrelationId: 4}
	setRecord := cupti.ActivityMemset4{Kind: cupti.ActivityKindMemset, Start: 10, End: 20, Bytes: 4096, DeviceId: 1, ContextId: 2, StreamId: 3, CorrelationId: 4}
	for _, data := range [][]byte{
		unsafe.Slice((*byte)(unsafe.Pointer(&copyRecord)), int(unsafe.Sizeof(copyRecord))),
		unsafe.Slice((*byte)(unsafe.Pointer(&setRecord)), int(unsafe.Sizeof(setRecord))),
	} {
		kind := (*cupti.Activity)(unsafe.Pointer(&data[0])).Kind
		r, err := decodeRecord(data, kind, nil)
		if err != nil || !r.Complete || r.Bytes != 4096 || r.DeviceID != 1 || r.ContextID != 2 || r.StreamID != 3 || r.CorrelationID != 4 {
			t.Fatalf("record=%+v error=%v", r, err)
		}
	}
}

func TestDecodeIteratorFailure(t *testing.T) {
	a := cupti.ActivityAPI{Kind: cupti.ActivityKindDriver}
	data := unsafe.Slice((*byte)(unsafe.Pointer(&a)), int(unsafe.Sizeof(a)))
	for _, repeated := range []bool{false, true} {
		failure := errors.New("iterator error")
		next := func(_ *uint8, _ uint, p **cupti.Activity) error {
			if !repeated {
				return failure
			}
			*p = (*cupti.Activity)(unsafe.Pointer(&a))
			return nil
		}
		err := decodeBuffer(data, map[cupti.ActivityKind]uint{a.Kind: uint(len(data))}, next, nil, func(Record) error { return nil })
		if err == nil || !repeated && !errors.Is(err, failure) {
			t.Fatalf("error=%v", err)
		}
	}
}
