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

func TestDecodeKernelLaunch(t *testing.T) {
	a := cupti.ActivityKernel11{Kind: cupti.ActivityKindConcurrentKernel, GridX: 7, GridY: 8, GridZ: 9, BlockX: 32, BlockY: 2, BlockZ: -1, GridId: -2, RegistersPerThread: 65535, StaticSharedMemory: -1, DynamicSharedMemory: 8192, LocalMemoryPerThread: 128, LocalMemoryTotal: 1, LocalMemoryTotalV2: 1<<63 + 9, LaunchType: 255, GraphId: 13, GraphNodeId: 1<<63 + 10, IsDeviceLaunched: 1}
	r, err := decodeRecord(unsafe.Slice((*byte)(unsafe.Pointer(&a)), int(unsafe.Sizeof(a))), a.Kind, nil)
	if err != nil || r.Grid != [3]int32{7, 8, 9} || r.Block != [3]int32{32, 2, -1} || r.GridID != -2 || r.RegistersPerThread != 65535 || r.StaticSharedMemory != -1 || r.DynamicSharedMemory != 8192 || r.LocalMemoryPerThread != 128 || r.LocalMemoryTotal != 1<<63+9 || r.LaunchType != 255 || r.GraphID != 13 || r.GraphNodeID != 1<<63+10 || !r.DeviceLaunched || r.Complete {
		t.Fatalf("launch=%+v err=%v", r, err)
	}
	a.GridX, a.GraphNodeId = 0, 0
	if r.Grid[0] != 7 || r.GraphNodeID != 1<<63+10 {
		t.Fatal("launch aliases native storage")
	}
}

func TestDecodeMemoryRecords(t *testing.T) {
	copyRecord := cupti.ActivityMemcpy6{Kind: cupti.ActivityKindMemcpy, Start: 10, End: 20, Bytes: 4096, DeviceId: 1, ContextId: 2, StreamId: 3, CorrelationId: 4, GraphId: 17, GraphNodeId: 1<<63 + 2, IsDeviceLaunched: 1}
	setRecord := cupti.ActivityMemset4{Kind: cupti.ActivityKindMemset, Start: 10, End: 20, Bytes: 4096, DeviceId: 1, ContextId: 2, StreamId: 3, CorrelationId: 4, GraphId: 17, GraphNodeId: 1<<63 + 2, IsDeviceLaunched: 1}
	for _, data := range [][]byte{
		unsafe.Slice((*byte)(unsafe.Pointer(&copyRecord)), int(unsafe.Sizeof(copyRecord))),
		unsafe.Slice((*byte)(unsafe.Pointer(&setRecord)), int(unsafe.Sizeof(setRecord))),
	} {
		kind := (*cupti.Activity)(unsafe.Pointer(&data[0])).Kind
		r, err := decodeRecord(data, kind, nil)
		if err != nil || !r.Complete || r.Bytes != 4096 || r.DeviceID != 1 || r.ContextID != 2 || r.StreamID != 3 || r.CorrelationID != 4 {
			t.Fatalf("record=%+v error=%v", r, err)
		}
		if r.GraphID != 17 || r.GraphNodeID != 1<<63+2 || !r.DeviceLaunched {
			t.Fatalf("lost graph identity: %+v", r)
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

func TestKernelIntervals(t *testing.T) {
	for _, tt := range []struct {
		name       string
		start, end uint64
		want       uint64
		ok         bool
	}{
		{"ordered", 10, 25, 15, true},
		{"measured zero", 10, 10, 0, true},
		{"missing start", 0, 25, 0, false},
		{"missing end", 10, 0, 0, false},
		{"reversed", 25, 10, 0, false},
		{"large native value", 1, 1<<63 + 1, 1 << 63, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for _, kind := range []cupti.ActivityKind{cupti.ActivityKindConcurrentKernel, cupti.ActivityKindMemcpy} {
				r := Record{Kind: kind, Queued: tt.start, Submitted: tt.end}
				ns, ok := r.QueuedToSubmitted()
				want, wantOK := tt.want, tt.ok
				if kind == cupti.ActivityKindMemcpy {
					want, wantOK = 0, false
				}
				if ns != want || ok != wantOK {
					t.Fatalf("queue interval=(%d,%v), want (%d,%v)", ns, ok, want, wantOK)
				}
				r.Submitted, r.Start = tt.start, tt.end
				ns, ok = r.SubmittedToStart()
				if ns != want || ok != wantOK {
					t.Fatalf("submission interval=(%d,%v), want (%d,%v)", ns, ok, want, wantOK)
				}
			}
		})
	}
}

func TestDecodeOptionalFields(t *testing.T) {
	for _, times := range [][2]uint64{{0, 0}, {10, 20}, {20, 10}} {
		a := cupti.ActivityKernel11{Kind: cupti.ActivityKindConcurrentKernel, Queued: times[0], Submitted: times[1], Start: 30, End: 40}
		r, err := decodeRecord(unsafe.Slice((*byte)(unsafe.Pointer(&a)), int(unsafe.Sizeof(a))), a.Kind, nil)
		if err != nil || !r.Complete || r.Queued != times[0] || r.Submitted != times[1] {
			t.Fatalf("record=%+v err=%v", r, err)
		}
	}
	for _, direction := range []uint8{0, 1, 2, 255} {
		a := cupti.ActivityMemcpy6{Kind: cupti.ActivityKindMemcpy, CopyKind: direction, Bytes: 1<<63 + 1, Start: 10, End: 20}
		r, err := decodeRecord(unsafe.Slice((*byte)(unsafe.Pointer(&a)), int(unsafe.Sizeof(a))), a.Kind, nil)
		if err != nil || r.CopyKind != cupti.ActivityMemcpyKind(direction) || r.Bytes != a.Bytes {
			t.Fatalf("record=%+v err=%v", r, err)
		}
	}
}

func TestDecodeMemcpyIdentity(t *testing.T) {
	for _, count := range []uint64{0, 1, 1<<63 + 9} {
		a := cupti.ActivityMemcpy6{Kind: cupti.ActivityKindMemcpy, CorrelationId: 7, RuntimeCorrelationId: 19, SrcKind: 255, DstKind: 254, CopyCount: count}
		r, err := decodeRecord(unsafe.Slice((*byte)(unsafe.Pointer(&a)), int(unsafe.Sizeof(a))), a.Kind, nil)
		if err != nil || r.CorrelationID != 7 || r.RuntimeCorrelationID != 19 || r.SourceMemoryKind != 255 || r.DestinationMemoryKind != 254 || r.CopyCount != count || r.Complete {
			t.Fatalf("record=%+v err=%v", r, err)
		}
		a.RuntimeCorrelationId, a.CopyCount = 0, 0
		if r.RuntimeCorrelationID != 19 || r.CopyCount != count {
			t.Fatal("record borrows native storage")
		}
	}
}

func TestDecodeExternalCorrelation(t *testing.T) {
	a := cupti.ActivityExternalCorrelation{Kind: cupti.ActivityKindExternalCorrelation, ExternalKind: cupti.ExternalCorrelationKindCustom0, ExternalId: 1<<63 + 7, CorrelationId: 42}
	data := unsafe.Slice((*byte)(unsafe.Pointer(&a)), int(unsafe.Sizeof(a)))
	r, err := decodeRecord(data, a.Kind, nil)
	if err != nil || r.ExternalID != a.ExternalId || r.ExternalKind != a.ExternalKind || r.CorrelationID != 42 || r.Complete || r.Start != 0 || r.End != 0 {
		t.Fatalf("record=%+v err=%v", r, err)
	}
	a.ExternalId = 9
	if r.ExternalID != 1<<63+7 {
		t.Fatal("external identity aliases native storage")
	}
}

func TestDecodeSynchronization(t *testing.T) {
	for _, tt := range []struct {
		name       string
		start, end uint64
		complete   bool
	}{
		{"complete", 10, 20, true},
		{"zero duration", 10, 10, true},
		{"unavailable", 0, 0, false},
		{"missing start", 0, 20, false},
		{"reversed", 20, 10, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a := cupti.ActivitySynchronization2{Kind: cupti.ActivityKindSynchronization, Type: cupti.ActivitySynchronizationTypeStreamWaitEvent, Start: tt.start, End: tt.end, CorrelationId: 7, ContextId: 8, StreamId: ^uint32(0), CudaEventId: ^uint32(0), CudaEventSyncId: 1<<63 + 9, ReturnValue: 600}
			r, err := decodeRecord(unsafe.Slice((*byte)(unsafe.Pointer(&a)), int(unsafe.Sizeof(a))), a.Kind, nil)
			if err != nil || r.Complete != tt.complete || r.Start != tt.start || r.End != tt.end || r.CorrelationID != 7 || r.ContextID != 8 || r.StreamID != ^uint32(0) || r.CudaEventID != ^uint32(0) || r.CudaEventSyncID != 1<<63+9 || r.ReturnValue != 600 || r.Synchronization != a.Type {
				t.Fatalf("record=%+v err=%v", r, err)
			}
			a.CudaEventSyncId = 0
			if r.CudaEventSyncID != 1<<63+9 {
				t.Fatal("record aliases native identity")
			}
		})
	}
}

func TestDecodeAllocationRecords(t *testing.T) {
	for _, op := range []cupti.ActivityMemoryOperationType{cupti.ActivityMemoryOperationTypeAllocation, cupti.ActivityMemoryOperationTypeRelease, 99} {
		a := cupti.ActivityMemory4{Kind: cupti.ActivityKindMemory2, MemoryOperationType: op, MemoryKind: cupti.ActivityMemoryKindDevice, CorrelationId: 41, Address: 1<<63 + 1, Bytes: 1<<63 + 2, Timestamp: 77, PC: 88, ProcessId: 1, DeviceId: 2, ContextId: 3, StreamId: 4, IsAsync: 1, Name: 5, Source: 6}
		data := unsafe.Slice((*byte)(unsafe.Pointer(&a)), int(unsafe.Sizeof(a)))
		r, err := decodeRecord(data, a.Kind, func(p uintptr) (string, error) {
			switch p {
			case 5:
				return "allocation", nil
			case 6:
				return "library.so", nil
			}
			t.Fatal("unexpected pointer")
			return "", nil
		})
		if err != nil || r.MemoryOperation != op || r.MemoryKind != a.MemoryKind || r.Address != a.Address || r.Bytes != a.Bytes || r.Timestamp != 77 || r.MemoryPC != 88 || r.ProcessID != 1 || r.DeviceID != 2 || r.ContextID != 3 || r.StreamID != 4 || !r.IsAsync || r.CorrelationID != 41 || r.Name != "allocation" || r.Source != "library.so" || r.Complete || r.Start != 0 || r.End != 0 {
			t.Fatalf("record=%+v err=%v", r, err)
		}
		if _, err := decodeRecord(data, a.Kind, nil); err == nil {
			t.Fatal("missing string copier accepted")
		}
		want := errors.New("name unavailable")
		if _, err := decodeRecord(data, a.Kind, func(uintptr) (string, error) { return "", want }); !errors.Is(err, want) {
			t.Fatal("lost string copy error")
		}
		a.Bytes = 1
		if r.Bytes != 1<<63+2 {
			t.Fatal("record aliases allocation storage")
		}
	}
}
