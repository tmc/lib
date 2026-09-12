package cuptiactivity

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestRecordLayouts(t *testing.T) {
	sizes := map[cupti.ActivityKind]uint{
		cupti.ActivityKindDriver:              uint(unsafe.Sizeof(cupti.ActivityAPI{})),
		cupti.ActivityKindRuntime:             uint(unsafe.Sizeof(cupti.ActivityAPI{})),
		cupti.ActivityKindConcurrentKernel:    uint(unsafe.Sizeof(cupti.ActivityKernel11{})),
		cupti.ActivityKindMemcpy:              uint(unsafe.Sizeof(cupti.ActivityMemcpy6{})),
		cupti.ActivityKindMemset:              uint(unsafe.Sizeof(cupti.ActivityMemset4{})),
		cupti.ActivityKindExternalCorrelation: uint(unsafe.Sizeof(cupti.ActivityExternalCorrelation{})),
		cupti.ActivityKindMemory2:             uint(unsafe.Sizeof(cupti.ActivityMemory4{})),
		cupti.ActivityKindSynchronization:     uint(unsafe.Sizeof(cupti.ActivitySynchronization2{})),
	}
	for _, version := range []uint32{0, 24, 130000, 130201, 130300} {
		if _, err := recordLayouts(version, func(cupti.ActivityKind, uint32) (uint, error) { t.Fatal("queried unknown layout"); return 0, nil }); err == nil {
			t.Fatalf("accepted version %d", version)
		}
	}
	got, err := recordLayouts(130200, func(kind cupti.ActivityKind, version uint32) (uint, error) {
		if version != 0 {
			t.Fatal("did not query current emitted layout")
		}
		return sizes[kind], nil
	})
	if err != nil || len(got) != len(sizes) {
		t.Fatalf("layouts=%v error=%v", got, err)
	}
	if _, err := recordLayouts(130200, func(cupti.ActivityKind, uint32) (uint, error) { return 1, nil }); err == nil {
		t.Fatal("accepted mismatched layout")
	}
	want := &cupti.SymbolError{Symbol: "cuptiActivityGetStructSize"}
	if _, err := recordLayouts(130200, func(cupti.ActivityKind, uint32) (uint, error) { return 0, want }); !errors.Is(err, want) {
		t.Fatalf("lost query error: %v", err)
	}
}

func TestRecordBytes(t *testing.T) {
	words := make([]uint64, 16)
	buffer := unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), len(words)*8)
	record := (*cupti.Activity)(unsafe.Pointer(&words[1]))
	got, err := recordBytes(buffer, record, 40)
	if err != nil || len(got) != 40 || &got[0] != &buffer[8] {
		t.Fatalf("length=%d error=%v", len(got), err)
	}
	for _, tt := range []struct {
		buffer []byte
		record *cupti.Activity
		size   uint
	}{
		{nil, record, 40}, {buffer, nil, 40}, {buffer, record, 0},
		{buffer, record, ^uint(0)}, {buffer[:16], record, 40},
		{buffer[16:], record, 8},
		{buffer, (*cupti.Activity)(unsafe.Pointer(&buffer[1])), 8},
	} {
		if _, err := recordBytes(tt.buffer, tt.record, tt.size); err == nil {
			t.Fatal("accepted invalid record bounds")
		}
	}
}
