package cuptiactivity

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// recordLayouts admits CUPTI 13.x and verifies the record sizes emitted by the
// loaded library before decoding. A different major version remains
// unqualified. Size checks are required for every admitted version because a
// version number alone does not prove that record layouts are unchanged.
func recordLayouts(version uint32, size func(cupti.ActivityKind, uint32) (uint, error)) (map[cupti.ActivityKind]uint, error) {
	if version < 130000 || version >= 140000 {
		return nil, fmt.Errorf("cuptiactivity: unqualified CUPTI record version %d (want 13.x)", version)
	}
	layouts := map[cupti.ActivityKind]uint{
		cupti.ActivityKindDriver:              uint(unsafe.Sizeof(cupti.ActivityAPI{})),
		cupti.ActivityKindRuntime:             uint(unsafe.Sizeof(cupti.ActivityAPI{})),
		cupti.ActivityKindConcurrentKernel:    uint(unsafe.Sizeof(cupti.ActivityKernel11{})),
		cupti.ActivityKindMemcpy:              uint(unsafe.Sizeof(cupti.ActivityMemcpy6{})),
		cupti.ActivityKindMemset:              uint(unsafe.Sizeof(cupti.ActivityMemset4{})),
		cupti.ActivityKindExternalCorrelation: uint(unsafe.Sizeof(cupti.ActivityExternalCorrelation{})),
		cupti.ActivityKindMemory2:             uint(unsafe.Sizeof(cupti.ActivityMemory4{})),
		cupti.ActivityKindSynchronization:     uint(unsafe.Sizeof(cupti.ActivitySynchronization2{})),
	}
	for kind, want := range layouts {
		got, err := size(kind, 0) // Ask about records actually emitted by this library.
		if err != nil {
			return nil, fmt.Errorf("cuptiactivity: query %s layout: %w", kind, err)
		}
		if got != want {
			return nil, fmt.Errorf("cuptiactivity: %s record size %d differs from Go layout %d", kind, got, want)
		}
	}
	return layouts, nil
}

// recordBytes bounds a native iterator's record pointer before decoding it.
// It does not infer record boundaries from kind alone or scan arbitrary memory.
func recordBytes(buffer []byte, record *cupti.Activity, size uint) ([]byte, error) {
	if len(buffer) == 0 || record == nil || size == 0 {
		return nil, errors.New("cuptiactivity: empty activity record")
	}
	base := uintptr(unsafe.Pointer(&buffer[0]))
	ptr := uintptr(unsafe.Pointer(record))
	if ptr < base {
		return nil, errors.New("cuptiactivity: record precedes activity buffer")
	}
	offset := ptr - base
	if offset > uintptr(len(buffer)) || uintptr(size) > uintptr(len(buffer))-offset {
		return nil, errors.New("cuptiactivity: record exceeds valid activity bytes")
	}
	if ptr%8 != 0 {
		return nil, errors.New("cuptiactivity: unaligned activity record")
	}
	return buffer[int(offset) : int(offset)+int(size)], nil
}
