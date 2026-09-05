package cuptiactivity

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// recordLayouts pins decoding to the headers used to verify the Go structures.
// Size equality alone does not prove that an unknown version has equal offsets.
// Extend this version table only with header/layout evidence for that release.
func recordLayouts(version uint32, size func(cupti.ActivityKind, uint32) (uint, error)) (map[cupti.ActivityKind]uint, error) {
	if version != 130200 {
		return nil, fmt.Errorf("cuptiactivity: unqualified CUPTI record version %d (headers are 130200)", version)
	}
	layouts := map[cupti.ActivityKind]uint{
		cupti.ActivityKindDriver:           uint(unsafe.Sizeof(cupti.ActivityAPI{})),
		cupti.ActivityKindRuntime:          uint(unsafe.Sizeof(cupti.ActivityAPI{})),
		cupti.ActivityKindConcurrentKernel: uint(unsafe.Sizeof(cupti.ActivityKernel11{})),
		cupti.ActivityKindMemcpy:           uint(unsafe.Sizeof(cupti.ActivityMemcpy6{})),
		cupti.ActivityKindMemset:           uint(unsafe.Sizeof(cupti.ActivityMemset4{})),
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
