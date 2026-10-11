package cuptiactivity

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// A LayoutError reports an activity record size that this build has not
// qualified for the loaded CUPTI version.
//
// The check spans every kind this package can decode, not only the kinds a
// session named in Config.Kinds. CUPTI's iterator supplies record boundaries;
// the checked sizes bound the Go decoder's reads within the valid buffer.
//
// Callers that want to tell "a kind I asked for" from "a kind I did not" should
// compare Kind against their own Config.Kinds. The package does not guess which
// of those two the caller cares about.
type LayoutError struct {
	Kind       cupti.ActivityKind
	NativeSize uint // record size reported by the loaded CUPTI
	GoSize     uint // size of the Go struct this build would decode it with
}

func (e *LayoutError) Error() string {
	return fmt.Sprintf("cuptiactivity: unqualified %s record size %d (Go base layout %d; every decodable kind is checked)", e.Kind, e.NativeSize, e.GoSize)
}

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
			var symbol *cupti.SymbolError
			if errors.As(err, &symbol) && symbol.Symbol == "cuptiActivityGetStructSize" && symbol.Err == nil {
				return nil, fmt.Errorf("cuptiactivity: CUPTI version %d cannot verify activity layouts: required cuptiActivityGetStructSize is absent: %w", version, err)
			}
			return nil, fmt.Errorf("cuptiactivity: CUPTI version %d query %s layout: %w", version, kind, err)
		}
		// Kinds whose record gained a field in a 13.x minor release are admitted
		// at known sizes for qualified versions. Record the native size to bound
		// decoding; CUPTI's iterator supplies the next record. Every kind not
		// listed here requires an exact size.
		var ok bool
		switch kind {
		case cupti.ActivityKindConcurrentKernel:
			ok = validKernelLayout(version, got)
		case cupti.ActivityKindMemcpy:
			ok = validMemcpyLayout(version, got)
		case cupti.ActivityKindMemset:
			ok = validMemsetLayout(version, got)
		default:
			ok = got == want
		}
		if !ok {
			return nil, &LayoutError{Kind: kind, NativeSize: got, GoSize: want}
		}
		layouts[kind] = got
	}
	return layouts, nil
}

// validMemcpyLayout and validMemsetLayout admit the record versions CUDA 13.x
// has shipped. Both records gained the same 16-byte CUDA-graph source-node tail
// in 13.4, appended with no shared field moved, which is why the decoder can go
// on reading the older struct out of the newer record. cupti's prefix-identity
// tests assert that "no shared field moved" rather than assuming it.
func validMemcpyLayout(version uint32, size uint) bool {
	switch size {
	case cupti.ActivityMemcpy6Size: // 96 bytes: CUDA 13.0–13.4
		return version >= 130000 && version < 140000
	case cupti.ActivityMemcpy7Size: // 112 bytes: CUDA 13.4
		return version >= 130400 && version < 140000
	default:
		return false
	}
}

func validMemsetLayout(version uint32, size uint) bool {
	switch size {
	case cupti.ActivityMemset4Size: // 88 bytes: CUDA 13.0–13.4
		return version >= 130000 && version < 140000
	case cupti.ActivityMemset5Size: // 104 bytes: CUDA 13.4
		return version >= 130400 && version < 140000
	default:
		return false
	}
}

func validKernelLayout(version uint32, size uint) bool {
	switch size {
	case cupti.ActivityKernel10Size: // 216 bytes: CUDA 13.0–13.4
		return version >= 130000 && version < 140000
	case cupti.ActivityKernel11Size: // 224 bytes: CUDA 13.0–13.4
		return version >= 130000 && version < 140000
	case cupti.ActivityKernel12Size: // 232 bytes: CUDA 13.4
		return version >= 130400 && version < 140000
	case cupti.ActivityKernel13Size: // 248 bytes: CUDA 13.4
		return version >= 130400 && version < 140000
	default:
		return false
	}
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
