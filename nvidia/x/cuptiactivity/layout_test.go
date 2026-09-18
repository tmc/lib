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
	for _, version := range []uint32{0, 24, 120901, 140000} {
		if _, err := recordLayouts(version, func(cupti.ActivityKind, uint32) (uint, error) { t.Fatal("queried unknown layout"); return 0, nil }); err == nil {
			t.Fatalf("accepted version %d", version)
		}
	}
	for _, version := range []uint32{130000, 130001, 130200, 130201, 130300, 130400, 130402} {
		got, err := recordLayouts(version, func(kind cupti.ActivityKind, version uint32) (uint, error) {
			if version != 0 {
				t.Fatal("did not query current emitted layout")
			}
			return sizes[kind], nil
		})
		if err != nil || len(got) != len(sizes) {
			t.Fatalf("version=%d layouts=%v error=%v", version, got, err)
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

	// Verify ActivityKernel versioned layout sizes
	kernelTests := []struct {
		name      string
		version   uint32
		kSize     uint
		wantValid bool
	}{
		{"K10 on CUDA 13.0", 130001, cupti.ActivityKernel10Size, true},
		{"K11 on CUDA 13.1", 130101, cupti.ActivityKernel11Size, true},
		{"K12 on CUDA 13.4", 130400, cupti.ActivityKernel12Size, true},
		{"K13 on CUDA 13.4", 130400, cupti.ActivityKernel13Size, true},
		{"K12 on CUDA 13.0 rejected", 130001, cupti.ActivityKernel12Size, false},
		{"K13 on CUDA 13.0 rejected", 130001, cupti.ActivityKernel13Size, false},
		{"unknown kernel size rejected", 130400, 200, false},
	}
	for _, kt := range kernelTests {
		t.Run(kt.name, func(t *testing.T) {
			customSizes := make(map[cupti.ActivityKind]uint)
			for k, v := range sizes {
				customSizes[k] = v
			}
			customSizes[cupti.ActivityKindConcurrentKernel] = kt.kSize

			got, err := recordLayouts(kt.version, func(kind cupti.ActivityKind, version uint32) (uint, error) {
				return customSizes[kind], nil
			})
			if kt.wantValid && err != nil {
				t.Fatalf("expected valid layout, got error: %v", err)
			}
			if !kt.wantValid && err == nil {
				t.Fatalf("expected error for invalid layout version=%d size=%d", kt.version, kt.kSize)
			}
			if kt.wantValid && got[cupti.ActivityKindConcurrentKernel] != kt.kSize {
				t.Fatalf("got kernel layout size %d, want %d", got[cupti.ActivityKindConcurrentKernel], kt.kSize)
			}
		})
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

// TestRecordLayoutsCUDA134 covers the toolkit this package previously refused.
// CUDA 13.4 widened the Memcpy record to 112 bytes and the Memset record to
// 104, so a session that asked only for kernels was rejected over a Memcpy
// layout. The sizes here are measured, not derived: compiled with gcc against
// /usr/local/cuda-13.4/targets/sbsa-linux/include on an sbsa-linux host.
func TestRecordLayoutsCUDA134(t *testing.T) {
	native := func(kind cupti.ActivityKind) uint {
		switch kind {
		case cupti.ActivityKindMemcpy:
			return cupti.ActivityMemcpy7Size
		case cupti.ActivityKindMemset:
			return cupti.ActivityMemset5Size
		case cupti.ActivityKindConcurrentKernel:
			return cupti.ActivityKernel13Size
		case cupti.ActivityKindDriver, cupti.ActivityKindRuntime:
			return uint(unsafe.Sizeof(cupti.ActivityAPI{}))
		case cupti.ActivityKindExternalCorrelation:
			return uint(unsafe.Sizeof(cupti.ActivityExternalCorrelation{}))
		case cupti.ActivityKindMemory2:
			return uint(unsafe.Sizeof(cupti.ActivityMemory4{}))
		default:
			return uint(unsafe.Sizeof(cupti.ActivitySynchronization2{}))
		}
	}
	got, err := recordLayouts(130400, func(kind cupti.ActivityKind, _ uint32) (uint, error) {
		return native(kind), nil
	})
	if err != nil {
		t.Fatalf("refused a CUDA 13.4 layout set: %v", err)
	}
	// The stride must be what CUPTI emits, not what Go lays out. A map holding
	// 96 here would advance the buffer walk 16 bytes short on every memcpy and
	// desynchronize every record after the first one.
	if got[cupti.ActivityKindMemcpy] != cupti.ActivityMemcpy7Size {
		t.Errorf("memcpy stride = %d, want %d", got[cupti.ActivityKindMemcpy], cupti.ActivityMemcpy7Size)
	}
	if got[cupti.ActivityKindMemset] != cupti.ActivityMemset5Size {
		t.Errorf("memset stride = %d, want %d", got[cupti.ActivityKindMemset], cupti.ActivityMemset5Size)
	}

	// The near-miss half: a 13.4-only record size reported by a CUPTI that
	// predates it. Each kind is widened ALONE, with every other kind at its
	// 13.0-13.3 size, because widening them all together lets the kernel gate
	// do the refusing -- and then this passes just as happily with the memcpy
	// and memset gates deleted, which is a control that proves nothing.
	for _, tt := range []struct {
		kind cupti.ActivityKind
		size uint
	}{
		{cupti.ActivityKindMemcpy, cupti.ActivityMemcpy7Size},
		{cupti.ActivityKindMemset, cupti.ActivityMemset5Size},
		{cupti.ActivityKindConcurrentKernel, cupti.ActivityKernel13Size},
	} {
		for _, version := range []uint32{130000, 130200, 130300} {
			_, err := recordLayouts(version, func(kind cupti.ActivityKind, _ uint32) (uint, error) {
				if kind == tt.kind {
					return tt.size, nil
				}
				return baseline132(kind), nil
			})
			var layout *LayoutError
			if !errors.As(err, &layout) {
				t.Errorf("version %d accepted a 13.4-only %s layout: %v", version, tt.kind, err)
				continue
			}
			if layout.Kind != tt.kind {
				t.Errorf("version %d refused %s, want the refusal to name %s", version, layout.Kind, tt.kind)
			}
		}
	}
}

// TestLayoutErrorIsTyped pins the refusal callers match on. A caller that has to
// parse the message cannot be told apart from one that matches the type, until
// the wording changes and only the first breaks.
func TestLayoutErrorIsTyped(t *testing.T) {
	// Only Memcpy disagrees. Every other kind reports the size this build
	// expects, so the refusal cannot be some other kind surfacing first --
	// recordLayouts walks a map, and map order would otherwise decide which
	// error the caller sees.
	_, err := recordLayouts(130200, func(kind cupti.ActivityKind, _ uint32) (uint, error) {
		if kind == cupti.ActivityKindMemcpy {
			return cupti.ActivityMemcpy7Size, nil
		}
		return baseline132(kind), nil
	})
	var layout *LayoutError
	if !errors.As(err, &layout) {
		t.Fatalf("refusal is not a *LayoutError: %v", err)
	}
	if layout.Kind != cupti.ActivityKindMemcpy || layout.NativeSize != cupti.ActivityMemcpy7Size {
		t.Errorf("LayoutError = %+v, want Memcpy at %d", layout, cupti.ActivityMemcpy7Size)
	}
	if layout.GoSize != cupti.ActivityMemcpy6Size {
		t.Errorf("GoSize = %d, want %d", layout.GoSize, cupti.ActivityMemcpy6Size)
	}
}

// baseline132 is the record size each kind has under CUDA 13.0-13.3, which is
// what this build's Go structs are laid out for.
func baseline132(kind cupti.ActivityKind) uint {
	switch kind {
	case cupti.ActivityKindMemcpy:
		return cupti.ActivityMemcpy6Size
	case cupti.ActivityKindMemset:
		return cupti.ActivityMemset4Size
	case cupti.ActivityKindConcurrentKernel:
		return uint(unsafe.Sizeof(cupti.ActivityKernel11{}))
	case cupti.ActivityKindDriver, cupti.ActivityKindRuntime:
		return uint(unsafe.Sizeof(cupti.ActivityAPI{}))
	case cupti.ActivityKindExternalCorrelation:
		return uint(unsafe.Sizeof(cupti.ActivityExternalCorrelation{}))
	case cupti.ActivityKindMemory2:
		return uint(unsafe.Sizeof(cupti.ActivityMemory4{}))
	default:
		return uint(unsafe.Sizeof(cupti.ActivitySynchronization2{}))
	}
}
