package nvml

import (
	"encoding/binary"
	"math"
	"math/bits"
	"runtime"
	"testing"
)

func TestValueMembers(t *testing.T) {
	// Poison bytes beyond the narrow member to catch accidental wide reads.
	data := [8]byte{0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5}
	binary.NativeEndian.PutUint32(data[:], 0xfffffffe)
	v := Value{Data: [1]uint64{binary.NativeEndian.Uint64(data[:])}}
	if got := v.Uint32(); got != 0xfffffffe {
		t.Fatalf("Uint32=%#x", got)
	}
	if got := v.Int32(); got != -2 {
		t.Fatalf("Int32=%d", got)
	}
	wantLong := v.Data[0]
	if runtime.GOOS == "windows" || bits.UintSize == 32 {
		wantLong = 0xfffffffe
	}
	if got := v.UnsignedLong(); got != wantLong {
		t.Fatalf("UnsignedLong=%#x, want %#x", got, wantLong)
	}
	binary.NativeEndian.PutUint16(data[:], 0xfedc)
	v.Data[0] = binary.NativeEndian.Uint64(data[:])
	if got := v.Uint16(); got != 0xfedc {
		t.Fatalf("Uint16=%#x", got)
	}
	v.Data[0] = 0xfffffffffffffffe
	if v.Uint64() != 0xfffffffffffffffe || v.Int64() != -2 {
		t.Fatalf("64-bit members: %x/%d", v.Uint64(), v.Int64())
	}
	for _, raw := range []uint64{0, 0x8000000000000000, 0x3ff8000000000000, 0x7ff0000000000000, 0x7ff8000000000042} {
		v.Data[0] = raw
		if got := math.Float64bits(v.Float64()); got != raw {
			t.Fatalf("Float64 bits=%x, want %x", got, raw)
		}
	}
}
