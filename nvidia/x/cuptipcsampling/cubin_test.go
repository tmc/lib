package cuptipcsampling

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// This header tests input validation and ABI plumbing, not native CUDA support.
func testCubin() []byte {
	b := make([]byte, 64)
	copy(b, "\x7fELF")
	b[4], b[5], b[6] = 2, 1, 1
	binary.LittleEndian.PutUint16(b[16:], uint16(elf.ET_EXEC))
	binary.LittleEndian.PutUint16(b[18:], uint16(elf.EM_CUDA))
	binary.LittleEndian.PutUint32(b[20:], 1)
	binary.LittleEndian.PutUint16(b[52:], 64)
	return b
}

func TestIdentifyCubin(t *testing.T) {
	for _, v := range []uint32{24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400} {
		for _, crc := range []uint64{0, 1<<63 + 7} {
			data := testCubin()
			got, err := identifyCubin(data, len(data), func() (uint32, error) { return v, nil }, func(p *cupti.GetCubinCrcParams) error {
				if p.Size != 32 || p.CubinSize != uint(len(data)) || p.Cubin == uintptr(unsafe.Pointer(&data[0])) {
					t.Fatalf("bad or borrowed params: %+v", p)
				}
				native := unsafe.Slice((*byte)(nativePointer(p.Cubin)), int(p.CubinSize))
				if !bytes.Equal(native, data) {
					t.Fatal("native bytes differ")
				}
				p.CubinCrc = crc
				return nil
			})
			want := CubinIdentity{CUPTIVersion: v, CRC: crc, SHA256: sha256.Sum256(data)}
			if err != nil || got != want {
				t.Fatalf("got %+v, %v; want %+v", got, err, want)
			}
		}
	}
}

func TestIdentifyCubinRejectsInput(t *testing.T) {
	host := testCubin()
	binary.LittleEndian.PutUint16(host[18:], uint16(elf.EM_X86_64))
	for _, tt := range []struct {
		name  string
		data  []byte
		limit int
	}{
		{"empty", nil, 64}, {"limit", testCubin(), 0}, {"oversize", testCubin(), 63}, {"text", []byte("not code"), 64}, {"host ELF", host, 64},
	} {
		t.Run(tt.name, func(t *testing.T) {
			out, err := identifyCubin(tt.data, tt.limit, func() (uint32, error) { t.Fatal("queried version for invalid input"); return 0, nil }, nil)
			if err == nil || out != (CubinIdentity{}) {
				t.Fatalf("got %+v, %v", out, err)
			}
		})
	}
}

func TestIdentifyCubinNativeFailure(t *testing.T) {
	failure := errors.New("native failure")
	for _, tt := range []struct {
		name    string
		version uint32
		mutate  func(*cupti.GetCubinCrcParams) error
	}{
		// 27 is refused: it is not an encoding any library reports. 28 is
		// CUDA 12.9's, and the rows below reach the native query.
		{"version", 27, nil},
		{"native error", 28, func(p *cupti.GetCubinCrcParams) error { return failure }},
		{"pointer", 28, func(p *cupti.GetCubinCrcParams) error { p.Cubin++; return nil }},
		{"length", 28, func(p *cupti.GetCubinCrcParams) error { p.CubinSize++; return nil }},
		{"bytes", 28, func(p *cupti.GetCubinCrcParams) error { *(*byte)(nativePointer(p.Cubin)) = 0; return nil }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			out, err := identifyCubin(testCubin(), 64, func() (uint32, error) { return tt.version, nil }, tt.mutate)
			if err == nil || out != (CubinIdentity{}) {
				t.Fatalf("got %+v, %v", out, err)
			}
			if tt.name == "native error" && !errors.Is(err, failure) {
				t.Fatal("lost native error")
			}
		})
	}
}
