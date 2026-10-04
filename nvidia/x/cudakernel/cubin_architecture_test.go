package cudakernel

import (
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"os"
	"strings"
	"testing"
)

func TestCUBINArchitectureABI(t *testing.T) {
	for _, class := range []elf.Class{elf.ELFCLASS32, elf.ELFCLASS64} {
		for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
			for _, tc := range []struct {
				name  string
				abi   byte
				flags uint32
				want  string
			}{
				{"legacy", 7, 0x005a0450, "sm_80"},
				{"modern", 8, 0x09005a04, "sm_90"},
				{"unknown", 9, 0x09005a04, ""},
				{"unspecified", 0, 0x005a0450, ""},
				{"zero", 8, 4, ""},
			} {
				t.Run(class.String()+"/"+order.String()+"/"+tc.name, func(t *testing.T) {
					image := architectureELF(class, order, tc.abi, tc.flags)
					info, err := ParseCUBIN(image)
					if err != nil {
						t.Fatal(err)
					}
					if info.Arch != tc.want {
						t.Fatalf("Arch=%q, want %q", info.Arch, tc.want)
					}
					order.PutUint16(image[18:20], uint16(elf.EM_X86_64))
					info, err = ParseCUBIN(image)
					if err == nil || info != nil {
						t.Fatalf("non-CUDA architecture: %+v, %v", info, err)
					}
				})
			}
		}
	}
}

func architectureELF(class elf.Class, order binary.ByteOrder, abi byte, flags uint32) []byte {
	size, offset := 64, 48
	if class == elf.ELFCLASS32 {
		size, offset = 52, 36
	}
	b := make([]byte, size)
	copy(b, "\x7fELF")
	b[4], b[5], b[6], b[7], b[8] = byte(class), byte(elf.ELFDATA2LSB), 1, 51, abi
	if order == binary.BigEndian {
		b[5] = byte(elf.ELFDATA2MSB)
	}
	order.PutUint16(b[16:18], uint16(elf.ET_EXEC))
	order.PutUint16(b[18:20], uint16(elf.EM_CUDA))
	order.PutUint32(b[20:24], 1)
	order.PutUint32(b[offset:offset+4], flags)
	if class == elf.ELFCLASS64 {
		order.PutUint16(b[52:54], uint16(size))
	} else {
		order.PutUint16(b[40:42], uint16(size))
	}
	return b
}

func TestCUBINArchitectureCUDA134Fixture(t *testing.T) {
	encoded, err := os.ReadFile("../cudacapture/testdata/kernels.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	image, err := hex.DecodeString(strings.TrimSpace(string(encoded)))
	if err != nil {
		t.Fatal(err)
	}
	if image[8] != 8 {
		t.Fatalf("fixture ABI=%d, want 8", image[8])
	}
	info, err := ParseCUBIN(image)
	if err != nil {
		t.Fatal(err)
	}
	if info.Arch != "sm_90" {
		t.Fatalf("real CUDA fixture Arch=%q, want sm_90", info.Arch)
	}
}
