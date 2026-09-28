package cudakernel

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"testing"
)

func TestParseCUBINArchitecture(t *testing.T) {
	// Flags below were checked against nvcc 12.6/13.4 output and the matching
	// cuobjdump ELF/nvdisasm target headers. ABI 8 is not limited to Blackwell.
	for _, tt := range []struct {
		name  string
		abi   byte
		flags uint32
		want  string
	}{
		{"ABI7 sm80", 7, 0x00500550, "sm_80"},
		{"ABI7 sm90", 7, 0x005a055a, "sm_90"},
		{"ABI8 sm80", 8, 0x06005004, "sm_80"},
		{"ABI8 sm90", 8, 0x06005a04, "sm_90"},
		{"ABI8 sm100", 8, 0x06006402, "sm_100"},
		{"ABI8 sm121", 8, 0x09007902, "sm_121"},
		{"unknown ABI", 9, 0x09007902, ""},
		{"unspecified ABI", 0, 0x005a055a, ""},
		{"unspecified SM", 8, 0x09000002, ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for _, class := range []elf.Class{elf.ELFCLASS32, elf.ELFCLASS64} {
				for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
					b := cubinHeader(t, class, order, tt.abi, tt.flags, elf.EM_CUDA)
					got, err := ParseCUBIN(b)
					if err != nil {
						t.Fatal(err)
					}
					if got.Arch != tt.want {
						t.Errorf("class=%v order=%v Arch=%q want %q", class, order, got.Arch, tt.want)
					}
				}
			}
		})
	}
}

func TestParseCUBINRejectsHostELF(t *testing.T) {
	b := cubinHeader(t, elf.ELFCLASS64, binary.LittleEndian, 7, 0x50, elf.EM_X86_64)
	if _, err := ParseCUBIN(b); err == nil {
		t.Fatal("host ELF accepted as CUDA")
	}
}

func TestParseCUBINTruncatedHeader(t *testing.T) {
	b := cubinHeader(t, elf.ELFCLASS64, binary.LittleEndian, 8, 0x9007902, elf.EM_CUDA)
	for n := 0; n < len(b); n++ {
		if _, err := ParseCUBIN(b[:n]); err == nil {
			t.Fatalf("accepted %d-byte header", n)
		}
	}
}

func cubinHeader(t *testing.T, class elf.Class, order binary.ByteOrder, abi byte, flags uint32, machine elf.Machine) []byte {
	t.Helper()
	var ident [16]byte
	copy(ident[:], "\x7fELF")
	ident[elf.EI_CLASS] = byte(class)
	ident[elf.EI_DATA] = byte(elf.ELFDATA2LSB)
	if order == binary.BigEndian {
		ident[elf.EI_DATA] = byte(elf.ELFDATA2MSB)
	}
	ident[elf.EI_VERSION] = byte(elf.EV_CURRENT)
	ident[elf.EI_OSABI] = 0x41
	if abi == 7 {
		ident[elf.EI_OSABI] = 0x33
	}
	ident[elf.EI_ABIVERSION] = abi
	var header any
	if class == elf.ELFCLASS64 {
		header = elf.Header64{Ident: ident, Type: uint16(elf.ET_EXEC), Machine: uint16(machine), Version: uint32(elf.EV_CURRENT), Flags: flags, Ehsize: 64}
	} else {
		header = elf.Header32{Ident: ident, Type: uint16(elf.ET_EXEC), Machine: uint16(machine), Version: uint32(elf.EV_CURRENT), Flags: flags, Ehsize: 52}
	}
	var b bytes.Buffer
	if err := binary.Write(&b, order, header); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestParseCUBINLegacyToolkitVersion(t *testing.T) {
	b := cubinHeader(t, elf.ELFCLASS64, binary.LittleEndian, 7, 0x00500550, elf.EM_CUDA)
	binary.LittleEndian.PutUint32(b[20:24], 126) // nvcc 12.6 legacy ELF witness
	original := bytes.Clone(b)
	got, err := ParseCUBIN(b)
	if err != nil || got.Arch != "sm_80" {
		t.Fatalf("info=%+v error=%v", got, err)
	}
	if !bytes.Equal(b, original) {
		t.Fatal("parser modified input artifact")
	}
	b[elf.EI_ABIVERSION] = 8
	if _, err := ParseCUBIN(b); err == nil {
		t.Fatal("accepted mismatched version in nonlegacy header")
	}
}
