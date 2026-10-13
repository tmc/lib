package cudainspect

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"strings"
	"testing"
)

// metadataELF is a synthetic ELF64 artifact with text and optional shared
// memory sections. No machine instructions or resource attributes are invented.
func metadataELF(shared bool, size uint64, machine elf.Machine) []byte {
	names := []byte("\x00.shstrtab\x00.text.kernel\x00.nv.shared.kernel\x00")
	count := 3
	if shared {
		count++
	}
	b := make([]byte, 64+count*64+len(names))
	copy(b, "\x7fELF")
	b[4], b[5], b[6] = 2, 1, 1
	b[elf.EI_OSABI], b[elf.EI_ABIVERSION] = 0x33, 7 // CUDA ABI 7: SM in flag bits 0..7
	order := binary.LittleEndian
	order.PutUint16(b[16:], uint16(elf.ET_EXEC))
	order.PutUint16(b[18:], uint16(machine))
	order.PutUint32(b[20:], 1)
	order.PutUint64(b[40:], 64)
	order.PutUint32(b[48:], 121)
	order.PutUint16(b[52:], 64)
	order.PutUint16(b[58:], 64)
	order.PutUint16(b[60:], uint16(count))
	order.PutUint16(b[62:], 1)
	section := func(index int, name string, kind elf.SectionType, offset, length uint64) {
		h := b[64+index*64:]
		order.PutUint32(h, uint32(bytes.Index(names, []byte(name))))
		order.PutUint32(h[4:], uint32(kind))
		order.PutUint64(h[24:], offset)
		order.PutUint64(h[32:], length)
	}
	section(1, ".shstrtab", elf.SHT_STRTAB, uint64(64+count*64), uint64(len(names)))
	section(2, ".text.kernel", elf.SHT_PROGBITS, 0, 0)
	if shared {
		section(3, ".nv.shared.kernel", elf.SHT_NOBITS, 0, size)
	}
	copy(b[64+count*64:], names)
	return b
}

func TestReportSharedMemoryPresence(t *testing.T) {
	i, _ := NewInspector(".")
	for _, tt := range []struct {
		name    string
		present bool
		size    uint64
		known   bool
	}{
		{"missing", false, 0, false},
		{"explicit_zero", true, 0, true},
		{"nonzero", true, 4096, true},
		{"overflow", true, 1 << 32, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := i.Inspect(metadataELF(tt.present, tt.size, elf.EM_CUDA), "kernel")
			if err != nil {
				t.Fatal(err)
			}
			if r.SharedMemoryKnown != tt.known || (tt.known && uint64(r.SharedMemory) != tt.size) {
				t.Fatalf("shared memory: %+v", r)
			}
			if strings.Contains(strings.Join(r.Unknowns, ";"), "static shared memory unavailable") == tt.known {
				t.Fatalf("unknowns: %v", r.Unknowns)
			}
			if r.DynamicSharedMemoryKnown || r.PeakGPRKnown {
				t.Fatal("invented runtime facts")
			}
		})
	}
}

func TestReportELFIdentityAndDeterminism(t *testing.T) {
	i, _ := NewInspector(".")
	data := metadataELF(true, 128, elf.EM_CUDA)
	a, err := i.Inspect(data, "kernel")
	if err != nil {
		t.Fatal(err)
	}
	b, _ := i.Inspect(data, "kernel")
	var x, y bytes.Buffer
	if err := a.WriteJSON(&x); err != nil {
		t.Fatal(err)
	}
	if err := b.WriteJSON(&y); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(x.Bytes(), y.Bytes()) {
		t.Fatal("nondeterministic JSON")
	}
	if !a.ELFParsed || a.ELFMachine != "EM_CUDA" || a.ELFClass != "ELF64" || a.ELFData != "ELFDATA2LSB" || a.ELFType != "ET_EXEC" {
		t.Fatalf("ELF identity: %+v", a)
	}
	if len(a.SectionDetails) != 4 || len(a.CodeSymbols) != 1 || a.CodeSymbols[0] != "kernel" {
		t.Fatalf("inventory: %+v", a)
	}
	last := a.SectionDetails[3]
	if last.Name != ".nv.shared.kernel" || last.Size != 128 || last.Type != "SHT_NOBITS" {
		t.Fatalf("section: %+v", last)
	}
	foreign, _ := i.Inspect(metadataELF(true, 128, elf.EM_X86_64), "kernel")
	if !foreign.ELFParsed || foreign.SharedMemoryKnown || foreign.Architecture != "" {
		t.Fatalf("foreign ELF given CUDA facts: %+v", foreign)
	}
	for _, corrupt := range [][]byte{[]byte("not ELF"), data[:32]} {
		r, err := i.Inspect(corrupt, "kernel")
		if err != nil {
			t.Fatal(err)
		}
		if r.ELFParsed || r.SharedMemoryKnown || len(r.Unknowns) == 0 {
			t.Fatalf("corrupt ELF accepted: %+v", r)
		}
	}
}
