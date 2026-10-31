package elfinspect

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"strings"
	"testing"
)

func put16(b []byte, i int, v uint16) { binary.LittleEndian.PutUint16(b[i:], v) }
func put32(b []byte, i int, v uint32) { binary.LittleEndian.PutUint32(b[i:], v) }
func put64(b []byte, i int, v uint64) { binary.LittleEndian.PutUint64(b[i:], v) }

// fixture is an in-memory ELF64 image, never executable or a runtime proof.
// It has section zero, .data, a shared string table, SYMTAB, and DYNSYM.
func fixture() []byte {
	b := make([]byte, 1024)
	copy(b, "\x7fELF\x02\x01\x01")
	put16(b, 16, uint16(elf.ET_DYN))
	put16(b, 18, uint16(elf.EM_X86_64))
	put32(b, 20, 1)
	put64(b, 32, 64)
	put64(b, 40, 512)
	put16(b, 52, 64)
	put16(b, 54, 56)
	put16(b, 56, 1)
	put16(b, 58, 64)
	put16(b, 60, 5)
	// A writable, allocated file-backed PT_LOAD segment with extra zero fill.
	put32(b, 64, uint32(elf.PT_LOAD))
	put32(b, 68, uint32(elf.PF_R|elf.PF_W))
	put64(b, 72, 256)
	put64(b, 80, 0x1000)
	put64(b, 96, 8)
	put64(b, 104, 16)
	put64(b, 112, 8)
	section := func(i int, typ elf.SectionType, flags elf.SectionFlag, addr, off, size uint64, link uint32, entsize uint64) {
		at := 512 + i*64
		put32(b, at+4, uint32(typ))
		put64(b, at+8, uint64(flags))
		put64(b, at+16, addr)
		put64(b, at+24, off)
		put64(b, at+32, size)
		put32(b, at+40, link)
		put64(b, at+56, entsize)
	}
	section(1, elf.SHT_PROGBITS, elf.SHF_ALLOC|elf.SHF_WRITE, 0x1000, 256, 8, 0, 0)
	copy(b[272:], "\x00cache\x00tls\x00abs\x00common\x00undef\x00")
	section(2, elf.SHT_STRTAB, 0, 0, 272, 28, 0, 0)
	section(3, elf.SHT_SYMTAB, 0, 0, 320, 6*24, 2, 24)
	section(4, elf.SHT_DYNSYM, 0, 0, 464, 2*24, 2, 24)
	symbol := func(at int, name uint32, bind byte, typ elf.SymType, sec elf.SectionIndex, value, size uint64) {
		put32(b, at, name)
		b[at+4] = bind<<4 | byte(typ)
		b[at+5] = byte(elf.STV_HIDDEN)
		put16(b, at+6, uint16(sec))
		put64(b, at+8, value)
		put64(b, at+16, size)
	}
	symbol(344, 1, 10, elf.STT_OBJECT, 1, 0x1000, 8)
	symbol(368, 7, 1, elf.STT_TLS, 1, 0, 4)
	symbol(392, 11, 1, elf.STT_OBJECT, elf.SHN_ABS, 12, 4)
	symbol(416, 15, 1, elf.STT_OBJECT, elf.SHN_COMMON, 8, 8)
	symbol(440, 22, 1, elf.STT_OBJECT, elf.SHN_UNDEF, 0, 0)
	symbol(488, 1, 1, elf.STT_OBJECT, 1, 0x1000, 8)
	return b
}

func TestTablesSectionsAndSegments(t *testing.T) {
	im, err := Read(bytes.NewReader(fixture()))
	if err != nil {
		t.Fatal(err)
	}
	if im.Header.Class != elf.ELFCLASS64 || im.Header.Type != elf.ET_DYN || im.TableState != TablesPresent || len(im.Tables) != 2 {
		t.Fatalf("wrong image: %+v", im)
	}
	if im.Tables[0].Section != 3 || im.Tables[0].Type != elf.SHT_SYMTAB || im.Tables[1].Section != 4 || im.Tables[1].Type != elf.SHT_DYNSYM {
		t.Fatal("table provenance lost")
	}
	s := im.Tables[0].Symbols
	if len(s) != 5 || s[0].Index != 1 || s[0].Name != "cache" || s[0].Binding != elf.SymBind(10) || s[0].Type != elf.STT_OBJECT || s[0].Other != byte(elf.STV_HIDDEN) {
		t.Fatalf("raw symbol lost: %+v", s)
	}
	if s[0].Section == nil || s[0].Section.Index != 1 || s[0].Section.Flags != elf.SHF_ALLOC|elf.SHF_WRITE || s[0].Section.Offset != 256 || s[0].Section.Addr != 0x1000 {
		t.Fatal("defined section lost")
	}
	if s[1].Type != elf.STT_TLS || s[1].Section == nil {
		t.Fatal("TLS metadata lost")
	}
	for i, sec := range []elf.SectionIndex{elf.SHN_ABS, elf.SHN_COMMON, elf.SHN_UNDEF} {
		if s[i+2].SectionIndex != sec || s[i+2].Section != nil {
			t.Fatal("reserved section reinterpreted")
		}
	}
	p := im.Segments[0]
	if p.Type != elf.PT_LOAD || p.Flags != elf.PF_R|elf.PF_W || p.Off != 256 || p.Vaddr != 0x1000 || p.Filesz != 8 || p.Memsz != 16 {
		t.Fatalf("segment metadata lost: %+v", p)
	}
	if len(im.Tables[1].Symbols) != 1 || im.Tables[1].Symbols[0].Binding != elf.STB_GLOBAL {
		t.Fatal("dual table entry conflated")
	}
}

func TestAbsentAndEmptyAreDistinct(t *testing.T) {
	for _, test := range []struct {
		name   string
		tables uint16
		want   TableState
		count  int
	}{{"absent", 3, TablesAbsent, 0}, {"null_only", 4, TablesPresent, 1}} {
		t.Run(test.name, func(t *testing.T) {
			b := fixture()
			put16(b, 60, test.tables)
			if test.count == 1 {
				put64(b, 512+3*64+32, 24)
			}
			im, err := Read(bytes.NewReader(b))
			if err != nil {
				t.Fatal(err)
			}
			if im.TableState != test.want || len(im.Tables) != test.count {
				t.Fatalf("missing state lost: %+v", im)
			}
			if test.count == 1 && len(im.Tables[0].Symbols) != 0 {
				t.Fatal("null symbol exposed")
			}
		})
	}
}

func TestMalformed(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func([]byte)
		want   string
	}{
		{"segment_overflow", func(b []byte) { put64(b, 72, ^uint64(0)-3) }, "program header"},
		{"virtual_overflow", func(b []byte) { put64(b, 80, ^uint64(0)-3) }, "segment"},
		{"filesz_gt_memsz", func(b []byte) { put64(b, 104, 4) }, "segment"},
		{"section_overflow", func(b []byte) { put64(b, 512+64+16, ^uint64(0)-3) }, "section"},
		{"symbol_overflow", func(b []byte) { put64(b, 344+8, ^uint64(0)-3) }, "symbol"},
		{"bad_section", func(b []byte) { put16(b, 344+6, 12) }, "section"},
		{"extended_section", func(b []byte) { put16(b, 344+6, uint16(elf.SHN_XINDEX)) }, "extended"},
		{"bad_name_offset", func(b []byte) { put32(b, 344, 9999) }, "name offset"},
		{"bad_link", func(b []byte) { put32(b, 512+3*64+40, 9) }, "link"},
		{"null_nonzero", func(b []byte) { b[320] = 1 }, "null"},
		{"bad_entsize", func(b []byte) { put64(b, 512+3*64+56, 8) }, "size"},
		{"multiple_tables", func(b []byte) { put32(b, 512+4*64+4, uint32(elf.SHT_SYMTAB)) }, "multiple"},
		{"compressed", func(b []byte) { put64(b, 512+3*64+8, uint64(elf.SHF_COMPRESSED)) }, "table"},
	} {
		t.Run(test.name, func(t *testing.T) {
			b := fixture()
			test.mutate(b)
			if _, err := Read(bytes.NewReader(b)); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got %v, want %s", err, test.want)
			}
		})
	}
	for _, b := range [][]byte{nil, []byte("not elf"), fixture()[:500]} {
		if _, err := Read(bytes.NewReader(b)); err == nil {
			t.Fatal("truncated/malformed input accepted")
		}
	}
}

func TestELF32BigEndian(t *testing.T) {
	b := make([]byte, 248)
	copy(b, "\x7fELF\x01\x02\x01")
	order := binary.BigEndian
	order.PutUint16(b[16:], uint16(elf.ET_REL))
	order.PutUint16(b[18:], uint16(elf.EM_386))
	order.PutUint32(b[20:], 1)
	order.PutUint32(b[32:], 128)
	order.PutUint16(b[40:], 52)
	order.PutUint16(b[46:], 40)
	order.PutUint16(b[48:], 3)
	copy(b[72:], "\x00z\x00")
	order.PutUint32(b[168+4:], uint32(elf.SHT_STRTAB))
	order.PutUint32(b[168+16:], 72)
	order.PutUint32(b[168+20:], 3)
	order.PutUint32(b[208+4:], uint32(elf.SHT_SYMTAB))
	order.PutUint32(b[208+16:], 80)
	order.PutUint32(b[208+20:], 32)
	order.PutUint32(b[208+24:], 1)
	order.PutUint32(b[208+36:], 16)
	order.PutUint32(b[96:], 1)
	order.PutUint32(b[100:], 0x100)
	order.PutUint32(b[104:], 4)
	b[108] = 10<<4 | byte(elf.STT_OBJECT)
	order.PutUint16(b[110:], uint16(elf.SHN_ABS))
	im, err := Read(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	s := im.Tables[0].Symbols[0]
	if im.Header.Class != elf.ELFCLASS32 || im.Header.Data != elf.ELFDATA2MSB || s.Name != "z" || s.Binding != 10 || s.Value != 0x100 || s.Size != 4 || s.SectionIndex != elf.SHN_ABS || s.Section != nil {
		t.Fatalf("32-bit metadata lost: %+v", s)
	}
	if _, err := Read(nil); err == nil {
		t.Fatal("nil reader accepted")
	}
}
