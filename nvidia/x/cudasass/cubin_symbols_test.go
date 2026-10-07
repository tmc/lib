package cudasass

import (
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"errors"
	"testing"
)

// symbolImage constructs an ELF64 fixture; it is not compiler output.
func symbolImage(info byte, section uint16, value uint64) []byte {
	b := make([]byte, 64+3*64+48+5)
	copy(b, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	binary.LittleEndian.PutUint16(b[16:], 1)
	binary.LittleEndian.PutUint16(b[18:], uint16(elf.EM_CUDA))
	binary.LittleEndian.PutUint32(b[20:], 1)
	binary.LittleEndian.PutUint64(b[40:], 64)
	binary.LittleEndian.PutUint16(b[52:], 64)
	binary.LittleEndian.PutUint16(b[58:], 64)
	binary.LittleEndian.PutUint16(b[60:], 3)
	s := b[128:192]
	binary.LittleEndian.PutUint32(s[4:], uint32(elf.SHT_SYMTAB))
	binary.LittleEndian.PutUint64(s[24:], 256)
	binary.LittleEndian.PutUint64(s[32:], 48)
	binary.LittleEndian.PutUint32(s[40:], 2)
	binary.LittleEndian.PutUint64(s[56:], 24)
	s = b[192:256]
	binary.LittleEndian.PutUint32(s[4:], uint32(elf.SHT_STRTAB))
	binary.LittleEndian.PutUint64(s[24:], 304)
	binary.LittleEndian.PutUint64(s[32:], 5)
	s = b[280:304]
	binary.LittleEndian.PutUint32(s, 1)
	s[4] = info
	binary.LittleEndian.PutUint16(s[6:], section)
	binary.LittleEndian.PutUint64(s[8:], value)
	binary.LittleEndian.PutUint64(s[16:], 4)
	copy(b[304:], []byte{0, 'r', 'a', 'w', 0})
	return b
}

func TestCubinSymbols(t *testing.T) {
	for _, tt := range []struct {
		name    string
		info    byte
		section uint16
		value   uint64
	}{{"zero", 0x21, 0, 0}, {"weak object", 0x21, 0, 64}, {"processor type", 0x1d, 0, 1024}, {"absolute", 0x11, uint16(elf.SHN_ABS), 3}, {"defined", 0x11, 2, 7}} {
		t.Run(tt.name, func(t *testing.T) {
			b := symbolImage(tt.info, tt.section, tt.value)
			want := sha256.Sum256(b)
			im, err := OpenCubin(b)
			if err != nil {
				t.Fatal(err)
			}
			b[280] = 99
			rows, err := im.Symbols("raw")
			if err != nil || len(rows) != 1 {
				t.Fatalf("rows=%v error=%v", rows, err)
			}
			r := rows[0]
			if r.Name != "raw" || r.Value != tt.value || r.Size != 4 || r.Type != elf.ST_TYPE(tt.info) || r.Binding != elf.ST_BIND(tt.info) || r.Section != elf.SectionIndex(tt.section) || im.SHA256() != want {
				t.Fatalf("metadata=%+v", r)
			}
			rows[0].Value = 999
			again, _ := im.Symbols("raw")
			if again[0].Value != tt.value {
				t.Fatal("result aliases image")
			}
			absent, err := im.Symbols("absent")
			if err != nil || len(absent) != 0 {
				t.Fatal(absent, err)
			}
		})
	}
}

func TestCubinSymbolRefusals(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func([]byte)
	}{
		{"empty table", func(b []byte) { binary.LittleEndian.PutUint64(b[160:], 0) }},
		{"name offset", func(b []byte) { binary.LittleEndian.PutUint32(b[280:], 99) }},
		{"unterminated name", func(b []byte) { b[308] = 1 }},
		{"entry size", func(b []byte) { binary.LittleEndian.PutUint64(b[184:], 1) }},
		{"oversize", func(b []byte) { binary.LittleEndian.PutUint64(b[160:], 9<<20) }},
		{"link", func(b []byte) { binary.LittleEndian.PutUint32(b[168:], 99) }},
		{"invalid section", func(b []byte) { binary.LittleEndian.PutUint16(b[286:], 99) }},
		{"extended section", func(b []byte) { binary.LittleEndian.PutUint16(b[286:], uint16(elf.SHN_XINDEX)) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := symbolImage(0x21, 0, 0)
			tt.change(b)
			im, err := OpenCubin(b)
			if err != nil {
				return
			}
			if _, err := im.Symbols("raw"); err == nil {
				t.Fatal("invalid metadata accepted")
			}
		})
	}
	b := symbolImage(0x21, 0, 0)
	binary.LittleEndian.PutUint32(b[132:], uint32(elf.SHT_PROGBITS))
	im, err := OpenCubin(b)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := im.Symbols("raw"); !errors.Is(err, elf.ErrNoSymbols) {
		t.Fatal(err)
	}
	im, _ = OpenCubin(symbolImage(0x21, 0, 0))
	if _, err := im.Symbols(""); err == nil {
		t.Fatal("empty query accepted")
	}
}

func TestCubinSymbolsDuplicates(t *testing.T) {
	original := symbolImage(0x21, 0, 0)
	b := make([]byte, len(original)+24)
	copy(b, original[:304])
	copy(b[304:328], original[280:304])
	copy(b[328:], original[304:])
	binary.LittleEndian.PutUint64(b[160:], 72)
	binary.LittleEndian.PutUint64(b[216:], 328)
	im, err := OpenCubin(b)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := im.Symbols("raw")
	if err != nil || len(rows) != 2 || rows[0] != rows[1] {
		t.Fatalf("duplicates = %v, %v", rows, err)
	}
}

func TestCubinSymbolNullEntry(t *testing.T) {
	b := symbolImage(0x21, 0, 0)
	b[256] = 1
	im, err := OpenCubin(b)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := im.Symbols("raw"); err == nil {
		t.Fatal("nonzero null symbol accepted")
	}
}

func TestCubinSymbolAggregateNames(t *testing.T) {
	const entries = 2200
	original := symbolImage(0x21, 0, 0)
	start := 256 + entries*24
	b := make([]byte, start+4098)
	copy(b, original[:256])
	binary.LittleEndian.PutUint64(b[160:], entries*24)
	binary.LittleEndian.PutUint64(b[216:], uint64(start))
	binary.LittleEndian.PutUint64(b[224:], 4098)
	for i := 1; i < entries; i++ {
		copy(b[256+i*24:], original[280:304])
	}
	for i := 1; i <= 4096; i++ {
		b[start+i] = 'x'
	}
	im, err := OpenCubin(b)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := im.Symbols("nonmatch"); err == nil {
		t.Fatal("aggregate decoded name budget ignored")
	}
}
