package ebpfhost

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// versionELF builds inert ELF metadata with two definitions of probe: hidden
// V1 at file offset 0x100, and default V2 at 0x110. It has no static symbols.
// The fixture is parsed as data and is never loaded or executed.
func versionELF() []byte {
	dynstr := []byte("\x00probe\x00V1\x00V2\x00")
	syms := []elf.Sym64{{},
		{Name: 1, Info: elf.ST_INFO(elf.STB_GLOBAL, elf.STT_FUNC), Shndx: 1, Value: 0x400100, Size: 1},
		{Name: 1, Info: elf.ST_INFO(elf.STB_GLOBAL, elf.STT_FUNC), Shndx: 1, Value: 0x400110, Size: 1},
	}
	encode := func(v any) []byte {
		var b bytes.Buffer
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			panic(err) // All fixture values have fixed-size binary representations.
		}
		return b.Bytes()
	}
	var versions []byte
	for i, name := range []string{"V1", "V2"} {
		var hash uint32
		for _, c := range []byte(name) {
			hash = hash<<4 + uint32(c)
			g := hash & 0xf0000000
			hash ^= g >> 24
			hash &^= g
		}
		var next uint32
		if i == 0 {
			next = 28
		}
		versions = append(versions, encode(struct {
			Version, Flags, Index, Count uint16
			Hash, Aux, Next              uint32
			Name, AuxNext                uint32
		}{1, 0, uint16(i + 2), 1, hash, 20, next, uint32(bytes.Index(dynstr, []byte(name))), 0})...)
	}
	shstr := []byte("\x00.text\x00.dynstr\x00.dynsym\x00.gnu.version\x00.gnu.version_d\x00.note.gnu.build-id\x00.shstrtab\x00.plt\x00.data\x00.iplt\x00")
	bodies := [][]byte{nil, make([]byte, 64), dynstr, encode(syms), encode([]uint16{0, 0x8002, 3}), versions,
		encode([]uint32{4, 4, 3, 0x00554e47, 0x04030201}), shstr, make([]byte, 16), make([]byte, 16)}
	sections := []elf.Section64{{},
		{Type: uint32(elf.SHT_PROGBITS), Flags: uint64(elf.SHF_ALLOC | elf.SHF_EXECINSTR), Addralign: 16},
		{Type: uint32(elf.SHT_STRTAB), Addralign: 1},
		{Type: uint32(elf.SHT_DYNSYM), Link: 2, Info: 1, Addralign: 8, Entsize: 24},
		{Type: uint32(elf.SHT_GNU_VERSYM), Link: 3, Addralign: 2, Entsize: 2},
		{Type: uint32(elf.SHT_GNU_VERDEF), Link: 2, Info: 2, Addralign: 4},
		{Type: uint32(elf.SHT_NOTE), Addralign: 4},
		{Type: uint32(elf.SHT_STRTAB), Addralign: 1},
		{Type: uint32(elf.SHT_PROGBITS), Flags: uint64(elf.SHF_ALLOC | elf.SHF_EXECINSTR), Addralign: 16},
		{Type: uint32(elf.SHT_PROGBITS), Flags: uint64(elf.SHF_ALLOC | elf.SHF_WRITE), Addralign: 8},
	}
	names := []string{"", ".text", ".dynstr", ".dynsym", ".gnu.version", ".gnu.version_d", ".note.gnu.build-id", ".shstrtab", ".plt", ".data"}
	data := make([]byte, 0x100)
	for i := 1; i < len(sections); i++ {
		for uint64(len(data))%sections[i].Addralign != 0 {
			data = append(data, 0)
		}
		sections[i].Name = uint32(bytes.Index(shstr, append([]byte(names[i]), 0)))
		sections[i].Off = uint64(len(data))
		sections[i].Addr = 0x400000 + sections[i].Off
		sections[i].Size = uint64(len(bodies[i]))
		data = append(data, bodies[i]...)
	}
	for len(data)%8 != 0 {
		data = append(data, 0)
	}
	header := elf.Header64{Ident: [16]byte{0x7f, 'E', 'L', 'F', byte(elf.ELFCLASS64), byte(elf.ELFDATA2LSB), 1},
		Type: uint16(elf.ET_DYN), Machine: uint16(elf.EM_X86_64), Version: 1,
		Phoff: 64, Shoff: uint64(len(data)), Ehsize: 64, Phentsize: 56, Phnum: 1, Shentsize: 64, Shnum: uint16(len(sections)), Shstrndx: 7}
	data = append(data, encode(sections)...)
	program := elf.Prog64{Type: uint32(elf.PT_LOAD), Flags: uint32(elf.PF_R | elf.PF_X), Vaddr: 0x400000, Filesz: uint64(len(data)), Memsz: uint64(len(data)), Align: 0x1000}
	copy(data, encode(header))
	copy(data[64:], encode(program))
	return data
}

func writeVersionELF(t testing.TB, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "probe.so")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestResolveImageDefaultVersion(t *testing.T) {
	path := writeVersionELF(t, versionELF())
	image, err := ResolveImage(path, "probe")
	if err != nil || image.Offset != 0x110 || image.BuildID != "01020304" {
		t.Fatalf("default image = %+v, %v", image, err)
	}
}

func TestResolveImageExactVersion(t *testing.T) {
	path := writeVersionELF(t, versionELF())
	for _, tt := range []struct {
		version, observed string
		offset            uint64
	}{
		{"V1", "V1", 0x100}, {"V2", "V2", 0x110}, {"", "V2", 0x110},
	} {
		t.Run(tt.version, func(t *testing.T) {
			image, err := ResolveImageVersion(path, "probe", tt.version)
			if err != nil || image.Offset != tt.offset || image.Symbol != "probe" || image.SymbolVersion != tt.observed || image.Size == 0 || len(image.SHA256) != 64 {
				t.Fatalf("image = %+v, %v", image, err)
			}
		})
	}
	for _, tt := range []struct{ symbol, version string }{{"probe", "V3"}, {"absent", "V2"}} {
		if _, err := ResolveImageVersion(path, tt.symbol, tt.version); err == nil || !strings.Contains(err.Error(), tt.symbol) {
			t.Fatalf("request %+v: error = %v", tt, err)
		}
	}
}

func TestResolveImageVersionMetadata(t *testing.T) {
	for _, name := range []string{"short-table", "long-table", "odd-table", "missing-table", "unknown-index", "missing-definition", "default-ambiguity", "no-default", "version-link", "version-link-range", "version-link-max", "dynamic-link-max", "version-entry-size", "definition-link", "definition-link-range", "definition-count", "duplicate-index", "aux-count", "aux-offset", "aux-name", "aux-next", "definition-next", "definition-format"} {
		t.Run(name, func(t *testing.T) {
			data := versionELF()
			f, err := elf.NewFile(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			shoff := binary.LittleEndian.Uint64(data[40:])
			switch name {
			case "short-table":
				binary.LittleEndian.PutUint64(data[shoff+4*64+32:], 4)
			case "long-table":
				binary.LittleEndian.PutUint64(data[shoff+4*64+32:], 8)
			case "odd-table":
				binary.LittleEndian.PutUint64(data[shoff+4*64+32:], 7)
			case "missing-table":
				binary.LittleEndian.PutUint32(data[shoff+4*64+4:], uint32(elf.SHT_PROGBITS))
			case "unknown-index":
				binary.LittleEndian.PutUint16(data[f.Section(".gnu.version").Offset+4:], 42)
			case "missing-definition":
				binary.LittleEndian.PutUint32(data[shoff+5*64+4:], uint32(elf.SHT_PROGBITS))
			case "default-ambiguity":
				binary.LittleEndian.PutUint16(data[f.Section(".gnu.version").Offset+2:], 2)
				binary.LittleEndian.PutUint64(data[f.Section(".dynsym").Offset+2*24+8:], 0x400100)
			case "no-default":
				binary.LittleEndian.PutUint16(data[f.Section(".gnu.version").Offset+4:], 0x8003)
			case "version-link":
				binary.LittleEndian.PutUint32(data[shoff+4*64+40:], 2)
			case "version-link-range":
				binary.LittleEndian.PutUint32(data[shoff+4*64+40:], 999)
			case "version-link-max":
				binary.LittleEndian.PutUint32(data[shoff+4*64+40:], ^uint32(0))
			case "dynamic-link-max":
				binary.LittleEndian.PutUint32(data[shoff+3*64+40:], ^uint32(0))
			case "version-entry-size":
				binary.LittleEndian.PutUint64(data[shoff+4*64+56:], 8)
			case "definition-link":
				binary.LittleEndian.PutUint32(data[shoff+5*64+40:], 3)
			case "definition-link-range":
				binary.LittleEndian.PutUint32(data[shoff+5*64+40:], 999)
			case "definition-count":
				binary.LittleEndian.PutUint32(data[shoff+5*64+44:], 1)
			case "duplicate-index":
				binary.LittleEndian.PutUint16(data[f.Section(".gnu.version_d").Offset+28+4:], 2)
			case "aux-count":
				binary.LittleEndian.PutUint16(data[f.Section(".gnu.version_d").Offset+28+6:], 2)
			case "aux-offset":
				binary.LittleEndian.PutUint32(data[f.Section(".gnu.version_d").Offset+12:], 0)
			case "aux-name":
				binary.LittleEndian.PutUint32(data[f.Section(".gnu.version_d").Offset+20:], 999)
			case "aux-next":
				binary.LittleEndian.PutUint32(data[f.Section(".gnu.version_d").Offset+24:], 8)
			case "definition-next":
				binary.LittleEndian.PutUint32(data[f.Section(".gnu.version_d").Offset+16:], 1)
			case "definition-format":
				binary.LittleEndian.PutUint16(data[f.Section(".gnu.version_d").Offset:], 2)
			}
			path := writeVersionELF(t, data)
			if _, err := ResolveImage(path, "probe"); err == nil || !strings.Contains(err.Error(), "probe") {
				t.Fatalf("unqualified default admitted: %v", err)
			}
		})
	}
}

func TestResolveImageUnversionedAndStaticOnly(t *testing.T) {
	for _, static := range []bool{false, true} {
		t.Run(fmt.Sprintf("static=%v", static), func(t *testing.T) {
			data := versionELF()
			shoff := binary.LittleEndian.Uint64(data[40:])
			binary.LittleEndian.PutUint64(data[shoff+3*64+32:], 48) // Null plus one function.
			for _, index := range []uint64{4, 5} {
				binary.LittleEndian.PutUint32(data[shoff+index*64+4:], uint32(elf.SHT_PROGBITS))
			}
			if static {
				binary.LittleEndian.PutUint32(data[shoff+3*64+4:], uint32(elf.SHT_SYMTAB))
			}
			path := writeVersionELF(t, data)
			image, err := ResolveImage(path, "probe")
			if err != nil || image.Offset != 0x100 || image.SymbolVersion != "" {
				t.Fatalf("unversioned image = %+v, %v", image, err)
			}
			if _, err := ResolveImageVersion(path, "probe", "V1"); err == nil {
				t.Fatal("unversioned symbol established requested version")
			}
		})
	}
}

func TestResolveImageVersionDependencies(t *testing.T) {
	for _, name := range []string{"valid", "link", "count", "aux-count", "duplicate-index", "aux-next", "library-name"} {
		t.Run(name, func(t *testing.T) {
			data := versionELF()
			shoff := binary.LittleEndian.Uint64(data[40:])
			section := data[shoff+9*64:]
			binary.LittleEndian.PutUint32(section[4:], uint32(elf.SHT_GNU_VERNEED))
			binary.LittleEndian.PutUint64(section[24:], uint64(len(data)))
			binary.LittleEndian.PutUint64(section[32:], 32)
			binary.LittleEndian.PutUint32(section[40:], 2)
			binary.LittleEndian.PutUint32(section[44:], 1)
			need := make([]byte, 32)
			binary.LittleEndian.PutUint16(need, 1)
			binary.LittleEndian.PutUint16(need[2:], 1)
			binary.LittleEndian.PutUint32(need[4:], 1) // Library name: probe.
			binary.LittleEndian.PutUint32(need[8:], 16)
			binary.LittleEndian.PutUint16(need[22:], 4)
			binary.LittleEndian.PutUint32(need[24:], 7) // Dependency name: V1.
			switch name {
			case "link":
				binary.LittleEndian.PutUint32(section[40:], 3)
			case "count":
				binary.LittleEndian.PutUint32(section[44:], 2)
			case "aux-count":
				binary.LittleEndian.PutUint16(need[2:], 2)
			case "duplicate-index":
				binary.LittleEndian.PutUint16(need[22:], 3)
			case "aux-next":
				binary.LittleEndian.PutUint32(need[28:], 16)
			case "library-name":
				binary.LittleEndian.PutUint32(need[4:], 999)
			}
			data = append(data, need...)
			image, err := ResolveImage(writeVersionELF(t, data), "probe")
			if name == "valid" {
				if err != nil || image.SymbolVersion != "V2" || image.Offset != 0x110 {
					t.Fatalf("valid dependencies: %+v, %v", image, err)
				}
			} else if err == nil {
				t.Fatalf("invalid dependency metadata admitted: %+v", image)
			}
		})
	}
}

func TestResolveImageRefusesIndirectAndInvalidCode(t *testing.T) {
	for _, name := range []string{"ifunc", "plt", "iplt", "non-code", "outside-section", "outside-file", "ambiguous", "undefined"} {
		t.Run(name, func(t *testing.T) {
			data := versionELF()
			f, err := elf.NewFile(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			dyn := f.Section(".dynsym").Offset
			second := dyn + 2*24
			put64 := func(off, value uint64) { binary.LittleEndian.PutUint64(data[off:], value) }
			switch name {
			case "ifunc":
				data[second+4] = elf.ST_INFO(elf.STB_GLOBAL, elf.STT_GNU_IFUNC)
			case "plt", "iplt", "non-code":
				index, section := uint16(8), f.Section(".plt")
				if name == "iplt" {
					shoff := binary.LittleEndian.Uint64(data[40:])
					strings, err := f.Section(".shstrtab").Data()
					if err != nil {
						t.Fatal(err)
					}
					binary.LittleEndian.PutUint32(data[shoff+8*64:], uint32(bytes.Index(strings, []byte(".iplt\x00"))))
				}
				if name == "non-code" {
					index, section = 9, f.Section(".data")
				}
				binary.LittleEndian.PutUint16(data[second+6:], index)
				put64(second+8, section.Addr)
			case "outside-section":
				put64(second+8, 0x400140)
			case "outside-file":
				put64(64+8, uint64(len(data)))
			case "ambiguous":
				binary.LittleEndian.PutUint16(data[f.Section(".gnu.version").Offset+2:], 3)
			case "undefined":
				binary.LittleEndian.PutUint16(data[second+6:], 0)
			}
			path := writeVersionELF(t, data)
			if _, err := ResolveImageVersion(path, "probe", "V2"); err == nil || !strings.Contains(err.Error(), "probe") {
				t.Fatalf("invalid code accepted: %v", err)
			}
		})
	}
}

func ExampleResolveImageVersion() {
	dir, err := os.MkdirTemp("", "elf-version-example-")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "probe.so")
	if err := os.WriteFile(path, versionELF(), 0600); err != nil {
		fmt.Println(err)
		return
	}
	image, err := ResolveImageVersion(path, "probe", "V1")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%s@%s at %#x\n", image.Symbol, image.SymbolVersion, image.Offset)
	// Output: probe@V1 at 0x100
}
