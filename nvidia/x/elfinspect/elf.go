package elfinspect

import (
	"bytes"
	"debug/elf"
	"fmt"
	"io"
	"strings"
)

// TableState distinguishes absent symbol tables from decoded tables.
type TableState string

const (
	// TablesAbsent means neither SHT_SYMTAB nor SHT_DYNSYM exists.
	TablesAbsent TableState = "absent"
	// TablesPresent means at least one symbol table was decoded successfully.
	TablesPresent TableState = "present"
)

// Image contains file metadata only. Its slices preserve file order.
type Image struct {
	Header     elf.FileHeader
	Sections   []Section
	Segments   []elf.ProgHeader
	TableState TableState
	Tables     []SymbolTable
}

// Section preserves the section index, including section zero, and raw header.
type Section struct {
	Index elf.SectionIndex
	elf.SectionHeader
}

// SymbolTable identifies the source section of its entries. The mandatory null
// symbol is validated but omitted from Symbols, as in debug/elf.
type SymbolTable struct {
	Section elf.SectionIndex
	Type    elf.SectionType
	Name    string
	Symbols []Symbol
}

// Symbol preserves an entry's raw fields and source-table entry index. Section
// is non-nil only for an ordinary defined section. Reserved indices, including
// UNDEF, ABS, COMMON, and processor-specific indices, remain in SectionIndex.
// Binding 10 is GNU-unique; Type STT_TLS remains TLS, not an ordinary object.
type Symbol struct {
	Index        uint64
	Name         string
	Info, Other  byte
	Binding      elf.SymBind
	Type         elf.SymType
	SectionIndex elf.SectionIndex
	Section      *Section
	Value, Size  uint64
}

// Read decodes metadata from r without taking ownership of it. Missing symbol
// tables return TablesAbsent, not an empty successful symbol census. Malformed
// tables, extended symbol section indices, compressed tables, and overflowing
// ranges return errors. Each symbol/string table is limited to 8 MiB, with at
// most 65536 symbols and 8 MiB of aggregate decoded names per table.
//
// Offset/size fields are file metadata: Read does not establish that symbol
// storage belongs to a PT_LOAD segment or that file storage is mapped at run time.
func Read(r io.ReaderAt) (*Image, error) {
	if r == nil {
		return nil, fmt.Errorf("elfinspect: nil reader")
	}
	f, err := elf.NewFile(r)
	if err != nil {
		return nil, fmt.Errorf("elfinspect: read elf: %w", err)
	}
	defer f.Close()
	im := &Image{Header: f.FileHeader, TableState: TablesAbsent}
	for i, s := range f.Sections {
		if overflow(s.Offset, s.FileSize) || overflow(s.Addr, s.Size) {
			return nil, fmt.Errorf("elfinspect: section %d range overflows", i)
		}
		im.Sections = append(im.Sections, Section{elf.SectionIndex(i), s.SectionHeader})
	}
	for i, p := range f.Progs {
		if overflow(p.Off, p.Filesz) || overflow(p.Vaddr, p.Memsz) || p.Type == elf.PT_LOAD && p.Filesz > p.Memsz {
			return nil, fmt.Errorf("elfinspect: segment %d range invalid", i)
		}
		im.Segments = append(im.Segments, p.ProgHeader)
	}
	seen := make(map[elf.SectionType]bool)
	for i, s := range f.Sections {
		if s.Type != elf.SHT_SYMTAB && s.Type != elf.SHT_DYNSYM {
			continue
		}
		if seen[s.Type] {
			return nil, fmt.Errorf("elfinspect: multiple %s sections", s.Type)
		}
		seen[s.Type] = true
		if err := validateTable(f, s); err != nil {
			return nil, fmt.Errorf("elfinspect: table %d: %w", i, err)
		}
		var syms []elf.Symbol
		if s.Type == elf.SHT_SYMTAB {
			syms, err = f.Symbols()
		} else {
			syms, err = f.DynamicSymbols()
		}
		if err != nil {
			return nil, fmt.Errorf("elfinspect: table %d: %w", i, err)
		}
		table := SymbolTable{Section: elf.SectionIndex(i), Type: s.Type, Name: s.Name}
		for j, sym := range syms {
			if overflow(sym.Value, sym.Size) {
				return nil, fmt.Errorf("elfinspect: symbol %d range overflows", j+1)
			}
			v := Symbol{Index: uint64(j + 1), Name: sym.Name, Info: sym.Info, Other: sym.Other, Binding: elf.ST_BIND(sym.Info), Type: elf.ST_TYPE(sym.Info), SectionIndex: sym.Section, Value: sym.Value, Size: sym.Size}
			if sym.Section == elf.SHN_XINDEX {
				return nil, fmt.Errorf("elfinspect: extended symbol section unsupported")
			}
			if sym.Section != elf.SHN_UNDEF && sym.Section < elf.SHN_LORESERVE {
				if int(sym.Section) >= len(im.Sections) {
					return nil, fmt.Errorf("elfinspect: symbol %d section invalid", j+1)
				}
				v.Section = &im.Sections[sym.Section]
			}
			table.Symbols = append(table.Symbols, v)
		}
		im.Tables = append(im.Tables, table)
		im.TableState = TablesPresent
	}
	return im, nil
}

func overflow(start, size uint64) bool { return size > ^uint64(0)-start }

// debug/elf silently drops invalid name offsets, so validate names before using
// its typed decoder. Keep raw table interpretation in the standard library.
func validateTable(f *elf.File, s *elf.Section) error {
	entry := uint64(24)
	if f.Class == elf.ELFCLASS32 {
		entry = 16
	}
	if s.Entsize != entry || s.Size < entry || s.Size%entry != 0 || s.Size/entry > 65536 || s.Size > 8<<20 {
		return fmt.Errorf("invalid or excessive symbol table size")
	}
	if s.Flags&elf.SHF_COMPRESSED != 0 || strings.HasPrefix(s.Name, ".zdebug") {
		return fmt.Errorf("compressed symbol table unsupported")
	}
	if uint64(s.Link) >= uint64(len(f.Sections)) {
		return fmt.Errorf("string table link invalid")
	}
	names := f.Sections[s.Link]
	if names.Type != elf.SHT_STRTAB || names.Flags&elf.SHF_COMPRESSED != 0 || strings.HasPrefix(names.Name, ".zdebug") || names.Size > 8<<20 {
		return fmt.Errorf("invalid or excessive symbol string table")
	}
	data, err := s.Data()
	if err != nil {
		return fmt.Errorf("read symbols: %w", err)
	}
	str, err := names.Data()
	if err != nil {
		return fmt.Errorf("read names: %w", err)
	}
	if !bytes.Equal(data[:entry], make([]byte, entry)) {
		return fmt.Errorf("nonzero null symbol")
	}
	if len(str) == 0 || str[0] != 0 || str[len(str)-1] != 0 {
		return fmt.Errorf("invalid string table sentinels")
	}
	var total uint64
	for i := uint64(0); i < uint64(len(data)); i += entry {
		off := uint64(f.ByteOrder.Uint32(data[i : i+4]))
		if off >= uint64(len(str)) {
			return fmt.Errorf("symbol name offset invalid")
		}
		n := bytes.IndexByte(str[off:], 0)
		if n < 0 || n > 4096 {
			return fmt.Errorf("invalid or excessive symbol name")
		}
		total += uint64(n)
		if total > 8<<20 {
			return fmt.Errorf("excessive decoded names")
		}
	}
	return nil
}
