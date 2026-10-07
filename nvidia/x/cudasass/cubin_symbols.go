package cudasass

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"fmt"
	"strings"
	"unicode/utf8"
)

// CubinSymbol preserves raw ELF symbol metadata. Value and Size are compiler
// metadata, not runtime reservation capacity or readable payload evidence.
type CubinSymbol struct {
	Name        string
	Type        elf.SymType
	Binding     elf.SymBind
	Section     elf.SectionIndex
	SectionName string
	Value, Size uint64
}

// SHA256 identifies the immutable bytes copied by OpenCubin.
func (im *CubinImage) SHA256() [32]byte { return sha256.Sum256(im.data) }

// Symbols returns ordinary symbol-table entries with exactly the given name,
// in table order, including duplicates and zero values. An empty result means
// no match; a missing table returns an error wrapping elf.ErrNoSymbols.
// Total decoded symbol names are limited to 8 MiB, including nonmatches.
// Reserved section indices remain raw and have no SectionName. This method
// reads no symbol payload and proves neither loading nor runtime semantics.
func (im *CubinImage) Symbols(name string) ([]CubinSymbol, error) {
	if name == "" || len(name) > 4096 || !utf8.ValidString(name) {
		return nil, fmt.Errorf("invalid symbol name")
	}
	if im == nil || im.file == nil {
		return nil, fmt.Errorf("nil cubin image")
	}
	var table *elf.Section
	for _, s := range im.file.Sections {
		if s.Type == elf.SHT_SYMTAB {
			if table != nil {
				return nil, fmt.Errorf("multiple symbol tables")
			}
			table = s
		}
	}
	if table == nil {
		return nil, fmt.Errorf("cubin symbols: %w", elf.ErrNoSymbols)
	}
	entry := uint64(24)
	if im.file.Class == elf.ELFCLASS32 {
		entry = 16
	}
	if table.Size < entry || table.Entsize != entry || table.Size%entry != 0 || table.Size/entry > 65536 || table.Size > 8<<20 || table.Flags&elf.SHF_COMPRESSED != 0 || strings.HasPrefix(table.Name, ".zdebug") {
		return nil, fmt.Errorf("unsupported or excessive symbol table")
	}
	if uint64(table.Link) >= uint64(len(im.file.Sections)) {
		return nil, fmt.Errorf("invalid symbol string table link")
	}
	strtab := im.file.Sections[table.Link]
	if strtab.Type != elf.SHT_STRTAB || strtab.Flags&elf.SHF_COMPRESSED != 0 || strings.HasPrefix(strtab.Name, ".zdebug") || strtab.Size > 8<<20-table.Size {
		return nil, fmt.Errorf("unsupported or excessive symbol string table")
	}
	for _, section := range []*elf.Section{table, strtab} {
		size := uint64(len(im.data))
		if section.Offset > size || section.Size > size-section.Offset || section.FileSize != section.Size {
			return nil, fmt.Errorf("invalid symbol table file extent")
		}
	}
	// Validate every st_name before the standard reader, which silently drops
	// invalid string offsets. Interpretation still belongs to debug/elf.
	tableData, err := table.Data()
	if err != nil {
		return nil, fmt.Errorf("reading symbol table: %w", err)
	}
	stringData, err := strtab.Data()
	if err != nil {
		return nil, fmt.Errorf("reading symbol strings: %w", err)
	}
	if !bytes.Equal(tableData[:entry], make([]byte, entry)) {
		return nil, fmt.Errorf("nonzero undefined symbol entry")
	}
	var decodedNames uint64
	if len(stringData) != 0 && (stringData[0] != 0 || stringData[len(stringData)-1] != 0) {
		return nil, fmt.Errorf("invalid symbol string table sentinels")
	}
	for i := uint64(0); i < uint64(len(tableData)); i += entry {
		offset := uint64(im.file.ByteOrder.Uint32(tableData[i : i+4]))
		if len(stringData) == 0 && offset == 0 {
			continue
		}
		if offset >= uint64(len(stringData)) {
			return nil, fmt.Errorf("invalid symbol name offset")
		}
		end := bytes.IndexByte(stringData[offset:], 0)
		if end >= 0 {
			decodedNames += uint64(end)
			if decodedNames > 8<<20 {
				return nil, fmt.Errorf("excessive aggregate decoded symbol names")
			}
		}
		if end < 0 || end > 4096 {
			return nil, fmt.Errorf("unterminated or excessive symbol name")
		}
	}
	symbols, err := im.file.Symbols()
	if err != nil {
		return nil, fmt.Errorf("cubin symbols: %w", err)
	}
	var out []CubinSymbol
	for _, s := range symbols {
		if len(s.Name) > 4096 {
			return nil, fmt.Errorf("excessive symbol name")
		}
		if s.Section == elf.SHN_XINDEX {
			return nil, fmt.Errorf("unsupported extended symbol section")
		}
		section := ""
		if s.Section != elf.SHN_UNDEF && s.Section < elf.SHN_LORESERVE {
			if int(s.Section) >= len(im.file.Sections) {
				return nil, fmt.Errorf("invalid symbol section")
			}
			section = im.file.Sections[s.Section].Name
			if len(section) > 4096 {
				return nil, fmt.Errorf("excessive section name")
			}
		}
		if s.Name != name {
			continue
		}
		if len(out) == 1024 {
			return nil, fmt.Errorf("excessive matching symbols")
		}
		out = append(out, CubinSymbol{s.Name, elf.ST_TYPE(s.Info), elf.ST_BIND(s.Info), s.Section, section, s.Value, s.Size})
	}
	return out, nil
}
