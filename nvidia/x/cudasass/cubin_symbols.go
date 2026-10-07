package cudasass

import (
	"crypto/sha256"
	"debug/elf"
	"fmt"
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
	if table.Entsize != entry || table.Size%entry != 0 || table.Size/entry > 65536 || table.Size > 8<<20 || table.Flags&elf.SHF_COMPRESSED != 0 {
		return nil, fmt.Errorf("unsupported or excessive symbol table")
	}
	if uint64(table.Link) >= uint64(len(im.file.Sections)) {
		return nil, fmt.Errorf("invalid symbol string table link")
	}
	strings := im.file.Sections[table.Link]
	if strings.Type != elf.SHT_STRTAB || strings.Flags&elf.SHF_COMPRESSED != 0 || strings.Size > 8<<20-table.Size {
		return nil, fmt.Errorf("unsupported or excessive symbol string table")
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
