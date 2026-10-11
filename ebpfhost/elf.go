package ebpfhost

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
)

// ResolveSymbol returns the executable file offset of an ELF symbol.
func ResolveSymbol(path, name string) (uint64, error) {
	image, err := ResolveImage(path, name)
	return image.Offset, err
}

// Image records the exact file used to resolve a probe.
type Image struct {
	Path, BuildID, SHA256 string
	Symbol, SymbolVersion string
	Size                  int64
	Offset                uint64
}

// ResolveImage resolves a regular function and records the image identity.
func ResolveImage(path, symbol string) (Image, error) {
	return ResolveImageVersion(path, symbol, "")
}

// ResolveImageVersion resolves an exact GNU ELF symbol version. An empty version
// selects the default dynamic version, or an unversioned symbol. SymbolVersion
// in the returned Image is read from the file, not copied from the request.
// PLT stubs and indirect functions are refused rather than followed.
func ResolveImageVersion(path, symbol, version string) (Image, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Image{}, fmt.Errorf("resolve symbol %q: %w", symbol, err)
	}
	f, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		return Image{}, fmt.Errorf("resolve symbol %q: %w", symbol, err)
	}
	defer f.Close()
	hits, err := imageSymbols(f, symbol, version)
	if err != nil {
		return Image{}, fmt.Errorf("resolve symbol %q: %w", symbol, err)
	}
	var offset uint64
	var selectedVersion string
	found := false
	for _, s := range hits {
		if elf.ST_TYPE(s.Info) != elf.STT_FUNC {
			return Image{}, fmt.Errorf("symbol %q is not a regular function", symbol)
		}
		if int(s.Section) >= len(f.Sections) {
			return Image{}, fmt.Errorf("symbol %q has no ordinary section", symbol)
		}
		section := f.Sections[s.Section]
		if section.Type == elf.SHT_NOBITS || section.Flags&elf.SHF_EXECINSTR == 0 || strings.HasPrefix(section.Name, ".plt") || section.Name == ".iplt" {
			return Image{}, fmt.Errorf("symbol %q is not ordinary executable code", symbol)
		}
		if s.Value < section.Addr || s.Value-section.Addr >= section.Size {
			return Image{}, fmt.Errorf("symbol %q lies outside its section", symbol)
		}
		for _, p := range f.Progs {
			if p.Type == elf.PT_LOAD && p.Flags&elf.PF_X != 0 && s.Value >= p.Vaddr && s.Value-p.Vaddr < p.Filesz {
				rel := s.Value - p.Vaddr
				if p.Off >= uint64(len(data)) || rel >= uint64(len(data))-p.Off {
					return Image{}, fmt.Errorf("symbol %q lies outside the file", symbol)
				}
				v := p.Off + rel
				if found && (v != offset || s.Version != selectedVersion) {
					return Image{}, fmt.Errorf("ambiguous ELF symbol %q", symbol)
				}
				offset = v
				selectedVersion = s.Version
				found = true
			}
		}
	}
	if !found {
		return Image{}, fmt.Errorf("no executable symbol %q with version %q", symbol, version)
	}
	sum := sha256.Sum256(data)
	result := Image{Path: path, SHA256: hex.EncodeToString(sum[:]), Size: int64(len(data)), Offset: offset, Symbol: symbol, SymbolVersion: selectedVersion}
	if sec := f.Section(".note.gnu.build-id"); sec != nil {
		note, e := sec.Data()
		if e != nil {
			return Image{}, fmt.Errorf("resolve symbol %q build ID: %w", symbol, e)
		}
		for len(note) >= 12 {
			ns := uint64(f.ByteOrder.Uint32(note))
			ds := uint64(f.ByteOrder.Uint32(note[4:]))
			kind := f.ByteOrder.Uint32(note[8:])
			na := (ns + 3) &^ 3
			da := (ds + 3) &^ 3
			if 12+na+da > uint64(len(note)) {
				return Image{}, fmt.Errorf("resolve symbol %q: truncated build ID note", symbol)
			}
			if kind == 3 && ns == 4 && string(note[12:16]) == "GNU\x00" {
				result.BuildID = hex.EncodeToString(note[12+na : 12+na+ds])
				break
			}
			note = note[12+na+da:]
		}
	}
	return result, nil
}

func imageSymbols(f *elf.File, name, version string) ([]elf.Symbol, error) {
	if err := checkVersionMetadata(f); err != nil {
		return nil, err
	}
	symbols, err := f.DynamicSymbols()
	if err != nil && !errors.Is(err, elf.ErrNoSymbols) {
		return nil, err
	}
	versions := f.SectionByType(elf.SHT_GNU_VERSYM)
	if versions != nil {
		data, err := versions.Data()
		if err != nil {
			return nil, err
		}
		if len(data)/2 != len(symbols)+1 || len(data)%2 != 0 {
			return nil, errors.New("ELF version table does not cover the dynamic symbols")
		}
	} else if f.SectionByType(elf.SHT_GNU_VERDEF) != nil || f.SectionByType(elf.SHT_GNU_VERNEED) != nil {
		return nil, errors.New("ELF version definitions lack a symbol version table")
	}
	var hits []elf.Symbol
	dynamicName := false
	for _, s := range symbols {
		if s.Name != name || s.Section == elf.SHN_UNDEF {
			continue
		}
		dynamicName = true
		if versions != nil && (!s.HasVersion || s.VersionIndex.Index() > 1 && s.Version == "") {
			return nil, errors.New("ELF symbol version is unresolved")
		}
		if version != "" {
			if !s.HasVersion || s.Version != version {
				continue
			}
		} else if s.HasVersion && (s.VersionIndex.IsHidden() || s.VersionIndex.Index() == 0) {
			continue
		}
		hits = append(hits, s)
	}
	// A static alias must not bypass dynamic version selection.
	if dynamicName || version != "" {
		return hits, nil
	}
	symbols, err = f.Symbols()
	if err != nil && !errors.Is(err, elf.ErrNoSymbols) {
		return nil, err
	}
	for _, s := range symbols {
		if s.Name == name && s.Section != elf.SHN_UNDEF {
			hits = append(hits, s)
		}
	}
	return hits, nil
}

// checkVersionMetadata validates the links and record counts that debug/elf
// does not use when looking up GNU version names.
func checkVersionMetadata(f *elf.File) error {
	var dynamic, versions *elf.Section
	var definitions []*elf.Section
	for _, s := range f.Sections {
		switch s.Type {
		case elf.SHT_DYNSYM:
			if dynamic != nil {
				return errors.New("multiple ELF dynamic symbol tables")
			}
			dynamic = s
		case elf.SHT_GNU_VERSYM:
			if versions != nil {
				return errors.New("multiple ELF symbol version tables")
			}
			versions = s
		case elf.SHT_GNU_VERDEF, elf.SHT_GNU_VERNEED:
			for _, old := range definitions {
				if old.Type == s.Type {
					return errors.New("multiple ELF version definition tables")
				}
			}
			definitions = append(definitions, s)
		}
	}
	if versions == nil && len(definitions) == 0 {
		return nil
	}
	if dynamic == nil || versions == nil || uint64(dynamic.Link) >= uint64(len(f.Sections)) || f.Sections[dynamic.Link].Type != elf.SHT_STRTAB {
		return errors.New("ELF version tables lack a dynamic symbol string table")
	}
	if uint64(versions.Link) >= uint64(len(f.Sections)) || f.Sections[versions.Link] != dynamic || versions.Entsize != 2 {
		return errors.New("invalid ELF symbol version table link or entry size")
	}
	strings, err := f.Sections[dynamic.Link].Data()
	if err != nil {
		return err
	}
	nameOK := func(off uint32) bool {
		return uint64(off) < uint64(len(strings)) && bytes.IndexByte(strings[off:], 0) > 0
	}
	indices := make(map[uint16]bool)
	for _, s := range definitions {
		if s.Link != dynamic.Link {
			return errors.New("invalid ELF version definition string table link")
		}
		data, err := s.Data()
		if err != nil {
			return err
		}
		need := s.Type == elf.SHT_GNU_VERNEED
		header, auxSize := uint64(20), uint64(8)
		if need {
			header, auxSize = 16, 16
		}
		off := uint64(0)
		for record := uint32(0); ; record++ {
			if record >= s.Info || off > uint64(len(data)) || header > uint64(len(data))-off {
				return errors.New("ELF version definition count conflicts with records")
			}
			b := data[off:]
			if f.ByteOrder.Uint16(b) != 1 {
				return errors.New("unsupported ELF version definition format")
			}
			var count uint16
			var aux, next uint64
			if need {
				count = f.ByteOrder.Uint16(b[2:])
				aux = uint64(f.ByteOrder.Uint32(b[8:]))
				next = uint64(f.ByteOrder.Uint32(b[12:]))
				if !nameOK(f.ByteOrder.Uint32(b[4:])) {
					return errors.New("invalid ELF version dependency name")
				}
			} else {
				count = f.ByteOrder.Uint16(b[6:])
				aux = uint64(f.ByteOrder.Uint32(b[12:]))
				next = uint64(f.ByteOrder.Uint32(b[16:]))
				index := f.ByteOrder.Uint16(b[4:])
				if index == 0 || index >= 0x8000 || indices[index] {
					return errors.New("duplicate or invalid ELF version index")
				}
				indices[index] = true
			}
			limit := uint64(len(data)) - off
			if next != 0 {
				if next < header || next > limit {
					return errors.New("invalid ELF version definition chain")
				}
				limit = next
			}
			if count == 0 || aux < header {
				return errors.New("invalid ELF version auxiliary count or offset")
			}
			for i := uint16(0); i < count; i++ {
				if aux > limit || auxSize > limit-aux {
					return errors.New("ELF version auxiliary records exceed definition")
				}
				a := b[aux:]
				name, step := f.ByteOrder.Uint32(a), uint64(f.ByteOrder.Uint32(a[4:]))
				if need {
					index := f.ByteOrder.Uint16(a[6:]) & 0x7fff
					if index < 2 || indices[index] {
						return errors.New("duplicate or invalid ELF version dependency index")
					}
					indices[index] = true
					name, step = f.ByteOrder.Uint32(a[8:]), uint64(f.ByteOrder.Uint32(a[12:]))
				}
				if !nameOK(name) || (i == count-1) != (step == 0) || step != 0 && (step < auxSize || step > limit-aux) {
					return errors.New("ELF version auxiliary count or name conflicts with records")
				}
				aux += step
			}
			if next == 0 {
				if record+1 != s.Info {
					return errors.New("ELF version definition count conflicts with records")
				}
				break
			}
			off += next
		}
	}
	return nil
}
