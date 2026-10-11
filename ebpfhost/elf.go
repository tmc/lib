package ebpfhost

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
)

// ResolveSymbol returns the executable file offset of an ELF symbol.
func ResolveSymbol(path, name string) (uint64, error) {
	image, err := ResolveImage(path, name)
	return image.Offset, err
}

// Image records the exact file used to resolve a probe.
type Image struct {
	Path, BuildID, SHA256 string
	Size                  int64
	Offset                uint64
}

// ResolveImage resolves a regular function and records the image identity.
func ResolveImage(path, symbol string) (Image, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Image{}, err
	}
	f, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		return Image{}, err
	}
	defer f.Close()
	var hits []elf.Symbol
	for _, read := range []func() ([]elf.Symbol, error){f.DynamicSymbols, f.Symbols} {
		ss, e := read()
		if e == nil {
			for _, s := range ss {
				if s.Name == symbol && s.Section != elf.SHN_UNDEF {
					hits = append(hits, s)
				}
			}
		}
	}
	var offset uint64
	found := false
	for _, s := range hits {
		if elf.ST_TYPE(s.Info) != elf.STT_FUNC {
			return Image{}, fmt.Errorf("symbol %q is not a regular function", symbol)
		}
		for _, p := range f.Progs {
			if p.Type == elf.PT_LOAD && p.Flags&elf.PF_X != 0 && s.Value >= p.Vaddr && s.Value-p.Vaddr < p.Filesz {
				v := p.Off + s.Value - p.Vaddr
				if found && v != offset {
					return Image{}, errors.New("ambiguous ELF symbol")
				}
				offset = v
				found = true
			}
		}
	}
	if !found {
		return Image{}, fmt.Errorf("no executable symbol %q", symbol)
	}
	sum := sha256.Sum256(data)
	result := Image{Path: path, SHA256: hex.EncodeToString(sum[:]), Size: int64(len(data)), Offset: offset}
	if sec := f.Section(".note.gnu.build-id"); sec != nil {
		note, e := sec.Data()
		if e != nil {
			return Image{}, e
		}
		for len(note) >= 12 {
			ns := uint64(f.ByteOrder.Uint32(note))
			ds := uint64(f.ByteOrder.Uint32(note[4:]))
			kind := f.ByteOrder.Uint32(note[8:])
			na := (ns + 3) &^ 3
			da := (ds + 3) &^ 3
			if 12+na+da > uint64(len(note)) {
				return Image{}, errors.New("truncated build ID note")
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
