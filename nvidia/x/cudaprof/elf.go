package cudaprof

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"os"
)

// Symbol represents an exported function symbol for ELF binary synthesis.
type Symbol struct {
	Name    string
	Address uint64
	Size    uint64
	Code    []byte
}

// SynthesizeELFBinary creates a valid ELF64 shared object binary containing symbol tables
// and instruction bytes for GPU kernels, allowing standard go tool pprof disassemblers
// to resolve and display kernel instruction disassembly.
func SynthesizeELFBinary(symbols []Symbol) ([]byte, error) {
	if len(symbols) == 0 {
		return nil, fmt.Errorf("no symbols provided")
	}

	buf := new(bytes.Buffer)

	// Build .shstrtab (section name string table)
	var shstrtabBuf bytes.Buffer
	shstrtabBuf.WriteByte(0) // 0-th byte is null string

	sTextOffset := uint32(shstrtabBuf.Len())
	shstrtabBuf.WriteString(".text\x00")

	sSymtabOffset := uint32(shstrtabBuf.Len())
	shstrtabBuf.WriteString(".symtab\x00")

	sStrtabOffset := uint32(shstrtabBuf.Len())
	shstrtabBuf.WriteString(".strtab\x00")

	sShstrtabOffset := uint32(shstrtabBuf.Len())
	shstrtabBuf.WriteString(".shstrtab\x00")

	// Build .strtab (symbol name string table)
	var strtabBuf bytes.Buffer
	strtabBuf.WriteByte(0) // 0-th byte is null string

	// Base virtual address
	baseAddr := symbols[0].Address
	for _, sym := range symbols {
		if sym.Address < baseAddr {
			baseAddr = sym.Address
		}
	}
	baseAddr = baseAddr &^ 0xFFF // 4KB page align

	type symEntry struct {
		nameOffset uint32
		addr       uint64
		size       uint64
	}
	var symEntries []symEntry

	var totalCode bytes.Buffer
	for _, sym := range symbols {
		nameOff := uint32(strtabBuf.Len())
		strtabBuf.WriteString(sym.Name)
		strtabBuf.WriteByte(0)

		symEntries = append(symEntries, symEntry{
			nameOffset: nameOff,
			addr:       sym.Address,
			size:       sym.Size,
		})

		// Pad up to sym.Address
		offset := sym.Address - baseAddr
		for uint64(totalCode.Len()) < offset {
			totalCode.WriteByte(0x90) // NOP padding
		}
		if len(sym.Code) > 0 {
			totalCode.Write(sym.Code)
		} else {
			for i := uint64(0); i < sym.Size; i++ {
				totalCode.WriteByte(0x90)
			}
		}
	}
	for totalCode.Len()%16 != 0 {
		totalCode.WriteByte(0x90)
	}

	// ELF Header (64 bytes)
	hdr := elf.Header64{
		Ident: [16]byte{
			0x7f, 'E', 'L', 'F',
			2, // 64-bit
			1, // Little endian
			1, // Current version
			0, // System V ABI
		},
		Type:      uint16(elf.ET_DYN),
		Machine:   uint16(elf.EM_X86_64),
		Version:   1,
		Entry:     baseAddr,
		Phoff:     0,
		Shoff:     64, // Section headers follow ELF header immediately
		Flags:     0,
		Ehsize:    64,
		Phentsize: 0,
		Phnum:     0,
		Shentsize: 64,
		Shnum:     5, // NULL, .text, .symtab, .strtab, .shstrtab
		Shstrndx:  4, // .shstrtab is section index 4
	}

	// Offsets
	shOffset := uint64(64)
	shTableSize := uint64(hdr.Shnum) * 64
	dataOffset := shOffset + shTableSize

	// 1. .text section
	textOffset := dataOffset
	textSize := uint64(totalCode.Len())

	// 2. .symtab section (24 bytes per symbol in ELF64)
	symtabOffset := textOffset + textSize
	numSymbols := 1 + len(symEntries) // +1 for null symbol
	symtabSize := uint64(numSymbols * 24)

	// 3. .strtab section
	strtabOffset := symtabOffset + symtabSize
	strtabSize := uint64(strtabBuf.Len())

	// 4. .shstrtab section
	shstrtabOffset := strtabOffset + strtabSize
	shstrtabSize := uint64(shstrtabBuf.Len())

	// Write ELF Header
	if err := binary.Write(buf, binary.LittleEndian, &hdr); err != nil {
		return nil, err
	}

	// Section Headers Table
	// 0: NULL
	var nullSec elf.Section64
	_ = binary.Write(buf, binary.LittleEndian, &nullSec)

	// 1: .text
	textSec := elf.Section64{
		Name:      sTextOffset,
		Type:      uint32(elf.SHT_PROGBITS),
		Flags:     uint64(elf.SHF_ALLOC | elf.SHF_EXECINSTR),
		Addr:      baseAddr,
		Off:       textOffset,
		Size:      textSize,
		Link:      0,
		Info:      0,
		Addralign: 16,
		Entsize:   0,
	}
	_ = binary.Write(buf, binary.LittleEndian, &textSec)

	// 2: .symtab
	symtabSec := elf.Section64{
		Name:      sSymtabOffset,
		Type:      uint32(elf.SHT_SYMTAB),
		Flags:     0,
		Addr:      0,
		Off:       symtabOffset,
		Size:      symtabSize,
		Link:      3, // Link to .strtab (section 3)
		Info:      1, // 1 local symbol (the null symbol)
		Addralign: 8,
		Entsize:   24,
	}
	_ = binary.Write(buf, binary.LittleEndian, &symtabSec)

	// 3: .strtab
	strtabSec := elf.Section64{
		Name:      sStrtabOffset,
		Type:      uint32(elf.SHT_STRTAB),
		Flags:     0,
		Addr:      0,
		Off:       strtabOffset,
		Size:      strtabSize,
		Link:      0,
		Info:      0,
		Addralign: 1,
		Entsize:   0,
	}
	_ = binary.Write(buf, binary.LittleEndian, &strtabSec)

	// 4: .shstrtab
	shstrtabSec := elf.Section64{
		Name:      sShstrtabOffset,
		Type:      uint32(elf.SHT_STRTAB),
		Flags:     0,
		Addr:      0,
		Off:       shstrtabOffset,
		Size:      shstrtabSize,
		Link:      0,
		Info:      0,
		Addralign: 1,
		Entsize:   0,
	}
	_ = binary.Write(buf, binary.LittleEndian, &shstrtabSec)

	// Write Section Contents
	buf.Write(totalCode.Bytes())

	// Write .symtab
	type elf64Sym struct {
		Name  uint32
		Info  uint8
		Other uint8
		Shndx uint16
		Value uint64
		Size  uint64
	}

	// Null symbol
	_ = binary.Write(buf, binary.LittleEndian, &elf64Sym{})

	// Function symbols
	for _, se := range symEntries {
		symRec := elf64Sym{
			Name:  se.nameOffset,
			Info:  (uint8(elf.STB_GLOBAL) << 4) | (uint8(elf.STT_FUNC) & 0xf),
			Other: uint8(elf.STV_DEFAULT),
			Shndx: 1, // .text section index
			Value: se.addr,
			Size:  se.size,
		}
		_ = binary.Write(buf, binary.LittleEndian, &symRec)
	}

	// Write .strtab
	buf.Write(strtabBuf.Bytes())

	// Write .shstrtab
	buf.Write(shstrtabBuf.Bytes())

	return buf.Bytes(), nil
}

// WriteELFBinary writes the synthesized ELF sidecar binary to path.
func WriteELFBinary(path string, symbols []Symbol) error {
	elfBytes, err := SynthesizeELFBinary(symbols)
	if err != nil {
		return err
	}
	return os.WriteFile(path, elfBytes, 0755)
}
