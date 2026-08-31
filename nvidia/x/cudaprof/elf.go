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

	// String tables
	shstrtab := newStringTable()
	shstrtab.Index("")
	sText := shstrtab.Index(".text")
	sSymtab := shstrtab.Index(".symtab")
	sStrtab := shstrtab.Index(".strtab")
	sShstrtab := shstrtab.Index(".shstrtab")

	strtab := newStringTable()
	strtab.Index("")

	// Compute base address and text size
	baseAddr := symbols[0].Address
	var totalCode bytes.Buffer

	for _, sym := range symbols {
		if sym.Address < baseAddr {
			baseAddr = sym.Address
		}
	}
	baseAddr = baseAddr &^ 0xFFF // 4KB page align

	type symEntry struct {
		nameIndex uint32
		addr      uint64
		size      uint64
	}
	var symEntries []symEntry

	for _, sym := range symbols {
		nameIdx := uint32(strtab.Index(sym.Name))
		symEntries = append(symEntries, symEntry{
			nameIndex: nameIdx,
			addr:      sym.Address,
			size:      sym.Size,
		})

		// Pad up to sym.Address
		offset := sym.Address - baseAddr
		for uint64(totalCode.Len()) < offset {
			totalCode.WriteByte(0x90) // NOP padding
		}
		if len(sym.Code) > 0 {
			totalCode.Write(sym.Code)
		} else {
			// Default NOP instructions if no raw bytes
			for i := uint64(0); i < sym.Size; i++ {
				totalCode.WriteByte(0x90)
			}
		}
	}
	// Pad total code to 16 bytes
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
		Shstrndx:  4, // .shstrtab is section 4
	}

	// We'll place sections after section header table:
	// Section 0: NULL (64 bytes)
	// Section 1: .text
	// Section 2: .symtab
	// Section 3: .strtab
	// Section 4: .shstrtab
	shOffset := uint64(64)
	shTableSize := uint64(hdr.Shnum) * 64
	dataOffset := shOffset + shTableSize

	// 1. .text section
	textOffset := dataOffset
	textSize := uint64(totalCode.Len())

	// 2. .symtab section (16 bytes per symbol entry in ELF64)
	symtabOffset := textOffset + textSize
	numSymbols := 1 + len(symEntries) // +1 for null symbol
	symtabSize := uint64(numSymbols * 24)

	// 3. .strtab section
	strtabOffset := symtabOffset + symtabSize
	var strtabBuf bytes.Buffer
	for _, s := range strtab.strings {
		strtabBuf.WriteString(s)
		strtabBuf.WriteByte(0)
	}
	strtabSize := uint64(strtabBuf.Len())

	// 4. .shstrtab section
	shstrtabOffset := strtabOffset + strtabSize
	var shstrtabBuf bytes.Buffer
	for _, s := range shstrtab.strings {
		shstrtabBuf.WriteString(s)
		shstrtabBuf.WriteByte(0)
	}
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
		Name:      uint32(sText),
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
		Name:      uint32(sSymtab),
		Type:      uint32(elf.SHT_SYMTAB),
		Flags:     0,
		Addr:      0,
		Off:       symtabOffset,
		Size:      symtabSize,
		Link:      3, // Link to .strtab
		Info:      1, // Number of local symbols
		Addralign: 8,
		Entsize:   24,
	}
	_ = binary.Write(buf, binary.LittleEndian, &symtabSec)

	// 3: .strtab
	strtabSec := elf.Section64{
		Name:      uint32(sStrtab),
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
		Name:      uint32(sShstrtab),
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

	// Write Section Data
	// .text
	buf.Write(totalCode.Bytes())

	// .symtab (Elf64_Sym: uint32 name, uint8 info, uint8 other, uint16 shndx, uint64 value, uint64 size)
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

	for _, se := range symEntries {
		symRec := elf64Sym{
			Name:  se.nameIndex,
			Info:  (uint8(elf.STB_GLOBAL) << 4) | (uint8(elf.STT_FUNC) & 0xf),
			Other: uint8(elf.STV_DEFAULT),
			Shndx: 1, // .text section index
			Value: se.addr,
			Size:  se.size,
		}
		_ = binary.Write(buf, binary.LittleEndian, &symRec)
	}

	// .strtab
	buf.Write(strtabBuf.Bytes())

	// .shstrtab
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
