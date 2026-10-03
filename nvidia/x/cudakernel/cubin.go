package cudakernel

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
)

// CUBINInfo contains metadata parsed from an NVIDIA CUBIN ELF binary.
type CUBINInfo struct {
	Arch               string            // Base SM (e.g. "sm_121"), without a/f suffix; empty for unknown ABI.
	Symbols            []string          // Kernel entry point symbols
	AllocatedRegisters map[string]uint32 // Symbol -> register count
	StaticSharedMemory map[string]uint32 // Symbol -> static shared memory in bytes
}

// Common NVIDIA ELF attribute tags in .nv.info sections.
const (
	eiattrMaxRegisters uint16 = 0x0204
	eiattrRegval       uint16 = 0x020a
	eiattrMinStack     uint16 = 0x0804
	eiattrFrameSize    uint16 = 0x0808
)

// ParseCUBIN parses and validates an NVIDIA CUBIN ELF image, extracting
// entry point symbols and hardware resource allocations from .nv.info
// and .nv.shared sections. It explicitly rejects malformed ELF inputs.
func ParseCUBIN(cubin []byte) (*CUBINInfo, error) {
	if len(cubin) < 16 {
		return nil, errors.New("cudakernel: cubin ELF too short")
	}
	if !bytes.HasPrefix(cubin, []byte("\x7fELF")) {
		return nil, errors.New("cudakernel: invalid ELF magic header")
	}

	// Legacy NVIDIA ELF uses e_version for the toolkit version (for example,
	// nvcc 12.6 emits 126), unlike the ELF version in e_ident. debug/elf
	// requires both to be EV_CURRENT. Normalize only that legacy CUDA header
	// in a private copy; code bytes and the caller's artifact stay unchanged.
	elfData := cubin
	if len(cubin) >= 24 && cubin[elf.EI_DATA] == byte(elf.ELFDATA2LSB) &&
		cubin[elf.EI_OSABI] == 0x33 && cubin[elf.EI_ABIVERSION] == 7 &&
		binary.LittleEndian.Uint16(cubin[18:20]) == uint16(elf.EM_CUDA) {
		version := binary.LittleEndian.Uint32(cubin[20:24])
		if version > uint32(elf.EV_CURRENT) {
			elfData = bytes.Clone(cubin)
			binary.LittleEndian.PutUint32(elfData[20:24], uint32(elf.EV_CURRENT))
		}
	}
	reader := bytes.NewReader(elfData)
	file, err := elf.NewFile(reader)
	if err != nil {
		return nil, fmt.Errorf("cudakernel: parse ELF: %w", err)
	}
	defer file.Close()
	if file.Machine != elf.EM_CUDA {
		return nil, errors.New("cudakernel: ELF is not a CUDA image")
	}

	info := &CUBINInfo{
		AllocatedRegisters: make(map[string]uint32),
		StaticSharedMemory: make(map[string]uint32),
	}

	// CUDA ABI 7 stores the SM in the low byte; ABI 8 stores it in
	// bits 8–15. Unknown ABI versions carry no architecture evidence.
	// See LLVM's CUDA ELF ABI update (llvm-project pull request 159451).
	var flags uint32
	switch file.Class {
	case elf.ELFCLASS64:
		flags = file.ByteOrder.Uint32(cubin[48:52])
	case elf.ELFCLASS32:
		flags = file.ByteOrder.Uint32(cubin[36:40])
	}
	var sm uint32
	if file.Machine == elf.EM_CUDA {
		switch cubin[8] {
		case 7:
			sm = flags & 0xff
		case 8:
			sm = (flags >> 8) & 0xff
		}
	}
	if sm != 0 {
		info.Arch = fmt.Sprintf("sm_%d", sm)
	}

	// Read symbols from ELF symbol table.
	syms, err := file.Symbols()
	if err == nil {
		for _, sym := range syms {
			if sym.Section < elf.SectionIndex(len(file.Sections)) {
				sec := file.Sections[sym.Section]
				if strings.HasPrefix(sec.Name, ".text.") && sym.Name != "" {
					info.Symbols = append(info.Symbols, sym.Name)
				}
			}
		}
	}

	// Inspect sections for text, shared memory, and .nv.info attributes.
	for _, sec := range file.Sections {
		if strings.HasPrefix(sec.Name, ".text.") {
			sym := strings.TrimPrefix(sec.Name, ".text.")
			if !containsString(info.Symbols, sym) {
				info.Symbols = append(info.Symbols, sym)
			}
		}

		if strings.HasPrefix(sec.Name, ".nv.shared.") {
			sym := strings.TrimPrefix(sec.Name, ".nv.shared.")
			info.StaticSharedMemory[sym] = uint32(sec.Size)
		}

		if strings.HasPrefix(sec.Name, ".nv.info") {
			sym := ""
			if strings.HasPrefix(sec.Name, ".nv.info.") {
				sym = strings.TrimPrefix(sec.Name, ".nv.info.")
			}
			secData, readErr := sec.Data()
			if readErr == nil && len(secData) > 0 {
				regs := parseNVInfoRegisters(secData)
				if regs > 0 {
					if sym != "" {
						info.AllocatedRegisters[sym] = regs
					} else {
						// Global .nv.info
						for _, s := range info.Symbols {
							if _, ok := info.AllocatedRegisters[s]; !ok {
								info.AllocatedRegisters[s] = regs
							}
						}
					}
				}
			}
		}
	}

	return info, nil
}

func parseNVInfoRegisters(data []byte) uint32 {
	r := bytes.NewReader(data)
	for r.Len() >= 4 {
		var tag uint16
		var fmtByte uint8
		var flagByte uint8

		if err := binary.Read(r, binary.LittleEndian, &tag); err != nil {
			break
		}
		if err := binary.Read(r, binary.LittleEndian, &fmtByte); err != nil {
			break
		}
		if err := binary.Read(r, binary.LittleEndian, &flagByte); err != nil {
			break
		}

		valSize := 4
		if fmtByte == 0x08 {
			valSize = 8
		} else if fmtByte == 0x02 {
			valSize = 2
		} else if fmtByte == 0x01 {
			valSize = 1
		}

		if r.Len() < valSize {
			break
		}

		var val uint64
		switch valSize {
		case 1:
			var v uint8
			_ = binary.Read(r, binary.LittleEndian, &v)
			val = uint64(v)
		case 2:
			var v uint16
			_ = binary.Read(r, binary.LittleEndian, &v)
			val = uint64(v)
		case 4:
			var v uint32
			_ = binary.Read(r, binary.LittleEndian, &v)
			val = uint64(v)
		case 8:
			_ = binary.Read(r, binary.LittleEndian, &val)
		default:
			return 0
		}

		if tag == eiattrMaxRegisters || tag == eiattrRegval {
			return uint32(val)
		}
	}
	return 0
}

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}
