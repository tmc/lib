package cudaoccupancy

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"io"
	"strings"
)

// KernelResource records static hardware resource allocations for a compiled CUDA kernel.
type KernelResource struct {
	Name      string // Kernel function name
	Registers int    // Physical 32-bit registers allocated per thread (from global .nv.info EIATTR_REGCOUNT)
	// StaticSharedMem is the physical static shared memory allocation charged per CTA in bytes.
	//
	// On sm_121a and modern architectures, the driver charges an extra 1024 bytes
	// (driver/hardware reservation) whenever a kernel declares shared memory:
	//
	//   Declared tile    ptxas -v          cuobjdump -res-usage / ELF .nv.shared size
	//   64 B             64 bytes smem     SHARED:1088
	//   4096 B           4096 bytes smem   SHARED:5120
	//   8192 B           8192 bytes smem   SHARED:9216
	//   none declared    (no line)         SHARED:0 (no .nv.shared section)
	//
	// StaticSharedMem records the actual ELF section size (e.g. 1088, 5120, 9216),
	// which is what cuOccupancyMaxActiveBlocksPerMultiprocessor charges against the SM.
	// (The same 1024 B arithmetic appears in MaxSharedMemoryOptin: 101376 B = 102400 - 1024 B,
	// though whether this shares the identical hardware reservation mechanism is unproven).
	StaticSharedMem  int
	StackFrameBytes  int // Local stack frame size per thread in bytes (EIATTR_FRAME_SIZE)
	ConstantBankSize int // Constant memory parameter bank 0 size in bytes (.nv.constant0 / EIATTR_PARAM_CBANK)
}

// InspectCubin parses a CUBIN ELF image and extracts resource usage for all kernels.
func InspectCubin(cubin []byte) ([]KernelResource, error) {
	return InspectCubinReader(bytes.NewReader(cubin), int64(len(cubin)))
}

// InspectCubinReader parses a CUBIN ELF image from an io.ReaderAt.
func InspectCubinReader(r io.ReaderAt, size int64) ([]KernelResource, error) {
	f, err := elf.NewFile(r)
	if err != nil {
		return nil, fmt.Errorf("cudaoccupancy: open cubin elf: %w", err)
	}
	defer f.Close()
	return ParseELF(f)
}

// ParseELF inspects an open CUBIN ELF file and extracts resource usage for all kernels.
func ParseELF(f *elf.File) ([]KernelResource, error) {
	syms, err := f.Symbols()
	if err != nil && err != elf.ErrNoSymbols {
		return nil, fmt.Errorf("cudaoccupancy: read symbols: %w", err)
	}

	kernelMap := make(map[string]*KernelResource)
	var kernelOrder []string

	registerKernel := func(name string) *KernelResource {
		if k, ok := kernelMap[name]; ok {
			return k
		}
		k := &KernelResource{Name: name}
		kernelMap[name] = k
		kernelOrder = append(kernelOrder, name)
		return k
	}

	// 1. Discover kernels from symbols and .text.<name> sections.
	for _, sym := range syms {
		if elf.ST_TYPE(sym.Info) == elf.STT_FUNC && sym.Section < elf.SectionIndex(len(f.Sections)) {
			sec := f.Sections[sym.Section]
			if strings.HasPrefix(sec.Name, ".text.") {
				registerKernel(sym.Name)
			}
		}
	}

	// Also check any .text.<name> section directly.
	for _, sec := range f.Sections {
		if strings.HasPrefix(sec.Name, ".text.") {
			name := strings.TrimPrefix(sec.Name, ".text.")
			if !strings.HasPrefix(name, "$__internal_") {
				registerKernel(name)
			}
		}
	}

	// 2. Read static shared memory (.nv.shared.<name>) and constant bank 0 (.nv.constant0.<name>).
	for _, sec := range f.Sections {
		if strings.HasPrefix(sec.Name, ".nv.shared.") {
			name := strings.TrimPrefix(sec.Name, ".nv.shared.")
			if k, ok := kernelMap[name]; ok {
				k.StaticSharedMem = int(sec.Size)
			}
		} else if strings.HasPrefix(sec.Name, ".nv.constant0.") {
			name := strings.TrimPrefix(sec.Name, ".nv.constant0.")
			if k, ok := kernelMap[name]; ok {
				k.ConstantBankSize = int(sec.Size)
			}
		}
	}

	// 3. Parse global .nv.info section.
	// In NVIDIA CUBIN ELF, .nv.info contains TLV records.
	// Format tag 0x04 has an 8-byte payload: [4-byte symbol index, 4-byte value].
	// Note: Go's debug/elf f.Symbols() omits STN_UNDEF (index 0), so ELF symbol index S maps to syms[S-1].
	nvinfoSec := f.Section(".nv.info")
	if nvinfoSec != nil {
		data, err := nvinfoSec.Data()
		if err == nil {
			parseNVInfoGlobal(data, syms, kernelMap)
		}
	}

	// 4. Parse per-kernel .nv.info.<name> sections for any kernel-specific attributes.
	for _, sec := range f.Sections {
		if strings.HasPrefix(sec.Name, ".nv.info.") {
			kname := strings.TrimPrefix(sec.Name, ".nv.info.")
			k, ok := kernelMap[kname]
			if !ok {
				continue
			}
			data, err := sec.Data()
			if err == nil {
				parseNVInfoPerKernel(data, k)
			}
		}
	}

	// Filter out internal compiler helper functions unless explicitly requested.
	var res []KernelResource
	for _, name := range kernelOrder {
		if strings.HasPrefix(name, "$__internal_") {
			continue
		}
		res = append(res, *kernelMap[name])
	}
	return res, nil
}

// ELF attribute IDs in NVIDIA .nv.info sections.
const (
	AttrParamCBank        = 0x0a // EIATTR_PARAM_CBANK (constant bank parameter offset & size)
	AttrFrameSize         = 0x11 // EIATTR_FRAME_SIZE (per-thread stack frame bytes)
	AttrMinStackSize      = 0x12 // EIATTR_MIN_STACK_SIZE
	AttrMaxRegCount       = 0x1b // EIATTR_MAXREG_COUNT (compiler register ceiling)
	AttrRegCount          = 0x2f // EIATTR_REGCOUNT / EIATTR_REG_COUNT (registers per thread)
	AttrSharedScratch     = 0x30 // EIATTR_SHARED_SCRATCH / EIATTR_SHARED_SIZE_SYSMEM
	AttrReservedSmem0Size = 0x3e // EIATTR_RESERVED_SMEM_0_SIZE
)

// Legacy and specification aliases for attribute IDs.
const (
	EIATTR_PARAM_CBANK        = AttrParamCBank
	EIATTR_FRAME_SIZE         = AttrFrameSize
	EIATTR_MIN_STACK_SIZE     = AttrMinStackSize
	EIATTR_MAXREG_COUNT       = AttrMaxRegCount
	EIATTR_REGCOUNT           = AttrRegCount
	EIATTR_REG_COUNT          = AttrRegCount
	EIATTR_SHARED_SIZE_SYSMEM = AttrSharedScratch
)

func parseNVInfoGlobal(data []byte, syms []elf.Symbol, kernels map[string]*KernelResource) {
	for i := 0; i+4 <= len(data); {
		fmtByte := data[i]
		attr := data[i+1]
		length := int(binary.LittleEndian.Uint16(data[i+2 : i+4]))
		totalLen := 4 + length
		if i+totalLen > len(data) {
			break
		}
		payload := data[i+4 : i+totalLen]
		i += totalLen

		if fmtByte == 0x04 && len(payload) >= 8 {
			symIdx := binary.LittleEndian.Uint32(payload[0:4])
			val := binary.LittleEndian.Uint32(payload[4:8])
			// Account for debug/elf omitting symbol 0 (STN_UNDEF).
			if symIdx > 0 && int(symIdx-1) < len(syms) {
				symName := syms[symIdx-1].Name
				if k, ok := kernels[symName]; ok {
					switch attr {
					case AttrRegCount:
						k.Registers = int(val)
					case AttrFrameSize:
						k.StackFrameBytes = int(val)
					case AttrSharedScratch:
						if k.StaticSharedMem == 0 {
							k.StaticSharedMem = int(val)
						}
					}
				}
			}
		}
	}
}

func parseNVInfoPerKernel(data []byte, k *KernelResource) {
	for i := 0; i+4 <= len(data); {
		_ = data[i]
		attr := data[i+1]
		length := int(binary.LittleEndian.Uint16(data[i+2 : i+4]))
		totalLen := 4 + length
		if i+totalLen > len(data) {
			break
		}
		payload := data[i+4 : i+totalLen]
		i += totalLen

		switch attr {
		case AttrRegCount:
			if len(payload) >= 4 {
				k.Registers = int(binary.LittleEndian.Uint32(payload[0:4]))
			}
		case AttrFrameSize:
			if len(payload) >= 4 {
				k.StackFrameBytes = int(binary.LittleEndian.Uint32(payload[0:4]))
			}
		case AttrSharedScratch:
			if k.StaticSharedMem == 0 && len(payload) >= 4 {
				k.StaticSharedMem = int(binary.LittleEndian.Uint32(payload[0:4]))
			}
		case AttrParamCBank:
			// EIATTR_PARAM_CBANK encodes constant bank 0 parameter size if not already set.
			if k.ConstantBankSize == 0 && len(payload) >= 8 {
				offset := binary.LittleEndian.Uint16(payload[4:6])
				size := binary.LittleEndian.Uint16(payload[6:8])
				k.ConstantBankSize = int(offset + size)
			}
		}
	}
}

// ParseCuobjdumpResUsage parses the text output of `cuobjdump -res-usage`.
// It returns a mapping of kernel symbol names to their reported resource usage.
func ParseCuobjdumpResUsage(text string) (map[string]KernelResource, error) {
	res := make(map[string]KernelResource)
	lines := strings.Split(text, "\n")
	var currentFunc string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "Function ") {
			// Format: "Function <name>:"
			currentFunc = strings.TrimPrefix(trimmed, "Function ")
			currentFunc = strings.TrimSuffix(currentFunc, ":")
			continue
		}

		if currentFunc == "" {
			continue
		}

		// Example line:
		// "REG:12 STACK:0 SHARED:5120 LOCAL:0 CONSTANT[0]:548 TEXTURE:0 SURFACE:0 SAMPLER:0"
		if strings.Contains(trimmed, "REG:") {
			kr := KernelResource{Name: currentFunc}
			fields := strings.Fields(trimmed)
			for _, f := range fields {
				kv := strings.SplitN(f, ":", 2)
				if len(kv) != 2 {
					continue
				}
				key, valStr := kv[0], kv[1]
				val := 0
				fmt.Sscanf(valStr, "%d", &val)
				switch key {
				case "REG":
					kr.Registers = val
				case "STACK":
					kr.StackFrameBytes = val
				case "SHARED":
					kr.StaticSharedMem = val
				case "CONSTANT[0]":
					kr.ConstantBankSize = val
				}
			}
			res[currentFunc] = kr
			currentFunc = ""
		}
	}

	return res, nil
}
