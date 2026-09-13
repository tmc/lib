package cudaoccupancy

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"io"
	"strings"
)

// ELF attribute IDs in NVIDIA .nv.info sections.
const (
	AttrReqNTID     = 0x10 // EIATTR_REQ_NTID (required CTA dimensions)
	EIATTR_REQ_NTID = AttrReqNTID
	EIATTR_REQNTID  = AttrReqNTID
)

// KernelResourceInfo represents static resource usage for a CUDA kernel
// parsed from a compiled CUBIN ELF image.
type KernelResourceInfo struct {
	Name             string // Kernel symbol or function name
	Registers        int    // 32-bit registers allocated per thread (EIATTR_REGCOUNT)
	RegCount         int    // Alias for Registers
	SharedMemBytes   int    // Static shared memory in bytes (EIATTR_SHARED_SIZE_SYSMEM or .nv.shared)
	FrameSizeBytes   int    // Per-thread stack frame size in bytes (EIATTR_FRAME_SIZE)
	FrameSize        int    // Alias for FrameSizeBytes
	MaxRegCount      int    // Compiler register ceiling (EIATTR_MAXREG_COUNT)
	ReqNTID          [3]int // Required CTA dimensions [x, y, z] (EIATTR_REQ_NTID)
	ConstantBankSize int    // Constant bank 0 parameter size in bytes (EIATTR_PARAM_CBANK)
}

// OccupancyResult describes the theoretical SM occupancy for a kernel configuration.
type OccupancyResult struct {
	Architecture     string  // Architecture code (e.g. "sm_90", "sm_121")
	Arch             string  // Architecture alias
	TheoreticalWarps int     // Theoretical active warps per SM
	ActiveWarps      int     // Theoretical active warps per SM (alias)
	MaxWarps         int     // Architecture maximum warps per SM
	Occupancy        float64 // Theoretical occupancy ratio [0.0, 1.0]
	ActiveBlocks     int     // Resident thread blocks (CTAs) per SM
	LimitingFactor   string  // Primary bottleneck: "registers", "shared_memory", "block_size", "blocks", "none"
	BlockLimit       int     // Ceiling imposed by max blocks per SM
	WarpLimit        int     // Ceiling imposed by max warps per SM
	RegLimit         int     // Ceiling imposed by register file capacity
	SmemLimit        int     // Ceiling imposed by shared memory capacity
}

// String returns a concise human-readable summary of the occupancy result.
func (r OccupancyResult) String() string {
	return fmt.Sprintf("%s: %d/%d warps (%.1f%%, %d blocks/SM) limited by %s",
		r.Architecture, r.TheoreticalWarps, r.MaxWarps, r.Occupancy*100, r.ActiveBlocks, r.LimitingFactor)
}

// Calculate determines theoretical occupancy for this kernel on the named GPU architecture.
func (k KernelResourceInfo) Calculate(arch string, threadsPerBlock int) (OccupancyResult, error) {
	return CalculateOccupancy(arch, k.Registers, k.SharedMemBytes, threadsPerBlock)
}

// ReadKernelInfo parses a CUBIN ELF image from an io.ReaderAt and extracts
// resource usage for all defined CUDA kernels.
func ReadKernelInfo(r io.ReaderAt) ([]KernelResourceInfo, error) {
	f, err := elf.NewFile(r)
	if err != nil {
		return nil, fmt.Errorf("cudaoccupancy: read cubin elf: %w", err)
	}
	defer f.Close()
	return parseELFInfo(f)
}

// ReadKernelInfoBytes parses a CUBIN ELF image from a byte slice.
func ReadKernelInfoBytes(cubin []byte) ([]KernelResourceInfo, error) {
	return ReadKernelInfo(bytes.NewReader(cubin))
}

// CalculateOccupancy computes theoretical active warps and thread blocks per SM
// on the given GPU architecture based on register, shared memory, and block size constraints.
func CalculateOccupancy(arch string, regCount, sharedMemBytes, threadsPerBlock int) (OccupancyResult, error) {
	dev, err := LookupDevice(arch)
	if err != nil {
		return OccupancyResult{}, fmt.Errorf("cudaoccupancy: calculate occupancy: %w", err)
	}

	k := KernelResource{
		Registers:       regCount,
		StaticSharedMem: sharedMemBytes,
	}
	rep := k.CalculateDevice(dev, threadsPerBlock, 0)
	return OccupancyResult{
		Architecture:     rep.Architecture,
		Arch:             rep.Architecture,
		TheoreticalWarps: rep.TheoreticalWarps,
		ActiveWarps:      rep.TheoreticalWarps,
		MaxWarps:         rep.MaxWarps,
		Occupancy:        rep.Occupancy,
		ActiveBlocks:     rep.ActiveBlocks,
		LimitingFactor:   rep.LimitingFactor,
		BlockLimit:       rep.BlockLimit,
		WarpLimit:        rep.WarpLimit,
		RegLimit:         rep.RegLimit,
		SmemLimit:        rep.SmemLimit,
	}, nil
}

func parseELFInfo(f *elf.File) ([]KernelResourceInfo, error) {
	syms, err := f.Symbols()
	if err != nil && err != elf.ErrNoSymbols {
		return nil, fmt.Errorf("cudaoccupancy: read symbols: %w", err)
	}

	kernelMap := make(map[string]*KernelResourceInfo)
	var kernelOrder []string

	registerKernel := func(name string) *KernelResourceInfo {
		if k, ok := kernelMap[name]; ok {
			return k
		}
		k := &KernelResourceInfo{Name: name}
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

	for _, sec := range f.Sections {
		if strings.HasPrefix(sec.Name, ".text.") {
			name := strings.TrimPrefix(sec.Name, ".text.")
			if !strings.HasPrefix(name, "$__internal_") {
				registerKernel(name)
			}
		}
	}

	for _, sec := range f.Sections {
		if strings.HasPrefix(sec.Name, ".nv.info.") {
			name := strings.TrimPrefix(sec.Name, ".nv.info.")
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
				k.SharedMemBytes = int(sec.Size)
			}
		} else if strings.HasPrefix(sec.Name, ".nv.constant0.") {
			name := strings.TrimPrefix(sec.Name, ".nv.constant0.")
			if k, ok := kernelMap[name]; ok {
				k.ConstantBankSize = int(sec.Size)
			}
		}
	}

	// 3. Parse global .nv.info section.
	nvinfoSec := f.Section(".nv.info")
	if nvinfoSec != nil {
		data, err := nvinfoSec.Data()
		if err == nil {
			parseNVInfoGlobalInfo(data, syms, kernelMap)
		}
	}

	// 4. Parse per-kernel .nv.info.<name> sections.
	for _, sec := range f.Sections {
		if strings.HasPrefix(sec.Name, ".nv.info.") {
			kname := strings.TrimPrefix(sec.Name, ".nv.info.")
			k, ok := kernelMap[kname]
			if !ok {
				continue
			}
			data, err := sec.Data()
			if err == nil {
				parseNVInfoPerKernelInfo(data, k)
			}
		}
	}

	var res []KernelResourceInfo
	for _, name := range kernelOrder {
		if strings.HasPrefix(name, "$__internal_") {
			continue
		}
		info := *kernelMap[name]
		info.RegCount = info.Registers
		info.FrameSize = info.FrameSizeBytes
		res = append(res, info)
	}
	return res, nil
}

func parseNVInfoGlobalInfo(data []byte, syms []elf.Symbol, kernels map[string]*KernelResourceInfo) {
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
			if symIdx > 0 && int(symIdx-1) < len(syms) {
				symName := syms[symIdx-1].Name
				if k, ok := kernels[symName]; ok {
					switch attr {
					case AttrRegCount:
						k.Registers = int(val)
					case AttrFrameSize:
						k.FrameSizeBytes = int(val)
					case AttrSharedScratch:
						if k.SharedMemBytes == 0 {
							k.SharedMemBytes = int(val)
						}
					case AttrMaxRegCount:
						k.MaxRegCount = int(val)
					case AttrReqNTID:
						if len(payload) >= 16 {
							k.ReqNTID = [3]int{
								int(binary.LittleEndian.Uint32(payload[4:8])),
								int(binary.LittleEndian.Uint32(payload[8:12])),
								int(binary.LittleEndian.Uint32(payload[12:16])),
							}
						} else if len(payload) >= 8 {
							k.ReqNTID = [3]int{int(val), 1, 1}
						}
					}
				}
			}
		}
	}
}

func parseNVInfoPerKernelInfo(data []byte, k *KernelResourceInfo) {
	for i := 0; i+4 <= len(data); {
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
				k.FrameSizeBytes = int(binary.LittleEndian.Uint32(payload[0:4]))
			}
		case AttrSharedScratch:
			if k.SharedMemBytes == 0 && len(payload) >= 4 {
				k.SharedMemBytes = int(binary.LittleEndian.Uint32(payload[0:4]))
			}
		case AttrMaxRegCount:
			if len(payload) == 4 {
				k.MaxRegCount = int(binary.LittleEndian.Uint32(payload[0:4]))
			} else if len(payload) == 2 {
				k.MaxRegCount = int(binary.LittleEndian.Uint16(payload[0:2]))
			} else if len(payload) == 1 {
				k.MaxRegCount = int(payload[0])
			}
		case AttrParamCBank:
			if k.ConstantBankSize == 0 && len(payload) >= 8 {
				offset := binary.LittleEndian.Uint16(payload[4:6])
				size := binary.LittleEndian.Uint16(payload[6:8])
				k.ConstantBankSize = int(offset + size)
			}
		case AttrReqNTID:
			if len(payload) >= 12 {
				k.ReqNTID = [3]int{
					int(binary.LittleEndian.Uint32(payload[0:4])),
					int(binary.LittleEndian.Uint32(payload[4:8])),
					int(binary.LittleEndian.Uint32(payload[8:12])),
				}
			} else if len(payload) >= 6 {
				k.ReqNTID = [3]int{
					int(binary.LittleEndian.Uint16(payload[0:2])),
					int(binary.LittleEndian.Uint16(payload[2:4])),
					int(binary.LittleEndian.Uint16(payload[4:6])),
				}
			} else if len(payload) >= 4 {
				k.ReqNTID = [3]int{int(binary.LittleEndian.Uint32(payload[0:4])), 1, 1}
			}
		}
	}
}
