// Package cudainspect implements Stage A CUBIN inspection and JSON Schema v1 reports.
package cudainspect

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

type ReportSchemaV1 struct {
	SchemaVersion string   `json:"schema_version"`
	Symbol        string   `json:"symbol"`
	Digest        string   `json:"digest"`
	RegCount      uint32   `json:"allocated_reg_count"`
	PeakGPR       uint32   `json:"measured_peak_gpr"`
	Toolchain     string   `json:"toolchain_stamp"`
	Architecture  string   `json:"architecture,omitempty"`
	Sections      []string `json:"sections,omitempty"`
	SharedMemory  uint32   `json:"static_shared_memory"`
	ByteLength    int      `json:"byte_length"`
	ELFClass      string   `json:"elf_class"`
	RegCountKnown bool     `json:"allocated_reg_count_known"`
	PeakGPRKnown  bool     `json:"peak_live_gpr_known"`
	Unknowns      []string `json:"unknowns,omitempty"`
	// ELFParsed means debug/elf read the header and section table. It does not
	// establish that the image is loadable or that section payloads are valid.
	ELFParsed                bool          `json:"elf_parsed"`
	ELFMachine               string        `json:"elf_machine,omitempty"`
	ELFType                  string        `json:"elf_type,omitempty"`
	ELFData                  string        `json:"elf_data,omitempty"`
	SectionDetails           []SectionInfo `json:"section_details,omitempty"`
	CodeSymbols              []string      `json:"code_symbols,omitempty"`
	SharedMemoryKnown        bool          `json:"static_shared_memory_known"`
	DynamicSharedMemoryKnown bool          `json:"dynamic_shared_memory_known"`
}

// SectionInfo describes an ELF section as stored in the supplied artifact.
// NOBITS sections have a logical size but no payload at their file offset.
type SectionInfo struct {
	Name   string `json:"name"`
	Type   string `json:"type"`
	Flags  string `json:"flags"`
	Offset uint64 `json:"offset"`
	Size   uint64 `json:"size"`
}

type Inspector struct {
	WorkDir string
}

func NewInspector(workDir string) (*Inspector, error) {
	if workDir == "" {
		return nil, fmt.Errorf("cudarecipe: work directory required")
	}
	return &Inspector{WorkDir: workDir}, nil
}

func (i *Inspector) Inspect(cubin []byte, symbol string) (*ReportSchemaV1, error) {
	if len(cubin) == 0 {
		return nil, fmt.Errorf("cudarecipe: empty cubin image")
	}
	if symbol == "" {
		return nil, fmt.Errorf("cudarecipe: empty symbol name")
	}
	h := sha256.Sum256(cubin)
	elfClass := "unknown"
	unknowns := []string{"target metadata unavailable", "allocated registers unavailable", "peak live GPR unavailable", "section inventory unavailable", "toolchain stamp unavailable", "static shared memory unavailable", "dynamic shared memory unavailable: requires launch configuration", "instruction widths unavailable: no disassembly supplied", "local-memory observations unavailable: no disassembly supplied"}
	var elfValid bool
	var elfMachine, elfType, elfData string
	var sectionDetails []SectionInfo
	var codeSymbols []string
	var sharedMemoryKnown bool
	var architecture string
	var sections []string
	var sharedMemory uint32
	var regCount uint32
	var regCountKnown bool
	var toolchain string
	if len(cubin) >= 5 && string(cubin[:4]) == "\x7fELF" {
		if cubin[4] == 1 {
			elfClass = "ELF32"
		} else if cubin[4] == 2 {
			elfClass = "ELF64"
		}
		if f, err := elf.NewFile(bytes.NewReader(cubin)); err == nil {
			elfValid = true
			elfMachine, elfType, elfData = f.Machine.String(), f.Type.String(), f.Data.String()
			for _, section := range f.Sections {
				sections = append(sections, section.Name)
				sectionDetails = append(sectionDetails, SectionInfo{Name: section.Name, Type: section.Type.String(), Flags: section.Flags.String(), Offset: section.Offset, Size: section.Size})
				if f.Machine == elf.EM_CUDA && section.Name == ".nv.shared."+symbol && section.Size <= uint64(^uint32(0)) {
					sharedMemory, sharedMemoryKnown = uint32(section.Size), true
				}
				if name, ok := strings.CutPrefix(section.Name, ".text."); ok && name != "" {
					codeSymbols = append(codeSymbols, name)
				}
			}
			sort.Strings(sections)
			sort.Strings(codeSymbols)
			_ = f.Close()
		} else {
			unknowns = append(unknowns, "section inventory unavailable: "+err.Error())
		}
		if elfValid && elfMachine == elf.EM_CUDA.String() {
			if info, err := cudakernel.ParseCUBIN(cubin); err == nil {
				architecture = info.Arch
			} else {
				unknowns = append(unknowns, "kernel resource metadata unavailable: "+err.Error())
			}
			// Use the allocation attribute parser, not a compiler register ceiling
			// or a live-register observation. Its zero cannot distinguish absence.
			if resources, err := cudaoccupancy.InspectCubin(cubin); err == nil {
				for _, resource := range resources {
					if resource.Name == symbol && resource.Registers > 0 && uint64(resource.Registers) <= uint64(^uint32(0)) {
						regCount, regCountKnown = uint32(resource.Registers), true
					}
				}
			} else {
				unknowns = append(unknowns, "allocation metadata unavailable: "+err.Error())
			}
			if stamps, err := cudasass.ReadToolchains(bytes.NewReader(cubin)); err == nil && len(stamps) > 0 {
				toolchain = strings.Join(func() []string {
					out := make([]string, len(stamps))
					for n, stamp := range stamps {
						out[n] = stamp.String()
					}
					return out
				}(), "; ")
			}
		}
	}
	if !elfValid {
		unknowns = append(unknowns, "ELF identity unavailable: invalid ELF image")
	}
	if sharedMemoryKnown {
		unknowns = removeUnknown(unknowns, "static shared memory unavailable")
	}
	if architecture != "" {
		unknowns = removeUnknown(unknowns, "target metadata unavailable")
	}
	if regCountKnown {
		unknowns = removeUnknown(unknowns, "allocated registers unavailable")
	}
	if len(sections) > 0 {
		unknowns = removeUnknown(unknowns, "section inventory unavailable")
	}
	if toolchain != "" {
		unknowns = removeUnknown(unknowns, "toolchain stamp unavailable")
	}
	return &ReportSchemaV1{
		SchemaVersion:     "v1.1.0",
		Symbol:            symbol,
		Digest:            hex.EncodeToString(h[:]),
		Architecture:      architecture,
		Sections:          sections,
		SharedMemory:      sharedMemory,
		RegCount:          regCount,
		RegCountKnown:     regCountKnown,
		Toolchain:         toolchain,
		ByteLength:        len(cubin),
		ELFClass:          elfClass,
		Unknowns:          unknowns,
		ELFParsed:         elfValid,
		ELFMachine:        elfMachine,
		ELFType:           elfType,
		ELFData:           elfData,
		SectionDetails:    sectionDetails,
		CodeSymbols:       codeSymbols,
		SharedMemoryKnown: sharedMemoryKnown,
	}, nil
}

func removeUnknown(unknowns []string, value string) []string {
	for n, unknown := range unknowns {
		if unknown == value {
			return append(unknowns[:n], unknowns[n+1:]...)
		}
	}
	return unknowns
}

func (r *ReportSchemaV1) WriteJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}
