// Package cudarecipe implements Stage A CUBIN inspection and JSON Schema v1 reports.
package cudarecipe

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
	unknowns := []string{"target metadata unavailable", "allocated registers unavailable", "peak live GPR unavailable", "section inventory unavailable", "toolchain stamp unavailable"}
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
			for _, section := range f.Sections {
				sections = append(sections, section.Name)
			}
			sort.Strings(sections)
			_ = f.Close()
		} else {
			unknowns = append(unknowns, "section inventory unavailable: "+err.Error())
		}
		if info, err := cudakernel.ParseCUBIN(cubin); err == nil {
			architecture = info.Arch
			if value, ok := info.AllocatedRegisters[symbol]; ok {
				regCount, regCountKnown = value, true
			}
			sharedMemory = info.StaticSharedMemory[symbol]
		} else {
			unknowns = append(unknowns, "kernel resource metadata unavailable: "+err.Error())
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
		SchemaVersion: "v1.0.0",
		Symbol:        symbol,
		Digest:        hex.EncodeToString(h[:]),
		Architecture:  architecture,
		Sections:      sections,
		SharedMemory:  sharedMemory,
		RegCount:      regCount,
		RegCountKnown: regCountKnown,
		Toolchain:     toolchain,
		ByteLength:    len(cubin),
		ELFClass:      elfClass,
		Unknowns:      unknowns,
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
