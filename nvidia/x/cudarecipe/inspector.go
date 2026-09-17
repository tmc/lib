// Package cudarecipe implements Stage A CUBIN inspection and JSON Schema v1 reports.
package cudarecipe

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
)

type ReportSchemaV1 struct {
	SchemaVersion string   `json:"schema_version"`
	Symbol        string   `json:"symbol"`
	Digest        string   `json:"digest"`
	RegCount      uint32   `json:"allocated_reg_count"`
	PeakGPR       uint32   `json:"measured_peak_gpr"`
	Toolchain     string   `json:"toolchain_stamp"`
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
	unknowns := []string{"target metadata unavailable", "allocated registers unavailable", "peak live GPR unavailable", "section inventory unavailable"}
	if len(cubin) >= 5 && string(cubin[:4]) == "\x7fELF" {
		if cubin[4] == 1 {
			elfClass = "ELF32"
		} else if cubin[4] == 2 {
			elfClass = "ELF64"
		}
	}
	return &ReportSchemaV1{
		SchemaVersion: "v1.0.0",
		Symbol:        symbol,
		Digest:        hex.EncodeToString(h[:]),
		ByteLength:    len(cubin),
		ELFClass:      elfClass,
		Unknowns:      unknowns,
	}, nil
}

func (r *ReportSchemaV1) WriteJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}
