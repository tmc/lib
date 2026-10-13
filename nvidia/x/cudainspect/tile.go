package cudainspect

import (
	"bytes"
	"context"
	"crypto/sha256"
	"debug/elf"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/tileir"
	"github.com/tmc/lib/nvidia/x/tileirdisasm"
)

// TileArtifact is extracted intermediate bytecode, not executable machine code.
// Bytecode owns the exact section payload. DecodeError and RoundTripError retain
// unsupported-format evidence without discarding those bytes. RoundTripExact
// means only that the decoder/re-encoder preserved bytes; it does not establish
// that an assembler accepts them or that their operations have modeled semantics.
type TileArtifact struct {
	ContainerDigest string
	Section         string
	Digest          string
	Bytecode        []byte
	Version         string
	Producer        string
	RoundTripExact  bool
	DecodeError     string
	RoundTripError  string
}

// TileReport retains extracted bytecode and a separate disassembly result. Raw
// output remains available even if the tool or text parser fails. A successful
// disassembly is inspection evidence, not a semantic or runtime qualification.
type TileReport struct {
	Artifact *TileArtifact
	Output   tileirdisasm.Output
	Module   *tileirdisasm.Module
}

// InspectTileIR extracts before invoking the separate Tile IR disassembler.
// The executable must be trusted. All scratch is created under WorkDir. Errors
// retain the intermediate artifact and any command evidence already obtained.
func (i *Inspector) InspectTileIR(ctx context.Context, image []byte, toolPath string) (*TileReport, error) {
	if i == nil || i.WorkDir == "" {
		return nil, fmt.Errorf("cudarecipe: work directory required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	artifact, err := ExtractTileIR(image)
	r := &TileReport{Artifact: artifact}
	if err != nil {
		return r, err
	}
	r.Module, r.Output, err = tileirdisasm.DisassembleWithOutput(ctx, artifact.Bytecode, tileirdisasm.WithPath(toolPath), tileirdisasm.WithTempDir(i.WorkDir))
	return r, err
}

// ExtractTileIR extracts exactly one Tile IR section using tileirdisasm's
// extraction API. Missing or ambiguous sections are errors, never an empty
// successful artifact. It retains raw bytes even when decoding fails. Pass
// Bytecode separately to tileirdisasm.Disassemble when tool execution is wanted.
// This function executes no external tool and does not assume CUBINs embed IR.
func ExtractTileIR(image []byte) (*TileArtifact, error) {
	f, err := elf.NewFile(bytes.NewReader(image))
	if err != nil {
		return nil, fmt.Errorf("cudarecipe: Tile IR container: %w", err)
	}
	defer f.Close()
	var name string
	for _, section := range f.Sections {
		if !strings.Contains(section.Name, "tileir") && !strings.Contains(section.Name, "tilebc") && section.Name != ".nv_tile_ir" {
			continue
		}
		if name != "" {
			return nil, fmt.Errorf("cudarecipe: multiple Tile IR sections require explicit selection")
		}
		name = section.Name
	}
	if name == "" {
		return nil, fmt.Errorf("cudarecipe: no Tile IR section")
	}
	payload, err := tileirdisasm.ExtractFromELF(image)
	if err != nil {
		return nil, err
	}
	h := sha256.Sum256(image)
	r := &TileArtifact{ContainerDigest: hex.EncodeToString(h[:]), Section: name, Bytecode: append([]byte(nil), payload...)}
	h = sha256.Sum256(payload)
	r.Digest = hex.EncodeToString(h[:])
	m, err := tileir.DecodeBytecodeBytes(payload)
	if err != nil {
		r.DecodeError = err.Error()
		return r, fmt.Errorf("cudarecipe: decode Tile IR: %w", err)
	}
	r.Version, r.Producer = m.Version.String(), m.Producer
	encoded, err := tileir.EncodeBytecode(m)
	if err != nil {
		r.RoundTripError = err.Error()
		return r, fmt.Errorf("cudarecipe: re-encode Tile IR: %w", err)
	}
	r.RoundTripExact = bytes.Equal(payload, encoded)
	if !r.RoundTripExact {
		r.RoundTripError = "decoded bytecode did not re-encode byte-for-byte"
	}
	return r, nil
}
