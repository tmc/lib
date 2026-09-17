// Package cudakernel defines immutable kernel artifact descriptors and parameter ABI schemas.
package cudakernel

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

type ArtifactKind int

const (
	ArtifactPTX ArtifactKind = iota
	ArtifactCUBIN
	ArtifactLTOIR
	ArtifactTileIR
)

type ParamDesc struct {
	Name      string `json:"name"`
	PTXType   string `json:"ptx_type"`
	ByteWidth uint32 `json:"byte_width"`
	Alignment uint32 `json:"alignment"`
	IsPointer bool   `json:"is_pointer"`
}

type DescriptorResources struct {
	StaticSharedBytes  uint32 `json:"static_shared_bytes"`
	DynamicSharedBytes uint32 `json:"dynamic_shared_bytes"`
	AllocatedRegisters uint32 `json:"allocated_registers"`
	PeakLiveGPR        uint32 `json:"peak_live_gpr"`
}

type Descriptor struct {
	Digest     string              `json:"digest"`
	Kind       ArtifactKind        `json:"kind"`
	Target     string              `json:"target"`
	EntryPoint string              `json:"entry_point"`
	Params     []ParamDesc         `json:"params"`
	Resources  DescriptorResources `json:"resources"`
	Toolchain  string              `json:"toolchain"`
	RawBytes   []byte              `json:"-"`
}

func NewDescriptor(kind ArtifactKind, target, entry string, raw []byte, params []ParamDesc, res DescriptorResources) (*Descriptor, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("cudakernel: raw artifact bytes cannot be empty")
	}
	h := sha256.Sum256(raw)
	return &Descriptor{
		Digest:     hex.EncodeToString(h[:]),
		Kind:       kind,
		Target:     target,
		EntryPoint: entry,
		Params:     params,
		Resources:  res,
		RawBytes:   raw,
	}, nil
}
