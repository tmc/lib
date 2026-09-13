package cuptipcsampling

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// CubinIdentity records CUPTI's code identifier and the digest of the exact
// input bytes. CRC is a native lookup key, not a collision-resistant artifact
// digest. It does not establish that this image was loaded in a given process.
type CubinIdentity struct {
	CUPTIVersion uint32
	CRC          uint64
	SHA256       [32]byte
}

// IdentifyCubin obtains CUPTI's native identifier for retained CUDA ELF bytes.
// It rejects empty inputs, inputs larger than maxBytes, and non-CUDA ELF files
// before calling CUPTI. maxBytes must be positive. The input is copied before
// native access; callers must not modify it concurrently with this function.
// It loads no CUDA module, launches no code and enables no collector.
//
// This query is for post-processing outside CUPTI callbacks. It admits the
// 64-bit parameter layout for CUPTI 27, 130001, 130200 and 130400. A matching sampled
// CRC is only a candidate code-image link: retain module/process generation,
// the artifact digest and function/offset evidence when resolving PCs.
func IdentifyCubin(cubin []byte, maxBytes int) (CubinIdentity, error) {
	return identifyCubin(cubin, maxBytes, cupti.GetVersion, cupti.GetCubinCrc)
}

func identifyCubin(cubin []byte, maxBytes int, version func() (uint32, error), query func(*cupti.GetCubinCrcParams) error) (CubinIdentity, error) {
	if maxBytes <= 0 || len(cubin) == 0 || len(cubin) > maxBytes {
		return CubinIdentity{}, errors.New("cuptipcsampling: invalid cubin size or limit")
	}
	data := bytes.Clone(cubin)
	digest := sha256.Sum256(data)
	f, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		return CubinIdentity{}, fmt.Errorf("cuptipcsampling: read cubin: %w", err)
	}
	defer f.Close()
	if f.Machine != elf.EM_CUDA {
		return CubinIdentity{}, errors.New("cuptipcsampling: cubin is not CUDA ELF")
	}
	v, err := version()
	if err != nil {
		return CubinIdentity{}, fmt.Errorf("cuptipcsampling: query version: %w", err)
	}
	if unsafe.Sizeof(uintptr(0)) != 8 || !qualifiedQueryVersion(v) {
		return CubinIdentity{}, fmt.Errorf("cuptipcsampling: unqualified cubin CRC layout for CUPTI %d", v)
	}
	var pin runtime.Pinner
	pin.Pin(&data[0])
	defer pin.Unpin()
	p := cupti.GetCubinCrcParams{Size: 32, CubinSize: uint(len(data)), Cubin: uintptr(unsafe.Pointer(&data[0]))}
	if err := query(&p); err != nil {
		return CubinIdentity{}, fmt.Errorf("cuptipcsampling: query cubin CRC: %w", err)
	}
	if p.Size != 32 || p.CubinSize != uint(len(data)) || p.Cubin != uintptr(unsafe.Pointer(&data[0])) {
		return CubinIdentity{}, errors.New("cuptipcsampling: native cubin query changed input storage")
	}
	if sha256.Sum256(data) != digest {
		return CubinIdentity{}, errors.New("cuptipcsampling: native cubin query changed input bytes")
	}
	return CubinIdentity{CUPTIVersion: v, CRC: p.CubinCrc, SHA256: digest}, nil
}
