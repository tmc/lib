// SPDX-License-Identifier: Apache-2.0

package cudaprepack

import (
	"fmt"
)

// Supported NVIDIA target architectures.
const (
	ArchSM121a = "sm_121a" // Blackwell GB10 (TMA, mbarrier, multimem; rejects tcgen05, wgmma)
	ArchSM100a = "sm_100a" // Blackwell GB200 / B100 (tcgen05, microscaling swizzle, 2:4 sparsity)
)

// Hardware alignment and box constraints.
const (
	MinAddressAlignment     = 16  // TMA requires 16-byte base address alignment
	MinBoxWidthBytes        = 16  // TMA box width must be at least 16 bytes
	MaxBoxWidthBytesSM121a  = 128 // sm_121a max box width in bytes
	MaxBoxWidthBytesSM100a  = 128 // sm_100a swizzled max box width in bytes
	BoxBoundaryBytesSM121a  = 128 // sm_121a box boundary constraint
)

// DataType identifies tensor element precision and packing format.
type DataType string

const (
	TypeIQ2_XXS  DataType = "IQ2_XXS"  // 2-bit codebook quantized super-blocks (256 elem/block)
	TypeQ8_0     DataType = "Q8_0"     // 8-bit symmetric with float16 scale (32 elem/block)
	TypeFP8_E4M3 DataType = "FP8_E4M3" // 8-bit float: 1s, 4e, 3m (max 448.0)
	TypeFP8_E5M2 DataType = "FP8_E5M2" // 8-bit float: 1s, 5e, 2m (IEEE-like)
	TypeMXFP4    DataType = "MXFP4"    // OCP Microscaling FP4 (E2M1) with per-block scale
	TypeFloat16  DataType = "Float16"  // IEEE 754 binary16
	TypeFloat32  DataType = "Float32"  // IEEE 754 binary32
)

// BitsPerElement returns the nominal number of bits per element.
func (d DataType) BitsPerElement() int {
	switch d {
	case TypeIQ2_XXS:
		return 2
	case TypeMXFP4:
		return 4
	case TypeQ8_0, TypeFP8_E4M3, TypeFP8_E5M2:
		return 8
	case TypeFloat16:
		return 16
	case TypeFloat32:
		return 32
	default:
		return 8
	}
}

// SubByte reports whether the format packs multiple elements per byte.
func (d DataType) SubByte() bool {
	return d == TypeIQ2_XXS || d == TypeMXFP4
}

// SwizzleMode describes TMA shared memory bank swizzling patterns.
type SwizzleMode uint32

const (
	SwizzleNone              SwizzleMode = 0
	Swizzle32B               SwizzleMode = 1
	Swizzle64B               SwizzleMode = 2
	Swizzle128B              SwizzleMode = 3
	Swizzle128BAtom32B       SwizzleMode = 4 // sm_100a tcgen05 atom swizzle
)

// String returns the human-readable swizzle mode name.
func (s SwizzleMode) String() string {
	switch s {
	case SwizzleNone:
		return "SwizzleNone"
	case Swizzle32B:
		return "Swizzle32B"
	case Swizzle64B:
		return "Swizzle64B"
	case Swizzle128B:
		return "Swizzle128B"
	case Swizzle128BAtom32B:
		return "Swizzle128BAtom32B"
	default:
		return fmt.Sprintf("SwizzleMode(%d)", s)
	}
}

// InterleaveMode describes tensor byte interleaving.
type InterleaveMode uint32

const (
	InterleaveNone InterleaveMode = 0
	Interleave16B  InterleaveMode = 1
	Interleave32B  InterleaveMode = 2
)

// BoxDescriptor describes a hardware-aligned 2D or 3D TMA tensor box.
type BoxDescriptor struct {
	Rank           int            // 2 or 3
	DataType       DataType       // Element data type
	GlobalAddress  uint64         // Base memory address (must be 16-byte aligned)
	GlobalDims     [3]uint64      // [dim0, dim1, dim2] in elements
	GlobalStrides  [3]int64       // Stride in bytes along dim1 and dim2 (multiples of 16)
	BoxDims        [3]uint32      // [box0, box1, box2] tile shape in elements
	ElementStrides [3]uint32      // Must be 1 for dim0
	Swizzle        SwizzleMode    // Shared memory bank swizzle
	Interleave     InterleaveMode // Byte interleaving layout
}

// Validate checks the descriptor against architectural constraints.
func (b *BoxDescriptor) Validate(arch string) error {
	if b.Rank != 2 && b.Rank != 3 {
		return fmt.Errorf("cudaprepack: invalid rank %d, must be 2 or 3", b.Rank)
	}
	if b.GlobalAddress%MinAddressAlignment != 0 {
		return fmt.Errorf("cudaprepack: global address 0x%x not %d-byte aligned", b.GlobalAddress, MinAddressAlignment)
	}
	if b.GlobalDims[0] == 0 || b.GlobalDims[1] == 0 {
		return fmt.Errorf("cudaprepack: global dimensions must be non-zero")
	}
	if b.Rank == 3 && b.GlobalDims[2] == 0 {
		return fmt.Errorf("cudaprepack: 3D global dim2 must be non-zero")
	}
	if b.BoxDims[0] == 0 || b.BoxDims[1] == 0 {
		return fmt.Errorf("cudaprepack: box dimensions must be non-zero")
	}
	if b.Rank == 3 && b.BoxDims[2] == 0 {
		return fmt.Errorf("cudaprepack: 3D box dim2 must be non-zero")
	}
	if b.ElementStrides[0] != 1 {
		return fmt.Errorf("cudaprepack: element stride 0 must be 1, got %d", b.ElementStrides[0])
	}

	// Compute box width in bytes
	bitsPerElem := b.DataType.BitsPerElement()
	boxWidthBytes := (int(b.BoxDims[0]) * bitsPerElem + 7) / 8
	if boxWidthBytes < MinBoxWidthBytes {
		return fmt.Errorf("cudaprepack: box width (%d bytes) less than minimum %d bytes", boxWidthBytes, MinBoxWidthBytes)
	}
	if boxWidthBytes%MinBoxWidthBytes != 0 {
		return fmt.Errorf("cudaprepack: box width (%d bytes) must be a multiple of %d bytes", boxWidthBytes, MinBoxWidthBytes)
	}

	// Row stride must be a multiple of 16 bytes
	if b.GlobalStrides[0]%MinAddressAlignment != 0 {
		return fmt.Errorf("cudaprepack: row stride %d must be multiple of %d bytes", b.GlobalStrides[0], MinAddressAlignment)
	}
	if b.Rank == 3 && b.GlobalStrides[1]%MinAddressAlignment != 0 {
		return fmt.Errorf("cudaprepack: plane stride %d must be multiple of %d bytes", b.GlobalStrides[1], MinAddressAlignment)
	}

	// Architecture-specific rules
	switch arch {
	case ArchSM121a:
		if boxWidthBytes > MaxBoxWidthBytesSM121a {
			return fmt.Errorf("cudaprepack: box width (%d bytes) exceeds sm_121a limit %d bytes", boxWidthBytes, MaxBoxWidthBytesSM121a)
		}
		if b.Swizzle == Swizzle128BAtom32B {
			return fmt.Errorf("cudaprepack: target sm_121a rejects tcgen05 swizzle mode %s", b.Swizzle)
		}
	case ArchSM100a:
		if boxWidthBytes > 256 {
			return fmt.Errorf("cudaprepack: box width (%d bytes) exceeds sm_100a limit 256 bytes", boxWidthBytes)
		}
		if b.Swizzle != SwizzleNone && boxWidthBytes > MaxBoxWidthBytesSM100a {
			return fmt.Errorf("cudaprepack: swizzled box width (%d bytes) exceeds sm_100a limit %d bytes", boxWidthBytes, MaxBoxWidthBytesSM100a)
		}
	default:
		return fmt.Errorf("cudaprepack: unsupported target architecture %q (supported: %q, %q)", arch, ArchSM121a, ArchSM100a)
	}

	return nil
}

// BoxSizeBytes returns the byte size of one 2D or 3D box tile.
func (b *BoxDescriptor) BoxSizeBytes() int {
	bitsPerElem := b.DataType.BitsPerElement()
	wBytes := (int(b.BoxDims[0]) * bitsPerElem + 7) / 8
	h := int(b.BoxDims[1])
	d := 1
	if b.Rank == 3 && b.BoxDims[2] > 0 {
		d = int(b.BoxDims[2])
	}
	return wBytes * h * d
}

// TotalSizeBytes returns the total memory footprint in bytes represented by the descriptor.
func (b *BoxDescriptor) TotalSizeBytes() int64 {
	if b.Rank == 2 {
		return b.GlobalStrides[0] * int64(b.GlobalDims[1])
	}
	return b.GlobalStrides[1] * int64(b.GlobalDims[2])
}

// EncodeTensorMap encodes the box descriptor into a 128-byte CUtensorMap representation.
func (b *BoxDescriptor) EncodeTensorMap() ([16]uint64, error) {
	var desc [16]uint64
	desc[0] = b.GlobalAddress
	desc[1] = b.GlobalDims[0] | (b.GlobalDims[1] << 32)
	desc[2] = b.GlobalDims[2]
	desc[3] = uint64(b.GlobalStrides[0])
	desc[4] = uint64(b.GlobalStrides[1])
	desc[5] = uint64(b.BoxDims[0]) | (uint64(b.BoxDims[1]) << 32)
	desc[6] = uint64(b.BoxDims[2]) |
		(uint64(b.ElementStrides[0]&0xFF) << 32) |
		(uint64(b.ElementStrides[1]&0xFF) << 40) |
		(uint64(b.ElementStrides[2]&0xFF) << 48)
	desc[7] = uint64(len(b.DataType)) |
		(uint64(b.Rank&0xFF) << 8) |
		(uint64(b.Swizzle&0xFF) << 16) |
		(uint64(b.Interleave&0xFF) << 24)
	return desc, nil
}

// PrepackedTensor contains the pre-packed weights, optional scale factors,
// optional sparsity metadata, and the associated TMA BoxDescriptor.
type PrepackedTensor struct {
	Arch     string
	Type     DataType
	BoxDesc  BoxDescriptor
	Payload  []byte // Hardware-aligned tile weights
	Scales   []byte // Contiguous or swizzled scale factors
	Metadata []byte // 2:4 structured sparsity metadata (if sparse)
	zeroCopy bool
}

// ZeroCopy reports whether the payload was sliced directly from an input buffer without copy.
func (p *PrepackedTensor) ZeroCopy() bool {
	return p.zeroCopy
}
