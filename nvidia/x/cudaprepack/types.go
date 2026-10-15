// SPDX-License-Identifier: Apache-2.0

package cudaprepack

import (
	"fmt"
	"math"
)

// Target names accepted by the host layout transformers.
const (
	ArchSM121a = "sm_121a"
	ArchSM100a = "sm_100a"
)

// Alignment and default box sizes used by the host layout transformers.
// The box-size defaults are not general tensor-map hardware limits.
const (
	MinAddressAlignment    = 16
	MinBoxWidthBytes       = 16
	MaxBoxWidthBytesSM121a = 128
	MaxBoxWidthBytesSM100a = 128
	BoxBoundaryBytesSM121a = 128
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
		return 0
	}
}

// SubByte reports whether the format packs multiple elements per byte.
func (d DataType) SubByte() bool {
	return d == TypeIQ2_XXS || d == TypeMXFP4
}

// SwizzleMode names a descriptor's requested swizzle, not a bank-conflict result.
type SwizzleMode uint32

const (
	SwizzleNone        SwizzleMode = 0
	Swizzle32B         SwizzleMode = 1
	Swizzle64B         SwizzleMode = 2
	Swizzle128B        SwizzleMode = 3
	Swizzle128BAtom32B SwizzleMode = 4
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

// BoxDescriptor describes host tensor storage and a traversal box.
// A valid host descriptor is not necessarily a CUDA tensor-map descriptor.
// Unused dimension, box and element-stride entries may be zero or one;
// unused global-stride entries must be zero.
type BoxDescriptor struct {
	Rank           int            // 2 or 3
	DataType       DataType       // Element data type
	GlobalAddress  uint64         // Base memory address (must be 16-byte aligned)
	GlobalDims     [3]uint64      // [dim0, dim1, dim2] in elements
	GlobalStrides  [3]int64       // Stride in bytes along dim1 and dim2 (multiples of 16)
	BoxDims        [3]uint32      // [box0, box1, box2] tile shape in elements
	ElementStrides [3]uint32      // Positive traversal strides
	Swizzle        SwizzleMode    // Shared memory bank swizzle
	Interleave     InterleaveMode // Byte interleaving layout
}

// Validate checks the host storage contract for a named transformer target.
// It checks shape, known formats and modes, aligned nonoverlapping strides,
// representable storage and address extents. It does not verify payload bytes,
// encode a tensor map, or establish eligibility for a device operation.
func (b *BoxDescriptor) Validate(arch string) error {
	if arch != ArchSM121a && arch != ArchSM100a {
		return fmt.Errorf("cudaprepack: unsupported target architecture %q", arch)
	}
	_, _, err := b.storageBytes()
	return err
}

func (b *BoxDescriptor) storageBytes() (int, int64, error) {
	if b == nil {
		return 0, 0, fmt.Errorf("cudaprepack: nil box descriptor")
	}
	if b.Rank != 2 && b.Rank != 3 {
		return 0, 0, fmt.Errorf("cudaprepack: invalid rank %d, must be 2 or 3", b.Rank)
	}
	if b.GlobalAddress%MinAddressAlignment != 0 {
		return 0, 0, fmt.Errorf("cudaprepack: global address 0x%x not %d-byte aligned", b.GlobalAddress, MinAddressAlignment)
	}
	bits := uint64(b.DataType.BitsPerElement())
	if bits == 0 {
		return 0, 0, fmt.Errorf("cudaprepack: unknown data type %q", b.DataType)
	}
	if b.Swizzle > Swizzle128BAtom32B || b.Interleave > Interleave32B {
		return 0, 0, fmt.Errorf("cudaprepack: unknown swizzle or interleave mode")
	}
	for i := 0; i < b.Rank; i++ {
		if b.GlobalDims[i] == 0 || b.BoxDims[i] == 0 || b.ElementStrides[i] == 0 {
			return 0, 0, fmt.Errorf("cudaprepack: dimension %d must have positive size, box and element stride", i)
		}
	}
	for i := b.Rank; i < len(b.GlobalDims); i++ {
		if b.GlobalDims[i] > 1 || b.BoxDims[i] > 1 || b.ElementStrides[i] > 1 {
			return 0, 0, fmt.Errorf("cudaprepack: dimension %d is unused at rank %d", i, b.Rank)
		}
	}
	for i := b.Rank - 1; i < len(b.GlobalStrides); i++ {
		if b.GlobalStrides[i] != 0 {
			return 0, 0, fmt.Errorf("cudaprepack: global stride %d is unused at rank %d", i, b.Rank)
		}
	}
	if b.GlobalDims[0] > (math.MaxUint64-7)/bits {
		return 0, 0, fmt.Errorf("cudaprepack: row extent overflows")
	}
	extent := (b.GlobalDims[0]*bits + 7) / 8
	for i := 0; i < b.Rank-1; i++ {
		stride := b.GlobalStrides[i]
		if stride <= 0 {
			return 0, 0, fmt.Errorf("cudaprepack: global stride %d must be positive", i)
		}
		if stride%MinAddressAlignment != 0 {
			if i == 0 {
				return 0, 0, fmt.Errorf("cudaprepack: row stride %d must be multiple of 16 bytes", stride)
			}
			return 0, 0, fmt.Errorf("cudaprepack: plane stride %d must be multiple of 16 bytes", stride)
		}
		if uint64(stride) < extent {
			return 0, 0, fmt.Errorf("cudaprepack: global stride %d overlaps the preceding extent", i)
		}
		if b.GlobalDims[i+1] > uint64(math.MaxInt64)/uint64(stride) {
			return 0, 0, fmt.Errorf("cudaprepack: storage extent overflows")
		}
		extent = uint64(stride) * b.GlobalDims[i+1]
	}
	if b.GlobalAddress > math.MaxUint64-extent {
		return 0, 0, fmt.Errorf("cudaprepack: storage address extent overflows")
	}
	box := (uint64(b.BoxDims[0])*bits + 7) / 8
	for i := 1; i < b.Rank; i++ {
		if box > uint64(math.MaxInt)/uint64(b.BoxDims[i]) {
			return 0, 0, fmt.Errorf("cudaprepack: box extent overflows")
		}
		box *= uint64(b.BoxDims[i])
	}
	return int(box), int64(extent), nil
}

// TensorMapOperation names the bounded tiled copy operations checked here.
type TensorMapOperation string

const (
	TensorMapTiledLoad  TensorMapOperation = "tiled-load"
	TensorMapTiledStore TensorMapOperation = "tiled-store"
)

// ValidateTensorMap checks descriptor fields for a tiled load or store under
// CUDA Driver API 13.4 cuTensorMapEncodeTiled and PTX ISA 9.4 tensor-copy rules.
// It supports Float16 and Float32, rank 2 or 3, the three interleave modes,
// and no swizzle or the standard 32-, 64- and 128-byte swizzles on the two named
// targets. Other packed formats and atom swizzles are explicitly unsupported.
// Stores are restricted to boxes contained in the logical tensor at the origin.
//
// Success does not encode an opaque tensor map or validate payload contents,
// allocation, transfer coordinates, shared addresses or completion protocols.
// Descriptor construction through the driver and native execution remain separate.
// See https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html.
func (b *BoxDescriptor) ValidateTensorMap(arch string, operation TensorMapOperation) error {
	if operation != TensorMapTiledLoad && operation != TensorMapTiledStore {
		return fmt.Errorf("cudaprepack: unsupported tensor-map operation %q", operation)
	}
	if err := b.Validate(arch); err != nil {
		return err
	}
	if b.DataType != TypeFloat16 && b.DataType != TypeFloat32 {
		return fmt.Errorf("cudaprepack: %s has no supported scalar tensor-map dtype", b.DataType)
	}
	for i := 0; i < b.Rank; i++ {
		if b.GlobalDims[i] > 1<<32 || b.BoxDims[i] > 256 || b.ElementStrides[i] > 8 {
			return fmt.Errorf("cudaprepack: dimension %d exceeds tiled tensor-map bounds", i)
		}
		if operation == TensorMapTiledStore && uint64(b.BoxDims[i]) > b.GlobalDims[i] {
			return fmt.Errorf("cudaprepack: tiled store box dimension %d exceeds logical extent", i)
		}
	}
	for i := 0; i < b.Rank-1; i++ {
		if b.GlobalStrides[i] >= 1<<40 {
			return fmt.Errorf("cudaprepack: global stride %d exceeds tiled tensor-map bounds", i)
		}
	}
	width := uint64(b.BoxDims[0]) * uint64(b.DataType.BitsPerElement()/8)
	var span uint64
	switch b.Swizzle {
	case SwizzleNone:
	case Swizzle32B:
		span = 32
	case Swizzle64B:
		span = 64
	case Swizzle128B:
		span = 128
	default:
		return fmt.Errorf("cudaprepack: tensor-map swizzle %s is unsupported for %s", b.Swizzle, operation)
	}
	if b.Interleave == InterleaveNone {
		if width%16 != 0 || span != 0 && width > span {
			return fmt.Errorf("cudaprepack: tensor-map inner box width violates alignment or swizzle span")
		}
	} else if b.Rank < 3 {
		return fmt.Errorf("cudaprepack: tensor-map interleave requires rank 3")
	}
	if b.Interleave == Interleave32B {
		if b.GlobalAddress%32 != 0 || b.Swizzle != Swizzle32B {
			return fmt.Errorf("cudaprepack: 32-byte interleave requires aligned address and 32-byte swizzle")
		}
		for i := 0; i < b.Rank-1; i++ {
			if b.GlobalStrides[i]%32 != 0 {
				return fmt.Errorf("cudaprepack: 32-byte interleave requires aligned global strides")
			}
		}
	}
	return nil
}

// BoxSizeBytes returns the host box size, or zero for malformed or
// unrepresentable storage. Validate returns the corresponding error.
func (b *BoxDescriptor) BoxSizeBytes() int {
	box, _, err := b.storageBytes()
	if err != nil {
		return 0
	}
	return box
}

// TotalSizeBytes returns the padded host storage size, or zero for malformed or
// unrepresentable storage. Validate returns the corresponding error.
func (b *BoxDescriptor) TotalSizeBytes() int64 {
	_, total, err := b.storageBytes()
	if err != nil {
		return 0
	}
	return total
}

// EncodeTensorMap returns an error because tensor maps require opaque
// descriptor construction through the CUDA Driver API.
func (b *BoxDescriptor) EncodeTensorMap() ([16]uint64, error) {
	return [16]uint64{}, fmt.Errorf("cudaprepack: tensor-map encoding requires the CUDA Driver API")
}

// PrepackedTensor contains host payload bytes, optional scales and sparsity
// metadata, and their layout description. It does not own device allocations.
type PrepackedTensor struct {
	Arch     string
	Type     DataType
	BoxDesc  BoxDescriptor
	Payload  []byte // Host tensor storage
	Scales   []byte // Contiguous or swizzled scale factors
	Metadata []byte // 2:4 structured sparsity metadata (if sparse)
	zeroCopy bool
}

// ZeroCopy reports whether the payload was sliced directly from an input buffer without copy.
func (p *PrepackedTensor) ZeroCopy() bool {
	return p.zeroCopy
}
