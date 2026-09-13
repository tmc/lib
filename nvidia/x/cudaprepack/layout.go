// SPDX-License-Identifier: Apache-2.0

package cudaprepack

import (
	"fmt"
)

// Option configures a LayoutTransformer.
type Option func(*LayoutTransformer)

// WithBoxDims sets the desired TMA box tile dimensions [box0, box1, box2].
func WithBoxDims(box0, box1, box2 uint32) Option {
	return func(lt *LayoutTransformer) {
		lt.boxDims = [3]uint32{box0, box1, box2}
		lt.customBox = true
	}
}

// WithSwizzle configures the shared memory bank swizzle mode.
func WithSwizzle(mode SwizzleMode) Option {
	return func(lt *LayoutTransformer) {
		lt.swizzle = mode
	}
}

// WithInterleave configures the tensor byte interleaving mode.
func WithInterleave(mode InterleaveMode) Option {
	return func(lt *LayoutTransformer) {
		lt.interleave = mode
	}
}

// WithAlignment sets the byte alignment for tensor strides (default: 16).
func WithAlignment(alignBytes int) Option {
	return func(lt *LayoutTransformer) {
		if alignBytes >= MinAddressAlignment && alignBytes%MinAddressAlignment == 0 {
			lt.alignBytes = alignBytes
		}
	}
}

// WithBaseAddress sets the mock or actual GPU global base address (must be 16-byte aligned).
func WithBaseAddress(addr uint64) Option {
	return func(lt *LayoutTransformer) {
		lt.baseAddress = addr
	}
}

// LayoutTransformer prepares tensor layouts into hardware-aligned TMA 2D/3D boxes.
type LayoutTransformer struct {
	arch        string
	dataType    DataType
	boxDims     [3]uint32
	customBox   bool
	swizzle     SwizzleMode
	interleave  InterleaveMode
	alignBytes  int
	baseAddress uint64
}

// NewLayoutTransformer creates a transformer for the target architecture and data type.
func NewLayoutTransformer(arch string, dt DataType, opts ...Option) (*LayoutTransformer, error) {
	if arch != ArchSM121a && arch != ArchSM100a {
		return nil, fmt.Errorf("cudaprepack: unsupported architecture %q", arch)
	}

	lt := &LayoutTransformer{
		arch:       arch,
		dataType:   dt,
		alignBytes: MinAddressAlignment,
	}

	for _, opt := range opts {
		opt(lt)
	}

	if arch == ArchSM121a && lt.swizzle == Swizzle128BAtom32B {
		return nil, fmt.Errorf("cudaprepack: sm_121a rejects tcgen05 swizzle mode %s", lt.swizzle)
	}
	if lt.baseAddress%MinAddressAlignment != 0 {
		return nil, fmt.Errorf("cudaprepack: base address 0x%x not 16-byte aligned", lt.baseAddress)
	}

	return lt, nil
}

// Transform2D transforms a 2D matrix into hardware-aligned TMA 2D tensor boxes.
// It pads row strides to satisfy 16-byte address alignment and 128-byte box boundary constraints.
func (lt *LayoutTransformer) Transform2D(src []byte, rows, cols int) (*PrepackedTensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("cudaprepack: rows (%d) and cols (%d) must be positive", rows, cols)
	}

	bitsPerElem := lt.dataType.BitsPerElement()
	rawRowBytes := (cols*bitsPerElem + 7) / 8
	expectedLen := rows * rawRowBytes
	if len(src) < expectedLen {
		return nil, fmt.Errorf("cudaprepack: source buffer too short (got %d bytes, want at least %d)", len(src), expectedLen)
	}

	// Align row stride to alignBytes (default 16 bytes, can be 128 bytes)
	strideAlign := lt.alignBytes
	if strideAlign < MinAddressAlignment {
		strideAlign = MinAddressAlignment
	}
	paddedRowStride := ((rawRowBytes + strideAlign - 1) / strideAlign) * strideAlign

	// Determine TMA box dimensions
	boxDims := lt.boxDims
	if !lt.customBox {
		// Box width in bytes must be a multiple of 16 bytes and <= 128 bytes for sm_121a
		boxWidthBytes := 128
		if rawRowBytes < 128 {
			boxWidthBytes = ((rawRowBytes + 15) / 16) * 16
		}
		if lt.arch == ArchSM121a && boxWidthBytes > MaxBoxWidthBytesSM121a {
			boxWidthBytes = MaxBoxWidthBytesSM121a
		}

		elemPerBox0 := (boxWidthBytes * 8) / bitsPerElem
		box0 := uint32(elemPerBox0)

		box1 := uint32(32)
		if box1 > uint32(rows) {
			box1 = uint32(rows)
		}
		boxDims = [3]uint32{box0, box1, 1}
	}

	// Allocate contiguous padded buffer
	paddedBuf := make([]byte, rows*paddedRowStride)
	for r := 0; r < rows; r++ {
		srcOffset := r * rawRowBytes
		dstOffset := r * paddedRowStride
		copy(paddedBuf[dstOffset:dstOffset+rawRowBytes], src[srcOffset:srcOffset+rawRowBytes])
	}

	desc := BoxDescriptor{
		Rank:           2,
		DataType:       lt.dataType,
		GlobalAddress:  lt.baseAddress,
		GlobalDims:     [3]uint64{uint64(cols), uint64(rows), 1},
		GlobalStrides:  [3]int64{int64(paddedRowStride), 0, 0},
		BoxDims:        boxDims,
		ElementStrides: [3]uint32{1, 1, 1},
		Swizzle:        lt.swizzle,
		Interleave:     lt.interleave,
	}

	if err := desc.Validate(lt.arch); err != nil {
		return nil, fmt.Errorf("cudaprepack: layout validation failed: %w", err)
	}

	return &PrepackedTensor{
		Arch:    lt.arch,
		Type:    lt.dataType,
		BoxDesc: desc,
		Payload: paddedBuf,
	}, nil
}

// Transform3D transforms a 3D volume into hardware-aligned TMA 3D tensor boxes.
func (lt *LayoutTransformer) Transform3D(src []byte, depth, rows, cols int) (*PrepackedTensor, error) {
	if depth <= 0 || rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("cudaprepack: depth, rows, cols must be positive")
	}

	bitsPerElem := lt.dataType.BitsPerElement()
	rawRowBytes := (cols*bitsPerElem + 7) / 8
	strideAlign := lt.alignBytes
	if strideAlign < MinAddressAlignment {
		strideAlign = MinAddressAlignment
	}
	paddedRowStride := ((rawRowBytes + strideAlign - 1) / strideAlign) * strideAlign
	rawPlaneBytes := rows * paddedRowStride
	paddedPlaneStride := ((rawPlaneBytes + strideAlign - 1) / strideAlign) * strideAlign

	expectedLen := depth * rows * rawRowBytes
	if len(src) < expectedLen {
		return nil, fmt.Errorf("cudaprepack: source buffer too short (got %d bytes, want at least %d)", len(src), expectedLen)
	}

	boxDims := lt.boxDims
	if !lt.customBox {
		boxWidthBytes := 128
		if rawRowBytes < 128 {
			boxWidthBytes = ((rawRowBytes + 15) / 16) * 16
		}
		if lt.arch == ArchSM121a && boxWidthBytes > MaxBoxWidthBytesSM121a {
			boxWidthBytes = MaxBoxWidthBytesSM121a
		}
		box0 := uint32((boxWidthBytes * 8) / bitsPerElem)
		box1 := uint32(16)
		if box1 > uint32(rows) {
			box1 = uint32(rows)
		}
		box2 := uint32(1)
		boxDims = [3]uint32{box0, box1, box2}
	}

	paddedBuf := make([]byte, depth*paddedPlaneStride)
	srcIdx := 0
	for d := 0; d < depth; d++ {
		planeOffset := d * paddedPlaneStride
		for r := 0; r < rows; r++ {
			rowOffset := planeOffset + r*paddedRowStride
			copy(paddedBuf[rowOffset:rowOffset+rawRowBytes], src[srcIdx:srcIdx+rawRowBytes])
			srcIdx += rawRowBytes
		}
	}

	desc := BoxDescriptor{
		Rank:           3,
		DataType:       lt.dataType,
		GlobalAddress:  lt.baseAddress,
		GlobalDims:     [3]uint64{uint64(cols), uint64(rows), uint64(depth)},
		GlobalStrides:  [3]int64{int64(paddedRowStride), int64(paddedPlaneStride), 0},
		BoxDims:        boxDims,
		ElementStrides: [3]uint32{1, 1, 1},
		Swizzle:        lt.swizzle,
		Interleave:     lt.interleave,
	}

	if err := desc.Validate(lt.arch); err != nil {
		return nil, fmt.Errorf("cudaprepack: 3D layout validation failed: %w", err)
	}

	return &PrepackedTensor{
		Arch:    lt.arch,
		Type:    lt.dataType,
		BoxDesc: desc,
		Payload: paddedBuf,
	}, nil
}
