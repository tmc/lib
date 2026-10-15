package cudaprepack

import (
	"fmt"
	"math"
)

// Q8Storage describes the host byte layout of Q8_0 blocks. Each block has
// 32 signed quantization bytes and a separate, unchanged two-byte scale.
// Byte counts describe conversion work and storage, not measured time.
type Q8Storage struct {
	Rows, Cols, RowStride int
	Blocks                int
	RawBytes              int
	PayloadBytes          int
	ScaleBytes            int
	WorkspaceBytes        int
}

func q8Storage(rows, cols, stride int) (Q8Storage, error) {
	if rows <= 0 || cols <= 0 || cols%32 != 0 {
		return Q8Storage{}, fmt.Errorf("cudaprepack: Q8_0 requires positive rows and columns divisible by 32")
	}
	if stride < cols || stride%MinAddressAlignment != 0 {
		return Q8Storage{}, fmt.Errorf("cudaprepack: invalid Q8_0 row stride %d", stride)
	}
	blocks, ok := checkedMul(rows, cols/32)
	if !ok {
		return Q8Storage{}, fmt.Errorf("cudaprepack: Q8_0 block count overflows")
	}
	raw, ok := checkedMul(blocks, 34)
	if !ok {
		return Q8Storage{}, fmt.Errorf("cudaprepack: Q8_0 source extent overflows")
	}
	payload, ok := checkedMul(rows, stride)
	if !ok {
		return Q8Storage{}, fmt.Errorf("cudaprepack: Q8_0 payload extent overflows")
	}
	scales, ok := checkedMul(blocks, 2)
	if !ok || payload > math.MaxInt-scales {
		return Q8Storage{}, fmt.Errorf("cudaprepack: Q8_0 workspace extent overflows")
	}
	return Q8Storage{rows, cols, stride, blocks, raw, payload, scales, payload + scales}, nil
}

// DescribeQ8 validates the row-major, separately scaled Q8_0 storage consumed
// by UnpackQ8_0. It does not validate a CUDA tensor map or own device memory.
// Swizzled, interleaved and sparse host payloads are unsupported.
func DescribeQ8(p *PrepackedTensor) (Q8Storage, error) {
	if p == nil {
		return Q8Storage{}, fmt.Errorf("cudaprepack: nil Q8_0 tensor")
	}
	b := p.BoxDesc
	if p.Type != TypeQ8_0 || b.DataType != TypeQ8_0 || b.Rank != 2 || b.GlobalDims[2] != 1 {
		return Q8Storage{}, fmt.Errorf("cudaprepack: invalid Q8_0 tensor dimensions or type")
	}
	if b.Swizzle != SwizzleNone || b.Interleave != InterleaveNone || b.ElementStrides != [3]uint32{1, 1, 1} || b.GlobalStrides[1] != 0 || b.GlobalStrides[2] != 0 || len(p.Metadata) != 0 {
		return Q8Storage{}, fmt.Errorf("cudaprepack: unsupported Q8_0 host layout")
	}
	if b.GlobalDims[0] > uint64(math.MaxInt) || b.GlobalDims[1] > uint64(math.MaxInt) || b.GlobalStrides[0] <= 0 || uint64(b.GlobalStrides[0]) > uint64(math.MaxInt) {
		return Q8Storage{}, fmt.Errorf("cudaprepack: Q8_0 extent cannot be represented")
	}
	s, err := q8Storage(int(b.GlobalDims[1]), int(b.GlobalDims[0]), int(b.GlobalStrides[0]))
	if err != nil {
		return Q8Storage{}, err
	}
	if len(p.Payload) < s.PayloadBytes || len(p.Scales) < s.ScaleBytes {
		return Q8Storage{}, fmt.Errorf("cudaprepack: Q8_0 payload or scales are truncated")
	}
	return s, nil
}
