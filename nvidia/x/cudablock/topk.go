package cudablock

import (
	"errors"
	"fmt"
	"math"
)

// ErrTopKUnsupported reports a row count, width or K outside the initial domain.
var ErrTopKUnsupported = errors.New("cudablock: unsupported topk configuration")

// TopKConfig describes FP32 rows with contiguous columns and a stride measured
// in elements. Results are sorted by decreasing value, then increasing source
// index. Signed zeros compare equal; the selected value retains its source bits.
// Infinities are admitted. Any NaN, even in a short row, rejects the invocation.
// Unused indices are -1 and unused values are negative infinity.
type TopKConfig struct {
	Rows, Width, RowStride, K int
	Values                    bool
}

// TopKExtents gives the required element counts, excluding caller redzones.
// ScratchValues and output Values are needed only when Values is true.
type TopKExtents struct {
	Input, RowMetadata, Results int
	ScratchBytes                uint64
}

// Extents checks the configuration before allocation or native calls. The
// initial domain is 1..65535 rows, width 1..4096, and K 1..32. Length zero is
// admitted by the per-row metadata, rather than by a zero-width allocation.
func (c TopKConfig) Extents() (TopKExtents, error) {
	if c.Rows < 1 || c.Rows > 65535 || c.Width < 1 || c.Width > 4096 || c.K < 1 || c.K > 32 {
		return TopKExtents{}, ErrTopKUnsupported
	}
	if c.RowStride < c.Width {
		return TopKExtents{}, fmt.Errorf("cudablock: topk row stride is smaller than width")
	}
	// Include only the live part of the final row, not its trailing padding.
	maxElements := int(^uint(0)>>1) / 4
	if c.Width > maxElements || c.Rows > 1 && c.RowStride > (maxElements-c.Width)/(c.Rows-1) {
		return TopKExtents{}, fmt.Errorf("cudablock: topk input extent overflows")
	}
	n := c.Rows * c.K
	bytes := uint64(n)*4 + uint64(c.Rows)*8
	if c.Values {
		bytes += uint64(n) * 4
	}
	return TopKExtents{Input: (c.Rows-1)*c.RowStride + c.Width, RowMetadata: c.Rows, Results: n, ScratchBytes: bytes}, nil
}

// TopKStatus is a per-row device validation result. Only TopKOK admits output.
type TopKStatus int32

const (
	TopKOK TopKStatus = iota
	TopKNaN
	TopKBadLength
	TopKBadOffset
)

// CheckTopKStatus refuses any nonzero or unknown device status. An empty table
// is missing evidence, not successful execution.
func CheckTopKStatus(status []int32) error {
	if len(status) == 0 {
		return fmt.Errorf("cudablock: missing topk device status")
	}
	for row, s := range status {
		if TopKStatus(s) != TopKOK {
			return fmt.Errorf("cudablock: topk row %d device status %d", row, s)
		}
	}
	return nil
}

func topkCheckRow(length, offset int32, width int) error {
	if length < 0 || int64(length) > int64(width) {
		return fmt.Errorf("cudablock: topk length out of range")
	}
	if offset < 0 || length > 0 && int64(offset)+int64(length)-1 > math.MaxInt32 {
		return fmt.Errorf("cudablock: topk index offset overflows")
	}
	return nil
}
