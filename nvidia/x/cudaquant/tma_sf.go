// SPDX-License-Identifier: Apache-2.0

package cudaquant

import (
	"fmt"
	"math"
)

// AlignedScaleRows pads rows up to a multiple of alignBytes (defaulting to 16
// when alignBytes == 0, matching 16-byte TMA descriptor alignment for packed
// UE8M0 scale columns).
func AlignedScaleRows(rows int, alignBytes int) (int, error) {
	if rows <= 0 {
		return 0, fmt.Errorf("cudaquant: rows (%d) must be positive", rows)
	}
	if alignBytes < 0 {
		return 0, fmt.Errorf("cudaquant: alignBytes (%d) must be non-negative", alignBytes)
	}
	if alignBytes == 0 {
		alignBytes = 16
	}
	rem := rows % alignBytes
	if rem == 0 {
		return rows, nil
	}
	pad := alignBytes - rem
	if rows > math.MaxInt-pad {
		return 0, fmt.Errorf("cudaquant: aligned rows overflow for rows=%d alignBytes=%d", rows, alignBytes)
	}
	return rows + pad, nil
}

// AlignedOutputStride returns the smallest element row stride >= cols whose
// byte width (stride * elemBytes) is a multiple of alignBytes (defaulting to
// 32 bytes when alignBytes == 0, matching DeepInfer's _OUTPUT_ALIGNMENT_BYTES).
func AlignedOutputStride(cols int, elemBytes int, alignBytes int) (int, error) {
	if cols <= 0 {
		return 0, fmt.Errorf("cudaquant: cols (%d) must be positive", cols)
	}
	if elemBytes <= 0 {
		return 0, fmt.Errorf("cudaquant: elemBytes (%d) must be positive", elemBytes)
	}
	if alignBytes < 0 {
		return 0, fmt.Errorf("cudaquant: alignBytes (%d) must be non-negative", alignBytes)
	}
	if alignBytes == 0 {
		alignBytes = 32
	}
	g := gcdInt(alignBytes, elemBytes)
	elemAlign := alignBytes / g
	rem := cols % elemAlign
	if rem == 0 {
		return cols, nil
	}
	pad := elemAlign - rem
	if cols > math.MaxInt-pad {
		return 0, fmt.Errorf("cudaquant: aligned output stride overflow for cols=%d elemBytes=%d alignBytes=%d", cols, elemBytes, alignBytes)
	}
	return cols + pad, nil
}

func gcdInt(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	if a <= 0 {
		return 1
	}
	return a
}

// TransformScaleLayout transposes a [rows, scaleCols] row-major packed scale
// matrix into a [scaleCols, alignedRows] TMA-aligned column-major layout.
func TransformScaleLayout(rowMajorScales []uint32, rows, scaleCols int, alignedRows int) ([]uint32, error) {
	return TransformScaleLayoutColMajor(rowMajorScales, rows, scaleCols, alignedRows)
}

// TransformScaleLayoutColMajor transposes a [rows, scaleCols] row-major packed
// scale matrix into a [scaleCols, alignedRows] TMA-aligned column-major layout,
// zero-filling padding rows in [rows, alignedRows).
func TransformScaleLayoutColMajor(rowMajorScales []uint32, rows, scaleCols int, alignedRows int) ([]uint32, error) {
	if rows <= 0 || scaleCols <= 0 || alignedRows < rows {
		return nil, fmt.Errorf("cudaquant: invalid scale layout dimensions rows=%d scaleCols=%d alignedRows=%d", rows, scaleCols, alignedRows)
	}
	if rows > math.MaxInt/scaleCols || alignedRows > math.MaxInt/scaleCols {
		return nil, fmt.Errorf("cudaquant: scale layout dimension overflow")
	}
	if len(rowMajorScales) != rows*scaleCols {
		return nil, fmt.Errorf("cudaquant: rowMajorScales length %d, want %d (rows=%d * scaleCols=%d)", len(rowMajorScales), rows*scaleCols, rows, scaleCols)
	}
	out := make([]uint32, scaleCols*alignedRows)
	for r := 0; r < rows; r++ {
		for c := 0; c < scaleCols; c++ {
			out[c*alignedRows+r] = rowMajorScales[r*scaleCols+c]
		}
	}
	return out, nil
}

// InverseTransformScaleLayoutColMajor converts a [scaleCols, alignedRows]
// column-major packed scale tensor back to a [rows, scaleCols] row-major slice,
// discarding any trailing padding rows in [rows, alignedRows).
func InverseTransformScaleLayoutColMajor(colMajorScales []uint32, rows, scaleCols int, alignedRows int) ([]uint32, error) {
	if rows <= 0 || scaleCols <= 0 || alignedRows < rows {
		return nil, fmt.Errorf("cudaquant: invalid scale layout dimensions rows=%d scaleCols=%d alignedRows=%d", rows, scaleCols, alignedRows)
	}
	if rows > math.MaxInt/scaleCols || alignedRows > math.MaxInt/scaleCols {
		return nil, fmt.Errorf("cudaquant: scale layout dimension overflow")
	}
	if len(colMajorScales) != scaleCols*alignedRows {
		return nil, fmt.Errorf("cudaquant: colMajorScales length %d, want %d (scaleCols=%d * alignedRows=%d)", len(colMajorScales), scaleCols*alignedRows, scaleCols, alignedRows)
	}
	out := make([]uint32, rows*scaleCols)
	for r := 0; r < rows; r++ {
		for c := 0; c < scaleCols; c++ {
			out[r*scaleCols+c] = colMajorScales[c*alignedRows+r]
		}
	}
	return out, nil
}
