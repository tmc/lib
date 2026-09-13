// SPDX-License-Identifier: Apache-2.0

package cudaprepack

import (
	"fmt"
	"math"
)

// Compress2to4 compresses a dense float32 matrix into 2:4 structured sparsity
// format and generates the corresponding metadata for sm_100a tcgen05.mma.ws.sp.
// For every 4 consecutive elements along cols, the 2 elements with the largest
// absolute values are preserved and the other 2 are zeroed out.
func Compress2to4(dense []float32, rows, cols int) (sparseWeights []float32, metadata []byte, err error) {
	if rows <= 0 || cols <= 0 {
		return nil, nil, fmt.Errorf("cudaprepack: rows (%d) and cols (%d) must be positive", rows, cols)
	}
	if cols%4 != 0 {
		return nil, nil, fmt.Errorf("cudaprepack: cols (%d) must be a multiple of 4 for 2:4 structured sparsity", cols)
	}
	if len(dense) < rows*cols {
		return nil, nil, fmt.Errorf("cudaprepack: dense slice too short (got %d, want %d)", len(dense), rows*cols)
	}

	sparseCols := cols / 2
	sparseWeights = make([]float32, rows*sparseCols)
	metaBytesPerRow := (cols + 7) / 8 // 1 byte per 8 elements (two 4-element groups)
	metadata = make([]byte, rows*metaBytesPerRow)

	sparseIdx := 0
	metaIdx := 0

	for r := 0; r < rows; r++ {
		rowOffset := r * cols
		for c := 0; c < cols; c += 8 {
			var metaByte byte

			// Process first group of 4
			g0Offset := rowOffset + c
			i0_0, i0_1 := selectTop2(
				dense[g0Offset],
				dense[g0Offset+1],
				dense[g0Offset+2],
				dense[g0Offset+3],
			)
			sparseWeights[sparseIdx] = dense[g0Offset+i0_0]
			sparseWeights[sparseIdx+1] = dense[g0Offset+i0_1]
			sparseIdx += 2

			nibble0 := byte((i0_0 & 0x03) | ((i0_1 & 0x03) << 2))
			metaByte |= nibble0

			// Process second group of 4 (if present within row)
			if c+4 < cols {
				g1Offset := rowOffset + c + 4
				i1_0, i1_1 := selectTop2(
					dense[g1Offset],
					dense[g1Offset+1],
					dense[g1Offset+2],
					dense[g1Offset+3],
				)
				sparseWeights[sparseIdx] = dense[g1Offset+i1_0]
				sparseWeights[sparseIdx+1] = dense[g1Offset+i1_1]
				sparseIdx += 2

				nibble1 := byte((i1_0 & 0x03) | ((i1_1 & 0x03) << 2))
				metaByte |= (nibble1 << 4)
			}

			metadata[metaIdx] = metaByte
			metaIdx++
		}
	}

	return sparseWeights, metadata, nil
}

// Decompress2to4 reconstructs the dense float32 matrix from 2:4 sparse weights and metadata.
func Decompress2to4(sparseWeights []float32, metadata []byte, rows, cols int) ([]float32, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("cudaprepack: rows (%d) and cols (%d) must be positive", rows, cols)
	}
	if cols%4 != 0 {
		return nil, fmt.Errorf("cudaprepack: cols (%d) must be a multiple of 4", cols)
	}

	sparseCols := cols / 2
	expectedSparseLen := rows * sparseCols
	if len(sparseWeights) < expectedSparseLen {
		return nil, fmt.Errorf("cudaprepack: sparseWeights too short (got %d, want %d)", len(sparseWeights), expectedSparseLen)
	}
	metaBytesPerRow := (cols + 7) / 8
	expectedMetaLen := rows * metaBytesPerRow
	if len(metadata) < expectedMetaLen {
		return nil, fmt.Errorf("cudaprepack: metadata too short (got %d, want %d)", len(metadata), expectedMetaLen)
	}

	dense := make([]float32, rows*cols)
	sparseIdx := 0
	metaIdx := 0

	for r := 0; r < rows; r++ {
		rowOffset := r * cols
		for c := 0; c < cols; c += 8 {
			metaByte := metadata[metaIdx]
			metaIdx++

			// First group of 4
			nibble0 := metaByte & 0x0F
			idx0_0 := int(nibble0 & 0x03)
			idx0_1 := int((nibble0 >> 2) & 0x03)

			dense[rowOffset+c+idx0_0] = sparseWeights[sparseIdx]
			dense[rowOffset+c+idx0_1] = sparseWeights[sparseIdx+1]
			sparseIdx += 2

			// Second group of 4
			if c+4 < cols {
				nibble1 := (metaByte >> 4) & 0x0F
				idx1_0 := int(nibble1 & 0x03)
				idx1_1 := int((nibble1 >> 2) & 0x03)

				dense[rowOffset+c+4+idx1_0] = sparseWeights[sparseIdx]
				dense[rowOffset+c+4+idx1_1] = sparseWeights[sparseIdx+1]
				sparseIdx += 2
			}
		}
	}

	return dense, nil
}

// Compress2to4Bytes compresses a dense byte slice (e.g. FP8 or INT8) into 2:4 structured sparsity.
func Compress2to4Bytes(dense []byte, rows, cols int) (sparseWeights []byte, metadata []byte, err error) {
	if rows <= 0 || cols <= 0 {
		return nil, nil, fmt.Errorf("cudaprepack: rows (%d) and cols (%d) must be positive", rows, cols)
	}
	if cols%4 != 0 {
		return nil, nil, fmt.Errorf("cudaprepack: cols (%d) must be a multiple of 4", cols)
	}
	if len(dense) < rows*cols {
		return nil, nil, fmt.Errorf("cudaprepack: dense slice too short")
	}

	sparseCols := cols / 2
	sparseWeights = make([]byte, rows*sparseCols)
	metaBytesPerRow := (cols + 7) / 8
	metadata = make([]byte, rows*metaBytesPerRow)

	sparseIdx := 0
	metaIdx := 0

	for r := 0; r < rows; r++ {
		rowOffset := r * cols
		for c := 0; c < cols; c += 8 {
			var metaByte byte

			g0Offset := rowOffset + c
			i0_0, i0_1 := selectTop2Byte(
				dense[g0Offset],
				dense[g0Offset+1],
				dense[g0Offset+2],
				dense[g0Offset+3],
			)
			sparseWeights[sparseIdx] = dense[g0Offset+i0_0]
			sparseWeights[sparseIdx+1] = dense[g0Offset+i0_1]
			sparseIdx += 2
			metaByte |= byte((i0_0 & 0x03) | ((i0_1 & 0x03) << 2))

			if c+4 < cols {
				g1Offset := rowOffset + c + 4
				i1_0, i1_1 := selectTop2Byte(
					dense[g1Offset],
					dense[g1Offset+1],
					dense[g1Offset+2],
					dense[g1Offset+3],
				)
				sparseWeights[sparseIdx] = dense[g1Offset+i1_0]
				sparseWeights[sparseIdx+1] = dense[g1Offset+i1_1]
				sparseIdx += 2
				metaByte |= byte(((i1_0 & 0x03) | ((i1_1 & 0x03) << 2)) << 4)
			}

			metadata[metaIdx] = metaByte
			metaIdx++
		}
	}

	return sparseWeights, metadata, nil
}

// Decompress2to4Bytes reconstructs a dense byte slice from 2:4 sparse bytes and metadata.
func Decompress2to4Bytes(sparseWeights []byte, metadata []byte, rows, cols int) ([]byte, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("cudaprepack: rows (%d) and cols (%d) must be positive", rows, cols)
	}
	if cols%4 != 0 {
		return nil, fmt.Errorf("cudaprepack: cols (%d) must be a multiple of 4", cols)
	}
	dense := make([]byte, rows*cols)
	sparseIdx := 0
	metaIdx := 0

	for r := 0; r < rows; r++ {
		rowOffset := r * cols
		for c := 0; c < cols; c += 8 {
			metaByte := metadata[metaIdx]
			metaIdx++

			nibble0 := metaByte & 0x0F
			idx0_0 := int(nibble0 & 0x03)
			idx0_1 := int((nibble0 >> 2) & 0x03)
			dense[rowOffset+c+idx0_0] = sparseWeights[sparseIdx]
			dense[rowOffset+c+idx0_1] = sparseWeights[sparseIdx+1]
			sparseIdx += 2

			if c+4 < cols {
				nibble1 := (metaByte >> 4) & 0x0F
				idx1_0 := int(nibble1 & 0x03)
				idx1_1 := int((nibble1 >> 2) & 0x03)
				dense[rowOffset+c+4+idx1_0] = sparseWeights[sparseIdx]
				dense[rowOffset+c+4+idx1_1] = sparseWeights[sparseIdx+1]
				sparseIdx += 2
			}
		}
	}
	return dense, nil
}

func selectTop2(v0, v1, v2, v3 float32) (int, int) {
	type item struct {
		val float64
		idx int
	}
	items := [4]item{
		{math.Abs(float64(v0)), 0},
		{math.Abs(float64(v1)), 1},
		{math.Abs(float64(v2)), 2},
		{math.Abs(float64(v3)), 3},
	}

	// Simple selection sort
	for i := 0; i < 2; i++ {
		maxIdx := i
		for j := i + 1; j < 4; j++ {
			if items[j].val > items[maxIdx].val {
				maxIdx = j
			}
		}
		items[i], items[maxIdx] = items[maxIdx], items[i]
	}

	idx0 := items[0].idx
	idx1 := items[1].idx
	if idx0 > idx1 {
		idx0, idx1 = idx1, idx0
	}
	return idx0, idx1
}

func selectTop2Byte(v0, v1, v2, v3 byte) (int, int) {
	type item struct {
		val int
		idx int
	}
	items := [4]item{
		{int(v0), 0},
		{int(v1), 1},
		{int(v2), 2},
		{int(v3), 3},
	}

	for i := 0; i < 2; i++ {
		maxIdx := i
		for j := i + 1; j < 4; j++ {
			if items[j].val > items[maxIdx].val {
				maxIdx = j
			}
		}
		items[i], items[maxIdx] = items[maxIdx], items[i]
	}

	idx0 := items[0].idx
	idx1 := items[1].idx
	if idx0 > idx1 {
		idx0, idx1 = idx1, idx0
	}
	return idx0, idx1
}

// SwizzleScales swizzles a 2D scale factor buffer for sm_100a tcgen05 shared memory bank alignment.
func SwizzleScales(scales []byte, rows, cols int, mode SwizzleMode) ([]byte, error) {
	if mode == SwizzleNone {
		out := make([]byte, len(scales))
		copy(out, scales)
		return out, nil
	}
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("cudaprepack: rows (%d) and cols (%d) must be positive", rows, cols)
	}
	if len(scales) < rows*cols {
		return nil, fmt.Errorf("cudaprepack: scales buffer too short")
	}

	out := make([]byte, rows*cols)

	for r := 0; r < rows; r++ {
		rowOffset := r * cols
		for c := 0; c < cols; c++ {
			var swizzledCol int
			switch mode {
			case Swizzle32B:
				// 32B swizzle: 2 chunks of 16B in 32B period
				chunk := (c / 16) % 2
				swizzledChunk := chunk ^ (r & 0x01)
				swizzledCol = (c / 32)*32 + swizzledChunk*16 + (c % 16)
			case Swizzle64B:
				// 64B swizzle: 4 chunks of 16B in 64B period
				chunk := (c / 16) % 4
				swizzledChunk := chunk ^ (r & 0x03)
				swizzledCol = (c / 64)*64 + swizzledChunk*16 + (c % 16)
			case Swizzle128B:
				// 128B swizzle: 8 chunks of 16B in 128B period
				chunk := (c / 16) % 8
				swizzledChunk := chunk ^ (r & 0x07)
				swizzledCol = (c / 128)*128 + swizzledChunk*16 + (c % 16)
			case Swizzle128BAtom32B:
				// 128B atom 32B swizzle: 4 atoms of 32B in 128B period
				atom := (c / 32) % 4
				swizzledAtom := atom ^ (r & 0x03)
				swizzledCol = (c / 128)*128 + swizzledAtom*32 + (c % 32)
			default:
				return nil, fmt.Errorf("cudaprepack: unknown swizzle mode %s", mode)
			}

			if swizzledCol < cols {
				out[rowOffset+swizzledCol] = scales[rowOffset+c]
			} else {
				out[rowOffset+c] = scales[rowOffset+c]
			}
		}
	}
	return out, nil
}

// UnswizzleScales restores the natural layout of a swizzled scale factor buffer.
// Because XOR swizzling is an involution (f(f(x)) = x), unswizzling applies the same mapping.
func UnswizzleScales(swizzled []byte, rows, cols int, mode SwizzleMode) ([]byte, error) {
	return SwizzleScales(swizzled, rows, cols, mode)
}
