// SPDX-License-Identifier: Apache-2.0

package quant

import (
	"encoding/binary"
	"fmt"
)

// SuperBlockSize is the number of elements a k-quant super-block covers. It is
// QK_K in the ds4 oracle (ds4.c:842).
const SuperBlockSize = 256

// Bytes per super-block, matching the packed C layouts asserted at ds4.c:888
// (block_q2_K, 84 bytes) and ds4.c:893 (block_iq2_xxs, 66 bytes).
const (
	blockBytesQ2_K    = 84
	blockBytesIQ2_XXS = 66
)

const (
	FormatQ2_K    QuantFormat = "Q2_K"    // 2-bit k-quant, 16 scale/min pairs per 256 elements
	FormatIQ2_XXS QuantFormat = "IQ2_XXS" // 2.0625-bit codebook quant, 256 elements per block
)

// BlockQ2_K is one Q2_K super-block, laid out as block_q2_K in ds4.c:846.
// Each of the 16 groups of 16 elements has a packed 4-bit scale and 4-bit
// minimum in Scales; Quants holds 256 two-bit values.
type BlockQ2_K struct {
	Scales   [SuperBlockSize / 16]byte // low nibble scales the quant, high nibble scales the minimum
	Quants   [SuperBlockSize / 4]byte  // 256 2-bit quants
	Scale    uint16                    // float16 scale for the quants (d)
	ScaleMin uint16                    // float16 scale for the minimums (dmin)
}

// BlockIQ2_XXS is one IQ2_XXS super-block, laid out as block_iq2_xxs in
// ds4.c:880. Quants holds eight 4-word groups; each group covers 32 elements
// as four codebook indices plus a packed word of sign indices and a scale.
type BlockIQ2_XXS struct {
	Scale  uint16                     // float16 scale for the whole super-block (d)
	Quants [SuperBlockSize / 8]uint16 // 8 groups of 4 words
}

// DequantizeQ2_K unpacks numElements values from Q2_K super-blocks.
// numElements must be a multiple of [SuperBlockSize].
//
// The arithmetic mirrors q2_k_value_f32 (ds4.c:3608): each element is
// d*(scale&0xf)*q - dmin*(scale>>4).
func DequantizeQ2_K(data []byte, numElements int) ([]float32, error) {
	if numElements < 0 || numElements%SuperBlockSize != 0 {
		return nil, fmt.Errorf("quant: %d elements is not a multiple of the %d-element super-block", numElements, SuperBlockSize)
	}
	numBlocks := numElements / SuperBlockSize
	if len(data) < numBlocks*blockBytesQ2_K {
		return nil, fmt.Errorf("quant: buffer length %d insufficient for %d Q2_K blocks", len(data), numBlocks)
	}

	out := make([]float32, numElements)
	for b := range numBlocks {
		block := data[b*blockBytesQ2_K : (b+1)*blockBytesQ2_K]
		scales := block[:SuperBlockSize/16]
		quants := block[SuperBlockSize/16 : SuperBlockSize/16+SuperBlockSize/4]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[80:82]))
		dmin := Float16ToFloat32(binary.LittleEndian.Uint16(block[82:84]))

		for i := range SuperBlockSize {
			group := i / 16
			l := i - group*16
			base := 32*(group/8) + 16*(group&1)
			shift := (group / 2 & 3) * 2
			q := quants[base+l] >> shift & 0x03
			sc := scales[group]
			out[b*SuperBlockSize+i] = d*float32(sc&0x0f)*float32(q) - dmin*float32(sc>>4)
		}
	}
	return out, nil
}

// DequantizeIQ2_XXS unpacks numElements values from IQ2_XXS super-blocks.
// numElements must be a multiple of [SuperBlockSize].
//
// Each 32-element group holds two little-endian uint32 words. The first
// supplies four codebook indices into iq2xxsGrid; the second supplies four
// 7-bit sign indices into iq2xxsSigns and, in its top four bits, the group
// scale. The arithmetic mirrors ds4_vec_dot_iq2_xxs_f32 (ds4.c:3950), whose
// per-group scale is 0.125*d*(2*(aux32[1]>>28)+1).
func DequantizeIQ2_XXS(data []byte, numElements int) ([]float32, error) {
	if numElements < 0 || numElements%SuperBlockSize != 0 {
		return nil, fmt.Errorf("quant: %d elements is not a multiple of the %d-element super-block", numElements, SuperBlockSize)
	}
	numBlocks := numElements / SuperBlockSize
	if len(data) < numBlocks*blockBytesIQ2_XXS {
		return nil, fmt.Errorf("quant: buffer length %d insufficient for %d IQ2_XXS blocks", len(data), numBlocks)
	}

	out := make([]float32, numElements)
	for b := range numBlocks {
		block := data[b*blockBytesIQ2_XXS : (b+1)*blockBytesIQ2_XXS]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[0:2]))
		qs := block[2:]

		for group := range SuperBlockSize / 32 {
			g := qs[group*8 : group*8+8]
			indices := binary.LittleEndian.Uint32(g[0:4])
			aux := binary.LittleEndian.Uint32(g[4:8])
			scale := 0.125 * d * float32(2*(aux>>28)+1)

			for l := range 4 {
				code := iq2xxsGrid[byte(indices>>(8*l))]
				signs := iq2xxsSigns[aux>>(7*l)&127]
				for j := range 8 {
					v := float32(byte(code >> (8 * j)))
					if signs>>j&1 != 0 {
						v = -v
					}
					out[b*SuperBlockSize+group*32+l*8+j] = scale * v
				}
			}
		}
	}
	return out, nil
}
