package dsv4cpu

import "encoding/binary"

// q4ScaleMin decodes the eight six-bit scale/min pairs in a Q4_K block.
func q4ScaleMin(scales []byte, group int) (uint8, uint8) {
	if group < 4 {
		return scales[group] & 63, scales[group+4] & 63
	}
	return (scales[group+4] & 15) | ((scales[group-4] >> 6) << 4), (scales[group+4] >> 4) | ((scales[group] >> 6) << 4)
}

// q4KDotBlock follows the scalar ds4_vec_dot_q4_K_q8_K block association.
// Its caller must validate the serialized extent and activation shape.
func q4KDotBlock(block []byte, activation Q8K) float32 {
	d := activation.Scale * half(binary.LittleEndian.Uint16(block))
	dm := -activation.Scale * half(binary.LittleEndian.Uint16(block[2:]))
	var integer, minima int32
	for group := 0; group < 8; group++ {
		scale, minimum := q4ScaleMin(block[4:16], group)
		minima += int32(minimum) * (int32(activation.Sums[group*2]) + int32(activation.Sums[group*2+1]))
		for lane := 0; lane < 32; lane++ {
			q := (block[16+(group/2)*32+lane] >> uint((group&1)*4)) & 15
			integer += int32(q) * int32(activation.Values[group*32+lane]) * int32(scale)
		}
	}
	return float32(d*float32(integer)) + float32(dm*float32(minima))
}

func q4KValue(block []byte, index int) float32 {
	group := index / 32
	scale, minimum := q4ScaleMin(block[4:16], group)
	q := (block[16+(group/2)*32+index%32] >> uint((group&1)*4)) & 15
	return float32(half(binary.LittleEndian.Uint16(block))*float32(scale)*float32(q)) - float32(half(binary.LittleEndian.Uint16(block[2:]))*float32(minimum))
}
