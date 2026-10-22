package dsv4cpu

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// MulQ8K evaluates routed Q2_K, Q4_K or IQ2_XXS weights against native Q8_K
// activations, retaining upstream scalar block accumulation order.
func (m *Matrix) MulQ8K(dst []float32, x []Q8K) error {
	if m == nil || m.source == nil || (m.encoding != Q2K && m.encoding != Q4K && m.encoding != IQ2XXS) || len(dst) != m.rows || m.cols%256 != 0 || len(x) != m.cols/256 {
		return errors.New("invalid Q8_K expert multiplication shape")
	}
	row := make([]byte, m.rowBytes)
	for r := range dst {
		n, err := m.source.ReadAt(row, m.offset+int64(r)*int64(m.rowBytes))
		if err == nil && n != len(row) {
			err = io.ErrUnexpectedEOF
		}
		if err != nil {
			return fmt.Errorf("read expert row %d: %w", r, err)
		}
		var sum float32
		for b, activation := range x {
			if m.encoding == Q4K {
				sum += q4KDotBlock(row[b*144:], activation)
				continue
			}
			if m.encoding == Q2K {
				q := row[b*84:]
				var isum, summs int32
				for j := 0; j < 256; j++ {
					group, l := j/16, j%16
					base := 32*(group/8) + 16*(group&1)
					v := (q[16+base+l] >> uint(((group/2)&3)*2)) & 3
					isum += int32(v) * int32(activation.Values[j]) * int32(q[group]&15)
				}
				for j, v := range activation.Sums {
					summs += int32(v) * int32(q[j]>>4)
				}
				dall := activation.Scale * half(binary.LittleEndian.Uint16(q[80:]))
				dmin := activation.Scale * half(binary.LittleEndian.Uint16(q[82:]))
				sum += float32(dall*float32(isum)) - float32(dmin*float32(summs))
			} else {
				q := row[b*66:]
				var bsum int32
				for group := 0; group < 8; group++ {
					codes := q[2+group*8:]
					aux := binary.LittleEndian.Uint32(codes[4:])
					var sumi int32
					for lane := 0; lane < 4; lane++ {
						grid := iq2xxs_grid[codes[lane]]
						signs := ksigns_iq2xs[(aux>>uint(7*lane))&127]
						for j := 0; j < 8; j++ {
							v := int32(byte(grid >> uint(j*8)))
							if signs&(1<<uint(j)) != 0 {
								v = -v
							}
							sumi += v * int32(activation.Values[group*32+lane*8+j])
						}
					}
					bsum += sumi * int32(2*(aux>>28)+1)
				}
				d := half(binary.LittleEndian.Uint16(q)) * activation.Scale
				sum += float32(d * float32(bsum))
			}
		}
		if m.encoding == IQ2XXS {
			sum *= .125
		}
		dst[r] = sum
	}
	return nil
}
