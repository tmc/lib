package nccltest

import "encoding/binary"

func (d DataType) smallFormat() (floatFormat, bool) {
	switch d {
	case Float16:
		return float16Format, true
	case BFloat16:
		return bfloat16Format, true
	case Float8E4M3:
		return float8e4m3Format, true
	case Float8E5M2:
		return float8e5m2Format, true
	}
	return floatFormat{}, false
}

func getSmall(data []byte, d DataType) uint16 {
	if d.Size() == 1 {
		return uint16(data[0])
	}
	return binary.NativeEndian.Uint16(data)
}

func putSmall(data []byte, d DataType, bits uint16) {
	if d.Size() == 1 {
		data[0] = byte(bits)
	} else {
		binary.NativeEndian.PutUint16(data, bits)
	}
}

// Sparse sums and power-of-two products keep every partial result exactly
// representable, regardless of the reduction tree. The active window rotates
// with element index so every rank contributes. Copies and min/max use varied
// finite normal values over the format's range.
func smallInput(c Case, ranks, rank, index int) float64 {
	f, _ := c.DataType.smallFormat()
	h := integerInput(Uint64, rank, index)
	sign := 1.0
	if h>>63 != 0 {
		sign = -1
	}
	if c.Operation == AllReduce || c.Operation == Reduce || c.Operation == ReduceScatter {
		distance := (rank - index%ranks + ranks) % ranks
		switch c.Reduction {
		case "", Sum, PreMulSum:
			if distance < 4 {
				return sign
			}
			return 0
		case Average:
			if distance < 2 {
				return sign
			}
			return 0
		case Product:
			if distance < 4 {
				return sign * float64(1+((h>>1)&1))
			}
			return sign
		}
	}
	exponentMask := uint64(1<<f.exponentBits) - 1
	fractionMask := uint64(1<<f.fractionBits) - 1
	exponent := 1 + (h>>(f.fractionBits+1))%(exponentMask-1)
	bits := uint16((h>>1)&fractionMask) | uint16(exponent)<<f.fractionBits
	if sign < 0 {
		bits |= 1 << (f.exponentBits + f.fractionBits)
	}
	return f.decode(bits)
}

// NCCL v2.31.2-1 hostToDevRedOp converts the reciprocal to float32 before
// converting it to the collective's element type (src/enqueue/enqueue.cc).
func smallAverageScale(d DataType, ranks int) float64 {
	f, _ := d.smallFormat()
	return f.decode(f.encode(float64(float32(1.0 / float64(ranks)))))
}

func expectedSmall(c Case, l Layout, ranks, rank, index int) float64 {
	if source, element, isCopy := sourceElement(c, l, ranks, rank, index); isCopy {
		return smallInput(c, ranks, source, element)
	}
	if c.Operation == ReduceScatter {
		index += rank * l.ParamCount
	}
	scale := 1.0
	if c.Reduction == Average {
		scale = smallAverageScale(c.DataType, ranks)
	}
	var value float64
	for r := 0; r < ranks; r++ {
		x := smallInput(c, ranks, r, index) * scale
		if c.Reduction == PreMulSum {
			x *= float64(1 + r%2)
		}
		if r == 0 {
			value = x
			continue
		}
		switch c.Reduction {
		case Product:
			value *= x
		case Minimum:
			if x < value {
				value = x
			}
		case Maximum:
			if x > value {
				value = x
			}
		default:
			value += x
		}
	}
	return value
}
