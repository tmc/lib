package nccltest

import "encoding/binary"

func (d DataType) integer() bool {
	switch d {
	case Int8, Uint8, Int32, Uint32, Int64, Uint64:
		return true
	}
	return false
}

func (d DataType) signed() bool { return d == Int8 || d == Int32 || d == Int64 }

func (d DataType) mask() uint64 { return ^uint64(0) >> (64 - 8*d.Size()) }

// Each width uses all its bits, including values not representable as float64.
// Odd inputs keep modular products nonzero even in large communicators.
func integerInput(d DataType, rank, index int) uint64 {
	x := uint64(rank+1)*0x9e3779b97f4a7c15 + uint64(index)
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return ((x ^ (x >> 31)) | 1) & d.mask()
}

func getInteger(data []byte, d DataType) uint64 {
	switch d.Size() {
	case 1:
		return uint64(data[0])
	case 4:
		return uint64(binary.NativeEndian.Uint32(data))
	default:
		return binary.NativeEndian.Uint64(data)
	}
}

func putInteger(data []byte, d DataType, value uint64) {
	switch d.Size() {
	case 1:
		data[0] = byte(value)
	case 4:
		binary.NativeEndian.PutUint32(data, uint32(value))
	case 8:
		binary.NativeEndian.PutUint64(data, value)
	}
}

func expectedInteger(c Case, l Layout, ranks, rank, index int) uint64 {
	if source, element, isCopy := sourceElement(c, l, ranks, rank, index); isCopy {
		return integerInput(c.DataType, source, element)
	}
	if c.Operation == ReduceScatter {
		index += rank * l.ParamCount
	}
	d := c.DataType
	value := integerInput(d, 0, index)
	for r := 1; r < ranks; r++ {
		x := integerInput(d, r, index)
		if c.Reduction == PreMulSum {
			x = (x * uint64(1+r%2)) & d.mask()
		}
		value = integerReduce(d, c.Reduction, value, x)
	}
	if c.Reduction == Average {
		if d.signed() {
			// Sign-extend the wrapped sum before division toward zero.
			shift := 64 - 8*d.Size()
			return uint64((int64(value<<shift)>>shift)/int64(ranks)) & d.mask()
		}
		return value / uint64(ranks)
	}
	return value
}

// The reference accumulates in the element type, then divides for integer avg.
// Unsigned arithmetic makes modulo-width overflow explicit for signed types too.
func integerReduce(d DataType, op Reduction, a, b uint64) uint64 {
	switch op {
	case Product:
		return (a * b) & d.mask()
	case Minimum, Maximum:
		x, y := a, b
		if d.signed() {
			sign := uint64(1) << (8*d.Size() - 1)
			x, y = x^sign, y^sign
		}
		if (op == Minimum && y < x) || (op == Maximum && y > x) {
			return b
		}
		return a
	default:
		return (a + b) & d.mask()
	}
}

// Integers have no NaN sentinel. Complement the expected result so every
// defined output starts wrong, including when a reduction's result is zero.
func poisonIntegers(c Case, l Layout, ranks, rank int, data []byte) {
	offset := 0
	if c.InPlace {
		offset = rank * l.RecvOffset
	}
	for i := 0; i < l.RecvCount; i++ {
		putInteger(data[(offset+i)*c.DataType.Size():], c.DataType, ^expectedInteger(c, l, ranks, rank, i))
	}
}
