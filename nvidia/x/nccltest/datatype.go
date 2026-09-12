package nccltest

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/nccl"
)

// DataType names an element representation using nccl-tests spellings.
// The zero value selects Float32.
type DataType string

const (
	Int8       DataType = "int8"
	Uint8      DataType = "uint8"
	Int32      DataType = "int32"
	Uint32     DataType = "uint32"
	Int64      DataType = "int64"
	Uint64     DataType = "uint64"
	Float16    DataType = "half"
	BFloat16   DataType = "bfloat16"
	Float8E4M3 DataType = "f8e4m3"
	Float8E5M2 DataType = "f8e5m2"
	Float32    DataType = "float"
	Float64    DataType = "double"
)

// Size returns the number of bytes per element, or zero for an unknown type.
func (d DataType) Size() int {
	switch d {
	case Int8, Uint8, Float8E4M3, Float8E5M2:
		return 1
	case Float16, BFloat16:
		return 2
	case "", Float32, Int32, Uint32:
		return 4
	case Float64, Int64, Uint64:
		return 8
	}
	return 0
}

// native is called only after Layout has validated the datatype.
func (d DataType) native() nccl.DataType {
	switch d {
	case Int8:
		return nccl.Int8
	case Uint8:
		return nccl.Uint8
	case Int32:
		return nccl.Int32
	case Uint32:
		return nccl.Uint32
	case Int64:
		return nccl.Int64
	case Uint64:
		return nccl.Uint64
	case Float16:
		return nccl.Float16
	case BFloat16:
		return nccl.Bfloat16
	case Float8E4M3:
		return nccl.Float8e4m3
	case Float8E5M2:
		return nccl.Float8e5m2
	case Float64:
		return nccl.Float64
	}
	return nccl.Float32
}

func inputBuffer(c Case, ranks, rank, count int) []byte {
	data := make([]byte, count*c.DataType.Size())
	for i := 0; i < count; i++ {
		if f, ok := c.DataType.smallFormat(); ok {
			putSmall(data[i*c.DataType.Size():], c.DataType, f.encode(smallInput(c, ranks, rank, i)))
		} else if c.DataType.integer() {
			putInteger(data[i*c.DataType.Size():], c.DataType, integerInput(c.DataType, rank, i))
		} else if c.DataType == Float64 {
			binary.NativeEndian.PutUint64(data[i*8:], math.Float64bits(inputValue[float64](c, rank, i)))
		} else {
			binary.NativeEndian.PutUint32(data[i*4:], math.Float32bits(inputValue[float32](c, rank, i)))
		}
	}
	return data
}

func poisonBuffer(c Case, count int) []byte {
	data := make([]byte, count*c.DataType.Size())
	// All-one bit patterns are NaNs in every supported floating format.
	for i := range data {
		data[i] = 0xff
	}
	return data
}

// CheckBytes checks a native-endian receive buffer for any supported datatype.
// Integer elements are compared bit for bit without floating-point conversion.
// It follows Check's rules for undefined non-root outputs.
func CheckBytes(c Case, ranks, rank int, data []byte) (int, int, error) {
	l, err := c.Layout(ranks)
	if err != nil {
		return 0, 0, err
	}
	if rank < 0 || rank >= ranks {
		return 0, 0, fmt.Errorf("nccltest: invalid rank %d", rank)
	}
	if (c.Operation == Reduce || c.Operation == Gather) && rank != c.Root {
		return 0, 0, nil
	}
	if len(data) != l.RecvCount*c.DataType.Size() {
		return 0, 0, fmt.Errorf("nccltest: received %d bytes, want %d", len(data), l.RecvCount*c.DataType.Size())
	}
	if c.DataType.integer() {
		wrong := 0
		for i := 0; i < l.RecvCount; i++ {
			if getInteger(data[i*c.DataType.Size():], c.DataType) != expectedInteger(c, l, ranks, rank, i) {
				wrong++
			}
		}
		return l.RecvCount, wrong, nil
	}
	if f, ok := c.DataType.smallFormat(); ok {
		wrong := 0
		for i := 0; i < l.RecvCount; i++ {
			value := f.decode(getSmall(data[i*c.DataType.Size():], c.DataType))
			if value != expectedSmall(c, l, ranks, rank, i) {
				wrong++
			}
		}
		return l.RecvCount, wrong, nil
	}
	if c.DataType == Float64 {
		got := make([]float64, len(data)/8)
		for i := range got {
			got[i] = math.Float64frombits(binary.NativeEndian.Uint64(data[i*8:]))
		}
		return Check64(c, ranks, rank, got)
	}
	got := make([]float32, len(data)/4)
	for i := range got {
		got[i] = math.Float32frombits(binary.NativeEndian.Uint32(data[i*4:]))
	}
	return Check(c, ranks, rank, got)
}
