package gomlx

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/tmc/lib/openxla"
)

type elementInfo struct {
	hlo  string
	pjrt openxla.Buffer_Type
	size int
}

func element(dtype dtypes.DType) (elementInfo, bool) {
	switch dtype {
	case dtypes.BFloat16:
		return elementInfo{"bf16", openxla.Buffer_Type_BF16, 2}, true
	case dtypes.Float16:
		return elementInfo{"f16", openxla.Buffer_Type_F16, 2}, true
	case dtypes.Uint8:
		return elementInfo{"ui8", openxla.Buffer_Type_U8, 1}, true
	case dtypes.Uint32:
		return elementInfo{"ui32", openxla.Buffer_Type_U32, 4}, true
	case dtypes.Uint64:
		return elementInfo{"ui64", openxla.Buffer_Type_U64, 8}, true
	case dtypes.Bool:
		return elementInfo{"i1", openxla.Buffer_Type_PRED, 1}, true
	case dtypes.Float64:
		return elementInfo{"f64", openxla.Buffer_Type_F64, 8}, true
	case dtypes.Float32:
		return elementInfo{"f32", openxla.Buffer_Type_F32, 4}, true
	case dtypes.Complex64:
		return elementInfo{"complex<f32>", openxla.Buffer_Type_C64, 8}, true
	case dtypes.Complex128:
		return elementInfo{"complex<f64>", openxla.Buffer_Type_C128, 16}, true
	case dtypes.Int8:
		return elementInfo{"i8", openxla.Buffer_Type_S8, 1}, true
	case dtypes.Int16:
		return elementInfo{"i16", openxla.Buffer_Type_S16, 2}, true
	case dtypes.Uint16:
		return elementInfo{"ui16", openxla.Buffer_Type_U16, 2}, true
	case dtypes.Int32:
		return elementInfo{"i32", openxla.Buffer_Type_S32, 4}, true
	case dtypes.Int64:
		return elementInfo{"i64", openxla.Buffer_Type_S64, 8}, true
	}
	return elementInfo{}, false
}
func elementDType(name string) dtypes.DType {
	switch name {
	case "bf16":
		return dtypes.BFloat16
	case "f16":
		return dtypes.Float16
	case "ui8":
		return dtypes.Uint8
	case "ui32":
		return dtypes.Uint32
	case "ui64":
		return dtypes.Uint64
	case "i1":
		return dtypes.Bool
	case "f64":
		return dtypes.Float64
	case "f32":
		return dtypes.Float32
	case "complex<f32>":
		return dtypes.Complex64
	case "complex<f64>":
		return dtypes.Complex128
	case "i8":
		return dtypes.Int8
	case "i16":
		return dtypes.Int16
	case "ui16":
		return dtypes.Uint16
	case "i32":
		return dtypes.Int32
	case "i64":
		return dtypes.Int64
	}
	return dtypes.InvalidDType
}
func flatShape(flat any) (dtypes.DType, int) {
	switch v := flat.(type) {
	case []bfloat16.BFloat16:
		return dtypes.BFloat16, len(v)
	case []float16.Float16:
		return dtypes.Float16, len(v)
	case []uint8:
		return dtypes.Uint8, len(v)
	case []uint32:
		return dtypes.Uint32, len(v)
	case []uint64:
		return dtypes.Uint64, len(v)
	case []bool:
		return dtypes.Bool, len(v)
	case []float64:
		return dtypes.Float64, len(v)
	case []float32:
		return dtypes.Float32, len(v)
	case []complex64:
		return dtypes.Complex64, len(v)
	case []complex128:
		return dtypes.Complex128, len(v)
	case []int8:
		return dtypes.Int8, len(v)
	case []int16:
		return dtypes.Int16, len(v)
	case []uint16:
		return dtypes.Uint16, len(v)
	case []int32:
		return dtypes.Int32, len(v)
	case []int64:
		return dtypes.Int64, len(v)
	}
	return dtypes.InvalidDType, 0
}
func newFlat(dtype dtypes.DType, n int) any {
	switch dtype {
	case dtypes.BFloat16:
		return make([]bfloat16.BFloat16, n)
	case dtypes.Float16:
		return make([]float16.Float16, n)
	case dtypes.Uint8:
		return make([]uint8, n)
	case dtypes.Uint32:
		return make([]uint32, n)
	case dtypes.Uint64:
		return make([]uint64, n)
	case dtypes.Bool:
		return make([]bool, n)
	case dtypes.Float64:
		return make([]float64, n)
	case dtypes.Float32:
		return make([]float32, n)
	case dtypes.Complex64:
		return make([]complex64, n)
	case dtypes.Complex128:
		return make([]complex128, n)
	case dtypes.Int8:
		return make([]int8, n)
	case dtypes.Int16:
		return make([]int16, n)
	case dtypes.Uint16:
		return make([]uint16, n)
	case dtypes.Int32:
		return make([]int32, n)
	case dtypes.Int64:
		return make([]int64, n)
	}
	return nil
}
func encodeFlat(flat any, raw []byte) {
	switch v := flat.(type) {
	case []bfloat16.BFloat16:
		for i, x := range v {
			binary.LittleEndian.PutUint16(raw[2*i:], x.Bits())
		}
	case []float16.Float16:
		for i, x := range v {
			binary.LittleEndian.PutUint16(raw[2*i:], x.Bits())
		}
	case []uint8:
		copy(raw, v)
	case []uint32:
		for i, x := range v {
			binary.LittleEndian.PutUint32(raw[4*i:], x)
		}
	case []uint64:
		for i, x := range v {
			binary.LittleEndian.PutUint64(raw[8*i:], x)
		}
	case []bool:
		for i, x := range v {
			raw[i] = 0
			if x {
				raw[i] = 1
			}
		}
	case []float64:
		for i, x := range v {
			binary.LittleEndian.PutUint64(raw[8*i:], math.Float64bits(x))
		}
	case []float32:
		for i, x := range v {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(x))
		}
	case []complex64:
		for i, x := range v {
			binary.LittleEndian.PutUint32(raw[8*i:], math.Float32bits(real(x)))
			binary.LittleEndian.PutUint32(raw[8*i+4:], math.Float32bits(imag(x)))
		}
	case []complex128:
		for i, x := range v {
			binary.LittleEndian.PutUint64(raw[16*i:], math.Float64bits(real(x)))
			binary.LittleEndian.PutUint64(raw[16*i+8:], math.Float64bits(imag(x)))
		}
	case []int8:
		for i, x := range v {
			raw[i] = byte(x)
		}
	case []int16:
		for i, x := range v {
			binary.LittleEndian.PutUint16(raw[2*i:], uint16(x))
		}
	case []uint16:
		for i, x := range v {
			binary.LittleEndian.PutUint16(raw[2*i:], x)
		}
	case []int32:
		for i, x := range v {
			binary.LittleEndian.PutUint32(raw[4*i:], uint32(x))
		}
	case []int64:
		for i, x := range v {
			binary.LittleEndian.PutUint64(raw[8*i:], uint64(x))
		}
	}
}
func decodeFlat(flat any, raw []byte) {
	switch v := flat.(type) {
	case []bfloat16.BFloat16:
		for i := range v {
			v[i] = bfloat16.FromBits(binary.LittleEndian.Uint16(raw[2*i:]))
		}
	case []float16.Float16:
		for i := range v {
			v[i] = float16.FromBits(binary.LittleEndian.Uint16(raw[2*i:]))
		}
	case []uint8:
		copy(v, raw)
	case []uint32:
		for i := range v {
			v[i] = binary.LittleEndian.Uint32(raw[4*i:])
		}
	case []uint64:
		for i := range v {
			v[i] = binary.LittleEndian.Uint64(raw[8*i:])
		}
	case []bool:
		for i := range v {
			v[i] = raw[i] != 0
		}
	case []float64:
		for i := range v {
			v[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[8*i:]))
		}
	case []float32:
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	case []complex64:
		for i := range v {
			v[i] = complex(math.Float32frombits(binary.LittleEndian.Uint32(raw[8*i:])), math.Float32frombits(binary.LittleEndian.Uint32(raw[8*i+4:])))
		}
	case []complex128:
		for i := range v {
			v[i] = complex(math.Float64frombits(binary.LittleEndian.Uint64(raw[16*i:])), math.Float64frombits(binary.LittleEndian.Uint64(raw[16*i+8:])))
		}
	case []int8:
		for i := range v {
			v[i] = int8(raw[i])
		}
	case []int16:
		for i := range v {
			v[i] = int16(binary.LittleEndian.Uint16(raw[2*i:]))
		}
	case []uint16:
		for i := range v {
			v[i] = binary.LittleEndian.Uint16(raw[2*i:])
		}
	case []int32:
		for i := range v {
			v[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	case []int64:
		for i := range v {
			v[i] = int64(binary.LittleEndian.Uint64(raw[8*i:]))
		}
	}
}
func constantLiteral(flat any, i int) string {
	switch v := flat.(type) {
	case []bfloat16.BFloat16:
		return fmt.Sprintf("0x%04X", v[i].Bits())
	case []float16.Float16:
		return fmt.Sprintf("0x%04X", v[i].Bits())
	case []uint8:
		return fmt.Sprint(v[i])
	case []uint32:
		return fmt.Sprint(v[i])
	case []uint64:
		return fmt.Sprint(v[i])
	case []bool:
		return fmt.Sprint(v[i])
	case []float64:
		return fmt.Sprintf("0x%016X", math.Float64bits(v[i]))
	case []float32:
		return fmt.Sprintf("0x%08X", math.Float32bits(v[i]))
	case []complex64:
		return fmt.Sprintf("(0x%08X, 0x%08X)", math.Float32bits(real(v[i])), math.Float32bits(imag(v[i])))
	case []complex128:
		return fmt.Sprintf("(0x%016X, 0x%016X)", math.Float64bits(real(v[i])), math.Float64bits(imag(v[i])))
	case []int8:
		return fmt.Sprint(v[i])
	case []int16:
		return fmt.Sprint(v[i])
	case []uint16:
		return fmt.Sprint(v[i])
	case []int32:
		return fmt.Sprint(v[i])
	case []int64:
		return fmt.Sprint(v[i])
	}
	return ""
}
