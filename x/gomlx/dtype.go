package gomlx

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/gomlx/compute/dtypes"
	"github.com/tmc/lib/openxla"
)

type elementInfo struct {
	hlo  string
	pjrt openxla.Buffer_Type
	size int
}

func element(dtype dtypes.DType) (elementInfo, bool) {
	switch dtype {
	case dtypes.Bool:
		return elementInfo{"i1", openxla.Buffer_Type_PRED, 1}, true
	case dtypes.Float32:
		return elementInfo{"f32", openxla.Buffer_Type_F32, 4}, true
	case dtypes.Int32:
		return elementInfo{"i32", openxla.Buffer_Type_S32, 4}, true
	case dtypes.Int64:
		return elementInfo{"i64", openxla.Buffer_Type_S64, 8}, true
	}
	return elementInfo{}, false
}
func elementDType(name string) dtypes.DType {
	switch name {
	case "i1":
		return dtypes.Bool
	case "f32":
		return dtypes.Float32
	case "i32":
		return dtypes.Int32
	case "i64":
		return dtypes.Int64
	}
	return dtypes.InvalidDType
}
func flatShape(flat any) (dtypes.DType, int) {
	switch v := flat.(type) {
	case []bool:
		return dtypes.Bool, len(v)
	case []float32:
		return dtypes.Float32, len(v)
	case []int32:
		return dtypes.Int32, len(v)
	case []int64:
		return dtypes.Int64, len(v)
	}
	return dtypes.InvalidDType, 0
}
func newFlat(dtype dtypes.DType, n int) any {
	switch dtype {
	case dtypes.Bool:
		return make([]bool, n)
	case dtypes.Float32:
		return make([]float32, n)
	case dtypes.Int32:
		return make([]int32, n)
	case dtypes.Int64:
		return make([]int64, n)
	}
	return nil
}
func encodeFlat(flat any, raw []byte) {
	switch v := flat.(type) {
	case []bool:
		for i, x := range v {
			raw[i] = 0
			if x {
				raw[i] = 1
			}
		}
	case []float32:
		for i, x := range v {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(x))
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
	case []bool:
		for i := range v {
			v[i] = raw[i] != 0
		}
	case []float32:
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
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
	case []bool:
		return fmt.Sprint(v[i])
	case []float32:
		return fmt.Sprintf("0x%08X", math.Float32bits(v[i]))
	case []int32:
		return fmt.Sprint(v[i])
	case []int64:
		return fmt.Sprint(v[i])
	}
	return ""
}
