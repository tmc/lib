package halfbits

import (
	"math"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cublaslt"
)

func converted(x float32) cublas.Scalar {
	return cublas.Float16BitsScalar(uint16(x)) // want `uint16\(x\) converts the float x to an integer; Float16BitsScalar takes the fp16 bit pattern`
}

func literal() cublaslt.Scalar {
	return cublaslt.Float16BitsScalar(uint16(1.0)) // want `uint16\(1.0\) converts the float 1.0 to an integer`
}

const one = 1.0

func namedConstant() cublas.Scalar {
	return cublas.Float16BitsScalar(uint16(one)) // want `uint16\(one\) converts the float one to an integer`
}

func chained(x float64) cublas.Scalar {
	return cublas.Float16BitsScalar(uint16(int32(x * 2))) // want `uint16\(int32\(x \* 2\)\) converts the float x \* 2 to an integer`
}

func variable(x float32) cublas.Scalar {
	b := uint16(x)
	return cublas.Float16BitsScalar(b) // want `b converts the float x to an integer`
}

func pattern() cublas.Scalar {
	return cublas.Float16BitsScalar(0x3c00)
}

func integer(n int) cublas.Scalar {
	return cublas.Float16BitsScalar(uint16(n))
}

func intConstant() cublas.Scalar {
	return cublas.Float16BitsScalar(uint16(1))
}

type f16 struct{}

func (f16) Bits(x float32) uint16 { return toHalfBits(x) }

func float16Bits(x float32) cublas.Scalar {
	var h f16
	return cublas.Float16BitsScalar(h.Bits(x))
}

func helper(x float32) cublas.Scalar {
	return cublas.Float16BitsScalar(toHalfBits(x))
}

func shifted(x float32) cublas.Scalar {
	return cublas.Float16BitsScalar(uint16(math.Float32bits(x) >> 16))
}

// roundHalf returns x rounded to the nearest half-precision value, as a
// float whose integer part a caller might mistake for bits.
func roundHalf(x float32) float32 { return x }

func namedForHalf(x float32) cublas.Scalar {
	return cublas.Float16BitsScalar(uint16(roundHalf(x)))
}

func toHalfBits(x float32) uint16 {
	b := math.Float32bits(x)
	return uint16(b>>16&0x8000 | (b>>13-(127-15)<<10)&0x7fff)
}
