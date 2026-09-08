package gomlx

import (
	"fmt"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) RNGBitGenerator(state compute.Value, shape shapes.Shape) (compute.Value, compute.Value, error) {
	if f.owner.returned {
		return nil, nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(state)
	if err != nil {
		return nil, nil, err
	}
	if x.Type.ElementType != "ui64" || !slices.Equal(x.Type.Shape, []int64{3}) {
		return nil, nil, fmt.Errorf("rng: state must be [3]uint64")
	}
	if err := validate(shape); err != nil {
		return nil, nil, err
	}
	if shape.DType != dtypes.Uint32 && shape.DType != dtypes.Uint64 {
		return nil, nil, fmt.Errorf("rng: unsigned 32 or 64 bit output required")
	}
	info, _ := element(shape.DType)
	next := hlo.Value{ID: fmt.Sprintf("%%rngstate%d", len(f.raw.Ops)), Type: x.Type}
	bits := hlo.Value{ID: fmt.Sprintf("%%rngbits%d", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(shape.Dimensions), ElementType: info.hlo}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s, %s = \"stablehlo.rng_bit_generator\"(%s) {rng_algorithm = #stablehlo<rng_algorithm PHILOX>} : (%s) -> (%s, %s)", next.ID, bits.ID, x.ID, x.Type, next.Type, bits.Type))
	return &node{f, next}, &node{f, bits}, nil
}
func (f *function) Abs(v compute.Value) (compute.Value, error)    { return f.unaryFloat(v, "abs") }
func (f *function) Cos(v compute.Value) (compute.Value, error)    { return f.unaryFloat(v, "cosine") }
func (f *function) Min(a, b compute.Value) (compute.Value, error) { return f.extreme(a, b, "minimum") }
func (f *function) Max(a, b compute.Value) (compute.Value, error) { return f.extreme(a, b, "maximum") }
func (f *function) extreme(a, b compute.Value, op string) (compute.Value, error) {
	return f.binary(a, b, func(x, y hlo.Value) (hlo.Value, error) {
		if x.Type.ElementType == "i1" {
			return hlo.Value{}, unsupported
		}
		z := hlo.Value{ID: fmt.Sprintf("%%extreme%d", len(f.raw.Ops)), Type: x.Type}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.%s %s, %s : %s", z.ID, op, x.ID, y.ID, x.Type))
		return z, nil
	})
}
