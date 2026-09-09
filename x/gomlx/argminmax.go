package gomlx

import (
	"fmt"
	"math"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) ArgMinMax(input compute.Value, axis int, outputDType dtypes.DType, isMin bool) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	s, err := f.Shape(input)
	if err != nil {
		return nil, err
	}
	out, err := shapeinference.ArgMinMax(s, axis, outputDType)
	if err != nil {
		return nil, err
	}
	if err := validate(out); err != nil {
		return nil, err
	}
	x, err := f.value(input)
	if err != nil {
		return nil, err
	}
	scalar := hlo.Type{ElementType: x.Type.ElementType}
	var initial hlo.Value
	if s.DType.IsFloat() {
		sign := -1
		if isMin {
			sign = 1
		}
		initial, err = f.raw.Constant(float32(math.Inf(sign)), scalar)
		if err != nil {
			return nil, err
		}
	} else {
		bits := 64
		switch s.DType {
		case dtypes.Int8, dtypes.Uint8:
			bits = 8
		case dtypes.Int16, dtypes.Uint16:
			bits = 16
		case dtypes.Int32, dtypes.Uint32:
			bits = 32
		}
		literal := "0"
		if s.DType.IsUnsigned() {
			if isMin {
				literal = fmt.Sprint(uint64(math.MaxUint64) >> (64 - bits))
			}
		} else if isMin {
			literal = fmt.Sprint(int64(math.MaxInt64) >> (64 - bits))
		} else {
			literal = fmt.Sprint(int64(-1) << (bits - 1))
		}
		initial = hlo.Value{ID: fmt.Sprintf("%%arginit%d", len(f.raw.Ops)), Type: scalar}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%s> : %s", initial.ID, literal, scalar))
	}
	indices := hlo.Value{ID: fmt.Sprintf("%%argiota%d", len(f.raw.Ops)), Type: hlo.Type{Shape: x.Type.Shape, ElementType: "i64"}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = \"stablehlo.iota\"() {iota_dimension = %d : i64} : () -> %s", indices.ID, axis, indices.Type))
	zero, err := f.raw.IntegerConstant(0, hlo.Type{ElementType: "i64"})
	if err != nil {
		return nil, err
	}
	direction := "GT"
	if isMin {
		direction = "LT"
	}
	comparison := "SIGNED"
	if s.DType.IsFloat() {
		comparison = "FLOAT"
	} else if s.DType.IsUnsigned() {
		comparison = "UNSIGNED"
	}
	var region strings.Builder
	fmt.Fprintf(&region, " %%better = stablehlo.compare %s, %%lv, %%rv, %s : (%s, %s) -> tensor<i1>\n", direction, comparison, scalar, scalar)
	fmt.Fprintf(&region, " %%equal = stablehlo.compare EQ, %%lv, %%rv, %s : (%s, %s) -> tensor<i1>\n", comparison, scalar, scalar)
	better, equal := "%better", "%equal"
	if s.DType.IsFloat() {
		fmt.Fprintf(&region, " %%lnan = stablehlo.compare NE, %%lv, %%lv, FLOAT : (%s, %s) -> tensor<i1>\n %%rnan = stablehlo.compare NE, %%rv, %%rv, FLOAT : (%s, %s) -> tensor<i1>\n", scalar, scalar, scalar, scalar)
		region.WriteString(" %nanbetter = stablehlo.or %better, %lnan : tensor<i1>\n %bothnan = stablehlo.and %lnan, %rnan : tensor<i1>\n %tie = stablehlo.or %equal, %bothnan : tensor<i1>\n")
		better, equal = "%nanbetter", "%tie"
	}
	region.WriteString(" %earlier = stablehlo.compare LE, %li, %ri, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>\n")
	fmt.Fprintf(&region, " %%left = stablehlo.select %s, %%earlier, %s : tensor<i1>, tensor<i1>\n", equal, better)
	fmt.Fprintf(&region, " %%value = stablehlo.select %%left, %%lv, %%rv : tensor<i1>, %s\n", scalar)
	region.WriteString(" %index = stablehlo.select %left, %li, %ri : tensor<i1>, tensor<i64>\n")
	fmt.Fprintf(&region, " stablehlo.return %%value, %%index : %s, tensor<i64>\n", scalar)
	result := hlo.Value{ID: fmt.Sprintf("%%arg%d#1", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(out.Dimensions), ElementType: "i64"}}
	valuesType := hlo.Type{Shape: result.Type.Shape, ElementType: x.Type.ElementType}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%%arg%d:2 = "stablehlo.reduce"(%s, %s, %s, %s) ({
 ^bb0(%%lv: %s, %%li: tensor<i64>, %%rv: %s, %%ri: tensor<i64>):
%s}) {dimensions = array<i64: %d>} : (%s, %s, %s, tensor<i64>) -> (%s, %s)`, len(f.raw.Ops), x.ID, indices.ID, initial.ID, zero.ID, scalar, scalar, region.String(), axis, x.Type, indices.Type, scalar, valuesType, result.Type))
	info, _ := element(outputDType)
	if info.hlo != "i64" {
		result, err = f.raw.Convert(result, info.hlo)
		if err != nil {
			return nil, err
		}
	}
	return &node{f, result}, nil
}
