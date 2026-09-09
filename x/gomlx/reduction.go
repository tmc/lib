package gomlx

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

func (f *function) ReduceMin(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduce(v, "minimum", axes)
}
func (f *function) ReduceBitwiseAnd(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceBits(v, "and", axes, false)
}
func (f *function) ReduceBitwiseOr(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceBits(v, "or", axes, false)
}
func (f *function) ReduceBitwiseXor(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceBits(v, "xor", axes, false)
}
func (f *function) ReduceLogicalAnd(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceBits(v, "and", axes, true)
}
func (f *function) ReduceLogicalOr(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceBits(v, "or", axes, true)
}
func (f *function) ReduceLogicalXor(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduceBits(v, "xor", axes, true)
}
func (f *function) reduceBits(v compute.Value, op string, axes []int, logical bool) (compute.Value, error) {
	s, err := f.Shape(v)
	if err != nil {
		return nil, err
	}
	if logical && s.DType != dtypes.Bool || !logical && !s.DType.IsInt() {
		return nil, unsupported
	}
	return f.reduce(v, op, axes)
}
