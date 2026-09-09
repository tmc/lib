package gomlx

import "github.com/gomlx/compute"

func (f *function) NotEqual(a, b compute.Value) (compute.Value, error) { return f.compare(a, b, "NE") }
func (f *function) EqualTotalOrder(a, b compute.Value) (compute.Value, error) {
	return f.compareOrder(a, b, "EQ", true)
}
func (f *function) NotEqualTotalOrder(a, b compute.Value) (compute.Value, error) {
	return f.compareOrder(a, b, "NE", true)
}
func (f *function) LessThanTotalOrder(a, b compute.Value) (compute.Value, error) {
	return f.compareOrder(a, b, "LT", true)
}
func (f *function) LessOrEqualTotalOrder(a, b compute.Value) (compute.Value, error) {
	return f.compareOrder(a, b, "LE", true)
}
func (f *function) GreaterThanTotalOrder(a, b compute.Value) (compute.Value, error) {
	return f.compareOrder(a, b, "GT", true)
}
func (f *function) GreaterOrEqualTotalOrder(a, b compute.Value) (compute.Value, error) {
	return f.compareOrder(a, b, "GE", true)
}
