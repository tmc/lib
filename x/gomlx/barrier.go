package gomlx

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) optimizationBarrier(operands ...compute.Value) ([]compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	if len(operands) == 0 {
		return nil, fmt.Errorf("optimization barrier requires at least one operand")
	}
	values := make([]hlo.Value, len(operands))
	for i, operand := range operands {
		v, err := f.value(operand)
		if err != nil {
			return nil, err
		}
		values[i] = v
	}
	base := len(f.raw.Ops)
	results := make([]compute.Value, len(values))
	resultIDs := make([]string, len(values))
	operandIDs := make([]string, len(values))
	types := make([]string, len(values))
	for i, v := range values {
		out := hlo.Value{ID: fmt.Sprintf("%%barrier%d_%d", base, i), Type: v.Type}
		results[i] = &node{f, out}
		resultIDs[i], operandIDs[i], types[i] = out.ID, v.ID, v.Type.String()
	}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.optimization_barrier"(%s) : (%s) -> (%s)`, strings.Join(resultIDs, ", "), strings.Join(operandIDs, ", "), strings.Join(types, ", "), strings.Join(types, ", ")))
	return results, nil
}

func (f *function) OptimizationBarrier(operands ...compute.Value) ([]compute.Value, error) {
	return f.optimizationBarrier(operands...)
}

func (f *function) SchedulingBarrier(operand compute.Value, dependencies ...compute.Value) (compute.Value, error) {
	if len(dependencies) == 0 {
		if f.returned {
			return nil, fmt.Errorf("function already returned")
		}
		if _, err := f.value(operand); err != nil {
			return nil, err
		}
		return operand, nil
	}
	all := make([]compute.Value, 1, 1+len(dependencies))
	all[0] = operand
	all = append(all, dependencies...)
	values, err := f.optimizationBarrier(all...)
	if err != nil {
		return nil, err
	}
	return values[0], nil
}
