package gomlx

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapeinference"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) DynamicSlice(operand compute.Value, starts []compute.Value, sizes []int) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	s, err := f.Shape(operand)
	if err != nil {
		return nil, err
	}
	out, err := shapeinference.DynamicSlice(s, sizes)
	if err != nil {
		return nil, err
	}
	if err := validate(out); err != nil {
		return nil, err
	}
	x, err := f.value(operand)
	if err != nil {
		return nil, err
	}
	indices := make([]hlo.Value, len(starts))
	for i, v := range starts {
		indices[i], err = f.value(v)
		if err != nil {
			return nil, err
		}
	}
	// Go-XLA also accepts a single vector containing one start per axis.
	vector := len(indices) == 1 && len(indices[0].Type.Shape) == 1
	if vector {
		if indices[0].Type.Shape[0] != int64(s.Rank()) {
			return nil, fmt.Errorf("dynamic slice: index vector length must match rank")
		}
	} else if len(indices) != s.Rank() {
		return nil, fmt.Errorf("dynamic slice: index count must match rank")
	}
	for i, v := range indices {
		if !elementDType(v.Type.ElementType).IsInt() || !vector && len(v.Type.Shape) != 0 {
			return nil, fmt.Errorf("dynamic slice: integer scalar indices required")
		}
		if i > 0 && v.Type.ElementType != indices[0].Type.ElementType {
			return nil, fmt.Errorf("dynamic slice: index types differ")
		}
	}
	if s.Rank() == 0 {
		return &node{f, x}, nil
	}
	if vector {
		v := indices[0]
		indices = make([]hlo.Value, s.Rank())
		for i := range indices {
			part := hlo.Value{ID: fmt.Sprintf("%%startvec%d", len(f.raw.Ops)), Type: hlo.Type{Shape: []int64{1}, ElementType: v.Type.ElementType}}
			f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.slice"(%s) {start_indices = array<i64: %d>, limit_indices = array<i64: %d>, strides = array<i64: 1>} : (%s) -> %s`, part.ID, v.ID, i, i+1, v.Type, part.Type))
			indices[i] = hlo.Value{ID: fmt.Sprintf("%%start%d", len(f.raw.Ops)), Type: hlo.Type{ElementType: v.Type.ElementType}}
			f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.reshape %s : (%s) -> %s", indices[i].ID, part.ID, part.Type, indices[i].Type))
		}
	}
	values, types := []string{x.ID}, []string{x.Type.String()}
	for _, v := range indices {
		values = append(values, v.ID)
		types = append(types, v.Type.String())
	}
	z := hlo.Value{ID: fmt.Sprintf("%%dynamic_slice%d", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(out.Dimensions), ElementType: x.Type.ElementType}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.dynamic_slice"(%s) {slice_sizes = array<i64: %s>} : (%s) -> %s`, z.ID, strings.Join(values, ", "), indexValues(sizes), strings.Join(types, ", "), z.Type))
	return &node{f, z}, nil
}
