package gomlx

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Sort(comparator compute.Function, axis int, stable bool, inputs ...compute.Value) ([]compute.Value, error) {
	if f.returned || f.owner.closed {
		return nil, fmt.Errorf("function closed")
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("sort: input required")
	}
	c, ok := comparator.(*function)
	if !ok || c == nil || c.parent != f {
		return nil, fmt.Errorf("sort: comparator must be a closure of this function")
	}
	types := make([]hlo.Type, len(inputs))
	ids := make([]string, len(inputs))
	args := make([]string, 0, 2*len(inputs))
	values := make([]compute.Value, 0, 2*len(inputs))
	for i, input := range inputs {
		v, err := f.value(input)
		if err != nil {
			return nil, err
		}
		if i > 0 && !slices.Equal(v.Type.Shape, types[0].Shape) {
			return nil, fmt.Errorf("sort: input dimensions must match")
		}
		types[i], ids[i] = v.Type, v.ID
		// GoMLX's graph API and executable backends use interleaved pairs,
		// matching StableHLO, despite the compute interface's grouped comment.
		for side := range 2 {
			arg := hlo.Value{ID: fmt.Sprintf("%%sortarg%d_%d", len(f.raw.Ops), 2*i+side), Type: hlo.Type{ElementType: v.Type.ElementType}}
			args = append(args, arg.ID+": "+arg.Type.String())
			values = append(values, &node{f, arg})
		}
	}
	rank := len(types[0].Shape)
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("sort: axis out of range")
	}
	_, rt, err := f.callRegion(comparator, values)
	if err != nil {
		return nil, err
	}
	if len(rt) != 1 || rt[0].ElementType != "i1" || len(rt[0].Shape) != 0 {
		return nil, fmt.Errorf("sort: comparator must return a scalar boolean")
	}
	code, err := f.sortCaptures(c, values, types[0])
	if err != nil {
		return nil, err
	}
	ids = append(ids, code.ids...)
	types = append(types, code.types...)
	args = append(args, code.args...)
	op := fmt.Sprintf("\"stablehlo.sort\"(%s) ({\n^bb0(%s):\n%s\n}) {dimension = %d : i64, is_stable = %t} : (%s) -> (%s)", strings.Join(ids, ", "), strings.Join(args, ", "), code.region, axis, stable, typeNames(types), typeNames(types))
	f.raw.Ops = append(f.raw.Ops, code.outer...)
	return f.regionResult("sort", op, types)[:len(inputs)], nil
}

type sortCode struct {
	outer  []string
	region string
	ids    []string
	types  []hlo.Type
	args   []string
}

// sortCaptures carries each scalar of a capture as an additional sort operand.
// The comparator reconstructs captured tensors from its local arguments: HLO
// sort computations cannot reference values in an enclosing region.
func (f *function) sortCaptures(c *function, values []compute.Value, inputType hlo.Type) (sortCode, error) {
	var code sortCode
	var inner, callIDs []string
	var callTypes []hlo.Type
	for _, value := range values {
		v := value.(*node).value
		callIDs = append(callIDs, v.ID)
		callTypes = append(callTypes, v.Type)
	}
	for ci, capture := range c.captures {
		v, e := f.value(capture)
		if e != nil {
			return sortCode{}, e
		}
		scalar := hlo.Type{ElementType: v.Type.ElementType}
		one := hlo.Type{Shape: []int64{1}, ElementType: v.Type.ElementType}
		count := int64(1)
		for _, dim := range v.Type.Shape {
			count *= dim
		}
		flat := hlo.Type{Shape: []int64{count}, ElementType: v.Type.ElementType}
		prefix := fmt.Sprintf("%%sortcapture%d_%d", len(f.raw.Ops), ci)
		flatID := v.ID
		if len(v.Type.Shape) > 0 && count > 0 {
			flatID = prefix + "flat"
			code.outer = append(code.outer, fmt.Sprintf("%s = \"stablehlo.reshape\"(%s) : (%s) -> %s", flatID, v.ID, v.Type, flat))
		}
		var pieces []string
		var pieceTypes []hlo.Type
		reconstructed := prefix + "value"
		for j := int64(0); j < count; j++ {
			scalarID := v.ID
			if len(v.Type.Shape) > 0 {
				sliceID := fmt.Sprintf("%sslice%d", prefix, j)
				scalarID = fmt.Sprintf("%sscalar%d", prefix, j)
				code.outer = append(code.outer, fmt.Sprintf("%s = \"stablehlo.slice\"(%s) {start_indices = array<i64: %d>, limit_indices = array<i64: %d>, strides = array<i64: 1>} : (%s) -> %s", sliceID, flatID, j, j+1, flat, one), fmt.Sprintf("%s = \"stablehlo.reshape\"(%s) : (%s) -> %s", scalarID, sliceID, one, scalar))
			}
			carrierType := hlo.Type{Shape: inputType.Shape, ElementType: v.Type.ElementType}
			carrierID := fmt.Sprintf("%scarrier%d", prefix, j)
			code.outer = append(code.outer, fmt.Sprintf("%s = \"stablehlo.broadcast_in_dim\"(%s) {broadcast_dimensions = array<i64>} : (%s) -> %s", carrierID, scalarID, scalar, carrierType))
			left := fmt.Sprintf("%sleft%d", prefix, j)
			right := fmt.Sprintf("%sright%d", prefix, j)
			code.args = append(code.args, left+": "+scalar.String(), right+": "+scalar.String())
			code.ids = append(code.ids, carrierID)
			code.types = append(code.types, carrierType)
			if len(v.Type.Shape) == 0 {
				reconstructed = left
			} else {
				piece := fmt.Sprintf("%spiece%d", prefix, j)
				inner = append(inner, fmt.Sprintf("%s = \"stablehlo.reshape\"(%s) : (%s) -> %s", piece, left, scalar, one))
				pieces = append(pieces, piece)
				pieceTypes = append(pieceTypes, one)
			}
		}
		if count == 0 {
			inner = append(inner, fmt.Sprintf("%s = \"stablehlo.constant\"() {value = dense<> : %s} : () -> %s", reconstructed, v.Type, v.Type))
		} else if len(v.Type.Shape) > 0 {
			joined := prefix + "joined"
			inner = append(inner, fmt.Sprintf("%s = \"stablehlo.concatenate\"(%s) {dimension = 0 : i64} : (%s) -> %s", joined, strings.Join(pieces, ", "), typeNames(pieceTypes), flat), fmt.Sprintf("%s = \"stablehlo.reshape\"(%s) : (%s) -> %s", reconstructed, joined, flat, v.Type))
		}
		callIDs = append(callIDs, reconstructed)
		callTypes = append(callTypes, v.Type)
	}
	predicate := fmt.Sprintf("%%sortpredicate%d", len(f.raw.Ops))
	inner = append(inner, fmt.Sprintf("%s = func.call @%s(%s) : (%s) -> tensor<i1>", predicate, c.raw.Name, strings.Join(callIDs, ", "), typeNames(callTypes)), "stablehlo.return "+predicate+" : tensor<i1>")
	code.region = strings.Join(inner, "\n")
	return code, nil
}
