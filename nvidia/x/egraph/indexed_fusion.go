package egraph

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func fusionTensor(expr *Expr) (FusionTensor, error) {
	if expr == nil {
		return FusionTensor{}, fmt.Errorf("egraph: nil indexed expression")
	}
	data, ok := expr.Data.(FusionTensor)
	if !ok {
		return data, fmt.Errorf("egraph: indexed expression %q requires FusionTensor metadata", expr.Op)
	}
	zero := false
	for _, dimension := range data.Shape {
		if dimension < 0 {
			return data, fmt.Errorf("egraph: indexed expression %q has negative dimension", expr.Op)
		}
		zero = zero || dimension == 0
	}
	count := uint64(1)
	if !zero {
		for _, dimension := range data.Shape {
			if count > math.MaxInt32/uint64(dimension) {
				return data, fmt.Errorf("egraph: indexed expression %q exceeds int32 extent", expr.Op)
			}
			count *= uint64(dimension)
		}
	}
	if strings.ToLower(expr.Op) != "transpose" && len(data.Permutation) != 0 {
		return data, fmt.Errorf("egraph: indexed expression %q has unexpected permutation", expr.Op)
	}
	return data, nil
}

func compileIndexedFusionRegion(region *ExtractedFusionRegion) (*cudafuse.FusedKernel, error) {
	ctx := cudafuse.NewLazyContext()
	arrays := make(map[*Expr]*cudafuse.LazyArray)
	var inputs []*cudafuse.LazyNode
	for _, expr := range region.Inputs {
		data, err := fusionTensor(expr)
		if err != nil {
			return nil, err
		}
		input := cudafuse.NewLazyArray(ctx, data.Shape, cudair.Float32)
		arrays[expr] = input
		inputs = append(inputs, input.Node)
	}
	var lower func(*Expr) (*cudafuse.LazyArray, error)
	lower = func(expr *Expr) (*cudafuse.LazyArray, error) {
		if array := arrays[expr]; array != nil {
			return array, nil
		}
		data, err := fusionTensor(expr)
		if err != nil {
			return nil, err
		}
		op := strings.ToLower(expr.Op)
		arity := 1
		switch op {
		case "add", "sub", "mul", "div":
			arity = 2
		case "relu", "silu", "exp", "sqrt", "neg", "reshape", "transpose":
		default:
			return nil, fmt.Errorf("egraph: unsupported indexed operator %q", expr.Op)
		}
		if len(expr.Children) != arity {
			return nil, fmt.Errorf("egraph: indexed operator %s requires %d children", op, arity)
		}
		a, err := lower(expr.Children[0])
		if err != nil {
			return nil, err
		}
		var result *cudafuse.LazyArray
		switch op {
		case "transpose":
			result = a.Transpose(data.Permutation)
		case "reshape":
			result = a.Reshape(data.Shape)
		case "add", "sub", "mul", "div":
			b, err := lower(expr.Children[1])
			if err != nil {
				return nil, err
			}
			if !slices.Equal(a.Shape, b.Shape) {
				return nil, fmt.Errorf("egraph: indexed binary operand shapes differ")
			}
			result = a.Map2(op, b)
		default:
			result = a.Map(op)
		}
		if result.Node.Err != nil {
			return nil, fmt.Errorf("egraph: indexed operator %s: %w", op, result.Node.Err)
		}
		if !slices.Equal(result.Shape, data.Shape) {
			return nil, fmt.Errorf("egraph: indexed operator %s contradicts output shape", op)
		}
		arrays[expr] = result
		return result, nil
	}
	root, err := lower(region.Root)
	if err != nil {
		return nil, err
	}
	kernel, err := cudafuse.FuseLazy(inputs, []*cudafuse.LazyNode{root.Node}, fmt.Sprintf("fused_egraph_region_%d", region.ID))
	if err != nil {
		return nil, err
	}
	return &kernel, nil
}
