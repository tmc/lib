package cudafuse

import (
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// shapeElements checks all dimensions before computing a logical count.
// A zero count does not imply that concrete layout strides are representable.
func shapeElements(shape []int) (uint64, error) {
	empty := false
	for i, dim := range shape {
		if dim < 0 {
			return 0, fmt.Errorf("%w: axis %d has extent %d", cudashape.ErrNegativeDimension, i, dim)
		}
		empty = empty || dim == 0
	}
	if empty {
		return 0, nil
	}
	count := uint64(1)
	for _, dim := range shape {
		if count > math.MaxUint64/uint64(dim) {
			return 0, cudashape.ErrShapeOverflow
		}
		count *= uint64(dim)
	}
	return count, nil
}

func elementCount(shape []int) (int, error) {
	n, err := shapeElements(shape)
	if err != nil {
		return 0, err
	}
	if n > math.MaxInt {
		return 0, cudashape.ErrShapeOverflow
	}
	return int(n), nil
}

// binaryShape follows cudair's ordered equal-shape-or-one-element rule.
func binaryShape(a, b []int) ([]int, error) {
	as, err := cudashape.New(a...)
	if err != nil {
		return nil, err
	}
	bs, err := cudashape.New(b...)
	if err != nil {
		return nil, err
	}
	an, _ := as.NumElementsSafe()
	bn, _ := bs.NumElementsSafe()
	switch {
	case slices.Equal(a, b), bn == 1:
		return slices.Clone(a), nil
	case an == 1:
		return slices.Clone(b), nil
	default:
		return nil, fmt.Errorf("%w: shapes %v and %v", cudair.ErrShapeMismatch, a, b)
	}
}

func arrayError(op string, arrays ...*LazyArray) error {
	var ctx *LazyContext
	for _, a := range arrays {
		if a == nil || a.Node == nil || a.Ctx == nil {
			return fmt.Errorf("%s: invalid operand", op)
		}
		if ctx != nil && a.Ctx != ctx {
			return fmt.Errorf("%s: operands belong to different contexts", op)
		}
		ctx = a.Ctx
		if a.Node.Err != nil {
			return fmt.Errorf("%s input: %w", op, a.Node.Err)
		}
		if !slices.Equal(a.Shape, a.Node.Shape) || a.DType != a.Node.DType {
			return fmt.Errorf("%s: array metadata differs from node", op)
		}
		if _, err := elementCount(a.Shape); err != nil {
			return fmt.Errorf("%s input: %w", op, err)
		}
	}
	return nil
}

func isIndexType(t cudair.DataType) bool {
	return t == cudair.Int32 || t == cudair.Int64 || t == cudair.Uint8 || t == cudair.Int8
}

// validateLazyShape also checks graphs built without the LazyArray methods.
func validateLazyShape(n *LazyNode) error {
	if n == nil {
		return fmt.Errorf("nil node")
	}
	if n.Err != nil {
		return n.Err
	}
	count, err := elementCount(n.Shape)
	if err != nil {
		return err
	}
	for _, in := range n.Inputs {
		if in == nil {
			return fmt.Errorf("nil input")
		}
		if in.Err != nil {
			return in.Err
		}
	}
	switch n.Op {
	case "input":
		if len(n.Inputs) != 0 || len(n.Attrs) != 0 {
			return fmt.Errorf("invalid input attributes")
		}
	case "add", "sub", "mul", "div":
		if len(n.Inputs) != 2 || len(n.Attrs) != 0 {
			return fmt.Errorf("%s requires two inputs and no attributes", n.Op)
		}
		if n.DType != n.Inputs[0].DType || n.DType != n.Inputs[1].DType {
			return cudair.ErrTypeMismatch
		}
		want, err := binaryShape(n.Inputs[0].Shape, n.Inputs[1].Shape)
		if err != nil {
			return err
		}
		if !slices.Equal(n.Shape, want) {
			return fmt.Errorf("%w: output %v, want %v", cudair.ErrShapeMismatch, n.Shape, want)
		}
	case "relu", "silu", "exp", "sqrt", "neg":
		if len(n.Inputs) != 1 || len(n.Attrs) != 0 || !slices.Equal(n.Shape, n.Inputs[0].Shape) {
			return fmt.Errorf("%w: unary output shape", cudair.ErrShapeMismatch)
		}
	case "reshape", "transpose", "cycle":
		if len(n.Inputs) != 1 || len(n.Attrs) != 1 {
			return fmt.Errorf("%s requires one input and one attribute", n.Op)
		}
		inputCount, err := elementCount(n.Inputs[0].Shape)
		if err != nil {
			return err
		}
		if n.Op == "transpose" {
			perm, ok := n.Attrs["permutation"].([]int)
			want, err := transposeShape(n.Inputs[0].Shape, perm)
			if !ok || err != nil || !slices.Equal(want, n.Shape) {
				return fmt.Errorf("invalid transpose shape or permutation")
			}
		} else {
			shape, ok := n.Attrs["target_shape"].([]int)
			if !ok || !slices.Equal(shape, n.Shape) {
				return fmt.Errorf("%s requires its exact target shape", n.Op)
			}
		}
		if n.Op == "cycle" {
			if len(n.Shape) == 0 || count == 0 || inputCount == 0 {
				return fmt.Errorf("cycle requires a ranked positive target and positive input")
			}
		} else if inputCount != count {
			return fmt.Errorf("%w: view changes element count", cudair.ErrShapeMismatch)
		}
	case "gather":
		if len(n.Inputs) != 2 || len(n.Attrs) != 0 || !isIndexType(n.Inputs[1].DType) || n.DType != n.Inputs[0].DType {
			return fmt.Errorf("gather requires data and integer indices")
		}
		indices, err := elementCount(n.Inputs[1].Shape)
		if err != nil {
			return err
		}
		input, err := elementCount(n.Inputs[0].Shape)
		if err != nil {
			return err
		}
		if !slices.Equal(n.Shape, []int{indices}) || input == 0 && indices != 0 {
			return fmt.Errorf("%w: gather extent", cudair.ErrShapeMismatch)
		}
	case "replicate":
		times, ok := n.Attrs["count"].(int)
		if len(n.Inputs) != 1 || len(n.Attrs) != 1 || !ok || times <= 0 {
			return fmt.Errorf("replicate requires a positive count")
		}
		input, err := elementCount(n.Inputs[0].Shape)
		if err != nil {
			return err
		}
		if input > math.MaxInt/times || !slices.Equal(n.Shape, []int{input * times}) {
			return fmt.Errorf("%w: replicate extent", cudair.ErrShapeMismatch)
		}
	}
	if n.DType < cudair.Float32 || n.DType > cudair.Int8 {
		return cudair.ErrTypeMismatch
	}
	if len(n.Inputs) == 1 && n.DType != n.Inputs[0].DType {
		return cudair.ErrTypeMismatch
	}
	return nil
}

func contextOf(a *LazyArray) *LazyContext {
	if a == nil {
		return nil
	}
	return a.Ctx
}
