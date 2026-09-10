package gomlx

import (
	"fmt"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

// unaryOp names the shape-preserving primitives shared by static and replay
// emission. The emitter validates the operation and its dtype before writing IR.
type unaryOp string

func emitUnary(raw *hlo.FuncBuilder, x hlo.Value, op unaryOp) (hlo.Value, error) {
	switch op {
	case "negate":
		if x.Type.ElementType == "i1" {
			return hlo.Value{}, unsupported
		}
	case "exponential", "log", "sqrt", "rsqrt", "sine", "cosine", "abs",
		"exponential_minus_one", "log_plus_one", "ceil", "floor", "round_nearest_even",
		"tanh", "logistic":
		if !elementDType(x.Type.ElementType).IsFloat() {
			return hlo.Value{}, unsupported
		}
	default:
		return hlo.Value{}, fmt.Errorf("unsupported unary operation %q", op)
	}
	z := hlo.Value{ID: fmt.Sprintf("%%unary%d", len(raw.Ops)), Type: x.Type}
	raw.Ops = append(raw.Ops, fmt.Sprintf("%s = stablehlo.%s %s : %s", z.ID, op, x.ID, x.Type))
	return z, nil
}

func (f *function) unary(v compute.Value, op unaryOp) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	z, err := emitUnary(f.raw, x, op)
	if err != nil {
		return nil, err
	}
	f.dynamicOps = append(f.dynamicOps, dynamicOp{kind: dynamicUnary, unary: op, operand: x, result: z})
	f.valueShapes[z.ID] = f.valueShape(x).Clone()
	return &node{f, z}, nil
}
