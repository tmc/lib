package egraph

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"strconv"
	"strings"
	"unicode/utf8"
)

var errPrimitiveNoValue = errors.New("primitive returned no value")
var errPrimitiveLog2Domain = errors.New("int64 log2 requires a positive value")

// functionPrimitiveSignature reports the input and output types for a pure
// primitive expression. Integer and floating-point division and remainder
// retain Egglog's fallible-operation semantics. Stored f64 values reject NaN
// during canonicalization.
func functionPrimitiveSignature(op FunctionMergeOp) (argument FunctionColumnKind, result FunctionColumnKind, arity int, err error) {
	argument, result, arity = FunctionInt64Column, FunctionInt64Column, 2
	switch op {
	case FunctionMergeInt64Add, FunctionMergeInt64Sub, FunctionMergeInt64Mul,
		FunctionMergeInt64Div, FunctionMergeInt64Mod, FunctionMergeInt64And,
		FunctionMergeInt64Or, FunctionMergeInt64Xor, FunctionMergeInt64ShiftLeft,
		FunctionMergeInt64ShiftRight, FunctionMergeInt64Min, FunctionMergeInt64Max:
	case FunctionMergeInt64Not, FunctionMergeInt64Log2:
		arity = 1
	case FunctionMergeInt64ToFloat64:
		result, arity = FunctionFloat64Column, 1
	case FunctionMergeInt64ToString:
		result, arity = FunctionStringColumn, 1
	case FunctionMergeStringCountMatches:
		argument, result = FunctionStringColumn, FunctionInt64Column
	case FunctionMergeStringConcat:
		argument, result, arity = FunctionStringColumn, FunctionStringColumn, -1
	case FunctionMergeStringReplace:
		argument, result, arity = FunctionStringColumn, FunctionStringColumn, 3
	case FunctionMergeInt64Equal, FunctionMergeInt64NotEqual,
		FunctionMergeInt64Less, FunctionMergeInt64LessEqual,
		FunctionMergeInt64Greater, FunctionMergeInt64GreaterEqual:
		result = FunctionBoolColumn
	case FunctionMergeBoolAnd, FunctionMergeBoolOr, FunctionMergeBoolXor, FunctionMergeBoolImplies:
		argument, result = FunctionBoolColumn, FunctionBoolColumn
	case FunctionMergeBoolNot:
		argument, result, arity = FunctionBoolColumn, FunctionBoolColumn, 1
	case FunctionMergeFloat64Add, FunctionMergeFloat64Sub, FunctionMergeFloat64Mul, FunctionMergeFloat64Div,
		FunctionMergeFloat64Mod, FunctionMergeFloat64Pow, FunctionMergeFloat64Min, FunctionMergeFloat64Max:
		argument, result = FunctionFloat64Column, FunctionFloat64Column
	case FunctionMergeFloat64Neg, FunctionMergeFloat64Abs:
		argument, result, arity = FunctionFloat64Column, FunctionFloat64Column, 1
	case FunctionMergeFloat64ToInt64:
		argument, result, arity = FunctionFloat64Column, FunctionInt64Column, 1
	case FunctionMergeFloat64Equal, FunctionMergeFloat64NotEqual,
		FunctionMergeFloat64Less, FunctionMergeFloat64LessEqual,
		FunctionMergeFloat64Greater, FunctionMergeFloat64GreaterEqual:
		argument, result = FunctionFloat64Column, FunctionBoolColumn
	case FunctionMergeValueEqual, FunctionMergeValueNotEqual:
		argument, result = 0, FunctionBoolColumn
	default:
		return 0, 0, 0, fmt.Errorf("invalid primitive operation %d", op)
	}
	return argument, result, arity, nil
}

func primitiveEqualityKind(kind FunctionColumnKind) bool {
	switch kind {
	case FunctionInt64Column, FunctionFloat64Column, FunctionBoolColumn, FunctionStringColumn,
		FunctionBigIntColumn, FunctionBigRatColumn, FunctionUnitColumn:
		return true
	default:
		return false
	}
}

func compareFloat64Values(a, b float64) int {
	aNaN, bNaN := math.IsNaN(a), math.IsNaN(b)
	switch {
	case aNaN && bNaN:
		return 0
	case aNaN:
		return 1
	case bNaN:
		return -1
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

func evaluateFunctionPrimitive(g *EGraph, op FunctionMergeOp, args []FunctionValue) (FunctionValue, error) {
	argument, result, arity, err := functionPrimitiveSignature(op)
	if err != nil {
		return FunctionValue{}, err
	}
	if arity >= 0 && len(args) != arity {
		return FunctionValue{}, fmt.Errorf("primitive operation %d expects %d arguments, got %d", op, arity, len(args))
	}
	if argument == 0 {
		if !primitiveEqualityKind(args[0].Kind) || args[0].Kind != args[1].Kind {
			return FunctionValue{}, fmt.Errorf("value equality requires arguments with the same primitive kind")
		}
		if g == nil {
			return FunctionValue{}, fmt.Errorf("value equality requires a graph")
		}
		left, err := g.canonicalTypedValue(FunctionColumnType{Kind: args[0].Kind}, args[0], "primitive equality argument 0")
		if err != nil {
			return FunctionValue{}, err
		}
		right, err := g.canonicalTypedValue(FunctionColumnType{Kind: args[1].Kind}, args[1], "primitive equality argument 1")
		if err != nil {
			return FunctionValue{}, err
		}
		equal := left == right
		if op == FunctionMergeValueNotEqual {
			equal = !equal
		}
		return BoolValue(equal), nil
	}
	for i, arg := range args {
		if arg.Kind != argument {
			return FunctionValue{}, fmt.Errorf("primitive operation %d argument %d has kind %d, want %d", op, i, arg.Kind, argument)
		}
	}
	if argument == FunctionBoolColumn {
		switch op {
		case FunctionMergeBoolAnd:
			return BoolValue(args[0].Bool && args[1].Bool), nil
		case FunctionMergeBoolOr:
			return BoolValue(args[0].Bool || args[1].Bool), nil
		case FunctionMergeBoolXor:
			return BoolValue(args[0].Bool != args[1].Bool), nil
		case FunctionMergeBoolImplies:
			return BoolValue(!args[0].Bool || args[1].Bool), nil
		case FunctionMergeBoolNot:
			return BoolValue(!args[0].Bool), nil
		}
	}
	if argument == FunctionFloat64Column {
		a, b := args[0].Float64, float64(0)
		if arity == 2 {
			b = args[1].Float64
		}
		comparison := compareFloat64Values(a, b)
		switch op {
		case FunctionMergeFloat64Add:
			return Float64Value(a + b), nil
		case FunctionMergeFloat64Sub:
			return Float64Value(a - b), nil
		case FunctionMergeFloat64Mul:
			return Float64Value(a * b), nil
		case FunctionMergeFloat64Div:
			if b == 0 {
				return FunctionValue{}, fmt.Errorf("%w: float64 division", errPrimitiveNoValue)
			}
			return Float64Value(a / b), nil
		case FunctionMergeFloat64Mod:
			if b == 0 {
				return FunctionValue{}, fmt.Errorf("%w: float64 remainder", errPrimitiveNoValue)
			}
			return Float64Value(math.Mod(a, b)), nil
		case FunctionMergeFloat64Pow:
			return Float64Value(math.Pow(a, b)), nil
		case FunctionMergeFloat64Neg:
			return Float64Value(-a), nil
		case FunctionMergeFloat64Abs:
			return Float64Value(math.Abs(a)), nil
		case FunctionMergeFloat64Min:
			if comparison > 0 {
				return Float64Value(b), nil
			}
			return Float64Value(a), nil
		case FunctionMergeFloat64Max:
			if comparison < 0 {
				return Float64Value(b), nil
			}
			return Float64Value(a), nil
		case FunctionMergeFloat64ToInt64:
			if math.IsNaN(a) {
				return Int64Value(0), nil
			}
			maxInt64 := int64(^uint64(0) >> 1)
			minInt64 := -maxInt64 - 1
			if a >= float64(maxInt64) {
				return Int64Value(maxInt64), nil
			}
			if a <= float64(minInt64) {
				return Int64Value(minInt64), nil
			}
			return Int64Value(int64(a)), nil
		case FunctionMergeFloat64Equal:
			return BoolValue(comparison == 0), nil
		case FunctionMergeFloat64NotEqual:
			return BoolValue(comparison != 0), nil
		case FunctionMergeFloat64Less:
			return BoolValue(comparison < 0), nil
		case FunctionMergeFloat64LessEqual:
			return BoolValue(comparison <= 0), nil
		case FunctionMergeFloat64Greater:
			return BoolValue(comparison > 0), nil
		case FunctionMergeFloat64GreaterEqual:
			return BoolValue(comparison >= 0), nil
		}
	}
	if argument == FunctionStringColumn {
		for _, arg := range args {
			if !utf8.ValidString(arg.String) {
				return FunctionValue{}, fmt.Errorf("%w: string primitive requires valid UTF-8", errPrimitiveNoValue)
			}
		}
		switch op {
		case FunctionMergeStringConcat:
			var result strings.Builder
			for _, arg := range args {
				result.WriteString(arg.String)
			}
			return StringValue(result.String()), nil
		case FunctionMergeStringReplace:
			return StringValue(strings.ReplaceAll(args[0].String, args[1].String, args[2].String)), nil
		case FunctionMergeStringCountMatches:
			return Int64Value(int64(strings.Count(args[0].String, args[1].String))), nil
		default:
			return FunctionValue{}, fmt.Errorf("primitive operation %d has unsupported string arguments", op)
		}
	}
	a, b := args[0].Int64, int64(0)
	if arity == 2 {
		b = args[1].Int64
	}
	switch op {
	case FunctionMergeInt64Add:
		max := int64(^uint64(0) >> 1)
		min := -max - 1
		if b > 0 && a > max-b || b < 0 && a < min-b {
			return FunctionValue{}, fmt.Errorf("%w: int64 addition overflow", errPrimitiveNoValue)
		}
		return Int64Value(a + b), nil
	case FunctionMergeInt64Sub:
		max := int64(^uint64(0) >> 1)
		min := -max - 1
		if b < 0 && a > max+b || b > 0 && a < min+b {
			return FunctionValue{}, fmt.Errorf("%w: int64 subtraction overflow", errPrimitiveNoValue)
		}
		return Int64Value(a - b), nil
	case FunctionMergeInt64Mul:
		max := int64(^uint64(0) >> 1)
		min := -max - 1
		if a != 0 && (b == min && a == -1 || a == min && b == -1 || (a*b)/a != b) {
			return FunctionValue{}, fmt.Errorf("%w: int64 multiplication overflow", errPrimitiveNoValue)
		}
		return Int64Value(a * b), nil
	case FunctionMergeInt64Div:
		if b == 0 || a == (-1<<63) && b == -1 {
			return FunctionValue{}, fmt.Errorf("%w: int64 division", errPrimitiveNoValue)
		}
		return Int64Value(a / b), nil
	case FunctionMergeInt64Mod:
		if b == 0 || a == (-1<<63) && b == -1 {
			return FunctionValue{}, fmt.Errorf("%w: int64 remainder", errPrimitiveNoValue)
		}
		return Int64Value(a % b), nil
	case FunctionMergeInt64And:
		return Int64Value(a & b), nil
	case FunctionMergeInt64Or:
		return Int64Value(a | b), nil
	case FunctionMergeInt64Xor:
		return Int64Value(a ^ b), nil
	case FunctionMergeInt64ShiftLeft:
		if b < 0 || b >= 64 {
			return FunctionValue{}, fmt.Errorf("%w: int64 shift count out of range", errPrimitiveNoValue)
		}
		return Int64Value(a << uint(b)), nil
	case FunctionMergeInt64ShiftRight:
		if b < 0 || b >= 64 {
			return FunctionValue{}, fmt.Errorf("%w: int64 shift count out of range", errPrimitiveNoValue)
		}
		return Int64Value(a >> uint(b)), nil
	case FunctionMergeInt64Not:
		return Int64Value(^a), nil
	case FunctionMergeInt64Log2:
		if a <= 0 {
			return FunctionValue{}, errPrimitiveLog2Domain
		}
		return Int64Value(int64(bits.Len64(uint64(a)) - 1)), nil
	case FunctionMergeInt64ToFloat64:
		return Float64Value(float64(a)), nil
	case FunctionMergeInt64ToString:
		return StringValue(strconv.FormatInt(a, 10)), nil
	case FunctionMergeInt64Min:
		if a <= b {
			return Int64Value(a), nil
		}
		return Int64Value(b), nil
	case FunctionMergeInt64Max:
		if a >= b {
			return Int64Value(a), nil
		}
		return Int64Value(b), nil
	case FunctionMergeInt64Equal:
		return BoolValue(a == b), nil
	case FunctionMergeInt64NotEqual:
		return BoolValue(a != b), nil
	case FunctionMergeInt64Less:
		return BoolValue(a < b), nil
	case FunctionMergeInt64LessEqual:
		return BoolValue(a <= b), nil
	case FunctionMergeInt64Greater:
		return BoolValue(a > b), nil
	case FunctionMergeInt64GreaterEqual:
		return BoolValue(a >= b), nil
	default:
		return FunctionValue{}, fmt.Errorf("primitive operation %d has unsupported result kind %d", op, result)
	}
}
