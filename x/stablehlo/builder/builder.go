package builder

import (
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
)

// Type represents a StableHLO tensor type.
type Type struct {
	Shape       []int64
	ElementType string
}

// String returns the MLIR representation of the type.
func (t Type) String() string {
	if len(t.Shape) == 0 {
		return "tensor<" + t.ElementType + ">"
	}
	var b strings.Builder
	b.WriteString("tensor<")
	for _, dim := range t.Shape {
		if dim < 0 {
			b.WriteString("?x")
		} else {
			b.WriteString(fmt.Sprintf("%dx", dim))
		}
	}
	b.WriteString(t.ElementType)
	b.WriteString(">")
	return b.String()
}

// Value represents an SSA value in MLIR.
type Value struct {
	ID   string
	Type Type
}

// FuncBuilder constructs a func.func containing StableHLO operations.
type FuncBuilder struct {
	Name    string
	Args    []Value
	Ops     []string
	Returns []Value
	nextID  int
}

// NewFunc creates a new FuncBuilder with the given name and argument types.
func NewFunc(name string, args ...Type) *FuncBuilder {
	b := &FuncBuilder{Name: name}
	for i, t := range args {
		b.Args = append(b.Args, Value{
			ID:   fmt.Sprintf("%%arg%d", i),
			Type: Type{Shape: slices.Clone(t.Shape), ElementType: t.ElementType},
		})
	}
	return b
}

// Arg returns the i-th argument.
func (b *FuncBuilder) Arg(i int) Value {
	return b.Args[i]
}

func (b *FuncBuilder) next() string {
	id := fmt.Sprintf("%%%d", b.nextID)
	b.nextID++
	return id
}

// Emit writes the function to the given io.Writer.
func (b *FuncBuilder) Emit(w io.Writer) error {
	var argStrs []string
	for _, arg := range b.Args {
		argStrs = append(argStrs, fmt.Sprintf("%s: %s", arg.ID, arg.Type))
	}

	var retStrs []string
	for _, ret := range b.Returns {
		retStrs = append(retStrs, ret.Type.String())
	}

	retSig := ""
	if len(retStrs) > 0 {
		if len(retStrs) == 1 {
			retSig = " -> " + retStrs[0]
		} else {
			retSig = " -> (" + strings.Join(retStrs, ", ") + ")"
		}
	}

	sig := fmt.Sprintf("  func.func @%s(%s)%s {\n", b.Name, strings.Join(argStrs, ", "), retSig)
	if _, err := io.WriteString(w, sig); err != nil {
		return fmt.Errorf("emit func signature: %w", err)
	}

	for _, op := range b.Ops {
		if _, err := io.WriteString(w, "    "+op+"\n"); err != nil {
			return fmt.Errorf("emit op: %w", err)
		}
	}

	var retValStrs []string
	for _, ret := range b.Returns {
		retValStrs = append(retValStrs, ret.ID)
	}
	retTypeStrs := ""
	if len(retStrs) > 0 {
		retTypeStrs = " : " + strings.Join(retStrs, ", ")
	}

	retStmt := fmt.Sprintf("    return %s%s\n", strings.Join(retValStrs, ", "), retTypeStrs)
	if _, err := io.WriteString(w, retStmt); err != nil {
		return fmt.Errorf("emit func return: %w", err)
	}

	if _, err := io.WriteString(w, "  }\n"); err != nil {
		return fmt.Errorf("emit func end: %w", err)
	}
	return nil
}

// Return sets the return values of the function.
func (b *FuncBuilder) Return(vs ...Value) {
	b.Returns = vs
}

// Add emits stablehlo.add
func (b *FuncBuilder) Add(lhs, rhs Value) (Value, error) {
	return b.binaryOp("stablehlo.add", lhs, rhs)
}

// Multiply emits stablehlo.multiply
func (b *FuncBuilder) Multiply(lhs, rhs Value) (Value, error) {
	return b.binaryOp("stablehlo.multiply", lhs, rhs)
}

// Divide emits stablehlo.divide
func (b *FuncBuilder) Divide(lhs, rhs Value) (Value, error) {
	return b.binaryOp("stablehlo.divide", lhs, rhs)
}

// Subtract emits stablehlo.subtract
func (b *FuncBuilder) Subtract(lhs, rhs Value) (Value, error) {
	return b.binaryOp("stablehlo.subtract", lhs, rhs)
}

func (b *FuncBuilder) binaryOp(op string, lhs, rhs Value) (Value, error) {
	if err := sameElements(lhs.Type, rhs.Type); err != nil {
		return Value{}, fmt.Errorf("%s: %w", op, err)
	}
	if lhs.Type.String() != rhs.Type.String() {
		return Value{}, fmt.Errorf("%s type mismatch: %s != %s", op, lhs.Type, rhs.Type)
	}
	out := Value{ID: b.next(), Type: lhs.Type}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = %s %s, %s : %s", out.ID, op, lhs.ID, rhs.ID, lhs.Type))
	return out, nil
}

// Rsqrt emits stablehlo.rsqrt
func (b *FuncBuilder) Rsqrt(v Value) (Value, error) {
	return b.unaryOp("stablehlo.rsqrt", v)
}

// Exponential emits stablehlo.exponential
func (b *FuncBuilder) Exponential(v Value) (Value, error) {
	return b.unaryOp("stablehlo.exponential", v)
}

func (b *FuncBuilder) unaryOp(op string, v Value) (Value, error) {
	if err := checkType(v.Type); err != nil {
		return Value{}, fmt.Errorf("%s: %w", op, err)
	}
	out := Value{ID: b.next(), Type: v.Type}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = %s %s : %s", out.ID, op, v.ID, v.Type))
	return out, nil
}

func fmtArr(arr []int64) string {
	if len(arr) == 0 {
		return "[]"
	}
	var strs []string
	for _, x := range arr {
		strs = append(strs, fmt.Sprintf("%d", x))
	}
	return "[" + strings.Join(strs, ", ") + "]"
}

func fmtI64Array(arr []int64) string {
	if len(arr) == 0 {
		return "array<i64>"
	}
	return "array<i64: " + strings.Trim(fmtArr(arr), "[]") + ">"
}

// DotGeneral emits stablehlo.dot_general
func (b *FuncBuilder) DotGeneral(lhs, rhs Value, lhsBatch, rhsBatch, lhsContract, rhsContract []int64, retType Type) (Value, error) {
	if err := sameElements(lhs.Type, rhs.Type, retType); err != nil {
		return Value{}, fmt.Errorf("dot_general: %w", err)
	}
	shape, err := dotShape(lhs.Type, rhs.Type, lhsBatch, rhsBatch, lhsContract, rhsContract)
	if err != nil {
		return Value{}, fmt.Errorf("dot_general: %w", err)
	}
	if err := checkShape(retType, shape); err != nil {
		return Value{}, fmt.Errorf("dot_general: %w", err)
	}

	out := Value{ID: b.next(), Type: retType}

	bDims := ""
	if len(lhsBatch) > 0 || len(rhsBatch) > 0 {
		bDims = fmt.Sprintf("batching_dims = %s x %s, ", fmtArr(lhsBatch), fmtArr(rhsBatch))
	}
	cDims := fmt.Sprintf("contracting_dims = %s x %s", fmtArr(lhsContract), fmtArr(rhsContract))

	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.dot_general %s, %s, %s%s : (%s, %s) -> %s",
		out.ID, lhs.ID, rhs.ID, bDims, cDims, lhs.Type, rhs.Type, retType))
	return out, nil
}

// Reshape emits stablehlo.reshape
func (b *FuncBuilder) Reshape(v Value, retType Type) (Value, error) {
	if err := sameElements(v.Type, retType); err != nil {
		return Value{}, fmt.Errorf("reshape: %w", err)
	}
	in, inKnown, err := elements(v.Type.Shape)
	if err != nil {
		return Value{}, fmt.Errorf("reshape: %w", err)
	}
	outCount, outKnown, err := elements(retType.Shape)
	if err != nil {
		return Value{}, fmt.Errorf("reshape: %w", err)
	}
	if inKnown && outKnown && in != outCount {
		return Value{}, fmt.Errorf("reshape: element count changes from %d to %d", in, outCount)
	}

	out := Value{ID: b.next(), Type: retType}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.reshape %s : (%s) -> %s", out.ID, v.ID, v.Type, retType))
	return out, nil
}

// Transpose emits stablehlo.transpose
func (b *FuncBuilder) Transpose(v Value, permutation []int64, retType Type) (Value, error) {
	if err := sameElements(v.Type, retType); err != nil {
		return Value{}, fmt.Errorf("transpose: %w", err)
	}
	if len(permutation) != len(v.Type.Shape) {
		return Value{}, fmt.Errorf("transpose: permutation length differs from rank")
	}
	if _, err := axes(len(v.Type.Shape), permutation); err != nil {
		return Value{}, fmt.Errorf("transpose: %w", err)
	}
	shape := make([]int64, len(permutation))
	for i, d := range permutation {
		shape[i] = v.Type.Shape[d]
	}
	if err := checkShape(retType, shape); err != nil {
		return Value{}, fmt.Errorf("transpose: %w", err)
	}

	out := Value{ID: b.next(), Type: retType}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.transpose %s, dims = %s : (%s) -> %s",
		out.ID, v.ID, fmtArr(permutation), v.Type, retType))
	return out, nil
}

// Slice emits stablehlo.slice
func (b *FuncBuilder) Slice(v Value, startIndices, limitIndices, strides []int64, retType Type) (Value, error) {
	if err := sameElements(v.Type, retType); err != nil {
		return Value{}, fmt.Errorf("slice: %w", err)
	}
	rank := len(v.Type.Shape)
	if len(startIndices) != rank || len(limitIndices) != rank || len(strides) != rank {
		return Value{}, fmt.Errorf("slice: index and stride lengths must match rank")
	}
	shape := make([]int64, rank)
	for i, n := range v.Type.Shape {
		start, limit, stride := startIndices[i], limitIndices[i], strides[i]
		if start < 0 || limit < start || stride <= 0 || (n >= 0 && limit > n) {
			return Value{}, fmt.Errorf("slice: invalid bounds or stride at axis %d", i)
		}
		if limit > start {
			shape[i] = 1 + (limit-start-1)/stride
		}
	}
	if err := checkShape(retType, shape); err != nil {
		return Value{}, fmt.Errorf("slice: %w", err)
	}

	out := Value{ID: b.next(), Type: retType}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = \"stablehlo.slice\"(%s) {start_indices = %s, limit_indices = %s, strides = %s} : (%s) -> %s",
		out.ID, v.ID, fmtI64Array(startIndices), fmtI64Array(limitIndices), fmtI64Array(strides), v.Type, retType))
	return out, nil
}

// Concatenate emits stablehlo.concatenate
func (b *FuncBuilder) Concatenate(vs []Value, dimension int64, retType Type) (Value, error) {
	if len(vs) == 0 {
		return Value{}, fmt.Errorf("concatenate: no operands")
	}
	if _, err := axes(len(retType.Shape), []int64{dimension}); err != nil {
		return Value{}, fmt.Errorf("concatenate: %w", err)
	}
	shape := slices.Clone(retType.Shape)
	shape[dimension] = 0
	for _, v := range vs {
		if err := sameElements(v.Type, retType); err != nil {
			return Value{}, fmt.Errorf("concatenate: %w", err)
		}
		if len(v.Type.Shape) != len(shape) {
			return Value{}, fmt.Errorf("concatenate: operand rank differs from result")
		}
		for i, n := range v.Type.Shape {
			if int64(i) == dimension {
				if n < 0 || shape[i] < 0 {
					shape[i] = -1
				} else {
					if n > math.MaxInt64-shape[i] {
						return Value{}, fmt.Errorf("concatenate: dimension overflows int64")
					}
					shape[i] += n
				}
			} else if !compatibleDim(n, shape[i]) {
				return Value{}, fmt.Errorf("concatenate: operand dimension %d differs", i)
			} else if shape[i] < 0 {
				shape[i] = n
			}
		}
	}
	if err := checkShape(retType, shape); err != nil {
		return Value{}, fmt.Errorf("concatenate: %w", err)
	}

	out := Value{ID: b.next(), Type: retType}

	var ids []string
	var types []string
	for _, v := range vs {
		ids = append(ids, v.ID)
		types = append(types, v.Type.String())
	}

	b.Ops = append(b.Ops, fmt.Sprintf("%s = \"stablehlo.concatenate\"(%s) {dimension = %d : i64} : (%s) -> %s",
		out.ID, strings.Join(ids, ", "), dimension, strings.Join(types, ", "), retType))
	return out, nil
}

// Constant emits stablehlo.constant
func (b *FuncBuilder) Constant(value float32, retType Type) (Value, error) {
	if err := checkType(retType); err != nil {
		return Value{}, fmt.Errorf("constant: %w", err)
	}
	if !floatType(retType.ElementType) {
		return Value{}, fmt.Errorf("constant: floating-point element type required")
	}
	for _, d := range retType.Shape {
		if d < 0 {
			return Value{}, fmt.Errorf("constant: static shape required")
		}
	}
	literal := strconv.FormatFloat(float64(value), 'g', -1, 32)
	if !strings.Contains(literal, ".") {
		if i := strings.IndexAny(literal, "eE"); i >= 0 {
			literal = literal[:i] + ".0" + literal[i:]
		} else {
			literal += ".0"
		}
	}

	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		switch retType.ElementType {
		case "f32":
			literal = fmt.Sprintf("0x%08X", math.Float32bits(value))
		case "f64":
			literal = fmt.Sprintf("0x%016X", math.Float64bits(float64(value)))
		case "f16", "bf16":
			bits := uint16(0x7c00)
			if retType.ElementType == "bf16" {
				bits = 0x7f80
			}
			if math.IsNaN(float64(value)) {
				bits |= 1
			}
			if math.Signbit(float64(value)) {
				bits |= 0x8000
			}
			literal = fmt.Sprintf("0x%04X", bits)
		}
	}
	out := Value{ID: b.next(), Type: retType}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%s> : %s", out.ID, literal, retType))
	return out, nil
}

// BroadcastInDim emits stablehlo.broadcast_in_dim
func (b *FuncBuilder) BroadcastInDim(v Value, broadcastDims []int64, retType Type) (Value, error) {
	if err := sameElements(v.Type, retType); err != nil {
		return Value{}, fmt.Errorf("broadcast_in_dim: %w", err)
	}
	if len(broadcastDims) != len(v.Type.Shape) {
		return Value{}, fmt.Errorf("broadcast_in_dim: mapping length differs from operand rank")
	}
	if _, err := axes(len(retType.Shape), broadcastDims); err != nil {
		return Value{}, fmt.Errorf("broadcast_in_dim: %w", err)
	}
	for i, d := range broadcastDims {
		if v.Type.Shape[i] != 1 && !compatibleDim(v.Type.Shape[i], retType.Shape[d]) {
			return Value{}, fmt.Errorf("broadcast_in_dim: incompatible dimension %d", i)
		}
	}

	out := Value{ID: b.next(), Type: retType}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.broadcast_in_dim %s, dims = %s : (%s) -> %s",
		out.ID, v.ID, fmtArr(broadcastDims), v.Type, retType))
	return out, nil
}

// Reduce emits stablehlo.reduce
func (b *FuncBuilder) Reduce(operand, initValue Value, dimensions []int64, retType Type, reducerOp string) (Value, error) {
	if err := sameElements(operand.Type, initValue.Type, retType); err != nil {
		return Value{}, fmt.Errorf("reduce: %w", err)
	}
	if len(initValue.Type.Shape) != 0 {
		return Value{}, fmt.Errorf("reduce: initial value must be scalar")
	}
	used, err := axes(len(operand.Type.Shape), dimensions)
	if err != nil {
		return Value{}, fmt.Errorf("reduce: %w", err)
	}
	var shape []int64
	for i, n := range operand.Type.Shape {
		if !used[i] {
			shape = append(shape, n)
		}
	}
	if err := checkShape(retType, shape); err != nil {
		return Value{}, fmt.Errorf("reduce: %w", err)
	}
	switch reducerOp {
	case "add", "multiply", "maximum", "minimum", "and", "or", "xor":
	default:
		return Value{}, fmt.Errorf("reduce: unsupported reducer %q", reducerOp)
	}

	out := Value{ID: b.next(), Type: retType}

	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.reduce(%s init: %s) applies stablehlo.%s across dimensions = %s : (%s, %s) -> %s",
		out.ID, operand.ID, initValue.ID, reducerOp, fmtArr(dimensions), operand.Type, initValue.Type, retType))

	return out, nil
}

// RMSNorm normalizes x along featureDim and multiplies by the scale gamma.
// F16 and BF16 inputs compute in F32, including the scale multiplication, and
// convert once at the end. F32 and F64 inputs retain their arithmetic type.
// An error leaves the builder unchanged.
func (b *FuncBuilder) RMSNorm(x, gamma Value, featureDim int64, epsilon float32) (out Value, err error) {
	if !floatType(x.Type.ElementType) {
		return Value{}, fmt.Errorf("rmsnorm: floating-point element type required")
	}
	for _, d := range x.Type.Shape {
		if d < 0 {
			return Value{}, fmt.Errorf("rmsnorm: static shape required")
		}
	}

	if err := sameElements(x.Type, gamma.Type); err != nil {
		return Value{}, fmt.Errorf("rmsnorm: %w", err)
	}
	if featureDim < 0 || featureDim >= int64(len(x.Type.Shape)) {
		return Value{}, fmt.Errorf("rmsnorm: feature axis outside rank")
	}
	if x.Type.Shape[featureDim] <= 0 {
		return Value{}, fmt.Errorf("rmsnorm: feature dimension must be positive and static")
	}
	if len(gamma.Type.Shape) != 1 || gamma.Type.Shape[0] != x.Type.Shape[featureDim] {
		return Value{}, fmt.Errorf("rmsnorm: gamma must match feature dimension")
	}
	if epsilon <= 0 || math.IsInf(float64(epsilon), 0) || math.IsNaN(float64(epsilon)) {
		return Value{}, fmt.Errorf("rmsnorm: epsilon must be positive and finite")
	}

	nops, next := len(b.Ops), b.nextID
	defer func() {
		if err != nil {
			b.Ops, b.nextID = b.Ops[:nops], next
		}
	}()
	element := x.Type.ElementType
	if element == "f16" || element == "bf16" {
		x, gamma = b.convert(x, "f32"), b.convert(gamma, "f32")
	}
	out, err = b.rmsNorm(x, gamma, featureDim, epsilon)
	if err != nil {
		return Value{}, err
	}
	return b.convert(out, element), nil
}

func (b *FuncBuilder) rmsNorm(x, gamma Value, featureDim int64, epsilon float32) (Value, error) {
	// 1. x_squared = x * x
	xSq, err := b.Multiply(x, x)
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm multiply: %w", err)
	}

	// 2. sum = reduce_sum(x_squared)
	scalarType := Type{ElementType: x.Type.ElementType}
	zero, err := b.Constant(0.0, scalarType)
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm constant zero: %w", err)
	}

	var reducedShape []int64
	for i, dim := range x.Type.Shape {
		if int64(i) != featureDim {
			reducedShape = append(reducedShape, dim)
		}
	}
	reducedType := Type{Shape: reducedShape, ElementType: x.Type.ElementType}

	sum, err := b.Reduce(xSq, zero, []int64{featureDim}, reducedType, "add")
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm reduce sum: %w", err)
	}

	// 3. mean = sum / features
	featureSize := float32(x.Type.Shape[featureDim])
	// We need a constant tensor of reducedType with value featureSize.
	featureSizeConst, err := b.Constant(featureSize, reducedType)
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm constant feature_size: %w", err)
	}

	mean, err := b.Divide(sum, featureSizeConst)
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm divide: %w", err)
	}

	// 4. mean + epsilon
	epsConst, err := b.Constant(epsilon, reducedType)
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm constant epsilon: %w", err)
	}

	meanEps, err := b.Add(mean, epsConst)
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm add epsilon: %w", err)
	}

	// 5. rsqrt
	invStdDev, err := b.Rsqrt(meanEps)
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm rsqrt: %w", err)
	}

	// 6. Broadcast invStdDev to x's shape
	var bcastDims []int64
	for i := range x.Type.Shape {
		if int64(i) != featureDim {
			bcastDims = append(bcastDims, int64(i))
		}
	}

	invStdDevBcast, err := b.BroadcastInDim(invStdDev, bcastDims, x.Type)
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm broadcast_in_dim: %w", err)
	}

	// 7. Multiply x by invStdDevBcast
	normed, err := b.Multiply(x, invStdDevBcast)
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm multiply inv_std_dev: %w", err)
	}

	// 8. Multiply by gamma
	gammaBcast, err := b.BroadcastInDim(gamma, []int64{featureDim}, x.Type)
	if err != nil {
		return Value{}, fmt.Errorf("rmsnorm broadcast gamma: %w", err)
	}

	return b.Multiply(normed, gammaBcast)
}
