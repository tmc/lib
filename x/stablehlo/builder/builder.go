package builder

import (
	"fmt"
	"io"
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
			Type: t,
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

// DotGeneral emits stablehlo.dot_general
func (b *FuncBuilder) DotGeneral(lhs, rhs Value, lhsBatch, rhsBatch, lhsContract, rhsContract []int64, retType Type) (Value, error) {
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
	out := Value{ID: b.next(), Type: retType}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.reshape %s : (%s) -> %s", out.ID, v.ID, v.Type, retType))
	return out, nil
}

// Transpose emits stablehlo.transpose
func (b *FuncBuilder) Transpose(v Value, permutation []int64, retType Type) (Value, error) {
	out := Value{ID: b.next(), Type: retType}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.transpose %s, permutation = %s : (%s) -> %s", 
		out.ID, v.ID, fmtArr(permutation), v.Type, retType))
	return out, nil
}

// Slice emits stablehlo.slice
func (b *FuncBuilder) Slice(v Value, startIndices, limitIndices, strides []int64, retType Type) (Value, error) {
	out := Value{ID: b.next(), Type: retType}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.slice %s, start_indices = %s, limit_indices = %s, strides = %s : (%s) -> %s", 
		out.ID, v.ID, fmtArr(startIndices), fmtArr(limitIndices), fmtArr(strides), v.Type, retType))
	return out, nil
}

// Concatenate emits stablehlo.concatenate
func (b *FuncBuilder) Concatenate(vs []Value, dimension int64, retType Type) (Value, error) {
	out := Value{ID: b.next(), Type: retType}
	
	var ids []string
	var types []string
	for _, v := range vs {
		ids = append(ids, v.ID)
		types = append(types, v.Type.String())
	}
	
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.concatenate %s, dimension = %d : (%s) -> %s", 
		out.ID, strings.Join(ids, ", "), dimension, strings.Join(types, ", "), retType))
	return out, nil
}

// Constant emits stablehlo.constant
func (b *FuncBuilder) Constant(value float32, retType Type) (Value, error) {
	out := Value{ID: b.next(), Type: retType}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%f> : %s", out.ID, value, retType))
	return out, nil
}

// BroadcastInDim emits stablehlo.broadcast_in_dim
func (b *FuncBuilder) BroadcastInDim(v Value, broadcastDims []int64, retType Type) (Value, error) {
	out := Value{ID: b.next(), Type: retType}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.broadcast_in_dim %s, dims = %s : (%s) -> %s", 
		out.ID, v.ID, fmtArr(broadcastDims), v.Type, retType))
	return out, nil
}

// Reduce emits stablehlo.reduce
func (b *FuncBuilder) Reduce(operand, initValue Value, dimensions []int64, retType Type, reducerOp string) (Value, error) {
	out := Value{ID: b.next(), Type: retType}

	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.reduce(%s init: %s) applies stablehlo.%s across dimensions = %s : (%s, %s) -> %s",
		out.ID, operand.ID, initValue.ID, reducerOp, fmtArr(dimensions), operand.Type, initValue.Type, retType))

	return out, nil
}

// RMSNorm builds an RMS Normalization subgraph.
func (b *FuncBuilder) RMSNorm(x, gamma Value, featureDim int64, epsilon float32) (Value, error) {
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
