package hopper

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/tileir"
)

// TileIRStep represents a hopper kernel step (RowKernel, TileGEMM, SkinnyGEMM, or PlanKernel)
// that can be lowered to a Tile IR Module and verified against CUDA host emulation.
type TileIRStep interface {
	stepKernel() PlanKernel
}

func (pk PlanKernel) stepKernel() PlanKernel { return pk }

func (k RowKernel) stepKernel() PlanKernel {
	rk := k
	return PlanKernel{Row: &rk}
}

func (g TileGEMM) stepKernel() PlanKernel {
	tg := g
	return PlanKernel{Tile: &tg}
}

func (g SkinnyGEMM) stepKernel() PlanKernel {
	sg := g
	return PlanKernel{GEMM: &sg}
}

// NewRMSNormKernel returns a RowKernel configured for RMSNorm over row width D:
//
//	out[m, c] = x[m, c] * rsqrt(mean(x[m, :]^2) + eps) * weight[c]
func NewRMSNormKernel(name string, d int, eps float32) RowKernel {
	if name == "" {
		name = "rmsnorm_f32"
	}
	if d <= 0 {
		d = 64
	}
	if eps <= 0 {
		eps = 1e-5
	}
	normBody := fmt.Sprintf("(mul (mul $0 (rsqrt (add (div $1 %d) %.9g))) $2)", d, eps)
	return RowKernel{
		Name: name,
		D:    d,
		Inputs: []RowTensor{
			{Name: "x", DType: F32, Kind: RowData},
			{Name: "weight", DType: F32, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "ss", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"x"}},
			{Name: "y", Body: MustLambda(normBody), Args: []string{"x", "ss", "weight"}},
		},
		Outputs: []RowTensor{
			{Name: "out", DType: F32, Kind: RowData, Value: "y"},
		},
	}
}

// NewResidualRMSNormKernel returns a RowKernel configured for fused ResidualAdd + RMSNorm:
//
//	h[m, c]   = x[m, c] + residual[m, c]
//	out[m, c] = h[m, c] * rsqrt(mean(h[m, :]^2) + eps) * weight[c]
func NewResidualRMSNormKernel(name string, d int, eps float32) RowKernel {
	if name == "" {
		name = "residual_rmsnorm_f32"
	}
	if d <= 0 {
		d = 64
	}
	if eps <= 0 {
		eps = 1e-5
	}
	normBody := fmt.Sprintf("(mul (mul $0 (rsqrt (add (div $1 %d) %.9g))) $2)", d, eps)
	return RowKernel{
		Name: name,
		D:    d,
		Inputs: []RowTensor{
			{Name: "x", DType: F32, Kind: RowData},
			{Name: "residual", DType: F32, Kind: RowData},
			{Name: "weight", DType: F32, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "h", Body: MustLambda("(add $0 $1)"), Args: []string{"x", "residual"}},
			{Name: "ss", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"h"}},
			{Name: "y", Body: MustLambda(normBody), Args: []string{"h", "ss", "weight"}},
		},
		Outputs: []RowTensor{
			{Name: "out", DType: F32, Kind: RowData, Value: "y"},
		},
	}
}

// NewSwiGLUKernel returns a RowKernel configured for elementwise SwiGLU:
//
//	out[m, c] = silu(gate[m, c]) * up[m, c]
func NewSwiGLUKernel(name string, d int) RowKernel {
	if name == "" {
		name = "swiglu_f32"
	}
	if d <= 0 {
		d = 64
	}
	return RowKernel{
		Name: name,
		D:    d,
		Inputs: []RowTensor{
			{Name: "gate", DType: F32, Kind: RowData},
			{Name: "up", DType: F32, Kind: RowData},
		},
		Values: []RowValue{
			{Name: "y", Body: MustLambda("(mul (silu $0) $1)"), Args: []string{"gate", "up"}},
		},
		Outputs: []RowTensor{
			{Name: "out", DType: F32, Kind: RowData, Value: "y"},
		},
	}
}

// FormatTileIR formats a Tile IR Module into its textual representation.
func FormatTileIR(mod *tileir.Module) string {
	if mod == nil {
		return ""
	}
	return mod.String()
}

// ParseTileIR parses textual Tile IR into a Module.
func ParseTileIR(src string) (*tileir.Module, error) {
	return tileir.ParseString(src)
}

// EncodeTileIR encodes a Tile IR Module into binary Tile IR bytecode.
func EncodeTileIR(mod *tileir.Module) ([]byte, error) {
	if mod == nil {
		return nil, fmt.Errorf("hopper: nil tileir module")
	}
	return mod.EncodeBytecode()
}

// StepToTileIR lowers a hopper step (RowKernel, TileGEMM, SkinnyGEMM, or PlanKernel)
// into a valid, bytecode-encodable *tileir.Module.
func StepToTileIR(step TileIRStep) (*tileir.Module, error) {
	if step == nil {
		return nil, fmt.Errorf("hopper: nil step")
	}
	pk := step.stepKernel()
	switch {
	case pk.Row != nil:
		return rowKernelToTileIR(pk.Row)
	case pk.Tile != nil:
		return tileGEMMToTileIR(pk.Tile, "tile_gemm")
	case pk.GEMM != nil:
		tg := &TileGEMM{
			Name:     pk.GEMM.Name,
			N:        pk.GEMM.N,
			K:        pk.GEMM.K,
			Weight:   pk.GEMM.Weight,
			Out:      pk.GEMM.Out,
			Epilogue: pk.GEMM.Epilogue,
			Extras:   pk.GEMM.Extras,
		}
		return tileGEMMToTileIR(tg, "skinny_gemm")
	default:
		return nil, fmt.Errorf("hopper: unsupported PlanKernel step (need Row, Tile, or GEMM)")
	}
}

// ToTileIRModules lowers all supported steps in a Plan into Tile IR modules,
// verifying that every module round-trips through Format -> Parse and encodes
// cleanly to Tile IR bytecode.
func (p *Plan) ToTileIRModules() ([]*tileir.Module, error) {
	if p == nil {
		return nil, fmt.Errorf("hopper: nil plan")
	}
	var modules []*tileir.Module
	for i, pk := range p.Kernels {
		if pk.Row == nil && pk.Tile == nil && pk.GEMM == nil {
			continue
		}
		mod, err := StepToTileIR(pk)
		if err != nil {
			return nil, fmt.Errorf("hopper: step %d: %w", i, err)
		}
		text := FormatTileIR(mod)
		parsed, err := ParseTileIR(text)
		if err != nil {
			return nil, fmt.Errorf("hopper: step %d parse round-trip: %w", i, err)
		}
		if _, err := EncodeTileIR(parsed); err != nil {
			return nil, fmt.Errorf("hopper: step %d bytecode encode: %w", i, err)
		}
		modules = append(modules, mod)
	}
	return modules, nil
}

type tileBuilder struct {
	ops    []*tileir.Operation
	nextID int
}

func (b *tileBuilder) ssa(prefix string) string {
	id := b.nextID
	b.nextID++
	return fmt.Sprintf("%%%s_%d", prefix, id)
}

func (b *tileBuilder) emit(results []string, opCode string, operands []string, attrStr string, ty tileir.Type) *tileir.Operation {
	fullOp := opCode
	if !strings.HasPrefix(fullOp, "cuda_tile.") {
		fullOp = "cuda_tile." + fullOp
	}
	var allParts []string
	allParts = append(allParts, operands...)
	if attrStr != "" {
		allParts = append(allParts, attrStr)
	}
	var raw strings.Builder
	if len(results) > 0 {
		raw.WriteString(strings.Join(results, ", "))
		raw.WriteString(" = ")
	}
	raw.WriteString(fullOp)
	if len(allParts) > 0 {
		raw.WriteString(" ")
		raw.WriteString(strings.Join(allParts, ", "))
	}
	var typeStr string
	if ty != nil {
		typeStr = ty.String()
		raw.WriteString(" : ")
		raw.WriteString(typeStr)
	}
	op := &tileir.Operation{
		Results:    results,
		OpCode:     fullOp,
		Operands:   operands,
		Attributes: attrStr,
		Type:       typeStr,
		TypeVal:    ty,
		Raw:        raw.String(),
	}
	b.ops = append(b.ops, op)
	return op
}

func (b *tileBuilder) constF32(prefix string, val float32, shape []int64) string {
	res := b.ssa(prefix)
	ty := tileir.NewTile(shape, tileir.Float32)
	valStr := strconv.FormatFloat(float64(val), 'e', 8, 32)
	if val == float32(int32(val)) && math.Abs(float64(val)) < 1e6 {
		valStr = fmt.Sprintf("%.1f", val)
	}
	attr := fmt.Sprintf("{value = %s}", valStr)
	b.emit([]string{res}, "constant", nil, attr, ty)
	return res
}

func (b *tileBuilder) constF64(prefix string, val float64, shape []int64) string {
	res := b.ssa(prefix)
	ty := tileir.NewTile(shape, tileir.Float64)
	valStr := fmt.Sprintf("%.1f", val)
	attr := fmt.Sprintf("{value = %s}", valStr)
	b.emit([]string{res}, "constant", nil, attr, ty)
	return res
}

func (b *tileBuilder) constI32(prefix string, val int, shape []int64) string {
	res := b.ssa(prefix)
	ty := tileir.NewTile(shape, tileir.Int32)
	attr := fmt.Sprintf("{value = %d}", val)
	b.emit([]string{res}, "constant", nil, attr, ty)
	return res
}

func (b *tileBuilder) constI32List(prefix string, vals []int, shape []int64) string {
	res := b.ssa(prefix)
	ty := tileir.NewTile(shape, tileir.Int32)
	strs := make([]string, len(vals))
	for i, v := range vals {
		strs[i] = strconv.Itoa(v)
	}
	attr := fmt.Sprintf("{value = [%s]}", strings.Join(strs, ", "))
	b.emit([]string{res}, "constant", nil, attr, ty)
	return res
}

func (b *tileBuilder) binOpF(opCode, lhs, rhs string, ty tileir.Type) string {
	res := b.ssa(opCode)
	b.emit([]string{res}, opCode, []string{lhs, rhs}, "{rounding_mode = nearest_even}", ty)
	return res
}

func (b *tileBuilder) binOpI(opCode, lhs, rhs string, ty tileir.Type) string {
	res := b.ssa(opCode)
	b.emit([]string{res}, opCode, []string{lhs, rhs}, "{overflow = no_signed_wrap}", ty)
	return res
}

func (b *tileBuilder) negF(x string, ty tileir.Type) string {
	res := b.ssa("negf")
	b.emit([]string{res}, "negf", []string{x}, "", ty)
	return res
}

func (b *tileBuilder) absF(x string, ty tileir.Type) string {
	res := b.ssa("absf")
	b.emit([]string{res}, "absf", []string{x}, "", ty)
	return res
}

func (b *tileBuilder) maxF(lhs, rhs string, ty tileir.Type) string {
	res := b.ssa("maxf")
	b.emit([]string{res}, "maxf", []string{lhs, rhs}, "", ty)
	return res
}

func (b *tileBuilder) minF(lhs, rhs string, ty tileir.Type) string {
	res := b.ssa("minf")
	b.emit([]string{res}, "minf", []string{lhs, rhs}, "", ty)
	return res
}

func (b *tileBuilder) cmpF(pred, lhs, rhs string, shape []int64) string {
	res := b.ssa("cmpf")
	ty := tileir.NewTile(shape, tileir.Int1)
	attr := fmt.Sprintf("{comparison_predicate = %s, comparison_ordering = ordered}", pred)
	b.emit([]string{res}, "cmpf", []string{lhs, rhs}, attr, ty)
	return res
}

func (b *tileBuilder) selectF(cond, tVal, fVal string, ty tileir.Type) string {
	res := b.ssa("sel")
	b.emit([]string{res}, "select", []string{cond, tVal, fVal}, "", ty)
	return res
}

func (b *tileBuilder) ftof(x string, dstTy tileir.Type) string {
	res := b.ssa("ftof")
	b.emit([]string{res}, "ftof", []string{x}, "{rounding_mode = nearest_even}", dstTy)
	return res
}

func (b *tileBuilder) reshape(x string, dstTy tileir.Type) string {
	res := b.ssa("reshape")
	b.emit([]string{res}, "reshape", []string{x}, "", dstTy)
	return res
}

func (b *tileBuilder) broadcast(x string, dstTy tileir.Type) string {
	res := b.ssa("bcast")
	b.emit([]string{res}, "broadcast", []string{x}, "", dstTy)
	return res
}

func (b *tileBuilder) extiU64(x string, shape []int64) string {
	res := b.ssa("exti")
	ty := tileir.NewTile(shape, tileir.Int64)
	b.emit([]string{res}, "exti", []string{x}, "{signedness = unsigned}", ty)
	return res
}

func (b *tileBuilder) offsetPtr(basePtr, idxI64 string, shape []int64) string {
	ptrTileTy := tileir.NewTile(shape, tileir.NewPointer(tileir.Float32))
	baseTile := basePtr
	if len(shape) > 0 {
		if len(shape) == 1 && shape[0] == 1 {
			baseTile = b.reshape(basePtr, ptrTileTy)
		} else {
			r1 := b.reshape(basePtr, tileir.NewTile([]int64{1}, tileir.NewPointer(tileir.Float32)))
			baseTile = b.broadcast(r1, ptrTileTy)
		}
	}
	res := b.ssa("ptr")
	b.emit([]string{res}, "offset", []string{baseTile, idxI64}, "", ptrTileTy)
	return res
}

func (b *tileBuilder) loadTile(ptrTile, tokIn string, shape []int64) (string, string) {
	val := b.ssa("ld")
	tokOut := b.ssa("tok")
	ty := tileir.NewTile(shape, tileir.Float32)
	tupleTy := tileir.NewFunction(nil, []tileir.Type{ty, tileir.Token})
	b.emit([]string{val, tokOut}, "load_ptr_tko", []string{ptrTile, tokIn}, "{memory_ordering_semantics = weak}", tupleTy)
	return val, tokOut
}

func (b *tileBuilder) storeTile(ptrTile, val, tokIn string, shape []int64) string {
	tokOut := b.ssa("tok")
	_ = shape
	b.emit([]string{tokOut}, "store_ptr_tko", []string{ptrTile, val, tokIn}, "{memory_ordering_semantics = weak}", tileir.Token)
	return tokOut
}

var rsqrtThresholds = []float32{
	4096.0,      // 4^6
	1024.0,      // 4^5
	256.0,       // 4^4
	64.0,        // 4^3
	16.0,        // 4^2
	4.0,         // 4^1
	1.0,         // 4^0
	0.25,        // 4^-1
	0.0625,      // 4^-2
	0.015625,    // 4^-3
	0.00390625,  // 4^-4
	9.765625e-4, // 4^-5
	2.441406e-4, // 4^-6
	6.103516e-5, // 4^-7
}

// emitRsqrt emits a range-switched initial guess followed by 6 Newton-Raphson iterations:
//
//	y_{k+1} = y_k * (1.5 - 0.5 * x * y_k * y_k)
func (b *tileBuilder) emitRsqrt(x string, shape []int64) string {
	ty := tileir.NewTile(shape, tileir.Float32)
	cHalf := b.constF32("half", 0.5, shape)
	cThreeHalf := b.constF32("three_half", 1.5, shape)

	y := b.constF32("g_init", float32(1.0/math.Sqrt(8192.0)), shape)
	for _, th := range rsqrtThresholds {
		thSSA := b.constF32("th", th, shape)
		gVal := float32(1.0 / math.Sqrt(0.5*float64(th)))
		gSSA := b.constF32("g", gVal, shape)
		y = b.selectF(b.cmpF("less_than", x, thSSA, shape), gSSA, y, ty)
	}

	halfX := b.binOpF("mulf", cHalf, x, ty)
	for i := 0; i < 6; i++ {
		y2 := b.binOpF("mulf", y, y, ty)
		xy2 := b.binOpF("mulf", halfX, y2, ty)
		term := b.binOpF("subf", cThreeHalf, xy2, ty)
		y = b.binOpF("mulf", y, term, ty)
	}
	return y
}

// emitExp emits range-clamped scaling + [2/2] Padé approximant + 5 squarings:
//
//	u = clamp(x, -20, 20) * (1/32)
//	exp(u) ≈ (1 + 0.5*u + (1/12)*u^2) / (1 - 0.5*u + (1/12)*u^2)
//	exp(x) ≈ exp(u)^32
func (b *tileBuilder) emitExp(x string, shape []int64) string {
	ty := tileir.NewTile(shape, tileir.Float32)
	cMin := b.constF32("emin", -20.0, shape)
	cMax := b.constF32("emax", 20.0, shape)
	xc := b.minF(b.maxF(x, cMin, ty), cMax, ty)

	cScale := b.constF32("escale", 1.0/32.0, shape)
	cOne := b.constF32("one", 1.0, shape)
	cHalf := b.constF32("half", 0.5, shape)
	cTwelfth := b.constF32("twelfth", 1.0/12.0, shape)

	u := b.binOpF("mulf", xc, cScale, ty)
	u2 := b.binOpF("mulf", u, u, ty)
	halfU := b.binOpF("mulf", cHalf, u, ty)
	twU2 := b.binOpF("mulf", cTwelfth, u2, ty)
	even := b.binOpF("addf", cOne, twU2, ty)
	num := b.binOpF("addf", even, halfU, ty)
	den := b.binOpF("subf", even, halfU, ty)
	y := b.binOpF("divf", num, den, ty)
	for i := 0; i < 5; i++ {
		y = b.binOpF("mulf", y, y, ty)
	}
	return y
}

func (b *tileBuilder) emitSigmoid(x string, shape []int64) string {
	ty := tileir.NewTile(shape, tileir.Float32)
	nx := b.negF(x, ty)
	ex := b.emitExp(nx, shape)
	cOne := b.constF32("one", 1.0, shape)
	den := b.binOpF("addf", cOne, ex, ty)
	return b.binOpF("divf", cOne, den, ty)
}

func (b *tileBuilder) emitSiLU(x string, shape []int64) string {
	ty := tileir.NewTile(shape, tileir.Float32)
	sig := b.emitSigmoid(x, shape)
	return b.binOpF("mulf", x, sig, ty)
}

func (b *tileBuilder) emitLambda(l Lambda, args []string, shape []int64) (string, error) {
	if l.root == nil {
		if len(args) == 0 {
			return b.constF32("zero", 0, shape), nil
		}
		return args[0], nil
	}
	ty := tileir.NewTile(shape, tileir.Float32)
	var walk func(n *lambdaNode) (string, error)
	walk = func(n *lambdaNode) (string, error) {
		switch n.op {
		case "$":
			if n.arg < 0 || n.arg >= len(args) {
				return "", fmt.Errorf("hopper: lambda arg $%d out of range (have %d)", n.arg, len(args))
			}
			return args[n.arg], nil
		case "#":
			return b.constF32("lit", n.value, shape), nil
		}
		var kids []string
		for _, k := range n.kids {
			kv, err := walk(k)
			if err != nil {
				return "", err
			}
			kids = append(kids, kv)
		}
		switch n.op {
		case "add":
			return b.binOpF("addf", kids[0], kids[1], ty), nil
		case "sub":
			return b.binOpF("subf", kids[0], kids[1], ty), nil
		case "mul":
			return b.binOpF("mulf", kids[0], kids[1], ty), nil
		case "div":
			return b.binOpF("divf", kids[0], kids[1], ty), nil
		case "max":
			return b.maxF(kids[0], kids[1], ty), nil
		case "min":
			return b.minF(kids[0], kids[1], ty), nil
		case "neg":
			return b.negF(kids[0], ty), nil
		case "abs":
			return b.absF(kids[0], ty), nil
		case "recip":
			one := b.constF32("one", 1.0, shape)
			return b.binOpF("divf", one, kids[0], ty), nil
		case "rsqrt":
			return b.emitRsqrt(kids[0], shape), nil
		case "sqrt":
			rs := b.emitRsqrt(kids[0], shape)
			return b.binOpF("mulf", kids[0], rs, ty), nil
		case "exp":
			return b.emitExp(kids[0], shape), nil
		case "sigmoid":
			return b.emitSigmoid(kids[0], shape), nil
		case "silu":
			return b.emitSiLU(kids[0], shape), nil
		case "bf16", "e4m3":
			return kids[0], nil
		default:
			return "", fmt.Errorf("hopper: unsupported lambda op %q in TileIR lowering", n.op)
		}
	}
	return walk(l.root)
}

func pow2TileWidth(n int) int {
	if n <= 0 {
		return 64
	}
	w := 1
	for w < n && w < 1024 {
		w <<= 1
	}
	if w > 1024 {
		w = 1024
	}
	return w
}

func rowKernelToTileIR(k *RowKernel) (*tileir.Module, error) {
	if _, err := k.check(); err != nil {
		return nil, err
	}
	width := pow2TileWidth(k.D)
	shape := []int64{int64(width)}
	tyF32 := tileir.NewTile(shape, tileir.Float32)
	tyI32 := tileir.NewTile(shape, tileir.Int32)
	tyI64 := tileir.NewTile(shape, tileir.Int64)
	ptrScalarTy := tileir.NewTile(nil, tileir.NewPointer(tileir.Float32))

	var b tileBuilder
	var params []*tileir.Param

	for _, o := range k.Outputs {
		pName := "%" + o.Name
		params = append(params, &tileir.Param{Name: pName, Type: ptrScalarTy.String(), TypeVal: ptrScalarTy})
	}
	seenBufs := make(map[string]bool)
	for _, in := range k.Inputs {
		buf := in.buffer()
		if seenBufs[buf] {
			continue
		}
		seenBufs[buf] = true
		pName := "%" + buf
		params = append(params, &tileir.Param{Name: pName, Type: ptrScalarTy.String(), TypeVal: ptrScalarTy})
	}

	tok := b.ssa("tok")
	b.emit([]string{tok}, "make_token", nil, "", tileir.Token)

	scalarI32 := tileir.NewTile(nil, tileir.Int32)
	tuple3I32 := tileir.NewFunction(nil, []tileir.Type{scalarI32, scalarI32, scalarI32})
	bx, by, bz := b.ssa("bx"), b.ssa("by"), b.ssa("bz")
	b.emit([]string{bx, by, bz}, "get_tile_block_id", nil, "", tuple3I32)
	_ = by
	_ = bz

	bx1 := b.reshape(bx, tileir.NewTile([]int64{1}, tileir.Int32))
	rowVec := b.broadcast(bx1, tyI32)
	dimK := b.constI32("dim_k", width, shape)
	rowBase := b.binOpI("muli", rowVec, dimK, tyI32)
	rowBaseI64 := b.extiU64(rowBase, shape)

	colOffI64 := b.ssa("col")
	b.emit([]string{colOffI64}, "iota", nil, "", tyI64)
	rowOffI64 := b.binOpI("addi", rowBaseI64, colOffI64, tyI64)
	zeroOffI64 := b.extiU64(b.constI32("zero_idx", 0, shape), shape)
	rowScalarOffI64 := b.extiU64(rowVec, shape)

	env := make(map[string]string)
	for _, in := range k.Inputs {
		srcParam := "%" + in.buffer()
		var baseIdxI64 string
		switch in.Kind {
		case RowData:
			idxI64 := rowOffI64
			if in.Offset != 0 {
				offI64 := b.extiU64(b.constI32("in_off", in.Offset, shape), shape)
				idxI64 = b.binOpI("addi", idxI64, offI64, tyI64)
			}
			baseIdxI64 = idxI64
		case ColVector:
			idxI64 := colOffI64
			if in.Offset != 0 {
				offI64 := b.extiU64(b.constI32("in_off", in.Offset, shape), shape)
				idxI64 = b.binOpI("addi", idxI64, offI64, tyI64)
			}
			baseIdxI64 = idxI64
		case RowScalar:
			baseIdxI64 = rowScalarOffI64
		case Scalar:
			baseIdxI64 = zeroOffI64
		default:
			baseIdxI64 = colOffI64
		}
		ptrTile := b.offsetPtr(srcParam, baseIdxI64, shape)
		val, nextTok := b.loadTile(ptrTile, tok, shape)
		tok = nextTok
		env[in.Name] = val
	}

	comments := []string{
		"// hopper.kind: row",
		fmt.Sprintf("// hopper.name: %s", k.Name),
		fmt.Sprintf("// hopper.d: %d", k.D),
	}

	for _, v := range k.Values {
		var argSSAs []string
		for _, a := range v.Args {
			ssa, ok := env[a]
			if !ok {
				return nil, fmt.Errorf("hopper: row kernel %s: unbound value arg %q", k.Name, a)
			}
			argSSAs = append(argSSAs, ssa)
		}
		valSSA, err := b.emitLambda(v.Body, argSSAs, shape)
		if err != nil {
			return nil, fmt.Errorf("hopper: row kernel %s value %s: %w", k.Name, v.Name, err)
		}
		if v.Reduce != "" {
			// Emit FP64 widen + accumulation + FP32 narrow (matches tileir/reduction.go SSA pattern).
			tyF64 := tileir.NewTile(shape, tileir.Float64)
			wide := b.ftof(valSSA, tyF64)
			var accWide string
			switch v.Reduce {
			case "sum":
				zero64 := b.constF64("zero64", 0.0, shape)
				accWide = b.binOpF("addf", wide, zero64, tyF64)
			case "max":
				negInf64 := b.constF64("ninf64", -1e30, shape)
				accWide = b.maxF(wide, negInf64, tyF64)
			default:
				return nil, fmt.Errorf("hopper: row kernel %s: unsupported reduction %q", k.Name, v.Reduce)
			}
			valSSA = b.ftof(accWide, tyF32)
			comments = append(comments, fmt.Sprintf("// hopper.reduce: %s=%s", accWide, v.Reduce))
		}
		env[v.Name] = valSSA
	}

	for _, o := range k.Outputs {
		valSSA, ok := env[o.Value]
		if !ok {
			return nil, fmt.Errorf("hopper: row kernel %s: unbound output value %q", k.Name, o.Value)
		}
		outPtr := b.offsetPtr("%"+o.Name, rowOffI64, shape)
		tok = b.storeTile(outPtr, valSSA, tok, shape)
	}
	_ = tok

	b.emit(nil, "return", nil, "", nil)

	mod := &tileir.Module{
		Name:     k.Name + "_module",
		Comments: comments,
		Functions: []*tileir.Function{
			{
				Name:       k.Name,
				Entry:      true,
				Params:     params,
				Operations: b.ops,
			},
		},
	}
	return mod, nil
}

func tileGEMMToTileIR(g *TileGEMM, kindTag string) (*tileir.Module, error) {
	if g.Name == "" {
		return nil, fmt.Errorf("hopper: GEMM Name must not be empty")
	}
	if g.N <= 0 || g.K <= 0 {
		return nil, fmt.Errorf("hopper: GEMM %s requires positive N (%d) and K (%d)", g.Name, g.N, g.K)
	}
	epilogue := g.Epilogue
	if epilogue.root == nil {
		epilogue = Identity
	}
	const (
		tm   = 16
		tn   = 16
		tk   = 16
		vecN = tm * tn // 256 (power-of-two 1D tile width for load_ptr_tko/store_ptr_tko)
	)
	shape1D := []int64{vecN}
	shape2D := []int64{tm, tn}
	ty1DF32 := tileir.NewTile(shape1D, tileir.Float32)
	ty2DF32 := tileir.NewTile(shape2D, tileir.Float32)
	ty1DI32 := tileir.NewTile(shape1D, tileir.Int32)
	scalarI32 := tileir.NewTile(nil, tileir.Int32)
	ptrScalarTy := tileir.NewTile(nil, tileir.NewPointer(tileir.Float32))

	var b tileBuilder
	params := []*tileir.Param{
		{Name: "%out", Type: ptrScalarTy.String(), TypeVal: ptrScalarTy},
		{Name: "%x", Type: ptrScalarTy.String(), TypeVal: ptrScalarTy},
		{Name: "%w", Type: ptrScalarTy.String(), TypeVal: ptrScalarTy},
	}
	for _, ex := range g.Extras {
		pName := "%" + ex.Name
		params = append(params, &tileir.Param{Name: pName, Type: ptrScalarTy.String(), TypeVal: ptrScalarTy})
	}

	tok := b.ssa("tok")
	b.emit([]string{tok}, "make_token", nil, "", tileir.Token)

	tuple3I32 := tileir.NewFunction(nil, []tileir.Type{scalarI32, scalarI32, scalarI32})
	bx, by, bz := b.ssa("bx"), b.ssa("by"), b.ssa("bz")
	b.emit([]string{bx, by, bz}, "get_tile_block_id", nil, "", tuple3I32)
	_ = bz

	bx1 := b.reshape(bx, tileir.NewTile([]int64{1}, tileir.Int32))
	by1 := b.reshape(by, tileir.NewTile([]int64{1}, tileir.Int32))
	bxVec := b.broadcast(bx1, ty1DI32)
	byVec := b.broadcast(by1, ty1DI32)

	aSub := make([]int, vecN)
	bSub := make([]int, vecN)
	for i := 0; i < tm; i++ {
		for j := 0; j < tn; j++ {
			aSub[i*tn+j] = i*tk + j
			bSub[i*tn+j] = j*tk + i
		}
	}
	aSubOff := b.constI32List("a_sub", aSub, shape1D)
	bSubOff := b.constI32List("b_sub", bSub, shape1D)

	strideTile := b.constI32("tile_stride", vecN, shape1D)
	aBase := b.binOpI("muli", byVec, strideTile, ty1DI32)
	aIdxI32 := b.binOpI("addi", aBase, aSubOff, ty1DI32)
	aIdxI64 := b.extiU64(aIdxI32, shape1D)

	bBase := b.binOpI("muli", bxVec, strideTile, ty1DI32)
	bIdxI32 := b.binOpI("addi", bBase, bSubOff, ty1DI32)
	bIdxI64 := b.extiU64(bIdxI32, shape1D)

	aPtr := b.offsetPtr("%x", aIdxI64, shape1D)
	aVec, nextTok := b.loadTile(aPtr, tok, shape1D)
	tok = nextTok
	aTile := b.reshape(aVec, ty2DF32)

	bPtr := b.offsetPtr("%w", bIdxI64, shape1D)
	bVec, nextTok := b.loadTile(bPtr, tok, shape1D)
	tok = nextTok
	bTile := b.reshape(bVec, ty2DF32)

	accInit := b.constF32("acc_zero", 0.0, shape2D)
	accTile := b.ssa("mma")
	b.emit([]string{accTile}, "mmaf", []string{aTile, bTile, accInit}, "", ty2DF32)

	var epilogueArgs []string
	acc0Vec := b.reshape(accTile, ty1DF32)
	epilogueArgs = append(epilogueArgs, acc0Vec)

	if g.GLU {
		bUpIdxI32 := b.binOpI("addi", bIdxI32, strideTile, ty1DI32)
		bUpIdxI64 := b.extiU64(bUpIdxI32, shape1D)
		bUpPtr := b.offsetPtr("%w", bUpIdxI64, shape1D)
		bUpVec, tokUp := b.loadTile(bUpPtr, tok, shape1D)
		tok = tokUp
		bUpTile := b.reshape(bUpVec, ty2DF32)
		accUpTile := b.ssa("mma_up")
		b.emit([]string{accUpTile}, "mmaf", []string{aTile, bUpTile, accInit}, "", ty2DF32)
		acc1Vec := b.reshape(accUpTile, ty1DF32)
		epilogueArgs = append(epilogueArgs, acc1Vec)
	}

	cIdxI32 := b.binOpI("addi", aBase, aSubOff, ty1DI32)
	cIdxI32 = b.binOpI("addi", cIdxI32, bBase, ty1DI32)
	cIdxI64 := b.extiU64(cIdxI32, shape1D)

	for _, ex := range g.Extras {
		exPtr := b.offsetPtr("%"+ex.Name, cIdxI64, shape1D)
		exVec, tokEx := b.loadTile(exPtr, tok, shape1D)
		tok = tokEx
		epilogueArgs = append(epilogueArgs, exVec)
	}

	outVec, err := b.emitLambda(epilogue, epilogueArgs, shape1D)
	if err != nil {
		return nil, fmt.Errorf("hopper: GEMM %s epilogue: %w", g.Name, err)
	}

	outPtr := b.offsetPtr("%out", cIdxI64, shape1D)
	tok = b.storeTile(outPtr, outVec, tok, shape1D)
	_ = tok

	b.emit(nil, "return", nil, "", nil)

	comments := []string{
		fmt.Sprintf("// hopper.kind: %s", kindTag),
		fmt.Sprintf("// hopper.name: %s", g.Name),
		fmt.Sprintf("// hopper.n: %d", g.N),
		fmt.Sprintf("// hopper.k: %d", g.K),
		fmt.Sprintf("// hopper.glu: %t", g.GLU),
		fmt.Sprintf("// hopper.epilogue: %s", epilogue.String()),
	}
	for _, ex := range g.Extras {
		comments = append(comments, fmt.Sprintf("// hopper.extra: %s:%d", ex.Name, ex.Broadcast))
	}

	mod := &tileir.Module{
		Name:     g.Name + "_module",
		Comments: comments,
		Functions: []*tileir.Function{
			{
				Name:       g.Name,
				Entry:      true,
				Params:     params,
				Operations: b.ops,
			},
		},
	}
	return mod, nil
}

var reConstValue = regexp.MustCompile(`value\s*=\s*([-+0-9.eE]+)`)

// EvalTileIRReference executes a lowered Tile IR Module on the given input buffers
// and problem dimensions (M, N, K) using a pure-Go Tile IR SSA interpreter.
func EvalTileIRReference(mod *tileir.Module, inputs map[string][]float32, M, N, K int, eps float32) ([]float32, error) {
	if mod == nil || len(mod.Functions) == 0 {
		return nil, fmt.Errorf("hopper: empty TileIR module")
	}
	fn := mod.Functions[0]

	hasMMA := false
	for _, op := range fn.Operations {
		if strings.TrimPrefix(op.OpCode, "cuda_tile.") == "mmaf" {
			hasMMA = true
			break
		}
	}
	if hasMMA {
		return evalTileIRGEMM(mod, fn, inputs, M, N, K)
	}
	return evalTileIRRow(mod, fn, inputs, M, K, eps)
}

func evalTileIRRow(mod *tileir.Module, fn *tileir.Function, inputs map[string][]float32, M, K int, _ float32) ([]float32, error) {
	if M <= 0 || K <= 0 {
		return nil, fmt.Errorf("hopper: invalid dimensions M=%d, K=%d", M, K)
	}
	reduceMap := make(map[string]string)
	for _, c := range mod.Comments {
		if strings.HasPrefix(c, "// hopper.reduce:") {
			spec := strings.TrimSpace(strings.TrimPrefix(c, "// hopper.reduce:"))
			parts := strings.SplitN(spec, "=", 2)
			if len(parts) == 2 {
				reduceMap[parts[0]] = parts[1]
			}
		}
	}

	outName := "%out"
	if len(fn.Params) > 0 {
		outName = fn.Params[0].Name
	}
	out := make([]float32, M*K)
	buffers := make(map[string][]float32, len(inputs)+2)
	for k, v := range inputs {
		buffers["%"+k] = v
	}
	buffers[outName] = out

	for m := 0; m < M; m++ {
		fEnv := make(map[string][]float64)
		iEnv := make(map[string][]int64)
		bEnv := make(map[string][]bool)
		ptrBase := make(map[string]string)
		ptrOff := make(map[string][]int64)

		for _, p := range fn.Params {
			ptrBase[p.Name] = p.Name
			ptrOff[p.Name] = []int64{0}
		}

		for _, op := range fn.Operations {
			shortOp := strings.TrimPrefix(op.OpCode, "cuda_tile.")
			switch shortOp {
			case "make_token", "join_tokens", "return":
				continue
			case "get_tile_block_id":
				if len(op.Results) >= 3 {
					iEnv[op.Results[0]] = resizeI64([]int64{int64(m)}, K)
					iEnv[op.Results[1]] = resizeI64([]int64{0}, K)
					iEnv[op.Results[2]] = resizeI64([]int64{0}, K)
				}
			case "iota":
				vec := make([]int64, K)
				for k := 0; k < K; k++ {
					vec[k] = int64(k)
				}
				iEnv[op.Results[0]] = vec
			case "constant":
				res := op.Results[0]
				if strings.HasPrefix(res, "%dim_k") {
					iEnv[res] = resizeI64([]int64{int64(K)}, K)
					continue
				}
				isInt := false
				if tt, ok := op.TypeVal.(*tileir.TileType); ok && tt.ElementType != nil && tt.ElementType.IsInt() {
					isInt = true
				}
				if match := reConstValue.FindStringSubmatch(op.Attributes); match != nil {
					if isInt {
						v, _ := strconv.ParseInt(match[1], 10, 64)
						iEnv[res] = resizeI64([]int64{v}, K)
					} else {
						v, _ := strconv.ParseFloat(match[1], 64)
						fEnv[res] = constVecF64(v, K)
					}
				}
			case "reshape", "broadcast", "exti":
				res := op.Results[0]
				src := op.Operands[0]
				if bName, ok := ptrBase[src]; ok {
					ptrBase[res] = bName
					ptrOff[res] = resizeI64(ptrOff[src], K)
				} else if iv, ok := iEnv[src]; ok {
					iEnv[res] = resizeI64(iv, K)
				} else if fv, ok := fEnv[src]; ok {
					fEnv[res] = resizeF64(fv, K)
				}
			case "addi", "subi", "muli":
				res := op.Results[0]
				lhs := resizeI64(iEnv[op.Operands[0]], K)
				rhs := resizeI64(iEnv[op.Operands[1]], K)
				vec := make([]int64, K)
				for k := 0; k < K; k++ {
					switch shortOp {
					case "addi":
						vec[k] = lhs[k] + rhs[k]
					case "subi":
						vec[k] = lhs[k] - rhs[k]
					case "muli":
						vec[k] = lhs[k] * rhs[k]
					}
				}
				iEnv[res] = vec
			case "offset":
				res := op.Results[0]
				base := op.Operands[0]
				idx := resizeI64(iEnv[op.Operands[1]], K)
				baseOff := resizeI64(ptrOff[base], K)
				vec := make([]int64, K)
				for k := 0; k < K; k++ {
					vec[k] = baseOff[k] + idx[k]
				}
				ptrBase[res] = ptrBase[base]
				ptrOff[res] = vec
			case "load_ptr_tko":
				res := op.Results[0]
				ptr := op.Operands[0]
				bName := ptrBase[ptr]
				buf, ok := buffers[bName]
				if !ok {
					return nil, fmt.Errorf("hopper: missing input buffer for parameter %s", bName)
				}
				offs := ptrOff[ptr]
				vec := make([]float64, K)
				for k := 0; k < K; k++ {
					idx := int(offs[k])
					if idx >= 0 && idx < len(buf) {
						vec[k] = float64(buf[idx])
					}
				}
				fEnv[res] = vec
			case "store_ptr_tko":
				ptr := op.Operands[0]
				val := resizeF64(fEnv[op.Operands[1]], K)
				bName := ptrBase[ptr]
				buf, ok := buffers[bName]
				if !ok {
					buf = make([]float32, M*K)
					buffers[bName] = buf
				}
				offs := ptrOff[ptr]
				for k := 0; k < K; k++ {
					idx := int(offs[k])
					if idx >= 0 && idx < len(buf) {
						buf[idx] = float32(val[k])
					}
				}
			case "negf", "absf":
				res := op.Results[0]
				src := resizeF64(fEnv[op.Operands[0]], K)
				vec := make([]float64, K)
				for k := 0; k < K; k++ {
					v := float32(src[k])
					if shortOp == "negf" {
						vec[k] = float64(-v)
					} else {
						vec[k] = float64(math.Abs(float64(v)))
					}
				}
				fEnv[res] = vec
			case "addf", "subf", "mulf", "divf", "maxf", "minf":
				res := op.Results[0]
				lhs := resizeF64(fEnv[op.Operands[0]], K)
				rhs := resizeF64(fEnv[op.Operands[1]], K)
				if redKind, isRed := reduceMap[res]; isRed {
					var red float64
					if redKind == "max" {
						red = -math.MaxFloat64
						for k := 0; k < K; k++ {
							if lhs[k] > red {
								red = lhs[k]
							}
						}
					} else {
						for k := 0; k < K; k++ {
							red += lhs[k]
						}
					}
					fEnv[res] = constVecF64(red, K)
					continue
				}
				vec := make([]float64, K)
				for k := 0; k < K; k++ {
					a, b := float32(lhs[k]), float32(rhs[k])
					var r float32
					switch shortOp {
					case "addf":
						r = a + b
					case "subf":
						r = a - b
					case "mulf":
						r = a * b
					case "divf":
						r = a / b
					case "maxf":
						if a > b {
							r = a
						} else {
							r = b
						}
					case "minf":
						if a < b {
							r = a
						} else {
							r = b
						}
					}
					vec[k] = float64(r)
				}
				fEnv[res] = vec
			case "ftof":
				res := op.Results[0]
				src := resizeF64(fEnv[op.Operands[0]], K)
				isDstF32 := false
				if tt, ok := op.TypeVal.(*tileir.TileType); ok && tt.ElementType != nil && tt.ElementType.Kind() == tileir.KindFloat32 {
					isDstF32 = true
				}
				vec := make([]float64, K)
				for k := 0; k < K; k++ {
					if isDstF32 {
						vec[k] = float64(float32(src[k]))
					} else {
						vec[k] = src[k]
					}
				}
				fEnv[res] = vec
			case "cmpf":
				res := op.Results[0]
				lhs := resizeF64(fEnv[op.Operands[0]], K)
				rhs := resizeF64(fEnv[op.Operands[1]], K)
				pred := "less_than"
				if strings.Contains(op.Attributes, "less_than_or_equal") {
					pred = "less_than_or_equal"
				} else if strings.Contains(op.Attributes, "greater_than_or_equal") {
					pred = "greater_than_or_equal"
				} else if strings.Contains(op.Attributes, "greater_than") {
					pred = "greater_than"
				} else if strings.Contains(op.Attributes, "not_equal") {
					pred = "not_equal"
				} else if strings.Contains(op.Attributes, "equal") {
					pred = "equal"
				}
				vec := make([]bool, K)
				for k := 0; k < K; k++ {
					a, b := float32(lhs[k]), float32(rhs[k])
					switch pred {
					case "less_than":
						vec[k] = a < b
					case "less_than_or_equal":
						vec[k] = a <= b
					case "greater_than":
						vec[k] = a > b
					case "greater_than_or_equal":
						vec[k] = a >= b
					case "equal":
						vec[k] = a == b
					case "not_equal":
						vec[k] = a != b
					}
				}
				bEnv[res] = vec
			case "select":
				res := op.Results[0]
				cond := bEnv[op.Operands[0]]
				tVal := resizeF64(fEnv[op.Operands[1]], K)
				fVal := resizeF64(fEnv[op.Operands[2]], K)
				vec := make([]float64, K)
				for k := 0; k < K; k++ {
					if cond[k] {
						vec[k] = tVal[k]
					} else {
						vec[k] = fVal[k]
					}
				}
				fEnv[res] = vec
			}
		}
	}
	return buffers[outName], nil
}

func evalTileIRGEMM(mod *tileir.Module, _ *tileir.Function, inputs map[string][]float32, M, N, K int) ([]float32, error) {
	if M <= 0 || N <= 0 || K <= 0 {
		return nil, fmt.Errorf("hopper: invalid GEMM dimensions M=%d, N=%d, K=%d", M, N, K)
	}
	epilogue := Identity
	glu := false
	var extras []Operand
	for _, c := range mod.Comments {
		switch {
		case strings.HasPrefix(c, "// hopper.epilogue:"):
			src := strings.TrimSpace(strings.TrimPrefix(c, "// hopper.epilogue:"))
			if src != "" {
				if l, err := ParseLambda(src); err == nil {
					epilogue = l
				}
			}
		case strings.HasPrefix(c, "// hopper.glu:"):
			glu = strings.TrimSpace(strings.TrimPrefix(c, "// hopper.glu:")) == "true"
		case strings.HasPrefix(c, "// hopper.extra:"):
			spec := strings.TrimSpace(strings.TrimPrefix(c, "// hopper.extra:"))
			parts := strings.SplitN(spec, ":", 2)
			if len(parts) == 2 {
				bc, _ := strconv.Atoi(parts[1])
				extras = append(extras, Operand{Name: parts[0], Broadcast: Broadcast(bc)})
			}
		}
	}

	xBuf, ok := inputs["x"]
	if !ok {
		return nil, fmt.Errorf("hopper: missing input 'x' for GEMM")
	}
	wBuf, ok := inputs["w"]
	if !ok {
		return nil, fmt.Errorf("hopper: missing input 'w' for GEMM")
	}

	nOut := N
	if glu {
		nOut = N / 2
	}
	out := make([]float32, M*nOut)

	const tm, tn, tk = 16, 16, 16
	for bm := 0; bm < M; bm += tm {
		for bn := 0; bn < nOut; bn += tn {
			var acc [tm][tn]float32
			var accUp [tm][tn]float32

			for kt := 0; kt < K; kt += tk {
				var aTile [tm][tk]float32
				var bTile [tk][tn]float32
				var bUpTile [tk][tn]float32

				for i := 0; i < tm; i++ {
					mIdx := bm + i
					for k := 0; k < tk; k++ {
						kIdx := kt + k
						if mIdx < M && kIdx < K {
							aTile[i][k] = BF16.Round(xBuf[mIdx*K+kIdx])
						}
					}
				}
				for k := 0; k < tk; k++ {
					kIdx := kt + k
					for j := 0; j < tn; j++ {
						nIdx := bn + j
						if nIdx < nOut && kIdx < K {
							bTile[k][j] = BF16.Round(wBuf[nIdx*K+kIdx])
							if glu {
								bUpTile[k][j] = BF16.Round(wBuf[(nIdx+nOut)*K+kIdx])
							}
						}
					}
				}

				for i := 0; i < tm; i++ {
					for j := 0; j < tn; j++ {
						sum := acc[i][j]
						for k := 0; k < tk; k++ {
							sum += aTile[i][k] * bTile[k][j]
						}
						acc[i][j] = sum
						if glu {
							sumUp := accUp[i][j]
							for k := 0; k < tk; k++ {
								sumUp += aTile[i][k] * bUpTile[k][j]
							}
							accUp[i][j] = sumUp
						}
					}
				}
			}

			for i := 0; i < tm; i++ {
				mIdx := bm + i
				if mIdx >= M {
					continue
				}
				for j := 0; j < tn; j++ {
					nIdx := bn + j
					if nIdx >= nOut {
						continue
					}
					args := []float32{acc[i][j]}
					if glu {
						args = append(args, accUp[i][j])
					}
					for _, ex := range extras {
						exBuf, ok := inputs[ex.Name]
						if !ok {
							return nil, fmt.Errorf("hopper: missing extra input %q", ex.Name)
						}
						var idx int
						switch ex.Broadcast {
						case PerCol:
							idx = nIdx
						case PerRow:
							idx = mIdx
						default:
							idx = mIdx*nOut + nIdx
						}
						args = append(args, exBuf[idx])
					}
					out[mIdx*nOut+nIdx] = evalLambdaTileIRApprox(epilogue, args)
				}
			}
		}
	}
	return out, nil
}

// evalLambdaTileIRApprox evaluates l using the exact Tile IR polynomial/NR
// approximations emitted by tileBuilder (Padé exp, Newton-Raphson rsqrt).
func evalLambdaTileIRApprox(l Lambda, args []float32) float32 {
	if l.root == nil {
		if len(args) == 0 {
			return 0
		}
		return args[0]
	}
	var walk func(n *lambdaNode) float32
	walk = func(n *lambdaNode) float32 {
		switch n.op {
		case "$":
			return args[n.arg]
		case "#":
			return n.value
		}
		a := walk(n.kids[0])
		var b float32
		if len(n.kids) > 1 {
			b = walk(n.kids[1])
		}
		switch n.op {
		case "add":
			return a + b
		case "sub":
			return a - b
		case "mul":
			return a * b
		case "div":
			return a / b
		case "max":
			if a > b {
				return a
			}
			return b
		case "min":
			if a < b {
				return a
			}
			return b
		case "neg":
			return -a
		case "abs":
			return float32(math.Abs(float64(a)))
		case "recip":
			return 1.0 / a
		case "rsqrt":
			return evalTileIRRsqrtScalar(a)
		case "sqrt":
			return a * evalTileIRRsqrtScalar(a)
		case "exp":
			return evalTileIRExpScalar(a)
		case "sigmoid":
			return 1.0 / (1.0 + evalTileIRExpScalar(-a))
		case "silu":
			return a / (1.0 + evalTileIRExpScalar(-a))
		case "bf16":
			return BF16.Round(a)
		case "e4m3":
			return E4M3.Round(a)
		default:
			return l.Eval(args...)
		}
	}
	return walk(l.root)
}

func evalTileIRExpScalar(x float32) float32 {
	if x < -20.0 {
		x = -20.0
	} else if x > 20.0 {
		x = 20.0
	}
	u := x * (1.0 / 32.0)
	u2 := u * u
	even := float32(1.0) + float32(1.0/12.0)*u2
	halfU := float32(0.5) * u
	y := (even + halfU) / (even - halfU)
	for i := 0; i < 5; i++ {
		y = y * y
	}
	return y
}

func evalTileIRRsqrtScalar(x float32) float32 {
	y := float32(1.0 / math.Sqrt(8192.0))
	for _, th := range rsqrtThresholds {
		if x < th {
			y = float32(1.0 / math.Sqrt(0.5*float64(th)))
		}
	}
	halfX := float32(0.5) * x
	for i := 0; i < 6; i++ {
		y = y * (float32(1.5) - halfX*y*y)
	}
	return y
}

func constVecF64(v float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func resizeF64(in []float64, n int) []float64 {
	if len(in) == n {
		return in
	}
	out := make([]float64, n)
	if len(in) == 0 {
		return out
	}
	for i := range out {
		out[i] = in[i%len(in)]
	}
	return out
}

func resizeI64(in []int64, n int) []int64 {
	if len(in) == n {
		return in
	}
	out := make([]int64, n)
	if len(in) == 0 {
		return out
	}
	for i := range out {
		out[i] = in[i%len(in)]
	}
	return out
}

// DualTargetReport summarizes the dual-target verification between the
// hopper CUDA C++ backend (executed via cudahostemu) and the Tile IR
// lowering path (formatted, parsed, bytecode-encoded, and evaluated).
type DualTargetReport struct {
	KernelName   string
	TileIRModule *tileir.Module
	TileIRText   string
	Bytecode     []byte
	CUDAOutput   []float32
	TileIROutput []float32
	MaxAbsDiff   float64
}

// VerifyDualTargetAgreement lowers step to both CUDA C++ (executed via cudahostemu.Run)
// and a *tileir.Module (verified through Format -> Parse -> EncodeBytecode and evaluated
// via EvalTileIRReference) on identical synthetic inputs, returning an error if
// maxAbsDiff >= 2e-2.
func VerifyDualTargetAgreement(ctx context.Context, step TileIRStep, M, N, K int, eps float32) (*DualTargetReport, error) {
	if step == nil {
		return nil, fmt.Errorf("hopper: nil step")
	}
	mod, err := StepToTileIR(step)
	if err != nil {
		return nil, fmt.Errorf("hopper: StepToTileIR: %w", err)
	}
	text := FormatTileIR(mod)
	parsed, err := ParseTileIR(text)
	if err != nil {
		return nil, fmt.Errorf("hopper: ParseTileIR round-trip: %w", err)
	}
	bc, err := EncodeTileIR(parsed)
	if err != nil {
		return nil, fmt.Errorf("hopper: EncodeTileIR: %w", err)
	}
	if !tileir.IsTileIRBytecode(bc) {
		return nil, fmt.Errorf("hopper: encoded bytes do not have TileIR magic header")
	}

	pk := step.stepKernel()
	rng := rand.New(rand.NewPCG(42, 43))
	randBuf := func(n int, scale float32, dt DType) []float32 {
		b := make([]float32, n)
		for i := range b {
			v := (rng.Float32()*2 - 1) * scale
			b[i] = dt.Round(v)
		}
		return b
	}

	inputs := make(map[string][]float32)
	var cudaOut []float32
	var kernelName string

	switch {
	case pk.Row != nil:
		rk := *pk.Row
		if rk.D > 0 {
			K = rk.D
		}
		kernelName = rk.Name
		seen := make(map[string]bool)
		for _, in := range rk.Inputs {
			buf := in.buffer()
			if seen[buf] {
				continue
			}
			seen[buf] = true
			sz := M * K
			switch in.Kind {
			case ColVector:
				sz = K
			case RowScalar:
				sz = M
			case Scalar:
				sz = 1
			}
			inputs[buf] = randBuf(sz, 1.0, in.DType)
		}
		cudaOut, err = runRowKernelHostEmu(ctx, rk, inputs, M, K)
		if err != nil {
			return nil, fmt.Errorf("hopper: cudahostemu row kernel %s: %w", rk.Name, err)
		}
	case pk.Tile != nil:
		tg := *pk.Tile
		if tg.N > 0 {
			N = tg.N
		}
		if tg.K > 0 {
			K = tg.K
		}
		kernelName = tg.Name
		inputs["x"] = randBuf(M*K, 0.5, BF16)
		inputs["w"] = randBuf(N*K, 0.25, BF16)
		nOut := N
		if tg.GLU {
			nOut = N / 2
		}
		for _, ex := range tg.Extras {
			sz := M * nOut
			if ex.Broadcast == PerCol {
				sz = nOut
			} else if ex.Broadcast == PerRow {
				sz = M
			}
			inputs[ex.Name] = randBuf(sz, 0.5, ex.DType)
		}
		cudaOut, err = runTileGEMMHostEmu(ctx, tg, inputs, M, nOut)
		if err != nil {
			return nil, fmt.Errorf("hopper: cudahostemu tile gemm %s: %w", tg.Name, err)
		}
	case pk.GEMM != nil:
		sg := *pk.GEMM
		if sg.N > 0 {
			N = sg.N
		}
		if sg.K > 0 {
			K = sg.K
		}
		kernelName = sg.Name
		inputs["x"] = randBuf(M*K, 0.5, BF16)
		inputs["w"] = randBuf(N*K, 0.25, BF16)
		for _, ex := range sg.Extras {
			sz := M * N
			if ex.Broadcast == PerCol {
				sz = N
			} else if ex.Broadcast == PerRow {
				sz = M
			}
			inputs[ex.Name] = randBuf(sz, 0.5, ex.DType)
		}
		cudaOut, err = runSkinnyGEMMHostEmu(ctx, sg, inputs, M)
		if err != nil {
			return nil, fmt.Errorf("hopper: cudahostemu skinny gemm %s: %w", sg.Name, err)
		}
	default:
		return nil, fmt.Errorf("hopper: unsupported PlanKernel step")
	}

	tileOut, err := EvalTileIRReference(parsed, inputs, M, N, K, eps)
	if err != nil {
		return nil, fmt.Errorf("hopper: EvalTileIRReference: %w", err)
	}
	if len(cudaOut) != len(tileOut) {
		return nil, fmt.Errorf("hopper: output length mismatch: cuda=%d vs tileir=%d", len(cudaOut), len(tileOut))
	}

	maxDiff := 0.0
	for i := range cudaOut {
		d := math.Abs(float64(cudaOut[i] - tileOut[i]))
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return nil, fmt.Errorf("hopper: NaN/Inf at index %d (cuda=%v, tileir=%v)", i, cudaOut[i], tileOut[i])
		}
		if d > maxDiff {
			maxDiff = d
		}
	}

	report := &DualTargetReport{
		KernelName:   kernelName,
		TileIRModule: parsed,
		TileIRText:   text,
		Bytecode:     bc,
		CUDAOutput:   cudaOut,
		TileIROutput: tileOut,
		MaxAbsDiff:   maxDiff,
	}
	const maxAllowedDiff = 2e-2
	if maxDiff >= maxAllowedDiff {
		return report, fmt.Errorf("hopper: dual-target disagreement for %s: maxAbsDiff=%g >= %g", kernelName, maxDiff, maxAllowedDiff)
	}
	return report, nil
}

func encodeTypedBytes(dt DType, vals []float32) []byte {
	switch dt {
	case BF16:
		b := make([]byte, 2*len(vals))
		for i, v := range vals {
			binary.LittleEndian.PutUint16(b[2*i:], BF16Bits(v))
		}
		return b
	case E4M3:
		b := make([]byte, len(vals))
		for i, v := range vals {
			b[i] = E4M3Bits(v)
		}
		return b
	default:
		b := make([]byte, 4*len(vals))
		for i, v := range vals {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
		}
		return b
	}
}

func decodeTypedBytes(dt DType, raw []byte) []float32 {
	switch dt {
	case BF16:
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = FromBF16Bits(binary.LittleEndian.Uint16(raw[2*i:]))
		}
		return out
	case E4M3:
		out := make([]float32, len(raw))
		for i, b := range raw {
			out[i] = FromE4M3Bits(b)
		}
		return out
	default:
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		return out
	}
}

func makeHostEmuWorkDir() (string, func(), error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", nil, err
	}
	base := home + "/tmp/lib-wt-llama-megakernel/20261026-tileir-bridge"
	if err := os.MkdirAll(base, 0o755); err != nil {
		return "", nil, err
	}
	dir, err := os.MkdirTemp(base, "emu-*")
	if err != nil {
		return "", nil, err
	}
	return dir, func() { _ = os.RemoveAll(dir) }, nil
}

func runRowKernelHostEmu(ctx context.Context, k RowKernel, inputs map[string][]float32, rows, d int) ([]float32, error) {
	src, err := k.CUDA()
	if err != nil {
		return nil, err
	}
	inTypes := make(map[string]DType)
	for _, in := range k.Inputs {
		inTypes[in.buffer()] = in.DType
	}
	outTypes := make(map[string]DType)
	for _, o := range k.Outputs {
		outTypes[o.Name] = o.DType
	}

	var cbufs []cudahostemu.Buffer
	var argNames []string
	for _, p := range k.params()[1:] {
		f := strings.Fields(p)
		name := f[len(f)-1]
		argNames = append(argNames, name)
		if dt, isOut := outTypes[name]; isOut {
			cbufs = append(cbufs, cudahostemu.Buffer{
				Name:  name,
				CType: dt.cType(),
				Data:  make([]byte, rows*d*dt.Bits()/8),
			})
		} else {
			dt := inTypes[name]
			cbufs = append(cbufs, cudahostemu.Buffer{
				Name:  name,
				CType: dt.cType(),
				Data:  encodeTypedBytes(dt, inputs[name]),
			})
		}
	}

	grid, block := k.Grid(rows)
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
		grid[0], block[0], k.Name, rows, strings.Join(argNames, ", "))

	dir, cleanup, err := makeHostEmuWorkDir()
	if err != nil {
		return nil, err
	}
	defer cleanup()

	res, err := cudahostemu.Run(ctx, dir, cudahostemu.Program{
		Source:  Prelude + src,
		Launch:  launch,
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		return nil, err
	}
	if err != nil {
		return nil, err
	}
	outSpec := k.Outputs[0]
	return decodeTypedBytes(outSpec.DType, res[outSpec.Name]), nil
}

func runTileGEMMHostEmu(ctx context.Context, g TileGEMM, inputs map[string][]float32, m, nOut int) ([]float32, error) {
	src, err := g.CUDA()
	if err != nil {
		return nil, err
	}
	var cbufs []cudahostemu.Buffer
	cbufs = append(cbufs,
		cudahostemu.Buffer{Name: "x", CType: "uint16_t", Data: encodeTypedBytes(BF16, inputs["x"])},
		cudahostemu.Buffer{Name: "w", CType: "uint16_t", Data: encodeTypedBytes(BF16, inputs["w"])},
	)
	args := []string{fmt.Sprint(m), "x", "w"}
	for _, ex := range g.Extras {
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  ex.Name,
			CType: ex.DType.cType(),
			Data:  encodeTypedBytes(ex.DType, inputs[ex.Name]),
		})
		args = append(args, ex.Name)
	}
	cbufs = append(cbufs, cudahostemu.Buffer{
		Name:  "out",
		CType: g.Out.cType(),
		Data:  make([]byte, m*nOut*g.Out.Bits()/8),
	})
	args = append(args, "out")

	grid, block := g.Grid(m, 1)
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %s)",
		grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, strings.Join(args, ", "))

	dir, cleanup, err := makeHostEmuWorkDir()
	if err != nil {
		return nil, err
	}
	defer cleanup()

	res, err := cudahostemu.Run(ctx, dir, cudahostemu.Program{
		Source:  Prelude + src,
		Launch:  launch,
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if err != nil {
		return nil, err
	}
	return decodeTypedBytes(g.Out, res["out"]), nil
}

func runSkinnyGEMMHostEmu(ctx context.Context, g SkinnyGEMM, inputs map[string][]float32, m int) ([]float32, error) {
	src, err := g.CUDA()
	if err != nil {
		return nil, err
	}
	grid, block, err := g.Grid()
	if err != nil {
		return nil, err
	}
	var cbufs []cudahostemu.Buffer
	cbufs = append(cbufs,
		cudahostemu.Buffer{Name: "x", CType: "uint16_t", Data: encodeTypedBytes(BF16, inputs["x"])},
		cudahostemu.Buffer{Name: "w", CType: "uint16_t", Data: encodeTypedBytes(BF16, inputs["w"])},
	)
	args := []string{"x", "w"}
	for _, ex := range g.Extras {
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  ex.Name,
			CType: ex.DType.cType(),
			Data:  encodeTypedBytes(ex.DType, inputs[ex.Name]),
		})
		args = append(args, ex.Name)
	}
	cbufs = append(cbufs, cudahostemu.Buffer{
		Name:  "out",
		CType: g.Out.cType(),
		Data:  make([]byte, m*g.N*g.Out.Bits()/8),
	})
	args = append(args, "out")
	if ws := g.WorkspaceBytes(); ws > 0 {
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  "ws",
			CType: "float",
			Data:  make([]byte, ws),
		})
		args = append(args, "ws")
	}

	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, %s)",
		grid[0], grid[1], block[0], g.Name, m, strings.Join(args, ", "))

	dir, cleanup, err := makeHostEmuWorkDir()
	if err != nil {
		return nil, err
	}
	defer cleanup()

	res, err := cudahostemu.Run(ctx, dir, cudahostemu.Program{
		Source:  Prelude + src,
		Launch:  launch,
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if err != nil {
		return nil, err
	}
	return decodeTypedBytes(g.Out, res["out"]), nil
}
