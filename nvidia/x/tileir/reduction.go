package tileir

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// A reductionRegion contains direct-input, rank-one/two FP32 reductions.
// Each tile lane owns one output; reduction positions are visited in order.
// The limits are source-generation bounds, not hardware resource guarantees.
type reductionRegion struct {
	inputs, outputs []*cudair.Node
	width           int64
	maxOutputs      int
}

func directReductionRegion(snap *cudair.Snapshot) bool {
	if snap == nil || len(snap.Outputs()) == 0 {
		return false
	}
	for _, n := range snap.Nodes() {
		if n.Kind != cudair.KindInput && (n.Kind != cudair.KindReduce || (n.Op != "sum" && n.Op != "max") || len(n.Inputs) != 1 || n.Inputs[0].Kind != cudair.KindInput) {
			return false
		}
	}
	for _, n := range snap.Outputs() {
		if n.Kind != cudair.KindReduce {
			return false
		}
	}
	return true
}

// compileTileSource selects the strict vector route where it is applicable.
// The existing bounded reference source remains available for other graphs.
func compileTileSource(snap *cudair.Snapshot, opts LowerOptions) (string, cudakernel.Schema, error) {
	if directReductionRegion(snap) {
		r, err := lowerReductionRegion(snap, opts)
		if err != nil {
			return "", cudakernel.Schema{}, err
		}
		return r.Source, r.Schema, nil
	}
	return GenerateSemanticTileCPP(snap, opts)
}

func checkedReductionRegion(snap *cudair.Snapshot, opts LowerOptions) (reductionRegion, error) {
	var r reductionRegion
	if !directReductionRegion(snap) || len(snap.Nodes()) > 256 {
		return r, fmt.Errorf("%w: ordered reductions require direct inputs and sum/max outputs", ErrUnsupportedTileOp)
	}
	r.width = opts.TileM
	if r.width == 0 {
		r.width = 16
	}
	if r.width <= 0 || r.width > 1<<20 || r.width&(r.width-1) != 0 {
		return r, fmt.Errorf("%w: reduction tile width must be a power of two within the source budget", ErrUnsupportedTileOp)
	}
	r.inputs, r.outputs = snap.Inputs(), snap.Outputs()
	for _, n := range snap.Nodes() {
		if n.Type != cudair.Float32 || n.Layout != cudashape.LayoutRowMajor || n.Effect != cudair.EffectReadOnly || len(n.Shape) < 1 || len(n.Shape) > 2 {
			return r, fmt.Errorf("%w: reduction type, rank, layout or effect at node %d", ErrUnsupportedTileOp, n.ID)
		}
		count := 1
		for _, d := range n.Shape {
			if d < 0 || d > 1<<20 || d != 0 && count > (1<<20)/d {
				return r, fmt.Errorf("%w: reduction shape exceeds source budget", ErrUnsupportedTileOp)
			}
			count *= d
		}
		if len(n.Strides) != 0 && !cudashape.IsContiguousRowMajor(n.Shape, n.Strides) {
			return r, fmt.Errorf("%w: reduction strides", ErrUnsupportedTileOp)
		}
		if n.Kind != cudair.KindReduce {
			continue
		}
		in := n.Inputs[0]
		if len(n.Axes) != 1 || n.Axes[0] < 0 || n.Axes[0] >= len(in.Shape) || in.Shape[n.Axes[0]] > 64 {
			return r, fmt.Errorf("%w: reduction axis or term budget", ErrUnsupportedTileOp)
		}
		axis := n.Axes[0]
		shape := append(append([]int(nil), in.Shape[:axis]...), in.Shape[axis+1:]...)
		if len(shape) == 0 {
			shape = []int{1}
		}
		if !sameTileShape(shape, n.Shape) {
			return r, fmt.Errorf("tileir: reduction output shape at node %d", n.ID)
		}
		if count > r.maxOutputs {
			r.maxOutputs = count
		}
	}
	return r, nil
}

// reductionBuilder emits one typed SSA chain and its corresponding Tile C++.
// The AST is an analysis representation; vendor bytecode encoding stays gated.
type reductionBuilder struct {
	module *Module
	fn     *Function
	source strings.Builder
	node   *cudair.Node
	serial int
	token  string
}

func (b *reductionBuilder) emit(code string, typ Type, operands []string, attrs, expression string) string {
	b.serial++
	name := fmt.Sprintf("v%d", b.serial)
	op := &Operation{Line: len(b.fn.Operations) + 1, OpCode: "cuda_tile." + code, Results: []string{"%" + name}, Operands: operands, Type: typ.String(), TypeVal: typ, Attributes: attrs}
	if b.node != nil {
		op.Attributes += fmt.Sprintf(" semantic_id = %d", b.node.ID)
		loc := b.node.SourceLoc
		if loc.File != "" && loc.Line > 0 {
			key := fmt.Sprintf("node%d", b.node.ID)
			b.module.Locations[key] = Location{File: loc.File, Line: loc.Line, Column: 1, Scope: loc.Function}
			op.LocRef, op.Loc = "#"+key, b.module.Locations[key]
		}
	}
	b.fn.Operations = append(b.fn.Operations, op)
	fmt.Fprintf(&b.source, "  auto %s = %s;\n", name, expression)
	return "%" + name
}

func cppName(ssa string) string { return strings.TrimPrefix(ssa, "%") }

func (b *reductionBuilder) constant(t Type, value string) string {
	cppType := "float"
	element := t
	if tile, ok := t.(*TileType); ok {
		element = tile.ElementType
	}
	if element.Kind() == KindInt32 {
		cppType = "unsigned int"
	} else if element.Kind() == KindFloat64 {
		cppType = "double"
	}
	if tile, ok := t.(*TileType); ok {
		if tile.ElementType.Kind() == KindFloat64 {
			cppType = "double"
		}
		return b.emit("constant", t, nil, "value = "+value, fmt.Sprintf("ct::full<ct::tile<%s, ct::shape<%d>>>(%s)", cppType, tile.Shape[0], value))
	}
	return b.emit("constant", t, nil, "value = "+value, value)
}

func (b *reductionBuilder) load(ptr, valid, padding string, typ Type) string {
	value := b.emit("load_ptr_tko", typ, []string{ptr, valid, padding, b.token}, "memory_ordering_semantics = weak", fmt.Sprintf("ct::load_masked(%s, %s, %s)", cppName(ptr), cppName(valid), cppName(padding)))
	b.serial++
	next := fmt.Sprintf("%%tok%d", b.serial)
	op := b.fn.Operations[len(b.fn.Operations)-1]
	op.Results = append(op.Results, next)
	op.TypeVal = &FunctionType{Results: []Type{typ, Token}}
	op.Type = op.TypeVal.String()
	b.token = next
	return value
}

func lowerReductionRegion(snap *cudair.Snapshot, opts LowerOptions) (*LoweringResult, error) {
	r, err := checkedReductionRegion(snap, opts)
	if err != nil {
		return nil, err
	}
	name := opts.KernelName
	if name == "" {
		name = "tile_kernel"
	}
	if len(name) > 128 {
		return nil, fmt.Errorf("tileir: kernel identifier too long")
	}
	for i, c := range name {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || i > 0 && c >= '0' && c <= '9') {
			return nil, fmt.Errorf("tileir: invalid kernel identifier")
		}
	}
	b := &reductionBuilder{module: &Module{Name: name + "_mod", Producer: "lib/nvidia/x/tileir", Locations: make(map[string]Location)}, fn: &Function{Name: name, Entry: true, OptHints: "ordered_reduction"}, token: "%tok0"}
	b.module.Functions = []*Function{b.fn}
	b.fn.Operations = []*Operation{{OpCode: "cuda_tile.make_token", Results: []string{b.token}, Type: Token.String(), TypeVal: Token}}
	b.source.WriteString("#include \"cuda_tile.h\"\n#include <limits>\nextern \"C\" __tile_global__ void " + name + "(")
	var params []cudakernel.Param
	inputArgs := make(map[int]string)
	for i, n := range append(append([]*cudair.Node(nil), r.inputs...), r.outputs...) {
		if i > 0 {
			b.source.WriteString(", ")
		}
		arg := fmt.Sprintf("arg%d", i)
		b.source.WriteString("float* " + arg)
		b.fn.Params = append(b.fn.Params, &Param{Name: "%" + arg, Type: NewPointer(Float32).String(), TypeVal: NewPointer(Float32)})
		paramName := n.Name
		if paramName == "" {
			paramName = arg
		}
		params = append(params, cudakernel.Param{Name: paramName, Type: "f32", ByteWidth: 8, Alignment: 16, IsPointer: true})
		if i < len(r.inputs) {
			inputArgs[n.ID] = "%" + arg
		}
	}
	b.source.WriteString(") {\n  namespace ct = ::cuda::tiles;\n")
	u := NewTile([]int64{r.width}, Int32)
	f := NewTile([]int64{r.width}, Float32)
	d := NewTile([]int64{r.width}, Float64)
	maskType := NewTile([]int64{r.width}, Int1)
	ptrType := NewTile([]int64{r.width}, NewPointer(Float32))
	block := b.emit("get_tile_block_id", Int32, nil, "axis = 0", "ct::bid().x")
	width := b.constant(Int32, strconv.FormatInt(r.width, 10)+"u")
	base := b.emit("muli", Int32, []string{block, width}, "overflow = wrap", cppName(block)+" * "+cppName(width))
	lanes := b.emit("iota", u, nil, "", fmt.Sprintf("ct::iota<ct::tile<unsigned int, ct::shape<%d>>>()", r.width))
	rows := b.emit("addi", u, []string{base, lanes}, "overflow = wrap", cppName(base)+" + "+cppName(lanes))
	zero := b.constant(u, "0u")
	for j, out := range r.outputs {
		b.node = out
		in, axis := out.Inputs[0], out.Axes[0]
		terms, count := in.Shape[axis], out.Shape[0]
		// Source identity does not depend on whether this output needs memory.
		if loc := out.SourceLoc; loc.File != "" && loc.Line > 0 {
			b.module.Locations[fmt.Sprintf("node%d", out.ID)] = Location{File: loc.File, Line: loc.Line, Column: 1, Scope: loc.Function}
		}
		// A different output can require a launch even when this one is empty.
		// Empty ABI objects may have null bindings: emit no pointer operations.
		if count == 0 {
			continue
		}
		limit := b.constant(Int32, fmt.Sprintf("%du", count))
		valid := b.emit("cmpi", maskType, []string{rows, limit}, "comparison_predicate = lt signedness = unsigned", cppName(rows)+" < "+cppName(limit))
		safeRows := b.emit("select", u, []string{valid, rows, zero}, "", fmt.Sprintf("ct::select(%s, %s, %s)", cppName(valid), cppName(rows), cppName(zero)))
		padding := b.constant(f, "0.0f")
		acc := b.constant(d, "0.0")
		if out.Op == "max" {
			acc = b.constant(f, "-std::numeric_limits<float>::infinity()")
		}
		for k := 0; k < terms; k++ {
			offset := safeRows
			if len(in.Shape) == 2 && axis == 1 {
				stride := b.constant(Int32, fmt.Sprintf("%du", terms))
				offset = b.emit("muli", u, []string{safeRows, stride}, "overflow = wrap", cppName(safeRows)+" * "+cppName(stride))
			}
			step := k
			if len(in.Shape) == 2 && axis == 0 {
				step *= in.Shape[1]
			}
			ix := b.constant(Int32, fmt.Sprintf("%du", step))
			offset = b.emit("addi", u, []string{offset, ix}, "overflow = wrap", cppName(offset)+" + "+cppName(ix))
			// Masks suppress accesses, not C++ pointer formation. Sanitize the
			// complete offset, including the reduction position, before adding.
			offset = b.emit("select", u, []string{valid, offset, zero}, "", fmt.Sprintf("ct::select(%s, %s, %s)", cppName(valid), cppName(offset), cppName(zero)))
			ptr := b.emit("offset", ptrType, []string{inputArgs[in.ID], offset}, "", cppName(inputArgs[in.ID])+" + "+cppName(offset))
			value := b.load(ptr, valid, padding, f)
			if out.Op == "sum" {
				wide := b.emit("ftof", d, []string{value}, "rounding_mode = nearest_even", "ct::element_cast<double>("+cppName(value)+")")
				acc = b.emit("addf", d, []string{acc, wide}, "rounding_mode = nearest_even subnormals = preserve", fmt.Sprintf("ct::add(%s, %s, ct::round_ties_to_even_t{}, ct::preserve_subnormals_t{})", cppName(acc), cppName(wide)))
			} else if k == 0 {
				acc = value
			} else {
				greater := b.emit("cmpf", maskType, []string{value, acc}, "comparison_predicate = gt comparison_ordering = ordered", cppName(value)+" > "+cppName(acc))
				next := b.emit("select", f, []string{greater, value, acc}, "", fmt.Sprintf("ct::select(%s, %s, %s)", cppName(greater), cppName(value), cppName(acc)))
				nan := b.emit("cmpf", maskType, []string{value, value}, "comparison_predicate = ne comparison_ordering = unordered", "ct::isnan("+cppName(value)+")")
				quiet := b.constant(f, "std::numeric_limits<float>::quiet_NaN()")
				acc = b.emit("select", f, []string{nan, quiet, next}, "", fmt.Sprintf("ct::select(%s, %s, %s)", cppName(nan), cppName(quiet), cppName(next)))
			}
		}
		if out.Op == "sum" {
			acc = b.emit("ftof", f, []string{acc}, "rounding_mode = nearest_even", "ct::element_cast<float>("+cppName(acc)+")")
		}
		arg := fmt.Sprintf("%%arg%d", len(r.inputs)+j)
		ptr := b.emit("offset", ptrType, []string{arg, safeRows}, "", cppName(arg)+" + "+cppName(safeRows))
		b.serial++
		nextToken := fmt.Sprintf("%%tok%d", b.serial)
		b.fn.Operations = append(b.fn.Operations, &Operation{Line: len(b.fn.Operations) + 1, OpCode: "cuda_tile.store_ptr_tko", Results: []string{nextToken}, Operands: []string{ptr, acc, valid, b.token}, Type: Token.String(), TypeVal: Token, Attributes: fmt.Sprintf("memory_ordering_semantics = weak semantic_id = %d", out.ID)})
		fmt.Fprintf(&b.source, "  ct::store_masked(%s, %s, %s);\n", cppName(ptr), cppName(acc), cppName(valid))
		store := b.fn.Operations[len(b.fn.Operations)-1]
		if out.SourceLoc.File != "" && out.SourceLoc.Line > 0 {
			key := fmt.Sprintf("node%d", out.ID)
			store.LocRef, store.Loc = "#"+key, b.module.Locations[key]
		}
		b.token = nextToken
	}
	b.fn.Operations = append(b.fn.Operations, &Operation{Line: len(b.fn.Operations) + 1, OpCode: "cuda_tile.return", Operands: []string{b.token}})
	b.source.WriteString("}\n")
	if b.source.Len() > 1<<20 {
		return nil, fmt.Errorf("tileir: reduction source byte budget exceeded")
	}
	hazards, shapes := b.fn.Verify()
	if len(hazards) != 0 || len(shapes) != 0 {
		return nil, fmt.Errorf("tileir: ordered reduction verification: %v %v", hazards, shapes)
	}
	schema, err := cudakernel.NewSchema(params...)
	if err != nil {
		return nil, err
	}
	grid := [3]uint32{uint32((int64(r.maxOutputs) + r.width - 1) / r.width), 1, 1}
	return &LoweringResult{Module: b.module, Schema: schema, Source: b.source.String(), Grid: grid}, nil
}

// verifyReductionSSA checks the emitted subset. Imported vendor functions and
// the older lowering retain their existing verification rules.
func (f *Function) verifyReductionSSA() []ShapeError {
	var errors []ShapeError
	values := make(map[string]Type)
	for _, p := range f.Params {
		values[p.Name] = p.TypeVal
	}
	fail := func(op *Operation, reason string) {
		errors = append(errors, ShapeError{Op: op, Reason: reason, Expected: "ordered reduction SSA contract", Actual: op.OpCode})
	}
	kind := func(t Type) TypeKind {
		if v, ok := t.(*TileType); ok {
			t = v.ElementType
		}
		if t == nil {
			return TypeKind(255)
		}
		return t.Kind()
	}
	arities := map[string]int{"make_token": 0, "constant": 0, "get_tile_block_id": 0, "iota": 0, "addi": 2, "muli": 2, "cmpi": 2, "cmpf": 2, "offset": 2, "ftof": 1, "addf": 2, "select": 3, "load_ptr_tko": 4, "store_ptr_tko": 4, "return": 1}
	var token string
	var width int64
	for _, op := range f.Operations {
		code := strings.TrimPrefix(op.OpCode, "cuda_tile.")
		arity, known := arities[code]
		if !known || len(op.Operands) != arity {
			fail(op, "unknown operation or operand count")
			continue
		}
		args := make([]Type, len(op.Operands))
		for i, operand := range op.Operands {
			args[i] = values[operand]
			if args[i] == nil {
				fail(op, "operand is not defined before use")
			}
		}
		results := []Type{op.TypeVal}
		if ft, ok := op.TypeVal.(*FunctionType); ok {
			results = ft.Results
		}
		if code == "return" {
			results = nil
		}
		expectedResults := 1
		if code == "load_ptr_tko" {
			expectedResults = 2
		}
		if code == "return" {
			expectedResults = 0
		}
		if len(results) != len(op.Results) || len(results) != expectedResults {
			fail(op, "result type count")
			continue
		}
		for _, t := range append(append([]Type(nil), args...), results...) {
			if tile, ok := t.(*TileType); ok {
				if len(tile.Shape) != 1 || tile.Shape[0] <= 0 || tile.Shape[0]&(tile.Shape[0]-1) != 0 {
					fail(op, "vector must have a power of two width")
					continue
				}
				if width == 0 {
					width = tile.Shape[0]
				} else if width != tile.Shape[0] {
					fail(op, "vector widths differ")
				}
			}
		}
		if len(results) > 0 && results[0] == nil {
			fail(op, "missing result type")
			continue
		}
		switch code {
		case "make_token":
			if token != "" || len(results) != 1 || kind(results[0]) != KindToken {
				fail(op, "initial token")
			}
			token = op.Results[0]
		case "load_ptr_tko", "store_ptr_tko":
			mask, value := 1, 2
			if code == "store_ptr_tko" {
				mask, value = 2, 1
			}
			if kind(args[0]) != KindPointer || kind(args[value]) != KindFloat32 || kind(args[mask]) != KindInt1 || kind(args[3]) != KindToken || op.Operands[3] != token {
				fail(op, "memory types or ordered token predecessor")
			}
			if code == "load_ptr_tko" && (len(results) != 2 || kind(results[0]) != KindFloat32 || kind(results[len(results)-1]) != KindToken) || code == "store_ptr_tko" && (len(results) != 1 || kind(results[0]) != KindToken) {
				fail(op, "memory result types")
			}
			token = op.Results[len(op.Results)-1]
		case "return":
			if op.Operands[0] != token || kind(args[0]) != KindToken {
				fail(op, "return must consume final token")
			}
		case "addf":
			if _, ok := results[0].(*TileType); !ok {
				fail(op, "sum result must be a vector")
			}
			if kind(args[0]) != KindFloat64 || kind(args[1]) != KindFloat64 || kind(results[0]) != KindFloat64 || !strings.Contains(op.Attributes, "rounding_mode = nearest_even") || !strings.Contains(op.Attributes, "subnormals = preserve") {
				fail(op, "ordered FP64 sum with nearest even rounding")
			}
		case "ftof":
			if !(kind(args[0]) == KindFloat32 && kind(results[0]) == KindFloat64 || kind(args[0]) == KindFloat64 && kind(results[0]) == KindFloat32) || !strings.Contains(op.Attributes, "rounding_mode = nearest_even") {
				fail(op, "explicit FP32/FP64 conversion")
			}
		case "select":
			if kind(args[0]) != KindInt1 || args[1] == nil || args[2] == nil || args[1].String() != args[2].String() || args[1].String() != results[0].String() {
				fail(op, "select alternatives must have the same type")
			}
		case "addi", "muli", "cmpi":
			if kind(args[0]) != KindInt32 || kind(args[1]) != KindInt32 || code != "cmpi" && kind(results[0]) != KindInt32 || code == "cmpi" && kind(results[0]) != KindInt1 {
				fail(op, "integer address or mask operation types")
			}
		case "cmpf":
			if !(strings.Contains(op.Attributes, "comparison_predicate = gt comparison_ordering = ordered") || strings.Contains(op.Attributes, "comparison_predicate = ne comparison_ordering = unordered")) {
				fail(op, "max comparison policy")
			}
			if kind(args[0]) != KindFloat32 || kind(args[1]) != KindFloat32 || kind(results[0]) != KindInt1 {
				fail(op, "FP32 max comparison types")
			}
		case "offset":
			if kind(args[0]) != KindPointer || kind(args[1]) != KindInt32 || kind(results[0]) != KindPointer {
				fail(op, "pointer offset types")
			}
		}
		for i, name := range op.Results {
			if values[name] != nil {
				fail(op, "duplicate SSA definition")
			}
			values[name] = results[i]
		}
	}
	return errors
}
