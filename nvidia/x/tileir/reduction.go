package tileir

import (
	"fmt"
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
	// Typed pointer tiles admit the bounded vector widths of ast_vector.go.
	if !vectorWidth([]int64{r.width}) {
		return r, fmt.Errorf("%w: reduction tile width must be a power of two from 1 to 1024", ErrUnsupportedTileOp)
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

// reductionBuilder emits one typed bytecode SSA chain and the corresponding
// Tile C++ reference source. Semantic identity is recorded in MemorySource
// sidecars rather than in bytecode attributes or debug locations.
type reductionBuilder struct {
	fn     *Function
	source strings.Builder
	cpp    map[string]string // SSA value to C++ variable
	serial int
	token  string
	ref    *MemorySource // record receiving emitted operations
}

func (b *reductionBuilder) name() string {
	b.serial++
	return fmt.Sprintf("%%v%d", b.serial)
}

func (b *reductionBuilder) emit(code string, typ Type, results, operands []string, attrs string) {
	if b.ref != nil {
		b.ref.Operations = append(b.ref.Operations, len(b.fn.Operations))
		b.ref.Values = append(b.ref.Values, results...)
	}
	b.fn.Operations = append(b.fn.Operations, &Operation{OpCode: "cuda_tile." + code, Results: results, Operands: operands, TypeVal: typ, Attributes: attrs})
}

// op emits a single-result operation whose C++ counterpart is expr.
// An empty expr aliases the first operand: extension, reshape and broadcast
// are explicit in bytecode but implicit in the C++ reference source.
func (b *reductionBuilder) op(code string, typ Type, operands []string, attrs, expr string) string {
	v := b.name()
	b.emit(code, typ, []string{v}, operands, attrs)
	if expr == "" {
		b.cpp[v] = b.cpp[operands[0]]
		return v
	}
	b.cpp[v] = strings.TrimPrefix(v, "%")
	fmt.Fprintf(&b.source, "  auto %s = %s;\n", b.cpp[v], expr)
	return v
}

func (b *reductionBuilder) c(v string) string { return b.cpp[v] }

// pointers forms the rank-one base pointer tile of a scalar pointer parameter.
func (b *reductionBuilder) pointers(arg string, typ *TileType) string {
	one := b.op("reshape", NewTile([]int64{1}, typ.ElementType), []string{arg}, "", "")
	return b.op("broadcast", typ, []string{one}, "", "")
}

const (
	reductionWeak       = "{memory_ordering_semantics = weak}"
	reductionNoWrap     = "{overflow = no_signed_wrap}"
	reductionNearest    = "{rounding_mode = nearest_even}"
	reductionBound      = "{comparison_predicate = less_than, signedness = unsigned}"
	reductionUnsigned   = "{signedness = unsigned}"
	reductionGreater    = "{comparison_predicate = greater_than, comparison_ordering = ordered}"
	reductionNaN        = "{comparison_predicate = not_equal, comparison_ordering = unordered}"
	reductionQuietNaN32 = `{value = dense<"0x0000c07f">}` // 0x7fc00000

	reductionNegInfBits   = 0xff800000
	reductionQuietNaNBits = 0x7fc00000
)

// floatTile returns a C++ expression for a width-w FP32 tile whose elements
// have the given bit pattern. The reference source cannot use
// std::numeric_limits: NVRTC has no <limits>, and nvcc rejects calls to its
// constexpr host functions from a __tile_global__ function.
func floatTile(w int64, bits uint32) string {
	return fmt.Sprintf("ct::element_bitcast<float>(ct::full<ct::tile<unsigned int, ct::shape<%d>>>(0x%08xu))", w, bits)
}

// lowerReductionRegion lowers direct-input FP32 sum and max reductions to a
// void kernel in the typed bytecode subset, and to equivalent Tile C++.
// Sums accumulate in FP64 with nearest-even rounding and no flush; max
// propagates NaN and keeps the first of equal values. Every lane forms only
// sanitized in-bounds addresses and accesses memory under its output mask.
// Result.Memory holds the source locations and semantic node IDs.
// When every output is empty, Grid is zero, the caller must not launch,
// and the Module has no pointer parameters or memory operations; Schema
// still describes the C++ reference kernel's ABI, so AssembleSemanticMemory
// refuses the result.
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
	w := r.width
	work := r.maxOutputs > 0
	b := &reductionBuilder{fn: &Function{Name: name, Entry: true}, cpp: make(map[string]string)}
	module := &Module{Name: name + "_mod", Producer: "lib/nvidia/x/tileir", Functions: []*Function{b.fn}}
	b.source.WriteString("#include \"cuda_tile.h\"\nextern \"C\" __tile_global__ void " + name + "(")
	var params []cudakernel.Param
	var refs []*MemorySource
	inputRefs := make(map[int]*MemorySource)
	ptr := NewTile(nil, NewPointer(Float32))
	for i, n := range append(append([]*cudair.Node(nil), r.inputs...), r.outputs...) {
		if i > 0 {
			b.source.WriteString(", ")
		}
		arg := fmt.Sprintf("arg%d", i)
		b.source.WriteString("float* " + arg)
		b.cpp["%"+arg] = arg
		if work {
			b.fn.Params = append(b.fn.Params, &Param{Name: "%" + arg, TypeVal: ptr})
		}
		paramName := n.Name
		if paramName == "" {
			paramName = arg
		}
		params = append(params, cudakernel.Param{Name: paramName, Type: "f32", ByteWidth: 8, Alignment: 8, IsPointer: true})
		if i < len(r.inputs) {
			ref := &MemorySource{NodeID: n.ID, Role: "input", ABIIndex: i, Location: n.SourceLoc}
			refs = append(refs, ref)
			inputRefs[n.ID] = ref
		}
	}
	b.source.WriteString(") {\n  namespace ct = ::cuda::tiles;\n")
	if work {
		i32, i64 := NewTile(nil, Int32), NewTile(nil, Int64)
		vi64, vf32, vf64 := NewTile([]int64{w}, Int64), NewTile([]int64{w}, Float32), NewTile([]int64{w}, Float64)
		mask, vptr := NewTile([]int64{w}, Int1), NewTile([]int64{w}, NewPointer(Float32))
		b.token = b.name()
		b.emit("make_token", Token, []string{b.token}, nil, "")
		bx, by, bz := b.name(), b.name(), b.name()
		b.emit("get_tile_block_id", NewFunction(nil, []Type{i32, i32, i32}), []string{bx, by, bz}, nil, "")
		b.cpp[bx] = strings.TrimPrefix(bx, "%")
		fmt.Fprintf(&b.source, "  auto %s = ct::bid().x;\n", b.cpp[bx])
		// Block IDs are below 2^24 (Ops.td:2386) and w <= 1024, so the
		// unmasked row index and every sanitized offset fit i64 unwrapped.
		block := b.op("exti", i64, []string{bx}, reductionUnsigned, "")
		width := b.op("constant", i64, nil, fmt.Sprintf("{value = %d}", w), fmt.Sprintf("%du", w))
		start := b.op("muli", i64, []string{block, width}, reductionNoWrap, b.c(block)+" * "+b.c(width))
		start1 := b.op("reshape", NewTile([]int64{1}, Int64), []string{start}, "", "")
		starts := b.op("broadcast", vi64, []string{start1}, "", "")
		lanes := b.op("iota", vi64, nil, "", fmt.Sprintf("ct::iota<ct::tile<unsigned int, ct::shape<%d>>>()", w))
		rows := b.op("addi", vi64, []string{starts, lanes}, reductionNoWrap, b.c(starts)+" + "+b.c(lanes))
		zero := b.op("constant", vi64, nil, "{value = 0}", fmt.Sprintf("ct::full<ct::tile<unsigned int, ct::shape<%d>>>(0u)", w))
		bases := make(map[string]string)
		for j, out := range r.outputs {
			in, axis := out.Inputs[0], out.Axes[0]
			terms, count := in.Shape[axis], out.Shape[0]
			opRef := &MemorySource{NodeID: out.ID, Role: out.Op, ABIIndex: -1, Location: out.SourceLoc}
			outRef := &MemorySource{NodeID: out.ID, Role: "output", ABIIndex: len(r.inputs) + j, Location: out.SourceLoc}
			refs = append(refs, opRef, outRef)
			// Empty ABI objects may have null bindings: emit no pointer operations.
			if count == 0 {
				continue
			}
			b.ref = opRef
			limit := b.op("constant", vi64, nil, fmt.Sprintf("{value = %d}", count), fmt.Sprintf("%du", count))
			valid := b.op("cmpi", mask, []string{rows, limit}, reductionBound, b.c(rows)+" < "+b.c(limit))
			safe := b.op("select", vi64, []string{valid, rows, zero}, "", fmt.Sprintf("ct::select(%s, %s, %s)", b.c(valid), b.c(rows), b.c(zero)))
			padding := b.op("constant", vf32, nil, "{value = 0.0}", fmt.Sprintf("ct::full<ct::tile<float, ct::shape<%d>>>(0.0f)", w))
			var acc string
			if out.Op == "sum" {
				acc = b.op("constant", vf64, nil, "{value = 0.0}", fmt.Sprintf("ct::full<ct::tile<double, ct::shape<%d>>>(0.0)", w))
			} else {
				acc = b.op("constant", vf32, nil, "{value = -inf}", floatTile(w, reductionNegInfBits))
			}
			arg := fmt.Sprintf("%%arg%d", indexOf(r.inputs, in))
			for k := 0; k < terms; k++ {
				// The row base is recomputed per term, as in the C++ reference
				// source this route has emitted since it was introduced.
				rowBase := safe
				if len(in.Shape) == 2 && axis == 1 {
					stride := b.op("constant", vi64, nil, fmt.Sprintf("{value = %d}", terms), fmt.Sprintf("%du", terms))
					rowBase = b.op("muli", vi64, []string{safe, stride}, reductionNoWrap, b.c(safe)+" * "+b.c(stride))
				}
				step := k
				if len(in.Shape) == 2 && axis == 0 {
					step *= in.Shape[1]
				}
				ix := b.op("constant", vi64, nil, fmt.Sprintf("{value = %d}", step), fmt.Sprintf("%du", step))
				offset := b.op("addi", vi64, []string{rowBase, ix}, reductionNoWrap, b.c(rowBase)+" + "+b.c(ix))
				// Masks suppress accesses, not pointer formation. Sanitize the
				// complete offset, including the reduction position, before adding.
				offset = b.op("select", vi64, []string{valid, offset, zero}, "", fmt.Sprintf("ct::select(%s, %s, %s)", b.c(valid), b.c(offset), b.c(zero)))
				b.ref = inputRefs[in.ID]
				if bases[arg] == "" {
					bases[arg] = b.pointers(arg, vptr)
				}
				p := b.op("offset", vptr, []string{bases[arg], offset}, "", b.c(bases[arg])+" + "+b.c(offset))
				value, token := b.name(), b.name()
				b.emit("load_ptr_tko", NewFunction(nil, []Type{vf32, Token}), []string{value, token}, []string{p, valid, padding, b.token}, reductionWeak)
				b.cpp[value] = strings.TrimPrefix(value, "%")
				fmt.Fprintf(&b.source, "  auto %s = ct::load_masked(%s, %s, %s);\n", b.c(value), b.c(p), b.c(valid), b.c(padding))
				b.token = token
				b.ref = opRef
				switch {
				case out.Op == "sum":
					wide := b.op("ftof", vf64, []string{value}, reductionNearest, "ct::element_cast<double>("+b.c(value)+")")
					acc = b.op("addf", vf64, []string{acc, wide}, reductionNearest, fmt.Sprintf("ct::add(%s, %s, ct::round_ties_to_even_t{}, ct::preserve_subnormals_t{})", b.c(acc), b.c(wide)))
				case k == 0:
					acc = value
				default:
					greater := b.op("cmpf", mask, []string{value, acc}, reductionGreater, b.c(value)+" > "+b.c(acc))
					next := b.op("select", vf32, []string{greater, value, acc}, "", fmt.Sprintf("ct::select(%s, %s, %s)", b.c(greater), b.c(value), b.c(acc)))
					nan := b.op("cmpf", mask, []string{value, value}, reductionNaN, "ct::isnan("+b.c(value)+")")
					quiet := b.op("constant", vf32, nil, reductionQuietNaN32, floatTile(w, reductionQuietNaNBits))
					acc = b.op("select", vf32, []string{nan, quiet, next}, "", fmt.Sprintf("ct::select(%s, %s, %s)", b.c(nan), b.c(quiet), b.c(next)))
				}
			}
			if out.Op == "sum" {
				acc = b.op("ftof", vf32, []string{acc}, reductionNearest, "ct::element_cast<float>("+b.c(acc)+")")
			}
			b.ref = outRef
			base := b.pointers(fmt.Sprintf("%%arg%d", len(r.inputs)+j), vptr)
			dst := b.op("offset", vptr, []string{base, safe}, "", b.c(base)+" + "+b.c(safe))
			stored := b.name()
			b.emit("store_ptr_tko", Token, []string{stored}, []string{dst, acc, valid, b.token}, reductionWeak)
			fmt.Fprintf(&b.source, "  ct::store_masked(%s, %s, %s);\n", b.c(dst), b.c(acc), b.c(valid))
			b.token = stored
			b.ref = nil
		}
	} else {
		for j, out := range r.outputs {
			refs = append(refs, &MemorySource{NodeID: out.ID, Role: out.Op, ABIIndex: -1, Location: out.SourceLoc},
				&MemorySource{NodeID: out.ID, Role: "output", ABIIndex: len(r.inputs) + j, Location: out.SourceLoc})
		}
	}
	b.emit("return", nil, nil, nil, "")
	b.source.WriteString("}\n")
	if b.source.Len() > 1<<20 {
		return nil, fmt.Errorf("tileir: reduction source byte budget exceeded")
	}
	schema, err := cudakernel.NewSchema(params...)
	if err != nil {
		return nil, err
	}
	grid := [3]uint32{uint32((int64(r.maxOutputs) + w - 1) / w), 1, 1}
	result := &LoweringResult{Module: module, Schema: schema, Source: b.source.String(), Grid: grid}
	data, err := module.EncodeBytecode()
	if err != nil {
		return nil, fmt.Errorf("tileir: ordered reduction bytecode: %w", err)
	}
	if errs := b.fn.verifyReductionSSA(); len(errs) != 0 {
		return nil, fmt.Errorf("tileir: ordered reduction policy: %v", errs)
	}
	sources := make([]MemorySource, len(refs))
	for i, ref := range refs {
		sources[i] = *ref
	}
	result.Memory = &SemanticMemoryBinding{byteDigest: memoryDigest(data), references: sources}
	result.Memory.contractDigest = memoryContract(result, sources)
	return result, nil
}

func indexOf(nodes []*cudair.Node, n *cudair.Node) int {
	for i, m := range nodes {
		if m.ID == n.ID {
			return i
		}
	}
	return -1
}

// verifyReductionSSA checks the ordered reduction policy that well-typed
// bytecode does not imply: FP64 nearest-even accumulation without flush,
// nearest-even conversions, unsigned bounds, nonwrapping index arithmetic,
// and masked weak accesses on one token chain. EncodeBytecode checks types.
func (f *Function) verifyReductionSSA() []ShapeError {
	var errors []ShapeError
	fail := func(op *Operation, reason string) {
		errors = append(errors, ShapeError{Op: op, Reason: reason, Expected: "ordered reduction policy", Actual: op.OpCode + " " + op.Attributes})
	}
	types := make(map[string]Type)
	for _, p := range f.Params {
		types[p.Name] = p.TypeVal
	}
	element := func(t Type) TypeKind {
		if tile, ok := t.(*TileType); ok && tile != nil && tile.ElementType != nil {
			return tile.ElementType.Kind()
		}
		return TypeKind(255)
	}
	fixed := map[string]string{
		"addf": reductionNearest, "ftof": reductionNearest,
		"addi": reductionNoWrap, "muli": reductionNoWrap,
		"cmpi": reductionBound, "exti": reductionUnsigned,
		"load_ptr_tko": reductionWeak, "store_ptr_tko": reductionWeak,
		"make_token": "", "get_tile_block_id": "", "reshape": "", "broadcast": "",
		"iota": "", "select": "", "offset": "", "return": "",
	}
	token := ""
	for _, op := range f.Operations {
		if op == nil {
			continue
		}
		code := strings.TrimPrefix(op.OpCode, "cuda_tile.")
		want, known := fixed[code]
		switch {
		case code == "constant":
			if !strings.HasPrefix(op.Attributes, "{value = ") {
				fail(op, "constant value")
			}
		case code == "cmpf":
			if op.Attributes != reductionGreater && op.Attributes != reductionNaN {
				fail(op, "max comparison policy")
			}
		case !known:
			fail(op, "operation outside the ordered reduction set")
		case op.Attributes != want:
			fail(op, "attribute policy")
		}
		switch code {
		case "addf":
			if element(op.TypeVal) != KindFloat64 {
				fail(op, "sum must accumulate in FP64")
			}
		case "make_token":
			token = op.Results[0]
		case "load_ptr_tko", "store_ptr_tko":
			mask := 1
			if code == "store_ptr_tko" {
				mask = 2
			}
			n := len(op.Operands)
			if n != 4 || element(types[op.Operands[mask]]) != KindInt1 || token == "" || op.Operands[n-1] != token {
				fail(op, "access must be masked and follow the ordered token chain")
			}
			if len(op.Results) > 0 {
				token = op.Results[len(op.Results)-1]
			}
		}
		results := []Type{op.TypeVal}
		if tuple, ok := op.TypeVal.(*FunctionType); ok {
			results = tuple.Results
		}
		for i, name := range op.Results {
			if i < len(results) {
				types[name] = results[i]
			}
		}
	}
	return errors
}
