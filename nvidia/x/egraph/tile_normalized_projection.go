package egraph

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/tileir"
)

// TileNormalizedProjection specifies a shape specialization in Expr.Data for
// tile_rmsnorm_proj, tile_rmsnorm_qkv and tile_rmsnorm_qkv_split (and fused_
// aliases). X is contiguous row-major M by K, normalization weights have K
// elements, projection weights are contiguous row-major K by N, and Out is
// contiguous row-major M by N. All buffers contain float32 and must be disjoint.
// Packed QKV uses one K by N matrix. Split QKV returns Q+(K+V) in one buffer,
// preserving the existing rewrite's addition; it does not return a tuple.
//
// The pointer ABI is X, W_norm, W_proj, Out, followed by scalar float32 eps;
// split replaces W_proj with W_q, W_k, W_v. The expression children must be
// symbolic leaves in rewrite order: X, W_norm, eps, then projection weights.
// eps must be positive and finite at launch. Shape is compiled into the kernel;
// there are no runtime dimension arguments. Rebinding shape requires lowering
// again. Launch exactly Grid(), with one row per grid y and 16 columns per x.
// No launch or runtime allocation validation occurs during source generation.
//
// M is in [1,65535] because rows map to CUDA grid y. Grid x must fit the
// documented 24-bit Tile block ID range. Each float32 buffer byte extent
// (four times its element count) must fit signed int64.
// Finite unrolling admits K while 64+30*K <= 262144 operations; 30 is the
// worst split recipe's operations per contraction term, with 64 reserved for
// fixed setup/output operations. Generated source is also limited to 64 MiB.
// These last two limits are explicit construction budgets, not compiler or
// hardware limits. Runtime K and larger recipes require loop-region support.
// The reduction and projections visit all K positions in order, use float32
// arithmetic, and preserve subnormals. Device rsqrt accuracy, compiler support
// and numerical error require separate qualification. This reference lowering
// makes no tensor-core or performance claim.
type TileNormalizedProjection struct {
	M, N, K int
}

// Grid returns the exact launch grid or an error for an unsupported shape.
func (s TileNormalizedProjection) Grid() ([3]uint32, error) {
	const maxElements = int64(1<<63-1) / 4 // float32 byte extents also fit int64
	if s.M < 1 || s.M > 65535 || s.N < 1 || s.K < 1 {
		return [3]uint32{}, fmt.Errorf("egraph: normalized projection requires M in [1,65535] and positive N,K")
	}
	if int64(s.N) > 16*(1<<24-1) || int64(s.M) > maxElements/int64(s.K) ||
		int64(s.N) > maxElements/int64(s.K) || int64(s.M) > maxElements/int64(s.N) {
		return [3]uint32{}, fmt.Errorf("egraph: normalized projection exceeds Tile block ID or signed 64-bit buffer byte extent range")
	}
	if s.K > (262144-64)/30 {
		return [3]uint32{}, fmt.Errorf("egraph: normalized projection finite-unrolling budget exceeded: 64+30*K must be <= 262144 operations; runtime or larger K requires loop-region lowering")
	}
	return [3]uint32{uint32((s.N + 15) / 16), uint32(s.M), 1}, nil
}

func normalizedProjectionOp(op string) bool {
	switch op {
	case "tile_rmsnorm_proj", "fused_rmsnorm_proj", "tile_rmsnorm_qkv", "fused_rmsnorm_qkv", "tile_rmsnorm_qkv_split", "fused_rmsnorm_qkv_split":
		return true
	}
	return false
}

func normalizedSwiGLUOp(op string) bool {
	switch op {
	case "tile_rmsnorm_gemv_swiglu", "tile_rmsnorm_swiglu", "fused_rmsnorm_gemv_swiglu", "fused_rmsnorm_swiglu":
		return true
	}
	return false
}

// A normalized projection cannot be silently treated as a materialized input
// by the legacy SwiGLU emitter when extraction chooses that nested form.
func normalizedSwiGLUNode(node *Expr) bool {
	if normalizedSwiGLUOp(node.Op) {
		return true
	}
	if (node.Op != "tile_swiglu" && node.Op != "fused_swiglu") || len(node.Children) == 0 {
		return false
	}
	pending := []*Expr{node.Children[0]}
	seen := make(map[*Expr]bool)
	for len(pending) > 0 {
		child := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		if child == nil || seen[child] {
			continue
		}
		seen[child] = true
		if normalizedProjectionOp(child.Op) || normalizedSwiGLUOp(child.Op) || child.Op == "rmsnorm" {
			return true
		}
		pending = append(pending, child.Children...)
	}
	return false
}

// projectionBuilder emits the same scalar/vector SSA and C++ Tile operations.
// Its only reduction is finite ordered addition, so no region encoding is needed.
type projectionBuilder struct {
	fn     *tileir.Function
	source strings.Builder
	cpp    map[string]string
	serial int
	token  string
}

func (b *projectionBuilder) emit(code string, typ tileir.Type, operands []string, attrs, cpp string) string {
	b.serial++
	v := fmt.Sprintf("%%p%d", b.serial)
	b.fn.Operations = append(b.fn.Operations, &tileir.Operation{
		OpCode: "cuda_tile." + code, Results: []string{v}, TypeVal: typ, Operands: operands, Attributes: attrs,
	})
	b.cpp[v] = strings.TrimPrefix(v, "%")
	if cpp != "" {
		fmt.Fprintf(&b.source, "  auto %s = %s;\n", b.cpp[v], cpp)
	} else if len(operands) > 0 {
		b.cpp[v] = b.cpp[operands[0]]
	}
	return v
}

func (b *projectionBuilder) c(v string) string { return b.cpp[v] }

const projectionNearest = "{rounding_mode = nearest_even}"
const projectionNoWrap = "{overflow = no_signed_wrap}"
const projectionWeak = "{memory_ordering_semantics = weak}"

func lowerNormalizedProjection(node *Expr, opts CodegenOptions) (*tileir.Module, string, error) {
	s, ok := node.Data.(TileNormalizedProjection)
	if !ok {
		return nil, "", fmt.Errorf("egraph: %s requires TileNormalizedProjection shape metadata in Expr.Data", node.Op)
	}
	if _, err := s.Grid(); err != nil {
		return nil, "", err
	}
	switch opts.Precision {
	case "", "float", "float32", "f32":
	default:
		return nil, "", fmt.Errorf("egraph: normalized projection supports only contiguous float32 buffers, got %q", opts.Precision)
	}
	weights := []string{"W_proj"}
	if strings.HasSuffix(node.Op, "qkv") {
		weights[0] = "W_qkv"
	}
	if strings.HasSuffix(node.Op, "qkv_split") {
		weights = []string{"W_q", "W_k", "W_v"}
	}
	if len(node.Children) != 3+len(weights) {
		return nil, "", fmt.Errorf("egraph: %s requires X, W_norm, eps and %d projection leaves", node.Op, len(weights))
	}
	for _, child := range node.Children {
		if child == nil || child.Op == "" || len(child.Children) != 0 || child.Data != nil && child.Data != child.Op {
			return nil, "", fmt.Errorf("egraph: normalized projection operands must be symbolic leaves; materialize composite or literal operands before lowering")
		}
	}
	name := opts.KernelName
	if name == "" {
		name = "NormalizedProjection"
	}
	if len(name) > 128 {
		return nil, "", fmt.Errorf("egraph: normalized projection kernel identifier too long")
	}
	for i, c := range name {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || i > 0 && c >= '0' && c <= '9') {
			return nil, "", fmt.Errorf("egraph: invalid normalized projection kernel identifier")
		}
	}
	b := &projectionBuilder{fn: &tileir.Function{Name: name, Entry: true}, cpp: make(map[string]string)}
	ptr := tileir.NewTile(nil, tileir.NewPointer(tileir.Float32))
	vptr := tileir.NewTile([]int64{16}, tileir.NewPointer(tileir.Float32))
	f := tileir.NewTile([]int64{16}, tileir.Float32)
	i := tileir.NewTile([]int64{16}, tileir.Int64)
	maskType := tileir.NewTile([]int64{16}, tileir.Int1)
	scalarI := tileir.NewTile(nil, tileir.Int64)
	scalarF := tileir.NewTile(nil, tileir.Float32)
	names := append([]string{"X", "W_norm"}, weights...)
	names = append(names, "Out")
	b.source.WriteString("#include \"cuda_tile.h\"\nextern \"C\" __tile_global__ void " + name + "(")
	bases := make(map[string]string)
	for j, n := range names {
		if j > 0 {
			b.source.WriteString(", ")
		}
		prefix := "const "
		if n == "Out" {
			prefix = ""
		}
		b.source.WriteString(prefix + "float* __restrict__ " + n)
		arg := "%" + n
		b.fn.Params = append(b.fn.Params, &tileir.Param{Name: arg, TypeVal: ptr})
		b.cpp[arg] = n
	}
	b.fn.Params = append(b.fn.Params, &tileir.Param{Name: "%eps", TypeVal: scalarF})
	b.cpp["%eps"] = "eps"
	b.source.WriteString(", float eps) {\n  namespace ct = ::cuda::tiles;\n")
	for _, n := range names {
		one := b.emit("reshape", tileir.NewTile([]int64{1}, ptr.ElementType), []string{"%" + n}, "", "")
		bases[n] = b.emit("broadcast", vptr, []string{one}, "", "")
	}
	b.token = b.emit("make_token", tileir.Token, nil, "", "")
	i32 := tileir.NewTile(nil, tileir.Int32)
	b.fn.Operations = append(b.fn.Operations, &tileir.Operation{OpCode: "cuda_tile.get_tile_block_id", Results: []string{"%bx", "%by", "%bz"}, TypeVal: tileir.NewFunction(nil, []tileir.Type{i32, i32, i32})})
	b.cpp["%bx"], b.cpp["%by"] = "ct::bid().x", "ct::bid().y"
	vec := func(v string, typ tileir.Type) string {
		t := typ.(*tileir.TileType)
		one := b.emit("reshape", tileir.NewTile([]int64{1}, t.ElementType), []string{v}, "", "")
		return b.emit("broadcast", typ, []string{one}, "", "")
	}
	index := func(n int64) string {
		return b.emit("constant", i, nil, fmt.Sprintf("{value = %d}", n), fmt.Sprintf("ct::full<ct::tile<long long, ct::shape<16>>>(%dll)", n))
	}
	integer := func(code, a, c, cpp string) string { return b.emit(code, i, []string{a, c}, projectionNoWrap, cpp) }
	zeroIndex := index(0)
	row := b.emit("exti", scalarI, []string{"%by"}, "{signedness = unsigned}", "")
	row = vec(row, i)
	block := b.emit("exti", scalarI, []string{"%bx"}, "{signedness = unsigned}", "")
	block = vec(block, i)
	width := index(16)
	start := integer("muli", block, width, b.c(block)+" * "+b.c(width))
	lanes := b.emit("iota", i, nil, "", "ct::iota<ct::tile<long long, ct::shape<16>>>()")
	col := integer("addi", start, lanes, b.c(start)+" + "+b.c(lanes))
	limit := index(int64(s.N))
	valid := b.emit("cmpi", maskType, []string{col, limit}, "{comparison_predicate = less_than, signedness = unsigned}", b.c(col)+" < "+b.c(limit))
	rowLimit := index(int64(s.M))
	rowValid := b.emit("cmpi", maskType, []string{row, rowLimit}, "{comparison_predicate = less_than, signedness = unsigned}", b.c(row)+" < "+b.c(rowLimit))
	valid = b.emit("andi", maskType, []string{valid, rowValid}, "", b.c(valid)+" & "+b.c(rowValid))
	constant := func(n float32) string {
		return b.emit("constant", f, nil, fmt.Sprintf("{value = %g}", n), fmt.Sprintf("ct::full<ct::tile<float, ct::shape<16>>>(%.9ef)", n))
	}
	zero := constant(0)
	floatOp := func(code, a, c string) string {
		call := map[string]string{"addf": "add", "mulf": "mul", "divf": "div"}[code]
		return b.emit(code, f, []string{a, c}, projectionNearest, fmt.Sprintf("ct::%s(%s, %s, ct::round_ties_to_even_t{}, ct::preserve_subnormals_t{})", call, b.c(a), b.c(c)))
	}
	address := func(base, offset string) string {
		safe := b.emit("select", i, []string{valid, offset, zeroIndex}, "", fmt.Sprintf("ct::select(%s, %s, %s)", b.c(valid), b.c(offset), b.c(zeroIndex)))
		return b.emit("offset", vptr, []string{bases[base], safe}, "", b.c(bases[base])+" + "+b.c(safe))
	}
	load := func(base, offset string) string {
		p := address(base, offset)
		v := b.emit("load_ptr_tko", tileir.NewFunction(nil, []tileir.Type{f, tileir.Token}), []string{p, valid, zero, b.token}, projectionWeak, fmt.Sprintf("ct::load_masked(%s, %s, %s)", b.c(p), b.c(valid), b.c(zero)))
		b.serial++
		b.token = fmt.Sprintf("%%p%d", b.serial)
		b.fn.Operations[len(b.fn.Operations)-1].Results = append(b.fn.Operations[len(b.fn.Operations)-1].Results, b.token)
		return v
	}
	kWidth := index(int64(s.K))
	rowBase := integer("muli", row, kWidth, b.c(row)+" * "+b.c(kWidth))
	xs := make([]string, s.K)
	squares := zero
	for k := 0; k < s.K; k++ {
		ki := index(int64(k))
		offset := integer("addi", rowBase, ki, b.c(rowBase)+" + "+b.c(ki))
		xs[k] = load("X", offset)
		squares = floatOp("addf", squares, floatOp("mulf", xs[k], xs[k]))
	}
	mean := floatOp("divf", squares, constant(float32(s.K)))
	eps := vec("%eps", f)
	denom := floatOp("addf", mean, eps)
	inv := b.emit("rsqrt", f, []string{denom}, "", "ct::rsqrt("+b.c(denom)+")")
	acc := make([]string, len(weights))
	for j := range acc {
		acc[j] = zero
	}
	for k, x := range xs {
		ki := index(int64(k))
		wn := load("W_norm", ki)
		h := floatOp("mulf", floatOp("mulf", x, inv), wn)
		weightBase := index(int64(k) * int64(s.N))
		offset := integer("addi", weightBase, col, b.c(weightBase)+" + "+b.c(col))
		for j, w := range weights {
			acc[j] = floatOp("addf", acc[j], floatOp("mulf", h, load(w, offset)))
		}
	}
	out := acc[0]
	if len(acc) == 3 {
		out = floatOp("addf", acc[0], floatOp("addf", acc[1], acc[2]))
	}
	nWidth := index(int64(s.N))
	outRow := integer("muli", row, nWidth, b.c(row)+" * "+b.c(nWidth))
	offset := integer("addi", outRow, col, b.c(outRow)+" + "+b.c(col))
	p := address("Out", offset)
	b.token = b.emit("store_ptr_tko", tileir.Token, []string{p, out, valid, b.token}, projectionWeak, "")
	fmt.Fprintf(&b.source, "  ct::store_masked(%s, %s, %s);\n}\n", b.c(p), b.c(out), b.c(valid))
	b.fn.Operations = append(b.fn.Operations, &tileir.Operation{OpCode: "cuda_tile.return"})
	if len(b.fn.Operations) > 262144 || b.source.Len() > 64<<20 {
		return nil, "", fmt.Errorf("egraph: normalized projection generated operation or source byte budget exceeded")
	}
	mod := &tileir.Module{Name: "kernels", Producer: "egraph_normalized_projection", Functions: []*tileir.Function{b.fn}}
	return mod, b.source.String(), nil
}
