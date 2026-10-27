package tileir

import (
	"fmt"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func matmulMemoryOptions(opts LowerOptions) bool {
	return (opts.TileM == 0 || opts.TileM == 1) && (opts.TileN == 0 || opts.TileN == 1) && (opts.TileK == 0 || opts.TileK == 1)
}

func matmulMemoryCandidate(snap *cudair.Snapshot) bool {
	if snap == nil {
		return false
	}
	nodes, inputs, outputs := snap.Nodes(), snap.Inputs(), snap.Outputs()
	if len(nodes) != 3 || len(inputs) != 2 || len(outputs) != 1 {
		return false
	}
	out := outputs[0]
	if out.Kind != cudair.KindMatMul || len(out.Inputs) != 2 || out.Inputs[0].ID == out.Inputs[1].ID || len(out.Shape) != 2 {
		return false
	}
	a, b := out.Inputs[0], out.Inputs[1]
	if len(a.Shape) != 2 || len(b.Shape) != 2 || a.Shape[1] != b.Shape[0] || out.Shape[0] != a.Shape[0] || out.Shape[1] != b.Shape[1] {
		return false
	}
	for _, n := range nodes {
		if n.Type != cudair.Float32 || len(n.Shape) != 2 || n.Effect != cudair.EffectReadOnly || n.StorageAlias != nil ||
			n.Layout != cudashape.LayoutRowMajor || (len(n.Strides) != 0 && !cudashape.IsContiguousRowMajor(n.Shape, n.Strides)) {
			return false
		}
		for _, dim := range n.Shape {
			if dim < 1 || dim > 4 {
				return false
			}
		}
		if n.ID != out.ID && n.Kind != cudair.KindInput {
			return false
		}
	}
	return (a.ID == inputs[0].ID && b.ID == inputs[1].ID) || (a.ID == inputs[1].ID && b.ID == inputs[0].ID)
}

// LowerSemanticMatMul lowers readonly contiguous row-major FP32 A[M,K] and
// B[K,N] to their sole matrix-product output. Each dimension must be 1..4.
// TileM/N/K must be zero or one; the scalar operations are fully unrolled,
// accumulating widened products in FP64 in increasing K order before one
// FP32 conversion per output. No Tensor Core instruction is promised.
//
// The pointer ABI is A, B, output, regardless of graph input order. The caller
// must launch exactly Grid(1,1,1), provide distinct FP32-aligned nonaliasing
// buffers of at least 4*M*K, 4*K*N and 4*M*N bytes without address wrap, and
// prevent concurrent writes. This function does not allocate, compile or
// execute device storage. Source locations remain in Memory's sidecar.
func LowerSemanticMatMul(snap *cudair.Snapshot, opts LowerOptions) (*LoweringResult, error) {
	if !matmulMemoryCandidate(snap) || !matmulMemoryOptions(opts) {
		return nil, fmt.Errorf("%w: readonly contiguous FP32 matmul with dimensions 1..4 and scalar tiles required", ErrUnsupportedTileOp)
	}
	name := opts.KernelName
	if name == "" {
		name = "tile_matmul"
	}
	if len(name) > 128 {
		return nil, fmt.Errorf("tileir: matmul kernel name too long")
	}
	for i, c := range name {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || i > 0 && c >= '0' && c <= '9') {
			return nil, fmt.Errorf("tileir: invalid matmul kernel name")
		}
	}
	out := snap.Outputs()[0]
	a, b := out.Inputs[0], out.Inputs[1]
	m, k, n := a.Shape[0], a.Shape[1], b.Shape[1]
	roles := []*cudair.Node{a, b, out}
	fn := &Function{Name: name, Entry: true}
	ptr := NewTile(nil, NewPointer(Float32))
	var params []cudakernel.Param
	names := make(map[string]bool)
	for i, node := range roles {
		if node.Name == "" || names[node.Name] {
			return nil, fmt.Errorf("tileir: matmul pointer ABI requires unique nonempty role names")
		}
		names[node.Name] = true
		params = append(params, cudakernel.Param{Name: node.Name, Type: "f32", IsPointer: true, ByteWidth: 8, Alignment: 8})
		fn.Params = append(fn.Params, &Param{Name: fmt.Sprintf("%%arg%d", i), TypeVal: ptr})
	}
	schema, err := cudakernel.NewSchema(params...)
	if err != nil {
		return nil, err
	}
	refs := []MemorySource{
		{NodeID: a.ID, Role: "input", ABIIndex: 0, Location: a.SourceLoc},
		{NodeID: b.ID, Role: "input", ABIIndex: 1, Location: b.SourceLoc},
		{NodeID: out.ID, Role: "matmul", ABIIndex: -1, Location: out.SourceLoc},
		{NodeID: out.ID, Role: "output", ABIIndex: 2, Location: out.SourceLoc},
	}
	// Names are allocated in emission order and have no semantic identity.
	// The detached sidecar records the actual source IDs and ABI roles.
	next := 0
	emit := func(ref int, code string, typ Type, operands []string, attrs string, count int) []string {
		values := make([]string, count)
		for i := range values {
			values[i] = fmt.Sprintf("%%v%d", next)
			next++
		}
		refs[ref].Operations = append(refs[ref].Operations, len(fn.Operations))
		refs[ref].Values = append(refs[ref].Values, values...)
		fn.Operations = append(fn.Operations, &Operation{OpCode: "cuda_tile." + code, TypeVal: typ, Results: values, Operands: operands, Attributes: attrs})
		return values
	}
	one := func(ref int, code string, typ Type, operands []string, attrs string) string {
		return emit(ref, code, typ, operands, attrs, 1)[0]
	}
	f32, f64, i64 := NewTile(nil, Float32), NewTile(nil, Float64), NewTile(nil, Int64)
	mask := one(2, "constant", NewTile(nil, Int1), nil, "{value = dense<\"0x01\">}")
	padding := one(2, "constant", f32, nil, "{value = 0.0}")
	token := one(2, "make_token", Token, nil, "")
	const weak = "{memory_ordering_semantics = weak}"
	const nearest = "{rounding_mode = nearest_even}"
	address := func(ref, index int) string {
		offset := one(ref, "constant", i64, nil, "{value = "+strconv.Itoa(index)+"}")
		return one(ref, "offset", ptr, []string{fmt.Sprintf("%%arg%d", refs[ref].ABIIndex), offset}, "")
	}
	load := func(ref, index int) string {
		p := address(ref, index)
		loaded := emit(ref, "load_ptr_tko", NewFunction(nil, []Type{f32, Token}), []string{p, mask, padding, token}, weak, 2)
		token = loaded[1]
		return loaded[0]
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			acc := one(2, "constant", f64, nil, "{value = 0.0}")
			for term := 0; term < k; term++ {
				left := load(0, i*k+term)
				right := load(1, term*n+j)
				left = one(2, "ftof", f64, []string{left}, nearest)
				right = one(2, "ftof", f64, []string{right}, nearest)
				product := one(2, "mulf", f64, []string{left, right}, nearest)
				acc = one(2, "addf", f64, []string{acc, product}, nearest)
			}
			value := one(2, "ftof", f32, []string{acc}, nearest)
			dst := address(3, i*n+j)
			token = one(3, "store_ptr_tko", Token, []string{dst, value, mask, token}, weak)
		}
	}
	emit(2, "return", nil, nil, "", 0)
	r := &LoweringResult{Module: &Module{Name: name + "_mod", Producer: "lib/nvidia/x/tileir", Functions: []*Function{fn}}, Schema: schema, Grid: [3]uint32{1, 1, 1}}
	data, err := r.Module.EncodeBytecode()
	if err != nil {
		return nil, err
	}
	r.Memory = &SemanticMemoryBinding{byteDigest: memoryDigest(data), references: refs}
	r.Memory.contractDigest = memoryContract(r, refs)
	return r, nil
}
