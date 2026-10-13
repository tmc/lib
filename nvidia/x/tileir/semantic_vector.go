package tileir

import (
	"fmt"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// SemanticVectorLayout describes a logical vector's source launch contract.
// Changes invalidate its SemanticMemoryBinding. BufferBytes is per ABI role.
type SemanticVectorLayout struct {
	Length       int64
	TileSize     int64
	BufferBytes  uint64
	NoDeviceWork bool
}

// ValidatePointers checks three numeric device-address ranges for extent,
// FP32 alignment and nonaliasing before ABI packing. It does not establish
// device placement, allocation ownership or lifetime. Empty layouts perform
// no address checks because they must not be launched.
func (v *SemanticVectorLayout) ValidatePointers(pointers [3]uint64) error {
	if v == nil {
		return fmt.Errorf("tileir: missing vector layout")
	}
	expected, _, err := vectorGeometry(v.Length, v.TileSize)
	if err != nil || *v != *expected {
		return fmt.Errorf("tileir: invalid vector layout")
	}
	if v.NoDeviceWork {
		return nil
	}
	var ends [3]uint64
	for i, ptr := range pointers {
		if ptr == 0 || ptr%4 != 0 || ptr > ^uint64(0)-v.BufferBytes {
			return fmt.Errorf("tileir: invalid vector pointer range %d", i)
		}
		ends[i] = ptr + v.BufferBytes
		for j := 0; j < i; j++ {
			if ptr < ends[j] && pointers[j] < ends[i] {
				return fmt.Errorf("tileir: vector pointer ranges %d and %d alias", j, i)
			}
		}
	}
	return nil
}

func vectorMemoryOptions(opts LowerOptions) bool {
	return (opts.TileN == 0 || opts.TileN == 1) && (opts.TileK == 0 || opts.TileK == 1)
}

func vectorMemoryCandidate(snap *cudair.Snapshot) bool {
	if snap == nil {
		return false
	}
	nodes, inputs, outputs := snap.Nodes(), snap.Inputs(), snap.Outputs()
	if len(nodes) != 3 || len(inputs) != 2 || len(outputs) != 1 {
		return false
	}
	out := outputs[0]
	if out.Kind != cudair.KindBinary || out.Op != "add" || len(out.Shape) != 1 {
		return false
	}
	for _, n := range nodes {
		if n.Type != cudair.Float32 || len(n.Shape) != 1 || n.Shape[0] != out.Shape[0] || n.Shape[0] < 0 || n.Effect != cudair.EffectReadOnly ||
			n.StorageAlias != nil || n.Layout != cudashape.LayoutRowMajor || (len(n.Strides) != 0 && !cudashape.IsContiguousRowMajor(n.Shape, n.Strides)) {
			return false
		}
		if n.ID != out.ID && n.Kind != cudair.KindInput {
			return false
		}
	}
	return len(out.Inputs) == 2 && out.Inputs[0].ID != out.Inputs[1].ID &&
		((out.Inputs[0].ID == inputs[0].ID && out.Inputs[1].ID == inputs[1].ID) || (out.Inputs[0].ID == inputs[1].ID && out.Inputs[1].ID == inputs[0].ID))
}

func vectorGeometry(length, width int64) (*SemanticVectorLayout, [3]uint32, error) {
	var grid [3]uint32
	if width == 0 {
		width = 128
	}
	if !vectorWidth([]int64{width}) || length < 0 {
		return nil, grid, fmt.Errorf("%w: vector length or tile width", ErrUnsupportedTileOp)
	}
	// Host byte extents must be representable before any multiplication.
	maxInt := uint64(^uint(0) >> 1)
	if uint64(length) > maxInt/4 {
		return nil, grid, fmt.Errorf("%w: FP32 buffer byte extent overflow", ErrUnsupportedTileOp)
	}
	blocks := length / width
	if length%width != 0 {
		blocks++
	}
	// Ops.td:2386 limits each Tile block-ID axis to 2^24-1. Do not
	// silently use the larger ordinary CUDA grid limit or truncate to u32.
	if blocks > (1<<24)-1 {
		return nil, grid, fmt.Errorf("%w: Tile block-ID grid limit", ErrUnsupportedTileOp)
	}
	// width<=1024 and blocks<=2^24-1 prove every unmasked logical
	// index (block*width+lane) and its storage-bit/byte scaling fit i64.
	grid = [3]uint32{uint32(blocks), 1, 1}
	return &SemanticVectorLayout{Length: length, TileSize: width, BufferBytes: uint64(length) * 4, NoDeviceWork: length == 0}, grid, nil
}

// LowerSemanticVector lowers a contiguous readonly FP32 vector add to typed
// pointer-memory bytecode. Inputs and output must have identical rank-one
// shapes. TileM selects a power-of-two width1..1024 (default128); TileN/K
// must be zero or one. The caller supplies three distinct, FP32-aligned,
// nonaliasing device buffers of Vector.BufferBytes and launches exactly Grid,
// without concurrent writes or pointer-address wrap. Empty vectors have no
// pointer ABI, memory instructions or device work: the caller skips launch.
// This function does not invoke a compiler, allocate or execute device memory.
func LowerSemanticVector(snap *cudair.Snapshot, opts LowerOptions) (*LoweringResult, error) {
	if !vectorMemoryCandidate(snap) || !vectorMemoryOptions(opts) {
		return nil, fmt.Errorf("%w: equal readonly contiguous FP32 vector add required", ErrUnsupportedTileOp)
	}
	inputs, out := snap.Inputs(), snap.Outputs()[0]
	layout, grid, err := vectorGeometry(int64(out.Shape[0]), opts.TileM)
	if err != nil {
		return nil, err
	}
	kernel := opts.KernelName
	if kernel == "" {
		kernel = "tile_vector_add"
	}
	for i, c := range kernel {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || i > 0 && c >= '0' && c <= '9') {
			return nil, fmt.Errorf("tileir: invalid vector kernel name")
		}
	}
	fn := &Function{Name: kernel, Entry: true}
	r := &LoweringResult{Module: &Module{Name: kernel + "_mod", Functions: []*Function{fn}}, Grid: grid, Vector: layout}
	refs := []MemorySource{{NodeID: inputs[0].ID, Role: "input", ABIIndex: -1, Location: inputs[0].SourceLoc},
		{NodeID: inputs[1].ID, Role: "input", ABIIndex: -1, Location: inputs[1].SourceLoc},
		{NodeID: out.ID, Role: "add", ABIIndex: -1, Location: out.SourceLoc}, {NodeID: out.ID, Role: "output", ABIIndex: -1, Location: out.SourceLoc}}
	if layout.NoDeviceWork {
		fn.Operations = []*Operation{{OpCode: "cuda_tile.return"}}
	} else {
		ptr := NewTile(nil, NewPointer(Float32))
		width := layout.TileSize
		i32, i64 := NewTile(nil, Int32), NewTile(nil, Int64)
		vi64, vf32, vptr := NewTile([]int64{width}, Int64), NewTile([]int64{width}, Float32), NewTile([]int64{width}, NewPointer(Float32))
		mask := NewTile([]int64{width}, Int1)
		var params []cudakernel.Param
		names := make(map[string]bool)
		for i, node := range append(inputs, out) {
			if node.Name == "" || names[node.Name] {
				return nil, fmt.Errorf("tileir: vector pointer ABI requires unique role names")
			}
			names[node.Name] = true
			params = append(params, cudakernel.Param{Name: node.Name, Type: "f32", IsPointer: true, ByteWidth: 8, Alignment: 8})
			fn.Params = append(fn.Params, &Param{Name: fmt.Sprintf("%%arg%d", i), TypeVal: ptr})
		}
		r.Schema, err = cudakernel.NewSchema(params...)
		if err != nil {
			return nil, err
		}
		emit := func(code string, typ Type, results, operands []string, attrs string) int {
			index := len(fn.Operations)
			fn.Operations = append(fn.Operations, &Operation{OpCode: "cuda_tile." + code, TypeVal: typ, Results: results, Operands: operands, Attributes: attrs})
			return index
		}
		one := func(code, name string, typ Type, operands []string, attrs string) int {
			return emit(code, typ, []string{name}, operands, attrs)
		}
		emit("get_tile_block_id", NewFunction(nil, []Type{i32, i32, i32}), []string{"%block_x", "%block_y", "%block_z"}, nil, "")
		one("exti", "%block64", i64, []string{"%block_x"}, "{signedness = unsigned}")
		one("constant", "%width", i64, nil, "{value = "+strconv.FormatInt(width, 10)+"}")
		one("muli", "%start", i64, []string{"%block64", "%width"}, "{overflow = no_signed_wrap}")
		one("reshape", "%start1", NewTile([]int64{1}, Int64), []string{"%start"}, "")
		one("broadcast", "%starts", vi64, []string{"%start1"}, "")
		one("iota", "%lanes", vi64, nil, "")
		one("addi", "%indices", vi64, []string{"%starts", "%lanes"}, "{overflow = no_signed_wrap}")
		one("constant", "%length", vi64, nil, "{value = "+strconv.FormatInt(layout.Length, 10)+"}")
		one("cmpi", "%mask", mask, []string{"%indices", "%length"}, "{comparison_predicate = less_than, signedness = unsigned}")
		one("constant", "%zero", vi64, nil, "{value = 0}")
		// Select the COMPLETE offset before pointer arithmetic. Even
		// inactive lanes form only in-bounds element0 addresses; masked
		// load/store alone would not sanitize out-of-bounds lane pointers.
		one("select", "%safe_indices", vi64, []string{"%mask", "%indices", "%zero"}, "")
		for i := 0; i < 3; i++ {
			one("reshape", fmt.Sprintf("%%ptr%d_1", i), NewTile([]int64{1}, NewPointer(Float32)), []string{fmt.Sprintf("%%arg%d", i)}, "")
			one("broadcast", fmt.Sprintf("%%ptr%d_base", i), vptr, []string{fmt.Sprintf("%%ptr%d_1", i)}, "")
			one("offset", fmt.Sprintf("%%ptr%d", i), vptr, []string{fmt.Sprintf("%%ptr%d_base", i), "%safe_indices"}, "")
		}
		one("constant", "%padding", vf32, nil, "{value = 0.0}")
		one("make_token", "%token0", Token, nil, "")
		weak := "{memory_ordering_semantics = weak}"
		load0 := emit("load_ptr_tko", NewFunction(nil, []Type{vf32, Token}), []string{"%x", "%token1"}, []string{"%ptr0", "%mask", "%padding", "%token0"}, weak)
		load1 := emit("load_ptr_tko", NewFunction(nil, []Type{vf32, Token}), []string{"%y", "%token2"}, []string{"%ptr1", "%mask", "%padding", "%token1"}, weak)
		left, right := "%x", "%y"
		if out.Inputs[0].ID == inputs[1].ID {
			left, right = right, left
		}
		add := one("addf", "%sum", vf32, []string{left, right}, "{rounding_mode = nearest_even}")
		store := one("store_ptr_tko", "%done", Token, []string{"%ptr2", "%sum", "%mask", "%token2"}, weak)
		emit("return", nil, nil, nil, "")
		refs[0].ABIIndex = 0
		refs[0].Operations = []int{12, 13, 14, load0}
		refs[0].Values = []string{"%arg0", "%ptr0", "%x", "%token1"}
		refs[1].ABIIndex = 1
		refs[1].Operations = []int{15, 16, 17, load1}
		refs[1].Values = []string{"%arg1", "%ptr1", "%y", "%token2"}
		refs[2].Operations = []int{add}
		refs[2].Values = []string{"%sum"}
		refs[3].ABIIndex = 2
		refs[3].Operations = []int{18, 19, 20, store}
		refs[3].Values = []string{"%arg2", "%ptr2", "%done"}
	}
	data, err := r.Module.EncodeBytecode()
	if err != nil {
		return nil, err
	}
	r.Memory = &SemanticMemoryBinding{byteDigest: memoryDigest(data), references: refs}
	r.Memory.contractDigest = memoryContract(r, refs)
	return r, nil
}
