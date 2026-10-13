package tileir

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// MemorySource records the semantic origin of emitted SSA values and ABI roles.
// Locations live in this sidecar; this route does not emit debug bytecode.
type MemorySource struct {
	NodeID     int
	Role       string
	ABIIndex   int // -1 for arithmetic rather than a pointer parameter
	Operations []int
	Values     []string
	Location   cudair.SourceLocation
}

// SemanticMemoryBinding binds a singleton lowering's bytes, ABI, launch grid
// and source map. Its zero value is not usable. Accessors return copies.
type SemanticMemoryBinding struct {
	byteDigest     string
	contractDigest string
	references     []MemorySource
}

// References returns detached semantic/source/ABI records.
func (b *SemanticMemoryBinding) References() []MemorySource {
	if b == nil {
		return nil
	}
	refs := append([]MemorySource(nil), b.references...)
	for i := range refs {
		refs[i].Operations = append([]int(nil), refs[i].Operations...)
		refs[i].Values = append([]string(nil), refs[i].Values...)
		refs[i].Location.Callers = append([]cudair.Frame(nil), refs[i].Location.Callers...)
	}
	return refs
}

// BytecodeDigest returns the SHA-256 digest of the admitted typed bytecode.
func (b *SemanticMemoryBinding) BytecodeDigest() string {
	if b == nil {
		return ""
	}
	return b.byteDigest
}

func memoryDigest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func memoryContract(r *LoweringResult, refs []MemorySource) string {
	// All fields below have deterministic JSON representations and no
	// unsupported values. Keep the emitted-source attribution out of wire attrs.
	data, _ := json.Marshal(struct {
		Module     *Module
		Schema     cudakernel.Schema
		Grid       [3]uint32
		References []MemorySource
		Vector     *SemanticVectorLayout `json:",omitempty"`
	}{r.Module, r.Schema, r.Grid, refs, r.Vector})
	return memoryDigest(data)
}

// Verify checks that the lowering still matches its bound bytes, source map,
// pointer ABI and grid. It does not validate caller device buffers or aliases.
func (b *SemanticMemoryBinding) Verify(r *LoweringResult) error {
	if b == nil || r == nil || r.Memory != b || r.Module == nil || b.byteDigest == "" {
		return fmt.Errorf("tileir: missing semantic memory binding")
	}
	data, err := r.Module.EncodeBytecode()
	if err != nil {
		return err
	}
	if memoryDigest(data) != b.byteDigest || memoryContract(r, b.references) != b.contractDigest {
		return fmt.Errorf("tileir: semantic memory binding mismatch")
	}
	return nil
}

func singletonMemoryCandidate(snap *cudair.Snapshot) bool {
	if snap == nil {
		return false
	}
	nodes, inputs, outputs := snap.Nodes(), snap.Inputs(), snap.Outputs()
	if len(nodes) != 3 || len(inputs) != 2 || len(outputs) != 1 {
		return false
	}
	for _, n := range nodes {
		if n.Type != cudair.Float32 || len(n.Shape) != 1 || n.Shape[0] != 1 {
			return false
		}
		if n.Effect != cudair.EffectReadOnly || n.StorageAlias != nil || n.Layout != cudashape.LayoutRowMajor ||
			(len(n.Strides) != 0 && !cudashape.IsContiguousRowMajor(n.Shape, n.Strides)) {
			return false
		}
	}
	return outputs[0].Kind == cudair.KindBinary && outputs[0].Op == "add"
}

func singletonMemoryOptions(opts LowerOptions) bool {
	return (opts.TileM == 0 || opts.TileM == 1) && (opts.TileN == 0 || opts.TileN == 1) && (opts.TileK == 0 || opts.TileK == 1)
}

// LowerSemanticMemory lowers exactly two readonly contiguous FP32 inputs of
// shape [1] and their single add output to a void pointer-memory kernel.
// The caller must launch Grid(1,1,1), supply three distinct nonaliasing device
// buffers with at least one FP32 each, and prevent concurrent writes. This
// function allocates no device storage and emits no C++ helper source.
func LowerSemanticMemory(snap *cudair.Snapshot, opts LowerOptions) (*LoweringResult, error) {
	if !singletonMemoryCandidate(snap) {
		return nil, fmt.Errorf("%w: singleton readonly FP32 add required", ErrUnsupportedTileOp)
	}
	if !singletonMemoryOptions(opts) {
		return nil, fmt.Errorf("%w: singleton tile dimensions must be one", ErrUnsupportedTileOp)
	}
	nodes, inputs, outputs := snap.Nodes(), snap.Inputs(), snap.Outputs()
	out := outputs[0]
	if len(out.Inputs) != 2 || out.Inputs[0].ID == out.Inputs[1].ID || inputs[0].ID == inputs[1].ID {
		return nil, fmt.Errorf("%w: two distinct add inputs required", ErrUnsupportedTileOp)
	}
	roles := map[int]int{inputs[0].ID: 0, inputs[1].ID: 1}
	for _, n := range nodes {
		if n.Effect != cudair.EffectReadOnly || n.StorageAlias != nil || n.Layout != cudashape.LayoutRowMajor ||
			(len(n.Strides) != 0 && !cudashape.IsContiguousRowMajor(n.Shape, n.Strides)) {
			return nil, fmt.Errorf("%w: contiguous readonly nonaliasing storage required", ErrUnsupportedTileOp)
		}
		if n.ID != out.ID && n.Kind != cudair.KindInput {
			return nil, fmt.Errorf("%w: extra semantic operation", ErrUnsupportedTileOp)
		}
	}
	left, lok := roles[out.Inputs[0].ID]
	right, rok := roles[out.Inputs[1].ID]
	if !lok || !rok {
		return nil, fmt.Errorf("%w: add must consume both external inputs", ErrUnsupportedTileOp)
	}
	kernel := opts.KernelName
	if kernel == "" {
		kernel = "tile_singleton_add"
	}
	for i, c := range kernel {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || i > 0 && c >= '0' && c <= '9') {
			return nil, fmt.Errorf("tileir: invalid singleton kernel name")
		}
	}
	ptr, f32, i1 := NewTile(nil, NewPointer(Float32)), NewTile(nil, Float32), NewTile(nil, Int1)
	fn := &Function{Name: kernel, Entry: true}
	var params []cudakernel.Param
	names := make(map[string]bool)
	for i, node := range append(inputs, out) {
		name := node.Name
		if name == "" || names[name] {
			return nil, fmt.Errorf("tileir: pointer ABI roles must have unique nonempty names")
		}
		names[name] = true
		params = append(params, cudakernel.Param{Name: name, Type: "f32", IsPointer: true, ByteWidth: 8, Alignment: 8})
		fn.Params = append(fn.Params, &Param{Name: fmt.Sprintf("%%arg%d", i), TypeVal: ptr})
	}
	schema, err := cudakernel.NewSchema(params...)
	if err != nil {
		return nil, err
	}
	weak := "{memory_ordering_semantics = weak}"
	fn.Operations = []*Operation{
		{OpCode: "cuda_tile.constant", Results: []string{"%mask"}, TypeVal: i1, Attributes: "{value = dense<\"0x01\">}"},
		{OpCode: "cuda_tile.constant", Results: []string{"%padding"}, TypeVal: f32, Attributes: "{value = 0.0}"},
		{OpCode: "cuda_tile.make_token", Results: []string{"%start"}, TypeVal: Token},
		{OpCode: "cuda_tile.load_ptr_tko", Results: []string{"%x", "%loaded_x"}, TypeVal: NewFunction(nil, []Type{f32, Token}), Operands: []string{"%arg0", "%mask", "%padding", "%start"}, Attributes: weak},
		{OpCode: "cuda_tile.load_ptr_tko", Results: []string{"%y", "%loaded_y"}, TypeVal: NewFunction(nil, []Type{f32, Token}), Operands: []string{"%arg1", "%mask", "%padding", "%loaded_x"}, Attributes: weak},
		{OpCode: "cuda_tile.addf", Results: []string{"%sum"}, TypeVal: f32, Operands: []string{[]string{"%x", "%y"}[left], []string{"%x", "%y"}[right]}, Attributes: "{rounding_mode = nearest_even}"},
		{OpCode: "cuda_tile.store_ptr_tko", Results: []string{"%stored"}, TypeVal: Token, Operands: []string{"%arg2", "%sum", "%mask", "%loaded_y"}, Attributes: weak},
		{OpCode: "cuda_tile.return"},
	}
	r := &LoweringResult{Module: &Module{Name: kernel + "_mod", Functions: []*Function{fn}}, Schema: schema, Grid: [3]uint32{1, 1, 1}}
	refs := []MemorySource{
		{NodeID: inputs[0].ID, Role: "input", ABIIndex: 0, Operations: []int{3}, Values: []string{"%arg0", "%x", "%loaded_x"}, Location: inputs[0].SourceLoc},
		{NodeID: inputs[1].ID, Role: "input", ABIIndex: 1, Operations: []int{4}, Values: []string{"%arg1", "%y", "%loaded_y"}, Location: inputs[1].SourceLoc},
		{NodeID: out.ID, Role: "add", ABIIndex: -1, Operations: []int{5}, Values: []string{"%sum"}, Location: out.SourceLoc},
		{NodeID: out.ID, Role: "output", ABIIndex: 2, Operations: []int{6}, Values: []string{"%arg2", "%stored"}, Location: out.SourceLoc},
	}
	data, err := r.Module.EncodeBytecode()
	if err != nil {
		return nil, err
	}
	r.Memory = &SemanticMemoryBinding{byteDigest: memoryDigest(data), references: refs}
	r.Memory.contractDigest = memoryContract(r, refs)
	return r, nil
}
