package tileir

import (
	"fmt"
	"strings"
	"sync"
)

// Category identifies the functional category of a Tile IR operation.
type Category uint8

const (
	CatUnknown        Category = iota
	CatTileMemory              // TMA loads/stores, views, partition views
	CatComputeMMA              // Tensor Core matrix multiply-accumulate
	CatComputeElement          // Elementwise vector arithmetic and transcendental functions
	CatComputeInteger          // Packed integer arithmetic
	CatShapeLayout             // Reshape, permute, broadcast, convert_layout
	CatReduction               // Reductions and scans across tile dimensions
	CatSyncToken               // Token creation and join operations
	CatControlFlow             // Loops, conditionals, returns
	CatIndexing                // Grid/Block ID and tensor shape queries
	CatTypeCast                // Bitcast, floating-point conversions
)

// String returns the human-readable category name.
func (c Category) String() string {
	switch c {
	case CatTileMemory:
		return "Tile Memory"
	case CatComputeMMA:
		return "Tensor Core MMA"
	case CatComputeElement:
		return "Compute Elementwise"
	case CatComputeInteger:
		return "Compute Integer"
	case CatShapeLayout:
		return "Shape & Layout"
	case CatReduction:
		return "Reduction"
	case CatSyncToken:
		return "Sync & Tokens"
	case CatControlFlow:
		return "Control Flow"
	case CatIndexing:
		return "Indexing & Context"
	case CatTypeCast:
		return "Type Conversion"
	default:
		return "Unknown"
	}
}

// IsCompute reports whether the category represents compute instructions.
func (c Category) IsCompute() bool {
	return c == CatComputeMMA || c == CatComputeElement || c == CatComputeInteger || c == CatReduction
}

// IsMemory reports whether the category represents memory or view operations.
func (c Category) IsMemory() bool {
	return c == CatTileMemory
}

// OpDoc represents structured reference documentation and semantic constraints
// for a Tile IR operation.
type OpDoc struct {
	OpCode           string   `json:"opcode"`                      // e.g. "cuda_tile.load_view_tko"
	Mnemonic         string   `json:"mnemonic"`                    // e.g. "load_view_tko"
	Category         Category `json:"category"`                    // Functional classification
	Summary          string   `json:"summary"`                     // 1-2 sentence overview
	Syntax           []string `json:"syntax,omitempty"`            // MLIR textual prototype(s)
	Description      string   `json:"description,omitempty"`       // Architectural behavior prose
	MinArch          string   `json:"min_arch,omitempty"`          // e.g. "sm_90" (Hopper) or "sm_100" (Blackwell)
	TokenInputs      int      `json:"token_inputs,omitempty"`      // Number of consumed tokens
	TokenOutputs     int      `json:"token_outputs,omitempty"`     // Number of produced tokens
	OperandTypes     []string `json:"operand_types,omitempty"`     // Acceptable operand type constraints
	ResultTypes      []string `json:"result_types,omitempty"`      // Acceptable result type constraints
	HardwareLowering string   `json:"hardware_lowering,omitempty"` // Target PTX/SASS instruction (e.g. "cp.async.bulk.tensor", "wgmma")
	FallbackHazards  []string `json:"fallback_hazards,omitempty"`  // Fallback hazards if rejected by compiler
}

var rawOpDocs = []*OpDoc{
	// Tile Memory & TMA
	{
		Mnemonic:         "make_tensor_view",
		Category:         CatTileMemory,
		Summary:          "Creates an N-dimensional tensor view over global memory with shapes and strides.",
		Syntax:           []string{"%view = cuda_tile.make_tensor_view %ptr, %shape, %strides : (!cuda_tile.ptr<T>, ...) -> !cuda_tile.tensor_view<...>"},
		MinArch:          "sm_90",
		HardwareLowering: "TMA descriptor setup",
	},
	{
		Mnemonic:         "make_partition_view",
		Category:         CatTileMemory,
		Summary:          "Creates a partitioned view slicing a tensor view into TMA-loadable coordinate boxes.",
		Syntax:           []string{"%part = cuda_tile.make_partition_view %view, %tile_shape : (!cuda_tile.tensor_view<...>) -> !cuda_tile.partition_view"},
		MinArch:          "sm_90",
		HardwareLowering: "TMA coordinate box partition",
	},
	{
		Mnemonic:         "load_view_tko",
		Category:         CatTileMemory,
		Summary:          "Asynchronously loads a multidimensional tile from a tensor or partition view via TMA into registers or shared memory.",
		Syntax:           []string{"%tile, %out_tok = cuda_tile.load_view_tko %view, %in_tok : (...) -> (!cuda_tile.tile<...>, !cuda_tile.token)"},
		MinArch:          "sm_90",
		TokenInputs:      1,
		TokenOutputs:     1,
		HardwareLowering: "cp.async.bulk.tensor",
		FallbackHazards:  []string{"Lacks TMA bulk acceleration if address alignment is violated"},
	},
	{
		Mnemonic:         "store_view_tko",
		Category:         CatTileMemory,
		Summary:          "Asynchronously stores a tile into global memory via TMA guided by an input dependency token.",
		Syntax:           []string{"%out_tok = cuda_tile.store_view_tko %tile, %view, %in_tok : (...) -> !cuda_tile.token"},
		MinArch:          "sm_90",
		TokenInputs:      1,
		TokenOutputs:     1,
		HardwareLowering: "cp.async.bulk.tensor",
		FallbackHazards:  []string{"Lacks TMA bulk acceleration if address alignment is violated"},
	},
	{
		Mnemonic:         "load_ptr_tko",
		Category:         CatTileMemory,
		Summary:          "Reads data directly through a pointer token; fallback when tensor views cannot be formed.",
		Syntax:           []string{"%tile, %out_tok = cuda_tile.load_ptr_tko %ptr, %in_tok : (...) -> (!cuda_tile.tile<...>, !cuda_tile.token)"},
		TokenInputs:      1,
		TokenOutputs:     1,
		HardwareLowering: "ld.global",
		FallbackHazards:  []string{"Scalar memory load fallback; bypasses TMA hardware pipeline"},
	},
	{
		Mnemonic:         "atomic_cas_tko",
		Category:         CatTileMemory,
		Summary:          "Atomic compare-and-swap operation on global memory using token synchronization.",
		Syntax:           []string{"%old, %out_tok = cuda_tile.atomic_cas_tko %ptr, %cmp, %val, %in_tok : ..."},
		TokenInputs:      1,
		TokenOutputs:     1,
		HardwareLowering: "atom.cas",
	},
	{
		Mnemonic:         "atomic_rmw_tko",
		Category:         CatTileMemory,
		Summary:          "Atomic read-modify-write operation on global memory using token synchronization.",
		Syntax:           []string{"%old, %out_tok = cuda_tile.atomic_rmw_tko %op, %ptr, %val, %in_tok : ..."},
		TokenInputs:      1,
		TokenOutputs:     1,
		HardwareLowering: "atom",
	},
	{
		Mnemonic: "offset",
		Category: CatTileMemory,
		Summary:  "Computes coordinate or memory offset within a tensor view.",
		Syntax:   []string{"%off = cuda_tile.offset %view, %coords : ..."},
	},

	// Tensor Core MMA
	{
		Mnemonic:         "mma_f",
		Category:         CatComputeMMA,
		Summary:          "Fused matrix multiply-accumulate on floating-point tiles via Tensor Cores.",
		Syntax:           []string{"%acc = cuda_tile.mma_f %A, %B, %C : (!cuda_tile.tile<MxKxf16>, !cuda_tile.tile<KxNxf16>, !cuda_tile.tile<MxNxf32>) -> !cuda_tile.tile<MxNxf32>"},
		MinArch:          "sm_90",
		HardwareLowering: "wgmma.mma_async / mma.sync",
		FallbackHazards:  []string{"May fall back to scalar fma if tile dimensions are unaligned or unsupported"},
	},
	{
		Mnemonic:         "mma_i",
		Category:         CatComputeMMA,
		Summary:          "Fused matrix multiply-accumulate on integer tiles via Tensor Cores.",
		Syntax:           []string{"%acc = cuda_tile.mma_i %A, %B, %C : (!cuda_tile.tile<MxKxs8>, !cuda_tile.tile<KxNxs8>, !cuda_tile.tile<MxNxs32>) -> !cuda_tile.tile<MxNxs32>"},
		MinArch:          "sm_90",
		HardwareLowering: "wgmma.mma_async / mma.sync",
	},
	{
		Mnemonic:         "mmaf_scaled",
		Category:         CatComputeMMA,
		Summary:          "Block-scaled matrix multiply-accumulate with exponent scale factors (NVFP4, MXFP8) on Blackwell.",
		Syntax:           []string{"%acc = cuda_tile.mmaf_scaled %A, %scaleA, %B, %scaleB, %C : ..."},
		MinArch:          "sm_100",
		HardwareLowering: "tcgen05.mma",
	},

	// Shape & Layout
	{
		Mnemonic: "reshape",
		Category: CatShapeLayout,
		Summary:  "Reinterprets the dimensional shape of a tile without altering the underlying data or element count.",
		Syntax:   []string{"%res = cuda_tile.reshape %tile : (!cuda_tile.tile<...>) -> !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "permute",
		Category: CatShapeLayout,
		Summary:  "Permutes the dimension axes of a multidimensional tile.",
		Syntax:   []string{"%res = cuda_tile.permute %tile {dims = [...]} : ..."},
	},
	{
		Mnemonic: "broadcast",
		Category: CatShapeLayout,
		Summary:  "Broadcasts singleton dimensions of a tile across larger dimensions.",
		Syntax:   []string{"%res = cuda_tile.broadcast %tile : ..."},
	},
	{
		Mnemonic: "cat",
		Category: CatShapeLayout,
		Summary:  "Concatenates two or more tiles along a specified dimension.",
		Syntax:   []string{"%res = cuda_tile.cat %t1, %t2 {dim = 0} : ..."},
	},
	{
		Mnemonic: "extract",
		Category: CatShapeLayout,
		Summary:  "Slices and extracts a sub-tile from a larger tile at specified coordinate offsets.",
		Syntax:   []string{"%res = cuda_tile.extract %tile {offsets = [...], sizes = [...]} : ..."},
	},
	{
		Mnemonic:         "convert_layout",
		Category:         CatShapeLayout,
		Summary:          "Transforms tile physical layout between row-major, column-major, and hardware swizzled layouts.",
		Syntax:           []string{"%res = cuda_tile.convert_layout %tile : ..."},
		HardwareLowering: "stmatrix / ldmatrix",
	},

	// Vector Floating-Point Arithmetic
	{
		Mnemonic:        "add_f",
		Category:        CatComputeElement,
		Summary:         "Elementwise floating-point addition across tile elements.",
		Syntax:          []string{"%res = cuda_tile.add_f %a, %b : !cuda_tile.tile<...>"},
		FallbackHazards: []string{"Unfused arithmetic fallback; prefer mma_f when computing matrix multiplications"},
	},
	{
		Mnemonic: "sub_f",
		Category: CatComputeElement,
		Summary:  "Elementwise floating-point subtraction across tile elements.",
		Syntax:   []string{"%res = cuda_tile.sub_f %a, %b : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic:        "mul_f",
		Category:        CatComputeElement,
		Summary:         "Elementwise floating-point multiplication across tile elements.",
		Syntax:          []string{"%res = cuda_tile.mul_f %a, %b : !cuda_tile.tile<...>"},
		FallbackHazards: []string{"Unfused arithmetic fallback; prefer mma_f when computing matrix multiplications"},
	},
	{
		Mnemonic: "div_f",
		Category: CatComputeElement,
		Summary:  "Elementwise floating-point division across tile elements.",
		Syntax:   []string{"%res = cuda_tile.div_f %a, %b : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic:         "fma",
		Category:         CatComputeElement,
		Summary:          "Fused multiply-add operation (a * b + c) across tile elements.",
		Syntax:           []string{"%res = cuda_tile.fma %a, %b, %c : !cuda_tile.tile<...>"},
		HardwareLowering: "fma",
	},
	{
		Mnemonic: "neg_f",
		Category: CatComputeElement,
		Summary:  "Elementwise floating-point negation.",
		Syntax:   []string{"%res = cuda_tile.neg_f %a : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "rem_f",
		Category: CatComputeElement,
		Summary:  "Elementwise floating-point remainder.",
		Syntax:   []string{"%res = cuda_tile.rem_f %a, %b : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "fpow_f",
		Category: CatComputeElement,
		Summary:  "Elementwise floating-point power function.",
		Syntax:   []string{"%res = cuda_tile.fpow_f %a, %b : !cuda_tile.tile<...>"},
	},

	// Vector Integer Arithmetic
	{
		Mnemonic:        "add_i",
		Category:        CatComputeInteger,
		Summary:         "Elementwise integer addition across tile elements.",
		Syntax:          []string{"%res = cuda_tile.add_i %a, %b : !cuda_tile.tile<...>"},
		FallbackHazards: []string{"Unfused arithmetic fallback; prefer mma_i when computing matrix multiplications"},
	},
	{
		Mnemonic: "sub_i",
		Category: CatComputeInteger,
		Summary:  "Elementwise integer subtraction across tile elements.",
		Syntax:   []string{"%res = cuda_tile.sub_i %a, %b : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic:        "mul_i",
		Category:        CatComputeInteger,
		Summary:         "Elementwise integer multiplication across tile elements.",
		Syntax:          []string{"%res = cuda_tile.mul_i %a, %b : !cuda_tile.tile<...>"},
		FallbackHazards: []string{"Unfused arithmetic fallback; prefer mma_i when computing matrix multiplications"},
	},
	{
		Mnemonic: "mulhi_i",
		Category: CatComputeInteger,
		Summary:  "Elementwise high-half integer multiplication.",
		Syntax:   []string{"%res = cuda_tile.mulhi_i %a, %b : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "div_i",
		Category: CatComputeInteger,
		Summary:  "Elementwise integer division across tile elements.",
		Syntax:   []string{"%res = cuda_tile.div_i %a, %b : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "neg_i",
		Category: CatComputeInteger,
		Summary:  "Elementwise integer negation.",
		Syntax:   []string{"%res = cuda_tile.neg_i %a : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "rem_i",
		Category: CatComputeInteger,
		Summary:  "Elementwise integer remainder or modulo.",
		Syntax:   []string{"%res = cuda_tile.rem_i %a, %b : !cuda_tile.tile<...>"},
	},

	// Math Intrinsics
	{
		Mnemonic: "exp",
		Category: CatComputeElement,
		Summary:  "Computes base-e exponential elementwise.",
		Syntax:   []string{"%res = cuda_tile.exp %a : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "exp2",
		Category: CatComputeElement,
		Summary:  "Computes base-2 exponential elementwise.",
		Syntax:   []string{"%res = cuda_tile.exp2 %a : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "log",
		Category: CatComputeElement,
		Summary:  "Computes natural logarithm elementwise.",
		Syntax:   []string{"%res = cuda_tile.log %a : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "log2",
		Category: CatComputeElement,
		Summary:  "Computes base-2 logarithm elementwise.",
		Syntax:   []string{"%res = cuda_tile.log2 %a : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "cos",
		Category: CatComputeElement,
		Summary:  "Computes cosine elementwise.",
		Syntax:   []string{"%res = cuda_tile.cos %a : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "cosh",
		Category: CatComputeElement,
		Summary:  "Computes hyperbolic cosine elementwise.",
		Syntax:   []string{"%res = cuda_tile.cosh %a : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "ceil",
		Category: CatComputeElement,
		Summary:  "Computes ceiling of floating-point elements.",
		Syntax:   []string{"%res = cuda_tile.ceil %a : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "floor",
		Category: CatComputeElement,
		Summary:  "Computes floor of floating-point elements.",
		Syntax:   []string{"%res = cuda_tile.floor %a : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "rsqrt",
		Category: CatComputeElement,
		Summary:  "Computes reciprocal square root (1/sqrt(x)) elementwise.",
		Syntax:   []string{"%res = cuda_tile.rsqrt %a : !cuda_tile.tile<...>"},
	},

	// Synchronization
	{
		Mnemonic:     "make_token",
		Category:     CatSyncToken,
		Summary:      "Creates a new asynchronous dependency token to order memory operations.",
		Syntax:       []string{"%tok = cuda_tile.make_token : !cuda_tile.token"},
		TokenOutputs: 1,
	},
	{
		Mnemonic:     "join_tokens",
		Category:     CatSyncToken,
		Summary:      "Joins multiple asynchronous dependency tokens into a single merged token (barrier fan-in).",
		Syntax:       []string{"%tok = cuda_tile.join_tokens %t0, %t1 : (!cuda_tile.token, !cuda_tile.token) -> !cuda_tile.token"},
		TokenInputs:  2,
		TokenOutputs: 1,
	},

	// Control & Context
	{
		Mnemonic: "module",
		Category: CatControlFlow,
		Summary:  "Top-level translation unit container in Tile IR.",
		Syntax:   []string{"cuda_tile.module @name { ... }"},
	},
	{
		Mnemonic: "entry",
		Category: CatControlFlow,
		Summary:  "Declares a kernel entry point callable from host device code.",
		Syntax:   []string{"cuda_tile.entry @kernel(%args...) { ... }"},
	},
	{
		Mnemonic: "func",
		Category: CatControlFlow,
		Summary:  "Declares an internal device function or subprogram.",
		Syntax:   []string{"cuda_tile.func @device_func(%args...) { ... }"},
	},
	{
		Mnemonic: "return",
		Category: CatControlFlow,
		Summary:  "Returns from a function or kernel execution.",
		Syntax:   []string{"cuda_tile.return"},
	},
	{
		Mnemonic: "for",
		Category: CatControlFlow,
		Summary:  "Structured counted loop executing over an index range.",
		Syntax:   []string{"cuda_tile.for %i = %lb to %ub step %step { ... }"},
	},
	{
		Mnemonic: "loop",
		Category: CatControlFlow,
		Summary:  "General iterative loop structure with loop-carried SSA values.",
		Syntax:   []string{"cuda_tile.loop { ... }"},
	},
	{
		Mnemonic: "if",
		Category: CatControlFlow,
		Summary:  "Conditional execution branch based on boolean condition.",
		Syntax:   []string{"cuda_tile.if %cond { ... }"},
	},
	{
		Mnemonic: "break",
		Category: CatControlFlow,
		Summary:  "Breaks out of the innermost enclosing loop.",
		Syntax:   []string{"cuda_tile.break"},
	},
	{
		Mnemonic: "continue",
		Category: CatControlFlow,
		Summary:  "Continues to the next iteration of the innermost loop.",
		Syntax:   []string{"cuda_tile.continue"},
	},
	{
		Mnemonic: "get_tile_block_id",
		Category: CatIndexing,
		Summary:  "Retrieves the multidimensional tile block coordinate within the launch grid.",
		Syntax:   []string{"%bid = cuda_tile.get_tile_block_id %dim : i32"},
	},
	{
		Mnemonic: "get_num_tile_blocks",
		Category: CatIndexing,
		Summary:  "Retrieves the total number of tile blocks in the launch grid along a dimension.",
		Syntax:   []string{"%n = cuda_tile.get_num_tile_blocks %dim : i32"},
	},
	{
		Mnemonic: "get_tensor_shape",
		Category: CatIndexing,
		Summary:  "Queries runtime extents of a dynamic tensor view.",
		Syntax:   []string{"%extent = cuda_tile.get_tensor_shape %view, %dim : i32"},
	},

	// Conversions & Bits
	{
		Mnemonic: "bitcast",
		Category: CatTypeCast,
		Summary:  "Reinterprets bit pattern from one type to another of identical bit width.",
		Syntax:   []string{"%res = cuda_tile.bitcast %a : !cuda_tile.tile<...> -> !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "ftof",
		Category: CatTypeCast,
		Summary:  "Converts floating-point precision between types (e.g. f16 to f32).",
		Syntax:   []string{"%res = cuda_tile.ftof %a : !cuda_tile.tile<...> -> !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "ftoi",
		Category: CatTypeCast,
		Summary:  "Converts floating-point numbers to signed or unsigned integers.",
		Syntax:   []string{"%res = cuda_tile.ftoi %a : !cuda_tile.tile<...> -> !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "itof",
		Category: CatTypeCast,
		Summary:  "Converts integers to floating-point numbers.",
		Syntax:   []string{"%res = cuda_tile.itof %a : !cuda_tile.tile<...> -> !cuda_tile.tile<...>"},
	},
	{
		Mnemonic:        "int_to_ptr",
		Category:        CatTypeCast,
		Summary:         "Casts integer address to a typed pointer.",
		Syntax:          []string{"%ptr = cuda_tile.int_to_ptr %val : i64 -> !cuda_tile.ptr<...>"},
		FallbackHazards: []string{"Scalar pointer arithmetic fallback; bypasses structured views"},
	},
	{
		Mnemonic:        "ptr_to_int",
		Category:        CatTypeCast,
		Summary:         "Casts typed pointer to a 64-bit integer address.",
		Syntax:          []string{"%val = cuda_tile.ptr_to_int %ptr : !cuda_tile.ptr<...> -> i64"},
		FallbackHazards: []string{"Scalar pointer arithmetic fallback; bypasses structured views"},
	},
	{
		Mnemonic: "ptr_to_ptr",
		Category: CatTypeCast,
		Summary:  "Casts pointer between different pointee types or memory spaces.",
		Syntax:   []string{"%p2 = cuda_tile.ptr_to_ptr %p1 : !cuda_tile.ptr<...> -> !cuda_tile.ptr<...>"},
	},
	{
		Mnemonic: "ext_i",
		Category: CatTypeCast,
		Summary:  "Sign-extends or zero-extends integer bit width.",
		Syntax:   []string{"%res = cuda_tile.ext_i %a : !cuda_tile.tile<...> -> !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "iota",
		Category: CatIndexing,
		Summary:  "Generates an incrementing sequence of indices across a tile dimension.",
		Syntax:   []string{"%tile = cuda_tile.iota : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "reduce",
		Category: CatReduction,
		Summary:  "Performs parallel associative reduction (sum, min, max) across tile axes.",
		Syntax:   []string{"%res = cuda_tile.reduce %tile {op = \"add\", axis = 1} : ..."},
	},
	{
		Mnemonic: "scan",
		Category: CatReduction,
		Summary:  "Performs parallel prefix scan across tile axes.",
		Syntax:   []string{"%res = cuda_tile.scan %tile {op = \"add\", axis = 1} : ..."},
	},
	{
		Mnemonic: "select",
		Category: CatComputeElement,
		Summary:  "Elementwise ternary selection based on boolean condition tile.",
		Syntax:   []string{"%res = cuda_tile.select %cond, %t, %f : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "cmp_f",
		Category: CatComputeElement,
		Summary:  "Elementwise floating-point comparison yielding boolean tile.",
		Syntax:   []string{"%cond = cuda_tile.cmp_f %pred, %a, %b : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "cmp_i",
		Category: CatComputeInteger,
		Summary:  "Elementwise integer comparison yielding boolean tile.",
		Syntax:   []string{"%cond = cuda_tile.cmp_i %pred, %a, %b : !cuda_tile.tile<...>"},
	},
	{
		Mnemonic: "constant",
		Category: CatComputeElement,
		Summary:  "Generates a tile populated with a constant scalar value.",
		Syntax:   []string{"%tile = cuda_tile.constant <val> : !cuda_tile.tile<...>"},
	},
}

var docDatabase = sync.OnceValue(func() map[string]*OpDoc {
	db := make(map[string]*OpDoc, len(rawOpDocs)*2)
	for _, d := range rawOpDocs {
		if d.OpCode == "" {
			d.OpCode = "cuda_tile." + d.Mnemonic
		}
		db[strings.ToLower(d.Mnemonic)] = d
		db[strings.ToLower(d.OpCode)] = d
	}
	return db
})

// LookupDoc returns documentation for a Tile IR operation.
// Lookup is case-insensitive and the "cuda_tile." prefix is optional.
// Example: LookupDoc("cuda_tile.mma_f"), LookupDoc("mma_f"), and LookupDoc("load_view_tko") all resolve.
func LookupDoc(opcode string) *OpDoc {
	opcode = strings.TrimSpace(opcode)
	if opcode == "" {
		return nil
	}
	db := docDatabase()
	lower := strings.ToLower(opcode)
	if d, ok := db[lower]; ok {
		return d
	}
	if strings.HasPrefix(lower, "cuda_tile.") {
		if d, ok := db[strings.TrimPrefix(lower, "cuda_tile.")]; ok {
			return d
		}
	} else {
		if d, ok := db["cuda_tile."+lower]; ok {
			return d
		}
	}
	return nil
}

// OpcodeCategory returns the functional category of the operation opcode.
func OpcodeCategory(opcode string) Category {
	if d := LookupDoc(opcode); d != nil {
		return d.Category
	}
	return CatUnknown
}

// Explain returns a formatted multi-line summary of an operation suitable for CLI diagnostics.
func Explain(opcode string) string {
	doc := LookupDoc(opcode)
	if doc == nil {
		return fmt.Sprintf("tileir: no documentation available for opcode %q", opcode)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s (%s)\n", doc.OpCode, doc.Category)
	if doc.Summary != "" {
		fmt.Fprintf(&b, "Summary: %s\n", doc.Summary)
	}
	if doc.HardwareLowering != "" {
		arch := ""
		if doc.MinArch != "" {
			arch = fmt.Sprintf(" [%s+]", doc.MinArch)
		}
		fmt.Fprintf(&b, "Hardware Lowering: %s%s\n", doc.HardwareLowering, arch)
	}
	if len(doc.FallbackHazards) > 0 {
		fmt.Fprintf(&b, "Fallback Hazards:\n")
		for _, h := range doc.FallbackHazards {
			fmt.Fprintf(&b, "  - %s\n", h)
		}
	}
	if len(doc.Syntax) > 0 {
		fmt.Fprintf(&b, "Syntax:\n")
		for _, syn := range doc.Syntax {
			fmt.Fprintf(&b, "  %s\n", syn)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// HoverInfo returns compact markdown documentation suitable for IDE LSP hover tooltips.
func HoverInfo(opcode string) string {
	doc := LookupDoc(opcode)
	if doc == nil {
		return fmt.Sprintf("### `%s`\n\n*Unknown Tile IR opcode*", opcode)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "### `%s` *(%s)*\n\n", doc.OpCode, doc.Category)
	if doc.Summary != "" {
		fmt.Fprintf(&b, "**%s**\n\n", doc.Summary)
	}
	if doc.MinArch != "" {
		fmt.Fprintf(&b, "- **Minimum Architecture:** `%s`\n", doc.MinArch)
	}
	if doc.HardwareLowering != "" {
		fmt.Fprintf(&b, "- **Hardware Lowering:** `%s`\n", doc.HardwareLowering)
	}
	if doc.TokenInputs > 0 || doc.TokenOutputs > 0 {
		fmt.Fprintf(&b, "- **Tokens:** in: %d, out: %d\n", doc.TokenInputs, doc.TokenOutputs)
	}
	if len(doc.FallbackHazards) > 0 {
		fmt.Fprintf(&b, "- **Fallback Hazards:**\n")
		for _, h := range doc.FallbackHazards {
			fmt.Fprintf(&b, "  - %s\n", h)
		}
	}
	if len(doc.Syntax) > 0 {
		fmt.Fprintf(&b, "\n```mlir\n%s\n```\n", strings.Join(doc.Syntax, "\n"))
	}
	return strings.TrimRight(b.String(), "\n")
}
