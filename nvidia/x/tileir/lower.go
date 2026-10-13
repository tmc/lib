package tileir

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var (
	// ErrUnsupportedTileOp is returned when a semantic region contains an operation
	// that cannot be lowered to Tile IR.
	ErrUnsupportedTileOp = errors.New("tileir: unsupported operation for tile lowering")

	// ErrInvalidHeader is returned when Tile C++ source violates the pinned contract.
	ErrInvalidHeader = errors.New("tileir: invalid tile c++ header: requires #include \"cuda_tile.h\"")

	// ErrInvalidTileGlobal is returned when Tile C++ source lacks __tile_global__.
	ErrInvalidTileGlobal = errors.New("tileir: tile c++ kernel must be marked __tile_global__")

	// ErrInvalidCompileFlags is returned when required NVRTC flags are missing or illegal.
	ErrInvalidCompileFlags = errors.New("tileir: invalid tile compilation flags")
)

// LowerOptions configures the translation from a validated cudair.Snapshot to tileir.Module.
type LowerOptions struct {
	KernelName string
	TileM      int64
	TileN      int64
	TileK      int64
}

// LoweringResult bundles the translated Module AST, ABI schema, and source attribution.
type LoweringResult struct {
	Module *Module
	Schema cudakernel.Schema
	Source string
	// Grid is the standalone launch grid for an ordered reduction region.
	// Other lowerings leave it zero because their launch policy is external.
	Grid [3]uint32
	// Memory binds a singleton pointer-memory module to its source/ABI map.
	// It is nil for legacy analysis and C++ lowerings.
	Memory *SemanticMemoryBinding
}

// LowerSemanticRegion lowers a validated semantic region (cudair.Snapshot) to a typed Tile IR Module.
//
// It validates operations, maps parameter ABIs, retains semantic node IDs in operation attributes,
// attaches Go source coordinates to Module.Locations, and validates token dependencies and shapes.
func LowerSemanticRegion(snap *cudair.Snapshot, opts LowerOptions) (*LoweringResult, error) {
	if snap == nil {
		return nil, errors.New("tileir: nil snapshot")
	}
	if singletonMemoryCandidate(snap) && singletonMemoryOptions(opts) {
		return LowerSemanticMemory(snap, opts)
	}
	for _, n := range snap.Nodes() {
		if n.Kind == cudair.KindReduce {
			return lowerReductionRegion(snap, opts)
		}
	}

	kernelName := opts.KernelName
	if kernelName == "" {
		kernelName = "tile_kernel"
	}
	tileM := opts.TileM
	if tileM <= 0 {
		tileM = 16
	}
	tileN := opts.TileN
	if tileN <= 0 {
		tileN = 16
	}
	tileK := opts.TileK
	if tileK <= 0 {
		tileK = 16
	}

	mod := &Module{
		Name:      kernelName + "_mod",
		Producer:  "lib/nvidia/x/tileir",
		Locations: make(map[string]Location),
	}

	fn := &Function{
		Name:  kernelName,
		Entry: true,
	}

	var schemaParams []cudakernel.Param
	paramTypes := make(map[string]Type)

	// Build parameters from external inputs and outputs
	inputs := snap.Inputs()
	for i, inp := range inputs {
		pName := fmt.Sprintf("%%arg%d", i)
		ptxType := ptxTypeFromDataType(inp.Type)

		schemaParams = append(schemaParams, cudakernel.Param{
			Name:      inp.Name,
			Type:      ptxType,
			ByteWidth: 8, // Device pointer
			Alignment: 16,
			IsPointer: true,
		})

		elemType := primitiveTypeFromDataType(inp.Type)
		ptrType := NewPointer(elemType)
		fn.Params = append(fn.Params, &Param{
			Name:    pName,
			Type:    ptrType.String(),
			TypeVal: ptrType,
		})
		paramTypes[inp.Name] = ptrType
	}

	outputs := snap.Outputs()
	for j, out := range outputs {
		pName := fmt.Sprintf("%%arg%d", len(inputs)+j)
		ptxType := ptxTypeFromDataType(out.Type)

		outParamName := out.Name
		if outParamName == "" {
			outParamName = fmt.Sprintf("out%d", j)
		}
		schemaParams = append(schemaParams, cudakernel.Param{
			Name:      outParamName,
			Type:      ptxType,
			ByteWidth: 8,
			Alignment: 16,
			IsPointer: true,
		})

		elemType := primitiveTypeFromDataType(out.Type)
		ptrType := NewPointer(elemType)
		fn.Params = append(fn.Params, &Param{
			Name:    pName,
			Type:    ptrType.String(),
			TypeVal: ptrType,
		})
		paramTypes[outParamName] = ptrType
	}

	schema, err := cudakernel.NewSchema(schemaParams...)
	if err != nil {
		return nil, fmt.Errorf("tileir: failed to construct ABI schema: %w", err)
	}

	// Validate admitted operations
	// Allowed: KindInput, KindConstant, KindBinary (add, mul, sub, div), KindUnary (relu, neg), KindMatMul
	for _, node := range snap.Nodes() {
		if node.Kind == cudair.KindInput {
			continue
		}
		switch node.Kind {
		case cudair.KindConstant:
			// Admitted
		case cudair.KindBinary:
			switch strings.ToLower(node.Op) {
			case "add", "mul", "sub", "div":
				// Admitted
			default:
				return nil, fmt.Errorf("%w: binary op %q (node %d)", ErrUnsupportedTileOp, node.Op, node.ID)
			}
		case cudair.KindUnary:
			switch strings.ToLower(node.Op) {
			case "relu", "neg":
				// Admitted
			default:
				return nil, fmt.Errorf("%w: unary op %q (node %d)", ErrUnsupportedTileOp, node.Op, node.ID)
			}
		case cudair.KindMatMul:
			// Admitted
		default:
			return nil, fmt.Errorf("%w: node %s (kind %v, op %q)", ErrUnsupportedTileOp, node.Name, node.Kind, node.Op)
		}
	}

	// Emit Tile IR operations
	// 1. Initial token: %tok0 = cuda_tile.make_token : !cuda_tile.token
	tokCounter := 0
	initialTok := fmt.Sprintf("%%tok%d", tokCounter)
	fn.Operations = append(fn.Operations, &Operation{
		Line:     len(fn.Operations) + 1,
		Results:  []string{initialTok},
		OpCode:   "cuda_tile.make_token",
		Type:     "!cuda_tile.token",
		TypeVal:  TokenType{},
		Operands: nil,
	})
	currentTok := initialTok

	// Map each cudair node to its resulting Tile IR SSA value
	nodeSSA := make(map[int]string)
	nodeTypes := make(map[int]Type)
	locCounter := 0

	// Helper to record source location
	recordLoc := func(n *cudair.Node) string {
		if n == nil || n.SourceLoc.File == "" {
			return ""
		}
		locCounter++
		locKey := fmt.Sprintf("loc%d", locCounter)
		mod.Locations[locKey] = Location{
			File:   n.SourceLoc.File,
			Line:   n.SourceLoc.Line,
			Column: 1,
			Scope:  n.SourceLoc.Function,
		}
		return "#" + locKey
	}

	// 2. Load input arguments via load_view_tko
	for i, inp := range inputs {
		argSSA := fmt.Sprintf("%%arg%d", i)
		valSSA := fmt.Sprintf("%%v%d", inp.ID)
		tokCounter++
		nextTok := fmt.Sprintf("%%tok%d", tokCounter)

		elemType := primitiveTypeFromDataType(inp.Type)
		var shape []int64
		if len(inp.Shape) >= 2 {
			shape = []int64{tileM, tileN}
		} else if len(inp.Shape) == 1 {
			shape = []int64{tileM}
		} else {
			shape = []int64{1}
		}
		tileType := NewTile(shape, elemType)

		locRef := recordLoc(inp)
		fn.Operations = append(fn.Operations, &Operation{
			Line:       len(fn.Operations) + 1,
			Results:    []string{valSSA, nextTok},
			OpCode:     "cuda_tile.load_view_tko",
			Operands:   []string{argSSA, currentTok},
			Type:       fmt.Sprintf("(%s, !cuda_tile.token)", tileType.String()),
			TypeVal:    tileType,
			Attributes: fmt.Sprintf(`{semantic_id = %d, name = %q}`, inp.ID, inp.Name),
			LocRef:     locRef,
			Loc:        mod.Locations[strings.TrimPrefix(locRef, "#")],
		})
		currentTok = nextTok
		nodeSSA[inp.ID] = valSSA
		nodeTypes[inp.ID] = tileType
	}

	// 3. Lower interior nodes in topological order
	for _, node := range snap.Nodes() {
		if node.Kind == cudair.KindInput {
			continue
		}

		locRef := recordLoc(node)
		nodeLoc := mod.Locations[strings.TrimPrefix(locRef, "#")]

		switch node.Kind {
		case cudair.KindConstant:
			resSSA := fmt.Sprintf("%%cst_%d", node.ID)
			elemType := primitiveTypeFromDataType(node.Type)
			tileType := NewTile([]int64{tileM, tileN}, elemType)

			fn.Operations = append(fn.Operations, &Operation{
				Line:       len(fn.Operations) + 1,
				Results:    []string{resSSA},
				OpCode:     "cuda_tile.constant",
				Type:       tileType.String(),
				TypeVal:    tileType,
				Attributes: fmt.Sprintf(`{semantic_id = %d, name = %q}`, node.ID, node.Name),
				LocRef:     locRef,
				Loc:        nodeLoc,
			})
			nodeSSA[node.ID] = resSSA
			nodeTypes[node.ID] = tileType

		case cudair.KindBinary:
			if len(node.Inputs) < 2 {
				return nil, fmt.Errorf("tileir: binary op %d has < 2 inputs", node.ID)
			}
			lhsSSA := nodeSSA[node.Inputs[0].ID]
			rhsSSA := nodeSSA[node.Inputs[1].ID]
			resSSA := fmt.Sprintf("%%b%d", node.ID)

			opName := "cuda_tile.addf"
			switch strings.ToLower(node.Op) {
			case "add":
				opName = "cuda_tile.addf"
			case "mul":
				opName = "cuda_tile.mulf"
			case "sub":
				opName = "cuda_tile.subf"
			case "div":
				opName = "cuda_tile.divf"
			}

			tileType := nodeTypes[node.Inputs[0].ID]
			if tileType == nil {
				tileType = NewTile([]int64{tileM, tileN}, primitiveTypeFromDataType(node.Type))
			}

			fn.Operations = append(fn.Operations, &Operation{
				Line:       len(fn.Operations) + 1,
				Results:    []string{resSSA},
				OpCode:     opName,
				Operands:   []string{lhsSSA, rhsSSA},
				Type:       tileType.String(),
				TypeVal:    tileType,
				Attributes: fmt.Sprintf(`{semantic_id = %d, name = %q}`, node.ID, node.Name),
				LocRef:     locRef,
				Loc:        nodeLoc,
			})
			nodeSSA[node.ID] = resSSA
			nodeTypes[node.ID] = tileType

		case cudair.KindUnary:
			if len(node.Inputs) < 1 {
				return nil, fmt.Errorf("tileir: unary op %d has no input", node.ID)
			}
			inSSA := nodeSSA[node.Inputs[0].ID]
			resSSA := fmt.Sprintf("%%u%d", node.ID)

			tileType := nodeTypes[node.Inputs[0].ID]
			if tileType == nil {
				tileType = NewTile([]int64{tileM, tileN}, primitiveTypeFromDataType(node.Type))
			}

			switch strings.ToLower(node.Op) {
			case "relu":
				// relu(x) -> max_f(x, 0)
				cstSSA := fmt.Sprintf("%%cst_zero_%d", node.ID)
				fn.Operations = append(fn.Operations, &Operation{
					Line:       len(fn.Operations) + 1,
					Results:    []string{cstSSA},
					OpCode:     "cuda_tile.constant",
					Type:       tileType.String(),
					TypeVal:    tileType,
					Attributes: `{value = 0.0}`,
					LocRef:     locRef,
					Loc:        nodeLoc,
				})
				fn.Operations = append(fn.Operations, &Operation{
					Line:       len(fn.Operations) + 1,
					Results:    []string{resSSA},
					OpCode:     "cuda_tile.maxf",
					Operands:   []string{inSSA, cstSSA},
					Type:       tileType.String(),
					TypeVal:    tileType,
					Attributes: fmt.Sprintf(`{semantic_id = %d, name = %q}`, node.ID, node.Name),
					LocRef:     locRef,
					Loc:        nodeLoc,
				})
			case "neg":
				fn.Operations = append(fn.Operations, &Operation{
					Line:       len(fn.Operations) + 1,
					Results:    []string{resSSA},
					OpCode:     "cuda_tile.negf",
					Operands:   []string{inSSA},
					Type:       tileType.String(),
					TypeVal:    tileType,
					Attributes: fmt.Sprintf(`{semantic_id = %d, name = %q}`, node.ID, node.Name),
					LocRef:     locRef,
					Loc:        nodeLoc,
				})
			}
			nodeSSA[node.ID] = resSSA
			nodeTypes[node.ID] = tileType

		case cudair.KindMatMul:
			if len(node.Inputs) < 2 {
				return nil, fmt.Errorf("tileir: matmul op %d has < 2 inputs", node.ID)
			}
			aSSA := nodeSSA[node.Inputs[0].ID]
			bSSA := nodeSSA[node.Inputs[1].ID]
			resSSA := fmt.Sprintf("%%mma%d", node.ID)

			accType := NewTile([]int64{tileM, tileN}, Float32)
			cstSSA := fmt.Sprintf("%%cst_mma_%d", node.ID)

			fn.Operations = append(fn.Operations, &Operation{
				Line:       len(fn.Operations) + 1,
				Results:    []string{cstSSA},
				OpCode:     "cuda_tile.constant",
				Type:       accType.String(),
				TypeVal:    accType,
				Attributes: `{value = 0.0}`,
				LocRef:     locRef,
				Loc:        nodeLoc,
			})

			fn.Operations = append(fn.Operations, &Operation{
				Line:       len(fn.Operations) + 1,
				Results:    []string{resSSA},
				OpCode:     "cuda_tile.mmaf",
				Operands:   []string{aSSA, bSSA, cstSSA},
				Type:       accType.String(),
				TypeVal:    accType,
				Attributes: fmt.Sprintf(`{semantic_id = %d, name = %q}`, node.ID, node.Name),
				LocRef:     locRef,
				Loc:        nodeLoc,
			})
			nodeSSA[node.ID] = resSSA
			nodeTypes[node.ID] = accType
		}
	}

	// 4. Store outputs via store_view_tko
	for j, out := range outputs {
		outArgSSA := fmt.Sprintf("%%arg%d", len(inputs)+j)
		valSSA := nodeSSA[out.ID]
		if valSSA == "" {
			return nil, fmt.Errorf("tileir: missing SSA result for output %s (node %d)", out.Name, out.ID)
		}
		tokCounter++
		nextTok := fmt.Sprintf("%%tok%d", tokCounter)

		locRef := recordLoc(out)
		fn.Operations = append(fn.Operations, &Operation{
			Line:       len(fn.Operations) + 1,
			Results:    []string{nextTok},
			OpCode:     "cuda_tile.store_view_tko",
			Operands:   []string{valSSA, outArgSSA, currentTok},
			Type:       "!cuda_tile.token",
			TypeVal:    TokenType{},
			Attributes: fmt.Sprintf(`{semantic_id = %d, name = %q}`, out.ID, out.Name),
			LocRef:     locRef,
			Loc:        mod.Locations[strings.TrimPrefix(locRef, "#")],
		})
		currentTok = nextTok
	}

	// 5. Final return operation
	fn.Operations = append(fn.Operations, &Operation{
		Line:     len(fn.Operations) + 1,
		OpCode:   "cuda_tile.return",
		Operands: []string{currentTok},
	})

	mod.Functions = append(mod.Functions, fn)

	// Verify the generated Tile IR module for token linearity and shape coherence
	hazards, shapes := fn.Verify()
	if len(hazards) > 0 {
		return nil, fmt.Errorf("tileir: lowering produced token hazards: %v", hazards)
	}
	if len(shapes) > 0 {
		return nil, fmt.Errorf("tileir: lowering produced shape errors: %v", shapes)
	}

	source, _, err := GenerateSemanticTileCPP(snap, opts)
	if err != nil {
		return nil, err
	}

	return &LoweringResult{
		Module: mod,
		Schema: schema,
		Source: source,
	}, nil
}

// GeneratePinnedTileCPP returns a deliberately unbuildable declaration. Shapes
// alone do not specify computation; use GenerateSemanticTileCPP for executable source.
func GeneratePinnedTileCPP(kernelName string, params []cudakernel.Param, tileM, tileN, tileK int64) string {
	var b strings.Builder
	b.WriteString("// Pinned Tile C++ contract: requires cuda_tile.h, __tile_global__, C++20\n")
	b.WriteString("#include \"cuda_tile.h\"\n\n")
	b.WriteString(fmt.Sprintf("extern \"C\" __tile_global__ void %s(\n", kernelName))

	for i, p := range params {
		comma := ","
		if i == len(params)-1 {
			comma = ""
		}
		cppType := "float*"
		if p.Type == "f16" {
			cppType = "__half*"
		} else if p.Type == "bf16" {
			cppType = "__nv_bfloat16*"
		}
		b.WriteString(fmt.Sprintf("    %s %s%s\n", cppType, p.Name, comma))
	}
	b.WriteString(") {\n")
	b.WriteString("#error shape-only declaration has no semantic body; use GenerateSemanticTileCPP\n")
	b.WriteString(fmt.Sprintf("    // Static tile parameters: TileM=%d, TileN=%d, TileK=%d\n", tileM, tileN, tileK))
	b.WriteString("}\n")
	return b.String()
}

// ValidateTileCPPSource verifies that a Tile C++ source strictly adheres to the pinned contract.
func ValidateTileCPPSource(src string) error {
	// Forbid hallucinated headers
	if strings.Contains(src, "<cuda/tiles.hpp>") || strings.Contains(src, "\"cuda/tiles.hpp\"") {
		return fmt.Errorf("%w: hallucinated header detected", ErrInvalidHeader)
	}
	if !strings.Contains(src, `#include "cuda_tile.h"`) && !strings.Contains(src, `#include <cuda_tile.h>`) {
		return fmt.Errorf("%w: missing required #include \"cuda_tile.h\"", ErrInvalidHeader)
	}
	if !strings.Contains(src, "__tile_global__") {
		return fmt.Errorf("%w: kernel missing __tile_global__ marker", ErrInvalidTileGlobal)
	}
	return nil
}

// ValidateTileCompileFlags ensures that NVRTC compilation options strictly adhere to
// the two-tier Option Splitting Model:
// Source flags (--enable-tile, --tile-only, --tilebc or --tilecubin, --std=c++20)
// are passed strictly to NVRTC; offline flags (-maxrregcount, -Xptxas, -O3) are forbidden.
func ValidateTileCompileFlags(flags []string) error {
	hasEnableTile := false
	hasTileOnly := false
	hasTileOutput := false
	hasStdCPP20 := false

	for _, f := range flags {
		if f == "--enable-tile" {
			hasEnableTile = true
		}
		if f == "--tile-only" {
			hasTileOnly = true
		}
		if f == "--tilebc" || f == "--tilecubin" {
			hasTileOutput = true
		}
		if f == "--std=c++20" || strings.HasPrefix(f, "-std=c++20") {
			hasStdCPP20 = true
		}

		// Reject offline/ptxas flags passed to NVRTC phase
		if strings.HasPrefix(f, "-maxrregcount") || strings.HasPrefix(f, "--maxrregcount") ||
			strings.HasPrefix(f, "-Xptxas") || strings.HasPrefix(f, "-O3") {
			return fmt.Errorf("%w: offline flag %q forbidden in NVRTC compile phase", ErrInvalidCompileFlags, f)
		}
	}

	if !hasEnableTile {
		return fmt.Errorf("%w: missing required --enable-tile flag", ErrInvalidCompileFlags)
	}
	if !hasTileOnly {
		return fmt.Errorf("%w: missing required --tile-only flag", ErrInvalidCompileFlags)
	}
	if !hasTileOutput {
		return fmt.Errorf("%w: missing required --tilebc or --tilecubin flag", ErrInvalidCompileFlags)
	}
	if !hasStdCPP20 {
		return fmt.Errorf("%w: missing required --std=c++20 flag", ErrInvalidCompileFlags)
	}

	return nil
}

func primitiveTypeFromDataType(dt cudair.DataType) Type {
	switch dt {
	case cudair.Float16:
		return Float16
	case cudair.BFloat16:
		return BFloat16
	case cudair.Int32:
		return Int32
	case cudair.Int64:
		return Int64
	case cudair.Uint8, cudair.Int8:
		return Int8
	case cudair.Float32:
		fallthrough
	default:
		return Float32
	}
}

func ptxTypeFromDataType(dt cudair.DataType) string {
	switch dt {
	case cudair.Float16:
		return "f16"
	case cudair.BFloat16:
		return "bf16"
	case cudair.Int32:
		return "s32"
	case cudair.Int64:
		return "s64"
	case cudair.Uint8:
		return "u8"
	case cudair.Int8:
		return "s8"
	case cudair.Float32:
		fallthrough
	default:
		return "f32"
	}
}

// GenerateSemanticTileCPP emits a CUDA 13.4 Tile C++ reference kernel and its
// pointer ABI from the existing semantic snapshot. Launch one tile block per
// output element (the largest output when there are several). The finite
// Float32 subset includes pointwise arithmetic, scalar constants, row-major
// matmul, and rank-one/two sum, max and mean reductions. Reductions are bounded
// to 64 terms and preserve their sequential accumulation order.
//
// This path emits actual loads, arithmetic and stores. It does not use the
// incomplete AST bytecode encoder, does not promise tensor-core execution,
// and does not qualify NaN, infinity, signed-zero or reassociation behavior.
func GenerateSemanticTileCPP(snap *cudair.Snapshot, opts LowerOptions) (string, cudakernel.Schema, error) {
	if snap == nil || len(snap.Nodes()) == 0 || len(snap.Nodes()) > 256 {
		return "", cudakernel.Schema{}, errors.New("tileir: missing snapshot or node budget exceeded")
	}
	name := opts.KernelName
	if name == "" {
		name = "tile_kernel"
	}
	for i, c := range name {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || i > 0 && c >= '0' && c <= '9') {
			return "", cudakernel.Schema{}, errors.New("tileir: invalid kernel identifier")
		}
	}
	if len(name) > 128 {
		return "", cudakernel.Schema{}, errors.New("tileir: kernel identifier too long")
	}
	counts := make(map[int]int)
	for _, n := range snap.Nodes() {
		if n.Type != cudair.Float32 || n.Layout != cudashape.LayoutRowMajor || len(n.Shape) > 2 || n.Effect != cudair.EffectReadOnly {
			return "", cudakernel.Schema{}, fmt.Errorf("%w: type, layout, rank or effect at node %d", ErrUnsupportedTileOp, n.ID)
		}
		count := 1
		for _, d := range n.Shape {
			if d <= 0 || d > (1<<20)/count {
				return "", cudakernel.Schema{}, errors.New("tileir: shape exceeds reference source budget")
			}
			count *= d
		}
		if len(n.Strides) != 0 && !cudashape.IsContiguousRowMajor(n.Shape, n.Strides) {
			return "", cudakernel.Schema{}, fmt.Errorf("%w: noncontiguous strides", ErrUnsupportedTileOp)
		}
		counts[n.ID] = count
	}
	inputs, outputs := snap.Inputs(), snap.Outputs()
	if len(outputs) == 0 {
		return "", cudakernel.Schema{}, errors.New("tileir: missing semantic outputs")
	}
	params := make([]cudakernel.Param, 0, len(inputs)+len(outputs))
	args := make(map[int]string)
	var b strings.Builder
	b.WriteString("#include \"cuda_tile.h\"\nextern \"C\" __tile_global__ void " + name + "(")
	for i, n := range append(append([]*cudair.Node(nil), inputs...), outputs...) {
		if i > 0 {
			b.WriteString(", ")
		}
		arg := fmt.Sprintf("arg%d", i)
		b.WriteString("float* " + arg)
		if i < len(inputs) {
			args[n.ID] = arg
		}
		paramName := n.Name
		if paramName == "" {
			paramName = arg
		}
		params = append(params, cudakernel.Param{Name: paramName, Type: "f32", ByteWidth: 8, Alignment: 16, IsPointer: true})
	}
	b.WriteString(") {\n  namespace ct = ::cuda::tiles;\n  auto idx = ct::bid().x;\n")
	schema, err := cudakernel.NewSchema(params...)
	if err != nil {
		return "", cudakernel.Schema{}, err
	}
	calls := 0
	var expr func(*cudair.Node, string, int) (string, error)
	expr = func(n *cudair.Node, index string, depth int) (string, error) {
		calls++
		if calls > 4096 || depth > 128 || len(index) > 4096 {
			return "", errors.New("tileir: expression expansion budget exceeded")
		}
		if _, ok := counts[n.ID]; !ok {
			return "", errors.New("tileir: input outside snapshot")
		}
		child := func(i int, ix string) (string, error) {
			if i >= len(n.Inputs) {
				return "", errors.New("tileir: missing semantic operand")
			}
			return expr(n.Inputs[i], ix, depth+1)
		}
		combine := func(op, a, c string) string { return "ct::" + op + "(" + a + ", " + c + ")" }
		switch n.Kind {
		case cudair.KindInput:
			arg, ok := args[n.ID]
			if !ok {
				return "", errors.New("tileir: unbound semantic input")
			}
			return arg + "[" + index + "]", nil
		case cudair.KindConstant:
			if len(n.ConstValues) != 1 || counts[n.ID] != 1 || math.IsNaN(float64(n.ConstValues[0])) || math.IsInf(float64(n.ConstValues[0]), 0) {
				return "", fmt.Errorf("%w: nonfinite or nonscalar constant", ErrUnsupportedTileOp)
			}
			s := strconv.FormatFloat(float64(n.ConstValues[0]), 'g', -1, 32)
			if !strings.ContainsAny(s, ".eE") {
				s += ".0"
			}
			return s + "f", nil
		case cudair.KindBinary, cudair.KindUnary:
			arity := 1
			if n.Kind == cudair.KindBinary {
				arity = 2
			}
			if len(n.Inputs) != arity {
				return "", errors.New("tileir: invalid operand count")
			}
			values := make([]string, arity)
			for i, in := range n.Inputs {
				ix := index
				if counts[in.ID] == 1 {
					ix = "0"
				} else if len(in.Shape) != len(n.Shape) || !sameTileShape(in.Shape, n.Shape) {
					return "", fmt.Errorf("%w: nonscalar broadcasting", ErrUnsupportedTileOp)
				}
				v, e := child(i, ix)
				if e != nil {
					return "", e
				}
				values[i] = v
			}
			switch n.Op {
			case "add", "sub", "mul", "div":
				if arity != 2 {
					break
				}
				return combine(n.Op, values[0], values[1]), nil
			case "neg":
				if arity == 1 {
					return "(-(" + values[0] + "))", nil
				}
			case "relu":
				if arity == 1 {
					return combine("max", values[0], "0.0f"), nil
				}
			}
		case cudair.KindReduce:
			if len(n.Inputs) != 1 || len(n.Axes) != 1 {
				return "", fmt.Errorf("%w: reduction arity", ErrUnsupportedTileOp)
			}
			in := n.Inputs[0]
			axis := n.Axes[0]
			if axis < 0 || axis >= len(in.Shape) || in.Shape[axis] > 64 {
				return "", fmt.Errorf("%w: reduction axis or term budget", ErrUnsupportedTileOp)
			}
			wantShape := append([]int(nil), in.Shape[:axis]...)
			wantShape = append(wantShape, in.Shape[axis+1:]...)
			if len(wantShape) == 0 {
				wantShape = []int{1}
			}
			if !sameTileShape(n.Shape, wantShape) {
				return "", errors.New("tileir: inconsistent reduction output shape")
			}
			if n.Op != "sum" && n.Op != "max" && n.Op != "mean" {
				break
			}
			var value string
			for k := 0; k < in.Shape[axis]; k++ {
				ix := strconv.Itoa(k)
				if len(in.Shape) == 2 {
					if axis == 1 {
						ix = fmt.Sprintf("((%s)*%d+%d)", index, in.Shape[1], k)
					} else {
						ix = fmt.Sprintf("(%d+%s)", k*in.Shape[1], index)
					}
				}
				v, e := child(0, ix)
				if e != nil {
					return "", e
				}
				if k == 0 {
					value = v
				} else {
					op := "add"
					if n.Op == "max" {
						op = "max"
					}
					value = combine(op, value, v)
				}
			}
			if n.Op == "mean" {
				value = combine("div", value, fmt.Sprintf("%d.0f", in.Shape[axis]))
			}
			return value, nil
		case cudair.KindMatMul:
			if len(n.Inputs) != 2 || len(n.Inputs[0].Shape) != 2 || len(n.Inputs[1].Shape) != 2 {
				return "", fmt.Errorf("%w: matmul ranks", ErrUnsupportedTileOp)
			}
			a, c := n.Inputs[0], n.Inputs[1]
			terms, cols := a.Shape[1], c.Shape[1]
			if !sameTileShape(n.Shape, []int{a.Shape[0], cols}) {
				return "", errors.New("tileir: inconsistent matmul output shape")
			}
			if terms != c.Shape[0] || terms > 64 {
				return "", fmt.Errorf("%w: matmul term budget", ErrUnsupportedTileOp)
			}
			value := "0.0f"
			for k := 0; k < terms; k++ {
				lhs, e := child(0, fmt.Sprintf("((%s)/%d*%d+%d)", index, cols, terms, k))
				if e != nil {
					return "", e
				}
				rhs, e := child(1, fmt.Sprintf("(%d+(%s)%%%d)", k*cols, index, cols))
				if e != nil {
					return "", e
				}
				value = combine("add", value, combine("mul", lhs, rhs))
			}
			return value, nil
		}
		return "", fmt.Errorf("%w: node %d op %q", ErrUnsupportedTileOp, n.ID, n.Op)
	}
	for i, n := range outputs {
		value, err := expr(n, "idx", 0)
		if err != nil {
			return "", cudakernel.Schema{}, err
		}
		fmt.Fprintf(&b, "  if (idx < %d) arg%d[idx] = %s;\n", counts[n.ID], len(inputs)+i, value)
		if b.Len() > 1<<20 {
			return "", cudakernel.Schema{}, errors.New("tileir: source byte budget exceeded")
		}
	}
	b.WriteString("}\n")
	return b.String(), schema, nil
}

func sameTileShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// CompileSemanticTile compiles the bounded semantic Tile C++ source through
// NVRTC into an authentic Tile IR artifact. It returns the successful compiler
// log; failed compilation diagnostics remain in the wrapped error.
// Compilation does not create a device context or execute a
// kernel; loading, output oracles and hardware qualification remain separate.
// The selected NVRTC library and headers must support CUDA 13.4 Tile C++.
func CompileSemanticTile(ctx context.Context, snap *cudair.Snapshot, opts LowerOptions, major, minor int) (*cudakernel.Artifact, string, error) {
	if ctx == nil || major < 8 || major > 99 || minor < 0 || minor > 9 {
		return nil, "", errors.New("tileir: missing compile context or invalid target")
	}
	if err := ctx.Err(); err != nil {
		return nil, "", err
	}
	source, schema, err := compileTileSource(snap, opts)
	if err != nil {
		return nil, "", err
	}
	options := []nvrtcx.Option{nvrtcx.WithArch(major, minor), nvrtcx.WithFlag("--std=c++20"), nvrtcx.WithOutput(nvrtcx.OutputTileIR), nvrtcx.WithToolkitIncludes(), nvrtcx.WithNoCache()}
	flags, err := nvrtcx.CompileFlags(options...)
	if err != nil {
		return nil, "", err
	}
	compiled, err := nvrtcx.CompileContext(ctx, source, options...)
	if err != nil {
		return nil, "", fmt.Errorf("tileir: compile semantic source: %w", err)
	}
	name := opts.KernelName
	if name == "" {
		name = "tile_kernel"
	}
	artifact, err := compiled.ToArtifact(name, schema, fmt.Sprintf("sm_%d%d", major, minor))
	if err != nil {
		return nil, compiled.Log(), err
	}
	compilerMajor, compilerMinor, err := nvrtc.Version()
	if err != nil {
		return nil, compiled.Log(), fmt.Errorf("tileir: compiler version: %w", err)
	}
	artifact.Toolchain = fmt.Sprintf("nvrtc %d.%d", compilerMajor, compilerMinor)
	// WithOutput adds this flag inside Compile after CompileFlags resolves includes.
	artifact.CompilerOpts = append(append([]string(nil), flags...), nvrtc.FlagTileOnly)
	return artifact, compiled.Log(), nil
}
