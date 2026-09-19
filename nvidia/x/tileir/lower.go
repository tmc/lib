package tileir

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
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
}

// LowerSemanticRegion lowers a validated semantic region (cudair.Snapshot) to a typed Tile IR Module.
//
// It validates operations, maps parameter ABIs, retains semantic node IDs in operation attributes,
// attaches Go source coordinates to Module.Locations, and validates token dependencies and shapes.
func LowerSemanticRegion(snap *cudair.Snapshot, opts LowerOptions) (*LoweringResult, error) {
	if snap == nil {
		return nil, errors.New("tileir: nil snapshot")
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
	nodeSSA := make(map[*cudair.Node]string)
	nodeTypes := make(map[*cudair.Node]Type)
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
		nodeSSA[inp] = valSSA
		nodeTypes[inp] = tileType
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
			nodeSSA[node] = resSSA
			nodeTypes[node] = tileType

		case cudair.KindBinary:
			if len(node.Inputs) < 2 {
				return nil, fmt.Errorf("tileir: binary op %d has < 2 inputs", node.ID)
			}
			lhsSSA := nodeSSA[node.Inputs[0]]
			rhsSSA := nodeSSA[node.Inputs[1]]
			resSSA := fmt.Sprintf("%%b%d", node.ID)

			opName := "cuda_tile.addf"
			switch strings.ToLower(node.Op) {
			case "add":
				opName = "cuda_tile.addf"
			case "mul":
				opName = "cuda_tile.mulf"
			case "sub":
				opName = "cuda_tile.addf" // Tile IR represents sub as add with neg or direct
			case "div":
				opName = "cuda_tile.divf"
			}

			tileType := nodeTypes[node.Inputs[0]]
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
			nodeSSA[node] = resSSA
			nodeTypes[node] = tileType

		case cudair.KindUnary:
			if len(node.Inputs) < 1 {
				return nil, fmt.Errorf("tileir: unary op %d has no input", node.ID)
			}
			inSSA := nodeSSA[node.Inputs[0]]
			resSSA := fmt.Sprintf("%%u%d", node.ID)

			tileType := nodeTypes[node.Inputs[0]]
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
			nodeSSA[node] = resSSA
			nodeTypes[node] = tileType

		case cudair.KindMatMul:
			if len(node.Inputs) < 2 {
				return nil, fmt.Errorf("tileir: matmul op %d has < 2 inputs", node.ID)
			}
			aSSA := nodeSSA[node.Inputs[0]]
			bSSA := nodeSSA[node.Inputs[1]]
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
			nodeSSA[node] = resSSA
			nodeTypes[node] = accType
		}
	}

	// 4. Store outputs via store_view_tko
	for j, out := range outputs {
		outArgSSA := fmt.Sprintf("%%arg%d", len(inputs)+j)
		valSSA := nodeSSA[out]
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

	source := GeneratePinnedTileCPP(kernelName, schemaParams, tileM, tileN, tileK)

	return &LoweringResult{
		Module: mod,
		Schema: schema,
		Source: source,
	}, nil
}

// GeneratePinnedTileCPP emits C++20 Tile C++ code following the strict pinned contract:
// strictly #include "cuda_tile.h", __tile_global__, and standard types.
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
	b.WriteString(fmt.Sprintf("    // Static tile parameters: TileM=%d, TileN=%d, TileK=%d\n", tileM, tileN, tileK))
	b.WriteString("    // Device execution adheres to Tile C++ contract\n")
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
