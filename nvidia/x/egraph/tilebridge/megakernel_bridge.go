package tilebridge

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/tileir"
)

// TileModule is a CUDA 13.1 Tile IR module ([tileir.Module]) representing a
// single-stage kernel or a multi-stage cross-layer megakernel.
type TileModule = tileir.Module

type megaSpecJSON struct {
	Name   string            `json:"name"`
	Inputs []hopper.Tensor   `json:"inputs"`
	Temps  []hopper.Tensor   `json:"temps"`
	Output hopper.Tensor     `json:"output"`
	Stages []megaStageSpecJS `json:"stages"`
}

type megaStageSpecJS struct {
	Name          string          `json:"name"`
	Kind          string          `json:"kind"`
	Grid          [3]int          `json:"grid"`
	Threads       int             `json:"threads"`
	BlockBase     int             `json:"block_base"`
	Blocks        int             `json:"blocks"`
	Shared        int             `json:"shared"`
	WaitSlots     []int           `json:"wait_slots"`
	SignalSlot    int             `json:"signal_slot"`
	Outputs       []hopper.Tensor `json:"outputs"`
	M             int             `json:"m,omitempty"`
	Rows          int             `json:"rows,omitempty"`
	Args          []string        `json:"args"`
	Row           *rowSpecJS      `json:"row,omitempty"`
	Tile          *tileSpecJS     `json:"tile,omitempty"`
	GEMM          *skinnySpecJS   `json:"gemm,omitempty"`
	StageComments []string        `json:"stage_comments,omitempty"`
}

type rowSpecJS struct {
	Name    string             `json:"name"`
	D       int                `json:"d"`
	Slots   int                `json:"slots,omitempty"`
	Cluster bool               `json:"cluster,omitempty"`
	Inputs  []hopper.RowTensor `json:"inputs"`
	Values  []rowValueSpecJS   `json:"values"`
	Outputs []hopper.RowTensor `json:"outputs"`
}

type rowValueSpecJS struct {
	Name   string   `json:"name"`
	Reduce string   `json:"reduce,omitempty"`
	Body   string   `json:"body"`
	Args   []string `json:"args"`
}

type tileSpecJS struct {
	Name        string              `json:"name"`
	N           int                 `json:"n"`
	K           int                 `json:"k"`
	BM          int                 `json:"bm"`
	BN          int                 `json:"bn"`
	Stages      int                 `json:"stages"`
	StreamK     int                 `json:"stream_k,omitempty"`
	Weight      hopper.WeightFormat `json:"weight"`
	Out         hopper.DType        `json:"out"`
	Swizzle128B bool                `json:"swizzle_128b,omitempty"`
	WGMMA       bool                `json:"wgmma,omitempty"`
	GLU         bool                `json:"glu,omitempty"`
	SumSq       bool                `json:"sum_sq,omitempty"`
	Epilogue    string              `json:"epilogue"`
	Extras      []hopper.Operand    `json:"extras,omitempty"`
}

type skinnySpecJS struct {
	Name         string              `json:"name"`
	N            int                 `json:"n"`
	K            int                 `json:"k"`
	MaxM         int                 `json:"max_m"`
	Weight       hopper.WeightFormat `json:"weight"`
	X            hopper.DType        `json:"x"`
	Out          hopper.DType        `json:"out"`
	RowsPerBlock int                 `json:"rows_per_block"`
	SplitK       int                 `json:"split_k"`
	GLU          bool                `json:"glu,omitempty"`
	Epilogue     string              `json:"epilogue"`
	Extras       []hopper.Operand    `json:"extras,omitempty"`
}

// LowerCrossLayerMegakernelToTileIR translates a multi-stage cross-layer
// [*hopper.Megakernel] into a single [*TileModule] containing one Tile IR
// entry function per stage while preserving stage dependencies (ready[stage]
// WaitSlots / SignalSlot producer-consumer barriers, CTA grid partitions,
// tile shapes, and epilogue operations).
func LowerCrossLayerMegakernelToTileIR(m *hopper.Megakernel) (*TileModule, error) {
	if m == nil {
		return nil, fmt.Errorf("tilebridge: nil megakernel")
	}
	if strings.TrimSpace(m.Name) == "" {
		return nil, fmt.Errorf("tilebridge: megakernel Name must not be empty")
	}
	if len(m.Stages) == 0 {
		return nil, fmt.Errorf("tilebridge: megakernel %s has no stages", m.Name)
	}

	spec := megaSpecJSON{
		Name:   m.Name,
		Inputs: slices.Clone(m.Inputs),
		Temps:  slices.Clone(m.Temps),
		Output: m.Output,
		Stages: make([]megaStageSpecJS, len(m.Stages)),
	}

	var comments []string
	comments = append(comments,
		"// hopper.kind: megakernel",
		fmt.Sprintf("// hopper.mega.name: %s", m.Name),
		fmt.Sprintf("// hopper.mega.stages: %d", len(m.Stages)),
		fmt.Sprintf("// hopper.mega.ready_words: %d", m.ReadyWords()),
	)

	var fns []*tileir.Function
	for i, st := range m.Stages {
		if strings.TrimSpace(st.Name) == "" {
			return nil, fmt.Errorf("tilebridge: megakernel %s stage %d has empty Name", m.Name, i)
		}
		stMod, err := StepHopperToTileIR(st.Kernel)
		if err != nil {
			return nil, fmt.Errorf("tilebridge: megakernel %s stage %d (%s): %w", m.Name, i, st.Name, err)
		}
		text := FormatTileIR(stMod)
		parsed, err := ParseTileIR(text)
		if err != nil {
			return nil, fmt.Errorf("tilebridge: megakernel %s stage %d (%s) parse check: %w", m.Name, i, st.Name, err)
		}
		bc, err := EncodeTileIR(parsed)
		if err != nil {
			return nil, fmt.Errorf("tilebridge: megakernel %s stage %d (%s) bytecode check: %w", m.Name, i, st.Name, err)
		}
		if !tileir.IsTileIRBytecode(bc) {
			return nil, fmt.Errorf("tilebridge: megakernel %s stage %d (%s) invalid bytecode header", m.Name, i, st.Name)
		}
		if len(stMod.Functions) == 0 {
			return nil, fmt.Errorf("tilebridge: megakernel %s stage %d (%s) lowered to empty function list", m.Name, i, st.Name)
		}

		fn := stMod.Functions[0]
		fn.Name = st.Name
		fn.Entry = true
		fns = append(fns, fn)

		stSpec, kindStr, epStr, tileShapeStr, err := buildStageSpecJS(st, stMod.Comments)
		if err != nil {
			return nil, fmt.Errorf("tilebridge: megakernel %s stage %d (%s): %w", m.Name, i, st.Name, err)
		}
		spec.Stages[i] = stSpec

		waitStr := formatIntList(st.WaitSlots)
		var outNames []string
		for _, o := range st.Outputs {
			outNames = append(outNames, o.Name)
		}
		comments = append(comments,
			fmt.Sprintf("// hopper.mega.stage.%d.name: %s", i, st.Name),
			fmt.Sprintf("// hopper.mega.stage.%d.kind: %s", i, kindStr),
			fmt.Sprintf("// hopper.mega.stage.%d.grid: %d,%d,%d", i, st.Grid[0], st.Grid[1], st.Grid[2]),
			fmt.Sprintf("// hopper.mega.stage.%d.block_base: %d", i, st.BlockBase),
			fmt.Sprintf("// hopper.mega.stage.%d.blocks: %d", i, st.Blocks),
			fmt.Sprintf("// hopper.mega.stage.%d.wait_slots: %s", i, waitStr),
			fmt.Sprintf("// hopper.mega.stage.%d.signal_slot: %d", i, st.SignalSlot),
			fmt.Sprintf("// hopper.mega.stage.%d.outputs: %s", i, strings.Join(outNames, ",")),
			fmt.Sprintf("// hopper.mega.stage.%d.tile_shape: %s", i, tileShapeStr),
		)
		if epStr != "" {
			comments = append(comments, fmt.Sprintf("// hopper.mega.stage.%d.epilogue: %s", i, epStr))
		}
		if i == 0 {
			comments = append(comments, stMod.Comments...)
		}
	}

	rawSpec, err := json.Marshal(spec)
	if err != nil {
		return nil, fmt.Errorf("tilebridge: marshal megakernel spec: %w", err)
	}
	comments = append(comments, fmt.Sprintf("// hopper.mega.spec: %s", string(rawSpec)))

	mod := &TileModule{
		Name:      m.Name + "_module",
		Comments:  comments,
		Functions: fns,
	}

	// Verify the combined multi-stage module round-trips through Format -> Parse -> EncodeBytecode.
	fullText := FormatTileIR(mod)
	fullParsed, err := ParseTileIR(fullText)
	if err != nil {
		return nil, fmt.Errorf("tilebridge: combined megakernel ParseTileIR: %w", err)
	}
	if _, err := EncodeTileIR(fullParsed); err != nil {
		return nil, fmt.Errorf("tilebridge: combined megakernel EncodeTileIR: %w", err)
	}
	return mod, nil
}

func buildStageSpecJS(st hopper.MegaStage, stComments []string) (spec megaStageSpecJS, kindStr, epStr, tileShapeStr string, err error) {
	spec = megaStageSpecJS{
		Name:          st.Name,
		Grid:          st.Grid,
		Threads:       st.Threads,
		BlockBase:     st.BlockBase,
		Blocks:        st.Blocks,
		Shared:        st.Shared,
		WaitSlots:     slices.Clone(st.WaitSlots),
		SignalSlot:    st.SignalSlot,
		Outputs:       slices.Clone(st.Outputs),
		M:             st.Kernel.M,
		Rows:          st.Kernel.Rows,
		Args:          slices.Clone(st.Kernel.Args),
		StageComments: slices.Clone(stComments),
	}
	pk := st.Kernel
	switch {
	case pk.Row != nil:
		rk := *pk.Row
		kindStr = "row"
		tileShapeStr = fmt.Sprintf("1x%d", pow2TileWidth(rk.D))
		var vals []rowValueSpecJS
		for _, v := range rk.Values {
			vals = append(vals, rowValueSpecJS{
				Name:   v.Name,
				Reduce: v.Reduce,
				Body:   v.Body.String(),
				Args:   slices.Clone(v.Args),
			})
		}
		spec.Kind = kindStr
		spec.Row = &rowSpecJS{
			Name:    rk.Name,
			D:       rk.D,
			Slots:   rk.Slots,
			Cluster: rk.Cluster,
			Inputs:  slices.Clone(rk.Inputs),
			Values:  vals,
			Outputs: slices.Clone(rk.Outputs),
		}
	case pk.Tile != nil:
		tg := *pk.Tile
		kindStr = "tile_gemm"
		epStr = tg.Epilogue.String()
		if epStr == "" {
			epStr = "$0"
		}
		bm, bn := tg.BM, tg.BN
		if bm <= 0 {
			bm = 64
		}
		if bn <= 0 {
			bn = 64
		}
		tileShapeStr = fmt.Sprintf("%dx%dx16", bm, bn)
		spec.Kind = kindStr
		spec.Tile = &tileSpecJS{
			Name:        tg.Name,
			N:           tg.N,
			K:           tg.K,
			BM:          tg.BM,
			BN:          tg.BN,
			Stages:      tg.Stages,
			StreamK:     tg.StreamK,
			Weight:      tg.Weight,
			Out:         tg.Out,
			Swizzle128B: tg.Swizzle128B,
			WGMMA:       tg.WGMMA,
			GLU:         tg.GLU,
			SumSq:       tg.SumSq,
			Epilogue:    epStr,
			Extras:      slices.Clone(tg.Extras),
		}
	case pk.GEMM != nil:
		sg := *pk.GEMM
		kindStr = "skinny_gemm"
		epStr = sg.Epilogue.String()
		if epStr == "" {
			epStr = "$0"
		}
		tileShapeStr = fmt.Sprintf("%dx%d", max(pk.M, 1), sg.RowsPerBlock)
		spec.Kind = kindStr
		spec.GEMM = &skinnySpecJS{
			Name:         sg.Name,
			N:            sg.N,
			K:            sg.K,
			MaxM:         sg.MaxM,
			Weight:       sg.Weight,
			X:            sg.X,
			Out:          sg.Out,
			RowsPerBlock: sg.RowsPerBlock,
			SplitK:       sg.SplitK,
			GLU:          sg.GLU,
			Epilogue:     epStr,
			Extras:       slices.Clone(sg.Extras),
		}
	default:
		return spec, "", "", "", fmt.Errorf("unsupported stage kernel (need Row, Tile, or GEMM)")
	}
	return spec, kindStr, epStr, tileShapeStr, nil
}

// LiftTileIRToCrossLayerMegakernel lifts a multi-stage [*TileModule] back into
// a persistent [*hopper.Megakernel] for the target GPU architecture arch,
// reconstructing and validating all stage dependencies (WaitSlots, SignalSlot,
// BlockBase, Blocks, Grid), tile shapes, and epilogue lambdas.
func LiftTileIRToCrossLayerMegakernel(mod *TileModule, arch hopper.Arch) (*hopper.Megakernel, error) {
	if mod == nil {
		return nil, fmt.Errorf("tilebridge: nil TileModule")
	}
	if len(mod.Functions) == 0 {
		return nil, fmt.Errorf("tilebridge: TileModule %q has no functions", mod.Name)
	}

	normArch, err := normalizeBridgeArch(arch)
	if err != nil {
		return nil, err
	}

	// Verify the module round-trips through Format -> Parse and encodes to bytecode.
	text := FormatTileIR(mod)
	parsed, err := ParseTileIR(text)
	if err != nil {
		return nil, fmt.Errorf("tilebridge: lift ParseTileIR: %w", err)
	}
	bc, err := EncodeTileIR(parsed)
	if err != nil {
		return nil, fmt.Errorf("tilebridge: lift EncodeTileIR: %w", err)
	}
	if !tileir.IsTileIRBytecode(bc) {
		return nil, fmt.Errorf("tilebridge: lift encoded bytes lack TileIR magic header")
	}

	var spec megaSpecJSON
	foundSpec := false
	stageWaitOverrides := make(map[int][]int)
	stageSignalOverrides := make(map[int]int)
	stageGridOverrides := make(map[int][3]int)
	stageBlockBaseOverrides := make(map[int]int)
	stageBlocksOverrides := make(map[int]int)
	stageEpilogueOverrides := make(map[int]string)
	var nameOverride string

	for _, rawC := range mod.Comments {
		c := cleanCommentLine(rawC)
		switch {
		case strings.HasPrefix(c, "hopper.mega.spec:"):
			js := strings.TrimSpace(strings.TrimPrefix(c, "hopper.mega.spec:"))
			if err := json.Unmarshal([]byte(js), &spec); err != nil {
				return nil, fmt.Errorf("tilebridge: invalid hopper.mega.spec metadata: %w", err)
			}
			foundSpec = true
		case strings.HasPrefix(c, "hopper.mega.name:"):
			nameOverride = strings.TrimSpace(strings.TrimPrefix(c, "hopper.mega.name:"))
		case strings.HasPrefix(c, "hopper.mega.stage."):
			rest := strings.TrimPrefix(c, "hopper.mega.stage.")
			dot := strings.IndexByte(rest, '.')
			colon := strings.IndexByte(rest, ':')
			if dot <= 0 || colon <= dot {
				continue
			}
			stIdx, err := strconv.Atoi(rest[:dot])
			if err != nil || stIdx < 0 {
				continue
			}
			field := strings.TrimSpace(rest[dot+1 : colon])
			val := strings.TrimSpace(rest[colon+1:])
			switch field {
			case "wait_slots":
				ws, err := parseIntList(val)
				if err != nil {
					return nil, fmt.Errorf("tilebridge: stage %d invalid wait_slots %q: %w", stIdx, val, err)
				}
				stageWaitOverrides[stIdx] = ws
			case "signal_slot":
				sig, err := strconv.Atoi(val)
				if err != nil {
					return nil, fmt.Errorf("tilebridge: stage %d invalid signal_slot %q: %w", stIdx, val, err)
				}
				stageSignalOverrides[stIdx] = sig
			case "grid":
				parts, err := parseIntList(val)
				if err != nil || len(parts) != 3 {
					return nil, fmt.Errorf("tilebridge: stage %d invalid grid %q", stIdx, val)
				}
				stageGridOverrides[stIdx] = [3]int{parts[0], parts[1], parts[2]}
			case "block_base":
				bb, err := strconv.Atoi(val)
				if err != nil {
					return nil, fmt.Errorf("tilebridge: stage %d invalid block_base %q: %w", stIdx, val, err)
				}
				stageBlockBaseOverrides[stIdx] = bb
			case "blocks":
				blks, err := strconv.Atoi(val)
				if err != nil {
					return nil, fmt.Errorf("tilebridge: stage %d invalid blocks %q: %w", stIdx, val, err)
				}
				stageBlocksOverrides[stIdx] = blks
			case "epilogue":
				stageEpilogueOverrides[stIdx] = val
			}
		}
	}

	if !foundSpec {
		return nil, fmt.Errorf("tilebridge: TileModule %q missing hopper.mega.spec cross-layer metadata", mod.Name)
	}
	if nameOverride != "" {
		spec.Name = nameOverride
	}
	if len(spec.Stages) != len(mod.Functions) {
		return nil, fmt.Errorf("tilebridge: stage count mismatch: metadata has %d stages, TileModule has %d functions",
			len(spec.Stages), len(mod.Functions))
	}

	stages := make([]hopper.MegaStage, len(spec.Stages))
	expectedBlockBase := 0
	for i, stSpec := range spec.Stages {
		fn := mod.Functions[i]
		if fn.Name != stSpec.Name {
			return nil, fmt.Errorf("tilebridge: stage %d function name %q != metadata stage name %q", i, fn.Name, stSpec.Name)
		}
		if ws, ok := stageWaitOverrides[i]; ok {
			stSpec.WaitSlots = ws
		}
		if sig, ok := stageSignalOverrides[i]; ok {
			stSpec.SignalSlot = sig
		}
		if g, ok := stageGridOverrides[i]; ok {
			stSpec.Grid = g
		}
		if bb, ok := stageBlockBaseOverrides[i]; ok {
			stSpec.BlockBase = bb
		}
		if blks, ok := stageBlocksOverrides[i]; ok {
			stSpec.Blocks = blks
		}
		if ep, ok := stageEpilogueOverrides[i]; ok {
			if stSpec.Tile != nil {
				stSpec.Tile.Epilogue = ep
			}
			if stSpec.GEMM != nil {
				stSpec.GEMM.Epilogue = ep
			}
		}

		// Validate WaitSlots and BlockBase invariants.
		for _, ws := range stSpec.WaitSlots {
			if ws < 0 || ws >= i {
				return nil, fmt.Errorf("tilebridge: stage %d (%s) has invalid wait slot %d", i, stSpec.Name, ws)
			}
		}
		if stSpec.Blocks <= 0 || stSpec.Grid[0]*stSpec.Grid[1]*stSpec.Grid[2] != stSpec.Blocks {
			return nil, fmt.Errorf("tilebridge: stage %d (%s) grid %v inconsistent with blocks %d",
				i, stSpec.Name, stSpec.Grid, stSpec.Blocks)
		}
		if stSpec.BlockBase != expectedBlockBase {
			return nil, fmt.Errorf("tilebridge: stage %d (%s) BlockBase %d != expected %d",
				i, stSpec.Name, stSpec.BlockBase, expectedBlockBase)
		}
		expectedBlockBase += stSpec.Blocks

		pk, err := reconstructPlanKernel(stSpec, fn)
		if err != nil {
			return nil, fmt.Errorf("tilebridge: stage %d (%s): %w", i, stSpec.Name, err)
		}

		stages[i] = hopper.MegaStage{
			Kernel:     pk,
			Name:       stSpec.Name,
			Grid:       stSpec.Grid,
			Threads:    stSpec.Threads,
			BlockBase:  stSpec.BlockBase,
			Blocks:     stSpec.Blocks,
			Shared:     stSpec.Shared,
			WaitSlots:  slices.Clone(stSpec.WaitSlots),
			SignalSlot: stSpec.SignalSlot,
			Outputs:    slices.Clone(stSpec.Outputs),
		}
	}

	mega := &hopper.Megakernel{
		Name:   spec.Name,
		Inputs: slices.Clone(spec.Inputs),
		Temps:  slices.Clone(spec.Temps),
		Output: spec.Output,
		Stages: stages,
	}

	if _, err := hopper.SelectArchDispatchPlan(normArch, nil); err != nil {
		return nil, fmt.Errorf("tilebridge: arch dispatch validation for %s: %w", normArch, err)
	}
	if _, err := mega.CUDA(); err != nil {
		return nil, fmt.Errorf("tilebridge: lifted megakernel CUDA validation: %w", err)
	}
	return mega, nil
}

func reconstructPlanKernel(stSpec megaStageSpecJS, fn *tileir.Function) (hopper.PlanKernel, error) {
	if fn == nil || len(fn.Operations) == 0 {
		return hopper.PlanKernel{}, fmt.Errorf("empty TileIR function")
	}
	hasReturn := false
	mmaCount := 0
	hasIota := false
	hasLoad := false
	hasStore := false
	for _, op := range fn.Operations {
		switch strings.TrimPrefix(op.OpCode, "cuda_tile.") {
		case "return":
			hasReturn = true
		case "mmaf":
			mmaCount++
		case "iota":
			hasIota = true
		case "load_ptr_tko":
			hasLoad = true
		case "store_ptr_tko":
			hasStore = true
		}
	}
	if !hasReturn {
		return hopper.PlanKernel{}, fmt.Errorf("function %s missing cuda_tile.return", fn.Name)
	}

	switch stSpec.Kind {
	case "row":
		if stSpec.Row == nil {
			return hopper.PlanKernel{}, fmt.Errorf("missing row kernel specification")
		}
		if !hasIota || !hasLoad || !hasStore || mmaCount != 0 {
			return hopper.PlanKernel{}, fmt.Errorf("function %s AST does not match row kernel structure (iota=%v, load=%v, store=%v, mma=%d)",
				fn.Name, hasIota, hasLoad, hasStore, mmaCount)
		}
		rs := stSpec.Row
		var vals []hopper.RowValue
		for _, v := range rs.Values {
			l, err := hopper.ParseLambda(v.Body)
			if err != nil {
				return hopper.PlanKernel{}, fmt.Errorf("row value %s lambda: %w", v.Name, err)
			}
			vals = append(vals, hopper.RowValue{
				Name:   v.Name,
				Reduce: v.Reduce,
				Body:   l,
				Args:   slices.Clone(v.Args),
			})
		}
		rk := hopper.RowKernel{
			Name:    rs.Name,
			D:       rs.D,
			Slots:   rs.Slots,
			Cluster: rs.Cluster,
			Inputs:  slices.Clone(rs.Inputs),
			Values:  vals,
			Outputs: slices.Clone(rs.Outputs),
		}
		if err := rk.Validate(); err != nil {
			return hopper.PlanKernel{}, err
		}
		return hopper.PlanKernel{
			Row:  &rk,
			Rows: stSpec.Rows,
			M:    stSpec.M,
			Args: slices.Clone(stSpec.Args),
		}, nil

	case "tile_gemm":
		if stSpec.Tile == nil {
			return hopper.PlanKernel{}, fmt.Errorf("missing tile_gemm specification")
		}
		ts := stSpec.Tile
		wantMMA := 1
		if ts.GLU {
			wantMMA = 2
		}
		if mmaCount != wantMMA || !hasLoad || !hasStore {
			return hopper.PlanKernel{}, fmt.Errorf("function %s AST mmaCount=%d (want %d), load=%v, store=%v",
				fn.Name, mmaCount, wantMMA, hasLoad, hasStore)
		}
		ep, err := hopper.ParseLambda(ts.Epilogue)
		if err != nil {
			return hopper.PlanKernel{}, fmt.Errorf("tile_gemm epilogue: %w", err)
		}
		tg := hopper.TileGEMM{
			Name:        ts.Name,
			N:           ts.N,
			K:           ts.K,
			BM:          ts.BM,
			BN:          ts.BN,
			Stages:      ts.Stages,
			StreamK:     ts.StreamK,
			Weight:      ts.Weight,
			Out:         ts.Out,
			Swizzle128B: ts.Swizzle128B,
			WGMMA:       ts.WGMMA,
			GLU:         ts.GLU,
			SumSq:       ts.SumSq,
			Epilogue:    ep,
			Extras:      slices.Clone(ts.Extras),
		}
		return hopper.PlanKernel{
			Tile: &tg,
			M:    stSpec.M,
			Rows: stSpec.Rows,
			Args: slices.Clone(stSpec.Args),
		}, nil

	case "skinny_gemm":
		if stSpec.GEMM == nil {
			return hopper.PlanKernel{}, fmt.Errorf("missing skinny_gemm specification")
		}
		if mmaCount != 1 || !hasLoad || !hasStore {
			return hopper.PlanKernel{}, fmt.Errorf("function %s AST mmaCount=%d (want 1), load=%v, store=%v",
				fn.Name, mmaCount, hasLoad, hasStore)
		}
		gs := stSpec.GEMM
		ep, err := hopper.ParseLambda(gs.Epilogue)
		if err != nil {
			return hopper.PlanKernel{}, fmt.Errorf("skinny_gemm epilogue: %w", err)
		}
		sg := hopper.SkinnyGEMM{
			Name:         gs.Name,
			N:            gs.N,
			K:            gs.K,
			MaxM:         gs.MaxM,
			Weight:       gs.Weight,
			X:            gs.X,
			Out:          gs.Out,
			RowsPerBlock: gs.RowsPerBlock,
			SplitK:       gs.SplitK,
			GLU:          gs.GLU,
			Epilogue:     ep,
			Extras:       slices.Clone(gs.Extras),
		}
		return hopper.PlanKernel{
			GEMM: &sg,
			M:    stSpec.M,
			Rows: stSpec.Rows,
			Args: slices.Clone(stSpec.Args),
		}, nil

	default:
		return hopper.PlanKernel{}, fmt.Errorf("unknown stage kind %q", stSpec.Kind)
	}
}

// VerifyCrossLayerBridgeEquivalence performs bidirectional verification across
// the Megakernel -> TileIR -> Megakernel round-trip:
//  1. Lowers m to a [*TileModule] via [LowerCrossLayerMegakernelToTileIR].
//  2. Round-trips the module through [FormatTileIR], [ParseTileIR], and [EncodeTileIR].
//  3. Lifts the parsed module back to a [*hopper.Megakernel] via [LiftTileIRToCrossLayerMegakernel].
//  4. Verifies structural, schedule, ready-barrier, and CUDA source equivalence.
//  5. Executes witness testing seeded by seed across all stages of the original
//     Megakernel, the lowered Tile IR SSA module ([EvalTileIRReference]), and
//     the lifted Megakernel, verifying numerical equivalence across the bridge.
func VerifyCrossLayerBridgeEquivalence(m *hopper.Megakernel, seed uint64) error {
	if m == nil {
		return fmt.Errorf("tilebridge: nil megakernel")
	}
	mod, err := LowerCrossLayerMegakernelToTileIR(m)
	if err != nil {
		return fmt.Errorf("tilebridge: lower megakernel to TileIR: %w", err)
	}
	text := FormatTileIR(mod)
	parsed, err := ParseTileIR(text)
	if err != nil {
		return fmt.Errorf("tilebridge: parse formatted TileIR: %w", err)
	}
	bc, err := EncodeTileIR(parsed)
	if err != nil {
		return fmt.Errorf("tilebridge: encode TileIR bytecode: %w", err)
	}
	if !tileir.IsTileIRBytecode(bc) {
		return fmt.Errorf("tilebridge: encoded bytecode missing magic header")
	}

	lifted, err := LiftTileIRToCrossLayerMegakernel(parsed, hopper.TargetSM90A)
	if err != nil {
		return fmt.Errorf("tilebridge: lift TileIR to megakernel: %w", err)
	}

	if err := verifyMegakernelStructuralEquivalence(m, lifted); err != nil {
		return err
	}

	// Execute witness evaluation across all stages for:
	//   (a) original Megakernel m
	//   (b) lowered TileIR module parsed (via EvalTileIRReference per stage)
	//   (c) lifted Megakernel lifted
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	initBuffers := make(map[string][]float32)

	allocTensor := func(t hopper.Tensor, fillRand bool) {
		n := tensorElemCount(t.Shape)
		buf := make([]float32, n)
		if fillRand {
			dt := DType(t.DType)
			if dt == 0 {
				dt = F32
			}
			for i := range buf {
				v := (rng.Float32()*1.6 - 0.8) * 0.5
				buf[i] = dt.Round(v)
			}
		}
		initBuffers[t.Name] = buf
	}

	for _, in := range m.Inputs {
		allocTensor(in, true)
	}
	for _, tmp := range m.Temps {
		allocTensor(tmp, false)
	}
	if m.Output.Name != "" {
		allocTensor(m.Output, false)
	}

	origBufs := cloneBufferMap(initBuffers)
	tileBufs := cloneBufferMap(initBuffers)
	liftBufs := cloneBufferMap(initBuffers)

	if err := evalMegakernelReference(m, origBufs); err != nil {
		return fmt.Errorf("tilebridge: evaluate original megakernel: %w", err)
	}
	if err := evalMegakernelTileIR(m, parsed, tileBufs); err != nil {
		return fmt.Errorf("tilebridge: evaluate TileIR module: %w", err)
	}
	if err := evalMegakernelReference(lifted, liftBufs); err != nil {
		return fmt.Errorf("tilebridge: evaluate lifted megakernel: %w", err)
	}

	// Check stage outputs and final Output tensor.
	var checkNames []string
	for _, st := range m.Stages {
		for _, o := range st.Outputs {
			if !slices.Contains(checkNames, o.Name) {
				checkNames = append(checkNames, o.Name)
			}
		}
	}
	if m.Output.Name != "" && !slices.Contains(checkNames, m.Output.Name) {
		checkNames = append(checkNames, m.Output.Name)
	}

	for _, name := range checkNames {
		ob := origBufs[name]
		lb := liftBufs[name]
		tb := tileBufs[name]
		if len(ob) != len(lb) || len(ob) != len(tb) {
			return fmt.Errorf("tilebridge: buffer %q length mismatch: orig=%d lifted=%d tileir=%d",
				name, len(ob), len(lb), len(tb))
		}
		for i := range ob {
			if math.IsNaN(float64(ob[i])) || math.IsNaN(float64(lb[i])) || math.IsNaN(float64(tb[i])) {
				return fmt.Errorf("tilebridge: NaN in buffer %q[%d] (orig=%v lifted=%v tileir=%v)",
					name, i, ob[i], lb[i], tb[i])
			}
			diffLift := math.Abs(float64(ob[i] - lb[i]))
			if diffLift > 1e-6 {
				return fmt.Errorf("tilebridge: round-trip mismatch in %q[%d]: orig=%g lifted=%g (diff=%g)",
					name, i, ob[i], lb[i], diffLift)
			}
			diffTile := math.Abs(float64(ob[i] - tb[i]))
			if diffTile > 2e-2 {
				return fmt.Errorf("tilebridge: TileIR witness mismatch in %q[%d]: orig=%g tileir=%g (diff=%g)",
					name, i, ob[i], tb[i], diffTile)
			}
		}
	}
	return nil
}

func verifyMegakernelStructuralEquivalence(a, b *hopper.Megakernel) error {
	if a.Name != b.Name {
		return fmt.Errorf("tilebridge: megakernel Name mismatch: %q != %q", a.Name, b.Name)
	}
	if len(a.Stages) != len(b.Stages) {
		return fmt.Errorf("tilebridge: megakernel stage count mismatch: %d != %d", len(a.Stages), len(b.Stages))
	}
	ga, ba := a.Grid()
	gb, bb := b.Grid()
	if ga != gb || ba != bb {
		return fmt.Errorf("tilebridge: megakernel Grid mismatch: (%v, %v) != (%v, %v)", ga, ba, gb, bb)
	}
	if a.ReadyWords() != b.ReadyWords() {
		return fmt.Errorf("tilebridge: megakernel ReadyWords mismatch: %d != %d", a.ReadyWords(), b.ReadyWords())
	}
	for i := range a.Stages {
		sa, sb := a.Stages[i], b.Stages[i]
		if sa.Name != sb.Name {
			return fmt.Errorf("tilebridge: stage %d Name mismatch: %q != %q", i, sa.Name, sb.Name)
		}
		if sa.Grid != sb.Grid || sa.BlockBase != sb.BlockBase || sa.Blocks != sb.Blocks {
			return fmt.Errorf("tilebridge: stage %d (%s) CTA partition mismatch: (%v, base=%d, blocks=%d) != (%v, base=%d, blocks=%d)",
				i, sa.Name, sa.Grid, sa.BlockBase, sa.Blocks, sb.Grid, sb.BlockBase, sb.Blocks)
		}
		if !slices.Equal(sa.WaitSlots, sb.WaitSlots) || sa.SignalSlot != sb.SignalSlot {
			return fmt.Errorf("tilebridge: stage %d (%s) ready barrier mismatch: (wait=%v, signal=%d) != (wait=%v, signal=%d)",
				i, sa.Name, sa.WaitSlots, sa.SignalSlot, sb.WaitSlots, sb.SignalSlot)
		}
		if !slices.Equal(sa.Kernel.Args, sb.Kernel.Args) {
			return fmt.Errorf("tilebridge: stage %d (%s) Args mismatch: %v != %v", i, sa.Name, sa.Kernel.Args, sb.Kernel.Args)
		}
	}
	srcA, err := a.CUDA()
	if err != nil {
		return fmt.Errorf("tilebridge: original CUDA(): %w", err)
	}
	srcB, err := b.CUDA()
	if err != nil {
		return fmt.Errorf("tilebridge: lifted CUDA(): %w", err)
	}
	if srcA != srcB {
		return fmt.Errorf("tilebridge: CUDA source mismatch after TileIR round-trip")
	}
	return nil
}

func evalMegakernelReference(m *hopper.Megakernel, bufs map[string][]float32) error {
	tensorMeta := make(map[string]hopper.Tensor)
	for _, t := range m.Inputs {
		tensorMeta[t.Name] = t
	}
	for _, t := range m.Temps {
		tensorMeta[t.Name] = t
	}
	if m.Output.Name != "" {
		tensorMeta[m.Output.Name] = m.Output
	}

	readySignaled := make(map[int]int)
	for i, st := range m.Stages {
		for _, ws := range st.WaitSlots {
			if readySignaled[ws] != m.Stages[ws].Blocks {
				return fmt.Errorf("stage %d (%s) wait slot %d not ready (have %d/%d blocks)",
					i, st.Name, ws, readySignaled[ws], m.Stages[ws].Blocks)
			}
		}
		if err := evalStageReference(st, bufs, tensorMeta); err != nil {
			return fmt.Errorf("stage %d (%s): %w", i, st.Name, err)
		}
		if st.SignalSlot >= 0 {
			readySignaled[st.SignalSlot] = st.Blocks
		}
	}
	return nil
}

func evalStageReference(st hopper.MegaStage, bufs map[string][]float32, tensorMeta map[string]hopper.Tensor) error {
	pk := st.Kernel
	tbStep, err := ToTileBridgeStep(pk, 0, false)
	if err != nil {
		return err
	}
	switch {
	case tbStep.Row != nil:
		rk := *tbStep.Row
		rows := pk.Rows
		if rows <= 0 {
			rows = pk.M
		}
		d := rk.D
		for r := 0; r < rows; r++ {
			env := make(map[string][]float32)
			for _, in := range rk.Inputs {
				src := bufs[in.buffer()]
				vec := make([]float32, d)
				for c := 0; c < d; c++ {
					var idx int
					switch in.Kind {
					case ColVector:
						idx = in.Offset + c
					case RowScalar:
						idx = in.Offset + r
					case Scalar:
						idx = in.Offset
					default:
						idx = in.Offset + r*d + c
					}
					if idx >= 0 && idx < len(src) {
						vec[c] = in.DType.Round(src[idx])
					}
				}
				env[in.Name] = vec
			}
			for _, v := range rk.Values {
				vec := make([]float32, d)
				callArgs := make([]float32, len(v.Args))
				for c := 0; c < d; c++ {
					for j, a := range v.Args {
						callArgs[j] = env[a][c]
					}
					vec[c] = evalLambdaTileIRApprox(v.Body, callArgs)
				}
				switch v.Reduce {
				case "sum":
					var sum float64
					for c := 0; c < d; c++ {
						sum += float64(vec[c])
					}
					for c := 0; c < d; c++ {
						vec[c] = float32(sum)
					}
				case "max":
					mx := float32(-1e30)
					for c := 0; c < d; c++ {
						if vec[c] > mx {
							mx = vec[c]
						}
					}
					for c := 0; c < d; c++ {
						vec[c] = mx
					}
				}
				env[v.Name] = vec
			}
			for _, o := range rk.Outputs {
				dst, ok := bufs[o.Name]
				if !ok || len(dst) < rows*d {
					dst = make([]float32, rows*d)
					bufs[o.Name] = dst
				}
				valVec := env[o.Value]
				dt := o.DType
				if dt == 0 {
					dt = F32
				}
				for c := 0; c < d; c++ {
					dst[r*d+c] = dt.Round(valVec[c])
				}
			}
		}
		return nil

	case tbStep.GEMM != nil:
		gm := *tbStep.GEMM
		M := pk.M
		if M <= 0 {
			M = pk.Rows
		}
		if len(pk.Args) < 3+len(gm.Extras) {
			return fmt.Errorf("gemm %s has insufficient args %v", gm.Name, pk.Args)
		}
		xName := pk.Args[0]
		wName := pk.Args[1]
		outName := pk.Args[2+len(gm.Extras)]

		gemmInputs := map[string][]float32{
			"x": bufs[xName],
			"w": bufs[wName],
		}
		for idx, ex := range gm.Extras {
			gemmInputs[ex.Name] = bufs[pk.Args[2+idx]]
		}
		stMod, err := GEMMToTileIR(&gm)
		if err != nil {
			return err
		}
		outVals, err := evalTileIRGEMM(stMod, stMod.Functions[0], gemmInputs, M, gm.N, gm.K)
		if err != nil {
			return err
		}
		outDT := gm.Out
		if t, ok := tensorMeta[outName]; ok && t.DType != 0 {
			outDT = DType(t.DType)
		}
		if outDT != 0 {
			for i, v := range outVals {
				outVals[i] = outDT.Round(v)
			}
		}
		bufs[outName] = outVals
		return nil

	default:
		return fmt.Errorf("unsupported stage step")
	}
}

func evalMegakernelTileIR(m *hopper.Megakernel, mod *TileModule, bufs map[string][]float32) error {
	var spec megaSpecJSON
	for _, rawC := range mod.Comments {
		c := cleanCommentLine(rawC)
		if strings.HasPrefix(c, "hopper.mega.spec:") {
			js := strings.TrimSpace(strings.TrimPrefix(c, "hopper.mega.spec:"))
			_ = json.Unmarshal([]byte(js), &spec)
			break
		}
	}

	for i, st := range m.Stages {
		fn := mod.Functions[i]
		var stComments []string
		if i < len(spec.Stages) && len(spec.Stages[i].StageComments) > 0 {
			stComments = spec.Stages[i].StageComments
		} else {
			stMod, err := StepHopperToTileIR(st.Kernel)
			if err != nil {
				return err
			}
			stComments = stMod.Comments
		}
		singleMod := &tileir.Module{
			Name:      fn.Name + "_module",
			Comments:  stComments,
			Functions: []*tileir.Function{fn},
		}

		pk := st.Kernel
		switch {
		case pk.Row != nil:
			rk := *pk.Row
			rows := pk.Rows
			if rows <= 0 {
				rows = pk.M
			}
			stageIn := make(map[string][]float32)
			for _, in := range rk.Inputs {
				bName := in.BufferName()
				src := bufs[bName]
				rounded := make([]float32, len(src))
				dt := DType(in.DType)
				for j, v := range src {
					rounded[j] = dt.Round(v)
				}
				stageIn[bName] = rounded
			}
			for _, o := range rk.Outputs {
				stageIn[o.Name] = make([]float32, rows*rk.D)
			}
			out0, err := EvalTileIRReference(singleMod, stageIn, rows, rk.D, rk.D, 1e-5)
			if err != nil {
				return fmt.Errorf("stage %d (%s) EvalTileIRReference: %w", i, st.Name, err)
			}
			for j, o := range rk.Outputs {
				var rawOut []float32
				if j == 0 {
					rawOut = out0
				} else {
					rawOut = stageIn[o.Name]
				}
				dt := DType(o.DType)
				dst := make([]float32, len(rawOut))
				for k, v := range rawOut {
					dst[k] = dt.Round(v)
				}
				bufs[o.Name] = dst
			}

		case pk.Tile != nil || pk.GEMM != nil:
			tbStep, err := ToTileBridgeStep(pk, 0, false)
			if err != nil {
				return err
			}
			gm := *tbStep.GEMM
			M := pk.M
			if M <= 0 {
				M = pk.Rows
			}
			xName := pk.Args[0]
			wName := pk.Args[1]
			outName := pk.Args[2+len(gm.Extras)]
			stageIn := map[string][]float32{
				"x": bufs[xName],
				"w": bufs[wName],
			}
			for idx, ex := range gm.Extras {
				stageIn[ex.Name] = bufs[pk.Args[2+idx]]
			}
			outVals, err := EvalTileIRReference(singleMod, stageIn, M, gm.N, gm.K, 1e-5)
			if err != nil {
				return fmt.Errorf("stage %d (%s) EvalTileIRReference: %w", i, st.Name, err)
			}
			outDT := gm.Out
			if outDT != 0 {
				for k, v := range outVals {
					outVals[k] = outDT.Round(v)
				}
			}
			bufs[outName] = outVals
		}
	}
	return nil
}

func normalizeBridgeArch(arch hopper.Arch) (hopper.Arch, error) {
	a := strings.TrimSpace(arch)
	switch a {
	case "", "sm_90", hopper.TargetSM90A:
		return hopper.TargetSM90A, nil
	case "sm_100", hopper.TargetSM100A:
		return hopper.TargetSM100A, nil
	case "sm_103", hopper.TargetSM103A:
		return hopper.TargetSM103A, nil
	case "sm_107", hopper.TargetSM107A:
		return hopper.TargetSM107A, nil
	case "sm_120", hopper.TargetSM120A:
		return hopper.TargetSM120A, nil
	case "sm_121", hopper.TargetSM121A:
		return hopper.TargetSM121A, nil
	default:
		if _, err := hopper.SelectArchDispatchPlan(a, nil); err != nil {
			return "", fmt.Errorf("tilebridge: unsupported arch %q: %w", arch, err)
		}
		return a, nil
	}
}

func cleanCommentLine(s string) string {
	s = strings.TrimSpace(s)
	for strings.HasPrefix(s, "//") {
		s = strings.TrimSpace(strings.TrimPrefix(s, "//"))
	}
	return s
}

func formatIntList(vals []int) string {
	if len(vals) == 0 {
		return "none"
	}
	parts := make([]string, len(vals))
	for i, v := range vals {
		parts[i] = strconv.Itoa(v)
	}
	return strings.Join(parts, ",")
}

func parseIntList(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" || s == "none" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

func tensorElemCount(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	n := 1
	for _, d := range shape {
		if d > 0 {
			n *= d
		}
	}
	return n
}

func cloneBufferMap(in map[string][]float32) map[string][]float32 {
	out := make(map[string][]float32, len(in))
	for k, v := range in {
		out[k] = slices.Clone(v)
	}
	return out
}
