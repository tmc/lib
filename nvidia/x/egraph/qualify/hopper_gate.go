package qualify

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/tilebridge"
)

// QualifyStep runs the unified host-side static and dynamic qualification gate
// on step.
func QualifyStep(ctx context.Context, step Step, M, N, K int, eps float32) (*QualificationReceipt, error) {
	if eps <= 0 {
		eps = 1e-5
	}
	spec, receipt, err := buildQualifyStepSpec(step, M, N, K, eps)
	if err != nil {
		return receipt, err
	}
	return QualifyStepSpec(ctx, spec)
}

// QualifyPlan runs [QualifyStep] across every step in plan.Steps(), and when
// plan is megakernel- or PDL-compatible, also verifies that plan.Megakernel
// and plan.EmitPDL succeed cleanly.
func QualifyPlan(ctx context.Context, plan *hopper.Plan, M, N, K int, eps float32) ([]QualificationReceipt, error) {
	if plan == nil || len(plan.Kernels) == 0 {
		return nil, fmt.Errorf("qualify plan: empty plan")
	}
	if _, err := plan.Source(); err != nil {
		return nil, fmt.Errorf("qualify plan %s source: %w", plan.Name, err)
	}

	steps := plan.Steps()
	receipts := make([]QualificationReceipt, 0, len(steps))
	for i, st := range steps {
		r, err := QualifyStep(ctx, st, M, N, K, eps)
		if err != nil {
			return receipts, fmt.Errorf("qualify plan %s step %d (%s): %w", plan.Name, i, stepName(st), err)
		}
		receipts = append(receipts, *r)
	}

	if canEmitPDL(plan) {
		pdlSrc, err := plan.EmitPDL(plan.Name)
		if err != nil {
			return receipts, fmt.Errorf("qualify plan %s EmitPDL: %w", plan.Name, err)
		}
		if code, err := pdlSrc.CUDA(); err != nil || strings.TrimSpace(code) == "" {
			return receipts, fmt.Errorf("qualify plan %s EmitPDL CUDA: %w", plan.Name, err)
		}
	}

	if canEmitMegakernel(plan) {
		mega, err := plan.Megakernel("")
		if err != nil {
			return receipts, fmt.Errorf("qualify plan %s Megakernel: %w", plan.Name, err)
		}
		if code, err := mega.Source(); err != nil || strings.TrimSpace(code) == "" {
			return receipts, fmt.Errorf("qualify plan %s Megakernel Source: %w", plan.Name, err)
		}
	}

	return receipts, nil
}

func buildQualifyStepSpec(step Step, M, N, K int, eps float32) (KernelStepSpec, *QualificationReceipt, error) {
	switch {
	case step.Row != nil:
		rk := *step.Row
		if rk.Name == "" {
			rk.Name = "row_kernel"
		}
		if rk.D <= 0 {
			switch {
			case K > 0:
				rk.D = K
			case N > 0:
				rk.D = N
			default:
				rk.D = 64
			}
		}
		rows := step.Rows
		if rows <= 0 {
			rows = M
		}
		if rows <= 0 {
			rows = 4
		}
		receipt := &QualificationReceipt{Name: rk.Name}
		src, err := rk.CUDA()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("row kernel %s CUDA: %w", rk.Name, err)
		}
		launcher, err := rk.Launcher()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("row kernel %s Launcher: %w", rk.Name, err)
		}
		hazRows := min(max(rows, 1), 8)
		grid, block := rk.Grid(hazRows)
		spec := KernelStepSpec{
			Name:            rk.Name,
			Kind:            "row kernel",
			CUDASource:      hopper.Prelude + "\n" + src,
			LauncherSource:  launcher,
			Launch:          cudahostemu.Launch{Kernel: rk.Name, Grid: cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]}, Block: cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]}},
			LeadingIntArgs:  []int{hazRows},
			ParamDecls:      rk.Params()[1:],
			DefaultBufBytes: max(hazRows*rk.D*8, 65536),
			Eps:             eps,
		}
		if canLowerRowToTileIR(rk) {
			dualM := min(max(rows, 1), 4)
			tb, err := tilebridge.ToTileBridgeStep(rk, dualM, true)
			if err != nil {
				return KernelStepSpec{}, receipt, err
			}
			spec.TileIRDualTarget = &tb
			spec.DualM, spec.DualN, spec.DualK = dualM, N, rk.D
		}
		return spec, receipt, nil

	case step.GEMM != nil:
		sg := *step.GEMM
		if sg.Name == "" {
			sg.Name = "skinny_gemm"
		}
		if sg.N <= 0 && N > 0 {
			sg.N = N
		}
		if sg.K <= 0 && K > 0 {
			sg.K = K
		}
		m := step.M
		if m <= 0 {
			m = M
		}
		if m <= 0 {
			m = 4
		}
		if sg.MaxM <= 0 {
			sg.MaxM = min(max(m, 4), 16)
		}
		if m > sg.MaxM {
			m = sg.MaxM
		}
		receipt := &QualificationReceipt{Name: sg.Name}
		normed, err := sg.Norm()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("skinny gemm %s norm: %w", sg.Name, err)
		}
		src, err := normed.CUDA()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("skinny gemm %s CUDA: %w", normed.Name, err)
		}
		launcher, err := normed.Launcher()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("skinny gemm %s Launcher: %w", normed.Name, err)
		}
		grid, block, err := normed.Grid()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("skinny gemm %s Grid: %w", normed.Name, err)
		}
		tbBC, err := tilebridge.ToTileBridgeStep(normed, 0, false)
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("skinny gemm %s TileIR: %w", normed.Name, err)
		}
		surrogate := hopper.SkinnyGEMM{
			Name:     normed.Name,
			N:        min(normed.N, 64),
			K:        min(normed.K, 256),
			MaxM:     min(max(m, 1), 4),
			Weight:   hopper.WeightBF16,
			Out:      normed.Out,
			Epilogue: normed.Epilogue,
			Extras:   normed.Extras,
		}
		if surrogate.K < 256 || surrogate.K%256 != 0 {
			surrogate.K = 256
		}
		if surrogate.N <= 0 || surrogate.N%16 != 0 {
			surrogate.N = 64
		}
		dualM := min(max(m, 1), 4)
		tbDual, err := tilebridge.ToTileBridgeStep(surrogate, dualM, true)
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("skinny gemm %s surrogate TileIR: %w", normed.Name, err)
		}
		return KernelStepSpec{
			Name:             normed.Name,
			Kind:             "skinny gemm",
			CUDASource:       hopper.Prelude + "\n" + src,
			LauncherSource:   launcher,
			Launch:           cudahostemu.Launch{Grid: cudahostemu.Dim3{X: min(grid[0], 2), Y: grid[1], Z: 1}, Block: cudahostemu.Dim3{X: block[0], Y: 1, Z: 1}},
			TemplateM:        m,
			ParamDecls:       normed.Params(),
			DefaultBufBytes:  max(normed.N*normed.K*2, max(normed.WorkspaceBytes(), 65536)),
			CheckNVFP4SASS:   normed.Weight == hopper.WeightNVFP4,
			TileIRBytecode:   &tbBC,
			TileIRDualTarget: &tbDual,
			DualM:            dualM,
			DualN:            surrogate.N,
			DualK:            surrogate.K,
			Eps:              eps,
		}, receipt, nil

	case step.Tile != nil:
		tg := *step.Tile
		if tg.Name == "" {
			tg.Name = "tile_gemm"
		}
		if tg.N <= 0 && N > 0 {
			tg.N = N
		}
		if tg.K <= 0 && K > 0 {
			tg.K = K
		}
		m := step.M
		if m <= 0 {
			m = M
		}
		if m <= 0 {
			m = 16
		}
		receipt := &QualificationReceipt{Name: tg.Name}
		normed, err := tg.Norm()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("tile gemm %s norm: %w", tg.Name, err)
		}
		src, err := normed.CUDA()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("tile gemm %s CUDA: %w", normed.Name, err)
		}
		launcher, err := normed.Launcher()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("tile gemm %s Launcher: %w", normed.Name, err)
		}
		experts := max(step.Experts, 1)
		grid, block := normed.Grid(m, experts)
		gx, gy, gz := min(grid[0], 1), min(grid[1], 1), min(grid[2], 1)
		if normed.StreamK > 0 {
			gx, gy, gz = normed.StreamK, 1, 1
		}
		var namedBufs map[string][]byte
		if normed.Grouped {
			offBytes := make([]byte, (experts+1)*4)
			for e := 0; e <= experts; e++ {
				binary.LittleEndian.PutUint32(offBytes[4*e:], uint32(e*m))
			}
			namedBufs = map[string][]byte{"offsets": offBytes}
		}
		tbBC, err := tilebridge.ToTileBridgeStep(normed, 0, false)
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("tile gemm %s TileIR: %w", normed.Name, err)
		}
		surrogateN := min(normed.N, 128)
		if surrogateN < 64 || surrogateN%64 != 0 {
			surrogateN = 64
		}
		if normed.GLU && surrogateN < 128 {
			surrogateN = 128
		}
		surrogateK := min(normed.K, 64)
		if surrogateK < 64 || surrogateK%64 != 0 {
			surrogateK = 64
		}
		surrogate := hopper.TileGEMM{
			Name:     normed.Name,
			N:        surrogateN,
			K:        surrogateK,
			BM:       64,
			BN:       64,
			Stages:   2,
			Weight:   hopper.WeightBF16,
			Out:      normed.Out,
			GLU:      normed.GLU,
			Epilogue: normed.Epilogue,
			Extras:   normed.Extras,
		}
		tbDual, err := tilebridge.ToTileBridgeStep(surrogate, 16, true)
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("tile gemm %s surrogate TileIR: %w", normed.Name, err)
		}
		return KernelStepSpec{
			Name:             normed.Name,
			Kind:             "tile gemm",
			CUDASource:       hopper.Prelude + "\n" + src,
			LauncherSource:   launcher,
			Launch:           cudahostemu.Launch{Kernel: normed.Name, Grid: cudahostemu.Dim3{X: gx, Y: gy, Z: gz}, Block: cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]}, SharedBytes: max(normed.SharedBytes(), 65536)},
			LeadingIntArgs:   []int{m},
			ParamDecls:       normed.Params()[1:],
			NamedBuffers:     namedBufs,
			DefaultBufBytes:  max(max(m, normed.BM)*normed.N*4, max(normed.N*normed.K*2*experts, max(normed.WorkspaceBytes(), 65536))),
			CheckNVFP4SASS:   normed.Weight == hopper.WeightNVFP4,
			TileIRBytecode:   &tbBC,
			TileIRDualTarget: &tbDual,
			DualM:            16,
			DualN:            surrogate.N,
			DualK:            surrogate.K,
			Eps:              eps,
		}, receipt, nil

	case step.Attention != nil:
		pa := *step.Attention
		if pa.Name == "" {
			pa.Name = "paged_attention"
		}
		receipt := &QualificationReceipt{Name: pa.Name}
		normed, err := pa.Norm()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("paged attention %s norm: %w", pa.Name, err)
		}
		src, err := normed.CUDA()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("paged attention %s CUDA: %w", normed.Name, err)
		}
		launcher, err := normed.Launcher()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("paged attention %s Launcher: %w", normed.Name, err)
		}
		batch := max(step.Batch, 1)
		maxKeys := step.MaxKeys
		if maxKeys <= 0 {
			maxKeys = min(normed.Chunk, 64)
		}
		maxPages := step.MaxPages
		if maxPages <= 0 {
			maxPages = max((maxKeys+normed.Page-1)/normed.Page, 4)
		}
		grid, block := normed.Grid(batch, maxKeys)
		gx, cluster := 1, 1
		if normed.ClusterCombine {
			gx, cluster = hopper.ClusterSize, hopper.ClusterSize
		}
		seqBytes := make([]byte, max(batch, 1)*4)
		for b := 0; b < max(batch, 1); b++ {
			binary.LittleEndian.PutUint32(seqBytes[4*b:], uint32(maxKeys))
		}
		oneBytes := make([]byte, 4)
		binary.LittleEndian.PutUint32(oneBytes, math.Float32bits(1.0))
		return KernelStepSpec{
			Name:            normed.Name,
			Kind:            "paged attention",
			CUDASource:      hopper.Prelude + "\n" + src,
			LauncherSource:  launcher,
			Launch:          cudahostemu.Launch{Kernel: normed.Name, Grid: cudahostemu.Dim3{X: gx, Y: min(max(grid[1], 1), 1), Z: 1}, Block: cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]}, Cluster: cluster, SharedBytes: max(normed.SharedBytes(), 65536)},
			ParamDecls:      normed.Params(),
			NamedInts:       map[string]int{"max_pages": maxPages, "chunk": normed.Chunk},
			NamedBuffers:    map[string][]byte{"seqlens": seqBytes, "k_scale": oneBytes, "v_scale": oneBytes},
			DefaultBufBytes: max(normed.WorkspaceBytes(batch, maxKeys), 65536),
		}, receipt, nil

	case step.TopK != nil:
		tk := *step.TopK
		if tk.Name == "" {
			tk.Name = "row_topk"
		}
		receipt := &QualificationReceipt{Name: tk.Name}
		normed, err := tk.Norm()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("row topk %s norm: %w", tk.Name, err)
		}
		src, err := normed.CUDA()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("row topk %s CUDA: %w", normed.Name, err)
		}
		launcher, err := normed.Launcher()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("row topk %s Launcher: %w", normed.Name, err)
		}
		hazRows := min(max(step.Rows, max(M, 1)), 8)
		grid, block := normed.Grid(hazRows)
		return KernelStepSpec{
			Name:            normed.Name,
			Kind:            "row topk",
			CUDASource:      hopper.Prelude + "\n" + src,
			LauncherSource:  launcher,
			Launch:          cudahostemu.Launch{Kernel: normed.Name, Grid: cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]}, Block: cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]}},
			LeadingIntArgs:  []int{hazRows},
			ParamDecls:      normed.Params()[1:],
			DefaultBufBytes: 65536,
		}, receipt, nil

	case step.Align != nil:
		al := *step.Align
		if al.Name == "" {
			al.Name = "expert_align"
		}
		receipt := &QualificationReceipt{Name: al.Name}
		normed, err := al.Norm()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("expert align %s norm: %w", al.Name, err)
		}
		src, err := normed.CUDA()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("expert align %s CUDA: %w", normed.Name, err)
		}
		launcher, err := normed.Launcher()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("expert align %s Launcher: %w", normed.Name, err)
		}
		tokens := min(max(step.Rows, max(M, 1)), 4)
		return KernelStepSpec{
			Name:            normed.Name,
			Kind:            "expert align",
			CUDASource:      hopper.Prelude + "\n" + src,
			LauncherSource:  launcher,
			Launch:          cudahostemu.Launch{Kernel: normed.Name, Grid: cudahostemu.Dim3{X: 1, Y: 1, Z: 1}, Block: cudahostemu.Dim3{X: 32 * normed.Warps, Y: 1, Z: 1}},
			LeadingIntArgs:  []int{tokens},
			ParamDecls:      normed.Params()[1:],
			DefaultBufBytes: 65536,
		}, receipt, nil

	case step.ExpertSum != nil:
		es := *step.ExpertSum
		if es.Name == "" {
			es.Name = "expert_sum"
		}
		receipt := &QualificationReceipt{Name: es.Name}
		normed, err := es.Norm()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("expert sum %s norm: %w", es.Name, err)
		}
		src, err := normed.CUDA()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("expert sum %s CUDA: %w", normed.Name, err)
		}
		launcher, err := normed.Launcher()
		if err != nil {
			return KernelStepSpec{}, receipt, fmt.Errorf("expert sum %s Launcher: %w", normed.Name, err)
		}
		tokens := min(max(step.Rows, max(M, 1)), 2)
		grid, block := normed.Grid(tokens)
		return KernelStepSpec{
			Name:            normed.Name,
			Kind:            "expert sum",
			CUDASource:      hopper.Prelude + "\n" + src,
			LauncherSource:  launcher,
			Launch:          cudahostemu.Launch{Kernel: normed.Name, Grid: cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]}, Block: cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]}},
			LeadingIntArgs:  []int{tokens},
			ParamDecls:      normed.Params()[1:],
			DefaultBufBytes: 65536,
		}, receipt, nil

	default:
		return KernelStepSpec{}, nil, fmt.Errorf("qualify step: empty step")
	}
}

func canLowerRowToTileIR(rk hopper.RowKernel) bool {
	if rk.Cluster || rk.Slots > 0 || len(rk.Outputs) != 1 {
		return false
	}
	if rk.Outputs[0].Index != "" || rk.Outputs[0].DType != hopper.F32 {
		return false
	}
	for _, in := range rk.Inputs {
		if in.DType != hopper.F32 || in.Padded > 0 || in.Group > 1 || (in.RowStride != 0 && in.RowStride != rk.D) {
			return false
		}
	}
	for _, v := range rk.Values {
		for _, a := range v.Args {
			if a == "col" || a == "sgn" || strings.HasPrefix(a, "~") {
				return false
			}
		}
	}
	return true
}

func stepName(st Step) string {
	switch {
	case st.Row != nil:
		return st.Row.Name
	case st.GEMM != nil:
		return st.GEMM.Name
	case st.Tile != nil:
		return st.Tile.Name
	case st.Attention != nil:
		return st.Attention.Name
	case st.TopK != nil:
		return st.TopK.Name
	case st.Align != nil:
		return st.Align.Name
	case st.ExpertSum != nil:
		return st.ExpertSum.Name
	default:
		return "unknown"
	}
}

func canEmitPDL(p *hopper.Plan) bool {
	if p == nil || len(p.Kernels) == 0 {
		return false
	}
	for _, k := range p.Kernels {
		if k.Attention != nil {
			return false
		}
		if k.Row != nil && k.Row.Cluster {
			return false
		}
	}
	return true
}

func canEmitMegakernel(p *hopper.Plan) bool {
	if !canEmitPDL(p) {
		return false
	}
	for _, k := range p.Kernels {
		if k.Tile != nil && k.Tile.WarpSpecialized {
			return false
		}
		if k.GEMM != nil && k.GEMM.Warps != 0 && k.GEMM.Warps != 8 {
			return false
		}
	}
	return true
}
