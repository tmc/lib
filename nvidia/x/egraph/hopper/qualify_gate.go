package hopper

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/cudatmem"
	"github.com/tmc/lib/nvidia/x/tileir"
)

// QualificationReceipt records the outcome of all host-side formal and
// adversarial checks for a compiled hopper Step or Plan.
type QualificationReceipt struct {
	Name              string
	CUDAValid         bool
	HazardsChecked    bool
	HazardCount       int
	Hazards           []cudahostemu.Hazard
	SASSEquivVerified bool
	TileIRAgreement   bool
	TileIRMaxAbsDiff  float64
	TMEMTraceVerified bool
	TMEMPeakColumns   uint32
}

// Passed reports whether the qualification receipt passed all applicable checks
// with zero concurrency/uniformity hazards.
func (r QualificationReceipt) Passed() bool {
	if !r.CUDAValid {
		return false
	}
	if r.HazardCount != 0 || len(r.Hazards) != 0 {
		return false
	}
	if !r.HazardsChecked && !r.TMEMTraceVerified {
		return false
	}
	if r.TileIRAgreement {
		if math.IsNaN(r.TileIRMaxAbsDiff) || math.IsInf(r.TileIRMaxAbsDiff, 0) || r.TileIRMaxAbsDiff >= 2e-2 {
			return false
		}
	}
	if r.TMEMTraceVerified {
		if r.TMEMPeakColumns == 0 || r.TMEMPeakColumns > 512 {
			return false
		}
	}
	return true
}

// QualifyStep runs the unified host-side static and dynamic qualification gate
// on step:
//  1. Emits and validates the CUDA C++ kernel and launcher source.
//  2. Runs cudahostemu.CheckHazards on a synthetic launch of step, verifying
//     0 shared-memory races, 0 DSMEM-before-wait violations, 0 C7519 WGMMA
//     warpgroup divergences, and 0 named barrier mismatches.
//  3. When step uses NVFP4 dequantization or E4M3 scale conversion, verifies
//     symbolic SASS equivalence via VerifyDequantProgramsSASS and
//     ProveScaleConversionSASS.
//  4. When step is supported by Tile IR lowering (RowKernel, SkinnyGEMM,
//     TileGEMM), lowers step to a *tileir.Module (\x7fTileIR\x00 bytecode) and
//     verifies dual-target numerical agreement against cudahostemu.Run.
func QualifyStep(ctx context.Context, step Step, M, N, K int, eps float32) (*QualificationReceipt, error) {
	if eps <= 0 {
		eps = 1e-5
	}
	switch {
	case step.Row != nil:
		return qualifyRowStep(ctx, *step.Row, step, M, N, K, eps)
	case step.GEMM != nil:
		return qualifySkinnyGEMMStep(ctx, *step.GEMM, step, M, N, K, eps)
	case step.Tile != nil:
		return qualifyTileGEMMStep(ctx, *step.Tile, step, M, N, K, eps)
	case step.Attention != nil:
		return qualifyPagedAttentionStep(ctx, *step.Attention, step, M)
	case step.TopK != nil:
		return qualifyTopKStep(ctx, *step.TopK, step, M)
	case step.Align != nil:
		return qualifyAlignStep(ctx, *step.Align, step, M)
	case step.ExpertSum != nil:
		return qualifyExpertSumStep(ctx, *step.ExpertSum, step, M)
	default:
		return nil, fmt.Errorf("qualify step: empty step")
	}
}

// QualifyPlan runs QualifyStep across every step in plan.Steps(), and when
// plan is megakernel- or PDL-compatible, also verifies that plan.Megakernel
// and plan.EmitPDL succeed cleanly.
func QualifyPlan(ctx context.Context, plan *Plan, M, N, K int, eps float32) ([]QualificationReceipt, error) {
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

// QualifySM100Kernels verifies the sm_100a Blackwell tcgen05 kernel suite
// (TCGen05Cluster2GEMM, TCGen05BlockScaledGEMM, and TCGen05TMEMFlashAttention)
// through CUDA C++ emission, NVFP4/UE4M3 SASS symbolic equivalence, and
// cudatmem.VerifyTrace(cudatmem.SM100A, ...).
func QualifySM100Kernels(ctx context.Context) (*QualificationReceipt, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	receipt := &QualificationReceipt{
		Name: "sm100a_tcgen05_suite",
	}

	c2 := TCGen05Cluster2GEMM{Name: "tcgen05_cluster2_gemm", M: 256, N: 128, K: 32}
	bs := TCGen05BlockScaledGEMM{Name: "tcgen05_block_scaled_gemm", M: 128, N: 128, K: 64}
	fa := TCGen05TMEMFlashAttention{Name: "tcgen05_tmem_flash_attention", M: 128, KV: 64, D: 64}

	for _, item := range []struct {
		name   string
		cudaFn func() (string, error)
		events []cudatmem.TraceEvent
	}{
		{name: c2.Name, cudaFn: c2.CUDA, events: c2.TraceEvents()},
		{name: bs.Name, cudaFn: bs.CUDA, events: bs.TraceEvents()},
		{name: fa.Name, cudaFn: fa.CUDA, events: fa.TraceEvents()},
	} {
		src, err := item.cudaFn()
		if err != nil || strings.TrimSpace(src) == "" {
			return receipt, fmt.Errorf("qualify sm100 %s CUDA: %w", item.name, err)
		}
		sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, item.events)
		if err != nil {
			return receipt, fmt.Errorf("qualify sm100 %s VerifyTrace: %w", item.name, err)
		}
		if sum.MaxAllocatedColumns > receipt.TMEMPeakColumns {
			receipt.TMEMPeakColumns = sum.MaxAllocatedColumns
		}
	}
	receipt.CUDAValid = true
	receipt.TMEMTraceVerified = true

	if err := verifyNVFP4SASS(); err != nil {
		return receipt, fmt.Errorf("qualify sm100 SASS equivalence: %w", err)
	}
	receipt.SASSEquivVerified = true

	return receipt, nil
}

// QualifySM100Suite is an alias for [QualifySM100Kernels].
func QualifySM100Suite(ctx context.Context) (*QualificationReceipt, error) {
	return QualifySM100Kernels(ctx)
}

func verifyNVFP4SASS() error {
	if err := VerifyDequantProgramsSASS(); err != nil {
		return fmt.Errorf("verify dequant programs SASS: %w", err)
	}
	if _, err := ProveScaleConversionSASS(); err != nil {
		return fmt.Errorf("prove scale conversion SASS: %w", err)
	}
	return nil
}

func qualifyRowStep(ctx context.Context, rk RowKernel, step Step, M, N, K int, eps float32) (*QualificationReceipt, error) {
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
		return receipt, fmt.Errorf("row kernel %s CUDA: %w", rk.Name, err)
	}
	if _, err := rk.Launcher(); err != nil {
		return receipt, fmt.Errorf("row kernel %s Launcher: %w", rk.Name, err)
	}
	receipt.CUDAValid = true

	hazRows := min(max(rows, 1), 8)
	grid, block := rk.Grid(hazRows)
	bufBytes := max(hazRows*rk.D*8, 65536)
	args := []cudahostemu.Arg{cudahostemu.IntArg(hazRows)}
	for _, p := range rk.params()[1:] {
		name, ctype := parseParamDecl(p)
		args = append(args, cudahostemu.ZeroBufferArg(name, ctype, bufBytes))
	}
	rep, err := cudahostemu.CheckHazards(ctx, Prelude+"\n"+src, cudahostemu.Launch{
		Kernel: rk.Name,
		Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
		Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
	}, args...)
	if err != nil {
		return receipt, fmt.Errorf("row kernel %s CheckHazards: %w", rk.Name, err)
	}
	receipt.HazardsChecked = true
	receipt.HazardCount = len(rep.Hazards)
	receipt.Hazards = rep.Hazards
	if receipt.HazardCount > 0 {
		return receipt, fmt.Errorf("row kernel %s: detected %d hazards: %+v", rk.Name, receipt.HazardCount, rep.Hazards)
	}

	if canLowerRowToTileIR(rk) {
		dtRep, err := VerifyDualTargetAgreement(ctx, rk, min(max(rows, 1), 4), N, rk.D, eps)
		if err != nil {
			return receipt, fmt.Errorf("row kernel %s VerifyDualTargetAgreement: %w", rk.Name, err)
		}
		receipt.TileIRAgreement = true
		receipt.TileIRMaxAbsDiff = dtRep.MaxAbsDiff
	}

	return receipt, nil
}

func canLowerRowToTileIR(rk RowKernel) bool {
	if rk.Cluster || rk.Slots > 0 || len(rk.Outputs) != 1 {
		return false
	}
	if rk.Outputs[0].Index != "" || rk.Outputs[0].DType != F32 {
		return false
	}
	for _, in := range rk.Inputs {
		if in.DType != F32 || in.Padded > 0 || in.Group > 1 || (in.RowStride != 0 && in.RowStride != rk.D) {
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

func qualifySkinnyGEMMStep(ctx context.Context, sg SkinnyGEMM, step Step, M, N, K int, eps float32) (*QualificationReceipt, error) {
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
	normed, err := sg.norm()
	if err != nil {
		return receipt, fmt.Errorf("skinny gemm %s norm: %w", sg.Name, err)
	}
	src, err := normed.CUDA()
	if err != nil {
		return receipt, fmt.Errorf("skinny gemm %s CUDA: %w", normed.Name, err)
	}
	if _, err := normed.Launcher(); err != nil {
		return receipt, fmt.Errorf("skinny gemm %s Launcher: %w", normed.Name, err)
	}
	receipt.CUDAValid = true

	grid, block, err := normed.Grid()
	if err != nil {
		return receipt, fmt.Errorf("skinny gemm %s Grid: %w", normed.Name, err)
	}
	gx := min(grid[0], 2)
	gy := grid[1]
	bufBytes := max(normed.N*normed.K*2, max(normed.WorkspaceBytes(), 65536))
	var args []cudahostemu.Arg
	var argNames []string
	for _, p := range normed.params() {
		name, ctype := parseParamDecl(p)
		args = append(args, cudahostemu.ZeroBufferArg(name, ctype, bufBytes))
		argNames = append(argNames, name)
	}
	launchStmt := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, 1), dim3(%d, 1, 1), 0, %s<%d>, %s)",
		gx, gy, block[0], normed.Name, m, strings.Join(argNames, ", "))
	rep, err := cudahostemu.CheckHazards(ctx, Prelude+"\n"+src, cudahostemu.Launch{
		Kernel: launchStmt,
	}, args...)
	if err != nil {
		return receipt, fmt.Errorf("skinny gemm %s CheckHazards: %w", normed.Name, err)
	}
	receipt.HazardsChecked = true
	receipt.HazardCount = len(rep.Hazards)
	receipt.Hazards = rep.Hazards
	if receipt.HazardCount > 0 {
		return receipt, fmt.Errorf("skinny gemm %s: detected %d hazards: %+v", normed.Name, receipt.HazardCount, rep.Hazards)
	}

	if normed.Weight == WeightNVFP4 {
		if err := verifyNVFP4SASS(); err != nil {
			return receipt, fmt.Errorf("skinny gemm %s SASS: %w", normed.Name, err)
		}
		receipt.SASSEquivVerified = true
	}

	if err := verifyStepTileIRBytecode(normed); err != nil {
		return receipt, fmt.Errorf("skinny gemm %s TileIR bytecode: %w", normed.Name, err)
	}
	surrogate := SkinnyGEMM{
		Name:     normed.Name,
		N:        min(normed.N, 64),
		K:        min(normed.K, 256),
		MaxM:     min(max(m, 1), 4),
		Weight:   WeightBF16,
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
	dtRep, err := VerifyDualTargetAgreement(ctx, surrogate, min(max(m, 1), 4), surrogate.N, surrogate.K, eps)
	if err != nil {
		return receipt, fmt.Errorf("skinny gemm %s VerifyDualTargetAgreement: %w", normed.Name, err)
	}
	receipt.TileIRAgreement = true
	receipt.TileIRMaxAbsDiff = dtRep.MaxAbsDiff

	return receipt, nil
}

func qualifyTileGEMMStep(ctx context.Context, tg TileGEMM, step Step, M, N, K int, eps float32) (*QualificationReceipt, error) {
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
	normed, err := tg.norm()
	if err != nil {
		return receipt, fmt.Errorf("tile gemm %s norm: %w", tg.Name, err)
	}
	src, err := normed.CUDA()
	if err != nil {
		return receipt, fmt.Errorf("tile gemm %s CUDA: %w", normed.Name, err)
	}
	if _, err := normed.Launcher(); err != nil {
		return receipt, fmt.Errorf("tile gemm %s Launcher: %w", normed.Name, err)
	}
	receipt.CUDAValid = true

	experts := max(step.Experts, 1)
	grid, block := normed.Grid(m, experts)
	gx, gy, gz := min(grid[0], 1), min(grid[1], 1), min(grid[2], 1)
	if normed.StreamK > 0 {
		gx, gy, gz = normed.StreamK, 1, 1
	}
	bufBytes := max(max(m, normed.BM)*normed.N*4, max(normed.N*normed.K*2*experts, max(normed.WorkspaceBytes(), 65536)))
	args := []cudahostemu.Arg{cudahostemu.IntArg(m)}
	for _, p := range normed.params()[1:] {
		name, ctype := parseParamDecl(p)
		if name == "offsets" && normed.Grouped {
			offBytes := make([]byte, (experts+1)*4)
			for e := 0; e <= experts; e++ {
				binary.LittleEndian.PutUint32(offBytes[4*e:], uint32(e*m))
			}
			args = append(args, cudahostemu.BufferArg(name, ctype, offBytes))
			continue
		}
		args = append(args, cudahostemu.ZeroBufferArg(name, ctype, bufBytes))
	}
	rep, err := cudahostemu.CheckHazards(ctx, Prelude+"\n"+src, cudahostemu.Launch{
		Kernel:      normed.Name,
		Grid:        cudahostemu.Dim3{X: gx, Y: gy, Z: gz},
		Block:       cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
		SharedBytes: max(normed.SharedBytes(), 65536),
	}, args...)
	if err != nil {
		return receipt, fmt.Errorf("tile gemm %s CheckHazards: %w", normed.Name, err)
	}
	receipt.HazardsChecked = true
	receipt.HazardCount = len(rep.Hazards)
	receipt.Hazards = rep.Hazards
	if receipt.HazardCount > 0 {
		return receipt, fmt.Errorf("tile gemm %s: detected %d hazards: %+v", normed.Name, receipt.HazardCount, rep.Hazards)
	}

	if normed.Weight == WeightNVFP4 {
		if err := verifyNVFP4SASS(); err != nil {
			return receipt, fmt.Errorf("tile gemm %s SASS: %w", normed.Name, err)
		}
		receipt.SASSEquivVerified = true
	}

	if err := verifyStepTileIRBytecode(normed); err != nil {
		return receipt, fmt.Errorf("tile gemm %s TileIR bytecode: %w", normed.Name, err)
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
	surrogate := TileGEMM{
		Name:     normed.Name,
		N:        surrogateN,
		K:        surrogateK,
		BM:       64,
		BN:       64,
		Stages:   2,
		Weight:   WeightBF16,
		Out:      normed.Out,
		GLU:      normed.GLU,
		Epilogue: normed.Epilogue,
		Extras:   normed.Extras,
	}
	dtRep, err := VerifyDualTargetAgreement(ctx, surrogate, 16, surrogate.N, surrogate.K, eps)
	if err != nil {
		return receipt, fmt.Errorf("tile gemm %s VerifyDualTargetAgreement: %w", normed.Name, err)
	}
	receipt.TileIRAgreement = true
	receipt.TileIRMaxAbsDiff = dtRep.MaxAbsDiff

	return receipt, nil
}

func qualifyPagedAttentionStep(ctx context.Context, pa PagedAttention, step Step, M int) (*QualificationReceipt, error) {
	if pa.Name == "" {
		pa.Name = "paged_attention"
	}
	receipt := &QualificationReceipt{Name: pa.Name}
	normed, _, err := pa.norm()
	if err != nil {
		return receipt, fmt.Errorf("paged attention %s norm: %w", pa.Name, err)
	}
	src, err := normed.CUDA()
	if err != nil {
		return receipt, fmt.Errorf("paged attention %s CUDA: %w", normed.Name, err)
	}
	if _, err := normed.Launcher(); err != nil {
		return receipt, fmt.Errorf("paged attention %s Launcher: %w", normed.Name, err)
	}
	receipt.CUDAValid = true

	batch := max(step.Batch, 1)
	if batch <= 0 && M > 0 {
		batch = 1
	}
	maxKeys := step.MaxKeys
	if maxKeys <= 0 {
		maxKeys = min(normed.Chunk, 64)
	}
	maxPages := step.MaxPages
	if maxPages <= 0 {
		maxPages = (maxKeys + normed.Page - 1) / normed.Page
		if maxPages <= 0 {
			maxPages = 4
		}
	}

	grid, block := normed.Grid(batch, maxKeys)
	gx := 1
	cluster := 1
	if normed.ClusterCombine {
		gx = ClusterSize
		cluster = ClusterSize
	}
	gy := min(max(grid[1], 1), 1)
	gz := 1

	bufBytes := max(normed.WorkspaceBytes(batch, maxKeys), 65536)
	seqBytes := make([]byte, max(batch, 1)*4)
	for b := 0; b < max(batch, 1); b++ {
		binary.LittleEndian.PutUint32(seqBytes[4*b:], uint32(maxKeys))
	}
	oneBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(oneBytes, math.Float32bits(1.0))

	var args []cudahostemu.Arg
	for _, p := range normed.params() {
		name, ctype := parseParamDecl(p)
		switch name {
		case "max_pages":
			args = append(args, cudahostemu.IntArg(maxPages))
		case "chunk":
			args = append(args, cudahostemu.IntArg(normed.Chunk))
		case "seqlens":
			args = append(args, cudahostemu.BufferArg(name, ctype, seqBytes))
		case "k_scale", "v_scale":
			args = append(args, cudahostemu.BufferArg(name, ctype, oneBytes))
		default:
			args = append(args, cudahostemu.ZeroBufferArg(name, ctype, bufBytes))
		}
	}

	rep, err := cudahostemu.CheckHazards(ctx, Prelude+"\n"+src, cudahostemu.Launch{
		Kernel:      normed.Name,
		Grid:        cudahostemu.Dim3{X: gx, Y: gy, Z: gz},
		Block:       cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
		Cluster:     cluster,
		SharedBytes: max(normed.SharedBytes(), 65536),
	}, args...)
	if err != nil {
		return receipt, fmt.Errorf("paged attention %s CheckHazards: %w", normed.Name, err)
	}
	receipt.HazardsChecked = true
	receipt.HazardCount = len(rep.Hazards)
	receipt.Hazards = rep.Hazards
	if receipt.HazardCount > 0 {
		return receipt, fmt.Errorf("paged attention %s: detected %d hazards: %+v", normed.Name, receipt.HazardCount, rep.Hazards)
	}

	return receipt, nil
}

func qualifyTopKStep(ctx context.Context, tk RowTopK, step Step, M int) (*QualificationReceipt, error) {
	if tk.Name == "" {
		tk.Name = "row_topk"
	}
	receipt := &QualificationReceipt{Name: tk.Name}
	normed, err := tk.norm()
	if err != nil {
		return receipt, fmt.Errorf("row topk %s norm: %w", tk.Name, err)
	}
	src, err := normed.CUDA()
	if err != nil {
		return receipt, fmt.Errorf("row topk %s CUDA: %w", normed.Name, err)
	}
	if _, err := normed.Launcher(); err != nil {
		return receipt, fmt.Errorf("row topk %s Launcher: %w", normed.Name, err)
	}
	receipt.CUDAValid = true

	rows := max(step.Rows, max(M, 1))
	hazRows := min(rows, 8)
	grid, block := normed.Grid(hazRows)
	args := []cudahostemu.Arg{cudahostemu.IntArg(hazRows)}
	for _, p := range normed.params()[1:] {
		name, ctype := parseParamDecl(p)
		args = append(args, cudahostemu.ZeroBufferArg(name, ctype, 65536))
	}
	rep, err := cudahostemu.CheckHazards(ctx, Prelude+"\n"+src, cudahostemu.Launch{
		Kernel: normed.Name,
		Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
		Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
	}, args...)
	if err != nil {
		return receipt, fmt.Errorf("row topk %s CheckHazards: %w", normed.Name, err)
	}
	receipt.HazardsChecked = true
	receipt.HazardCount = len(rep.Hazards)
	receipt.Hazards = rep.Hazards
	if receipt.HazardCount > 0 {
		return receipt, fmt.Errorf("row topk %s: detected %d hazards: %+v", normed.Name, receipt.HazardCount, rep.Hazards)
	}
	return receipt, nil
}

func qualifyAlignStep(ctx context.Context, al ExpertAlign, step Step, M int) (*QualificationReceipt, error) {
	if al.Name == "" {
		al.Name = "expert_align"
	}
	receipt := &QualificationReceipt{Name: al.Name}
	normed, err := al.norm()
	if err != nil {
		return receipt, fmt.Errorf("expert align %s norm: %w", al.Name, err)
	}
	src, err := normed.CUDA()
	if err != nil {
		return receipt, fmt.Errorf("expert align %s CUDA: %w", normed.Name, err)
	}
	if _, err := normed.Launcher(); err != nil {
		return receipt, fmt.Errorf("expert align %s Launcher: %w", normed.Name, err)
	}
	receipt.CUDAValid = true

	tokens := min(max(step.Rows, max(M, 1)), 4)
	args := []cudahostemu.Arg{cudahostemu.IntArg(tokens)}
	for _, p := range normed.params()[1:] {
		name, ctype := parseParamDecl(p)
		args = append(args, cudahostemu.ZeroBufferArg(name, ctype, 65536))
	}
	rep, err := cudahostemu.CheckHazards(ctx, Prelude+"\n"+src, cudahostemu.Launch{
		Kernel: normed.Name,
		Grid:   cudahostemu.Dim3{X: 1, Y: 1, Z: 1},
		Block:  cudahostemu.Dim3{X: 32 * normed.Warps, Y: 1, Z: 1},
	}, args...)
	if err != nil {
		return receipt, fmt.Errorf("expert align %s CheckHazards: %w", normed.Name, err)
	}
	receipt.HazardsChecked = true
	receipt.HazardCount = len(rep.Hazards)
	receipt.Hazards = rep.Hazards
	if receipt.HazardCount > 0 {
		return receipt, fmt.Errorf("expert align %s: detected %d hazards: %+v", normed.Name, receipt.HazardCount, rep.Hazards)
	}
	return receipt, nil
}

func qualifyExpertSumStep(ctx context.Context, es ExpertSum, step Step, M int) (*QualificationReceipt, error) {
	if es.Name == "" {
		es.Name = "expert_sum"
	}
	receipt := &QualificationReceipt{Name: es.Name}
	normed, err := es.norm()
	if err != nil {
		return receipt, fmt.Errorf("expert sum %s norm: %w", es.Name, err)
	}
	src, err := normed.CUDA()
	if err != nil {
		return receipt, fmt.Errorf("expert sum %s CUDA: %w", normed.Name, err)
	}
	if _, err := normed.Launcher(); err != nil {
		return receipt, fmt.Errorf("expert sum %s Launcher: %w", normed.Name, err)
	}
	receipt.CUDAValid = true

	tokens := min(max(step.Rows, max(M, 1)), 2)
	grid, block := normed.Grid(tokens)
	args := []cudahostemu.Arg{cudahostemu.IntArg(tokens)}
	for _, p := range normed.params()[1:] {
		name, ctype := parseParamDecl(p)
		args = append(args, cudahostemu.ZeroBufferArg(name, ctype, 65536))
	}
	rep, err := cudahostemu.CheckHazards(ctx, Prelude+"\n"+src, cudahostemu.Launch{
		Kernel: normed.Name,
		Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
		Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
	}, args...)
	if err != nil {
		return receipt, fmt.Errorf("expert sum %s CheckHazards: %w", normed.Name, err)
	}
	receipt.HazardsChecked = true
	receipt.HazardCount = len(rep.Hazards)
	receipt.Hazards = rep.Hazards
	if receipt.HazardCount > 0 {
		return receipt, fmt.Errorf("expert sum %s: detected %d hazards: %+v", normed.Name, receipt.HazardCount, rep.Hazards)
	}
	return receipt, nil
}

func verifyStepTileIRBytecode(step TileIRStep) error {
	mod, err := StepToTileIR(step)
	if err != nil {
		return err
	}
	text := FormatTileIR(mod)
	parsed, err := ParseTileIR(text)
	if err != nil {
		return err
	}
	bc, err := EncodeTileIR(parsed)
	if err != nil {
		return err
	}
	if !tileir.IsTileIRBytecode(bc) {
		return fmt.Errorf("missing TileIR bytecode magic header")
	}
	return nil
}

func parseParamDecl(decl string) (name, ctype string) {
	fields := strings.Fields(decl)
	if len(fields) == 0 {
		return "arg", "uint8_t"
	}
	name = strings.TrimLeft(fields[len(fields)-1], "*")
	if idx := strings.LastIndexByte(decl, '*'); idx >= 0 {
		prefix := decl[:idx]
		var kept []string
		for _, f := range strings.Fields(prefix) {
			if f == "const" || f == "volatile" || f == "__restrict__" || f == "__restrict" {
				continue
			}
			kept = append(kept, f)
		}
		if len(kept) > 0 {
			ctype = strings.Join(kept, " ")
		}
	}
	if ctype == "" {
		ctype = "uint8_t"
	}
	return name, ctype
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

func canEmitPDL(p *Plan) bool {
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

func canEmitMegakernel(p *Plan) bool {
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
