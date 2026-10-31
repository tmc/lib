package qualify

import (
	"context"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

type mockPTXASCompilerFunc func(ctx context.Context, cudaSource, arch string) (ptxasinfo.Report, error)

func (f mockPTXASCompilerFunc) CompilePTXAS(ctx context.Context, cudaSource, arch string) (ptxasinfo.Report, error) {
	return f(ctx, cudaSource, arch)
}

type mockSASSAnalyzerFunc func(sassText string) ([]StageSASSMetrics, error)

func (f mockSASSAnalyzerFunc) ParseSASSDisassembly(sassText string) ([]StageSASSMetrics, error) {
	return f(sassText)
}

func parseCleanPTXASReport(t *testing.T, fnNames ...string) ptxasinfo.Report {
	t.Helper()
	var b strings.Builder
	b.WriteString("ptxas info    : 0 bytes gmem\n")
	for _, name := range fnNames {
		b.WriteString("ptxas info    : Compiling entry function '" + name + "' for 'sm_90a'\n")
		b.WriteString("ptxas info    : Function properties for " + name + "\n")
		b.WriteString("    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n")
		b.WriteString("ptxas info    : Used 64 registers, 16384 bytes smem, 384 bytes cmem[0]\n")
	}
	rep, err := ptxasinfo.Parse(strings.NewReader(b.String()))
	if err != nil {
		t.Fatalf("ptxasinfo.Parse: %v", err)
	}
	return *rep
}

func TestSASSLoopParseSASSDisassembly(t *testing.T) {
	const rawSASS = `
	//--------------------- .text.tile_unswizzled --------------------------
	.global tile_unswizzled
	/*0000*/ LDG.E.128 R4, [R2] ;
	/*0010*/ LDG.E.U8 R8, [R10] ;
	/*0020*/ STS.128 [R12], R4 ;
	/*0030*/ LDS.128 R16, [R14] ;
	/*0040*/ STL [R20], R16 ;
	/*0050*/ LDL R22, [R20] ;
	/*0060*/ WARPGROUP.DEPBAR.LE 0x0 ; {!8}
	/*0070*/ HMMA.16816.F32 R24, R16, R22, R24 ;
	/*0080*/ EXIT ;

	//--------------------- .text.tile_swizzled --------------------------
	.global tile_swizzled
	/*0000*/ LDG.E.128 R4, [R2] ;
	/*0010*/ LOP3.LUT R12, R10, 0x3c, R11, 0x3c ; // swizzled shared address
	/*0020*/ STS.128 [R12], R4 ;
	/*0030*/ LDS.128 R16, [R12] ;
	/*0040*/ HMMA.16816.F32 R24, R16, R4, R24 ;
	/*0050*/ EXIT ;
`
	metrics, err := ParseSASSDisassembly(rawSASS)
	if err != nil {
		t.Fatalf("ParseSASSDisassembly: %v", err)
	}
	if len(metrics) != 2 {
		t.Fatalf("got %d stage metrics, want 2", len(metrics))
	}

	m0 := metrics[0]
	if m0.StageName != "tile_unswizzled" {
		t.Errorf("m0.StageName = %q, want tile_unswizzled", m0.StageName)
	}
	if m0.TotalInsts != 9 {
		t.Errorf("m0.TotalInsts = %d, want 9", m0.TotalInsts)
	}
	if !m0.SharedBankConflictRisk {
		t.Errorf("m0.SharedBankConflictRisk = false, want true for unswizzled STS.128/LDS.128")
	}
	if m0.UncoalescedGlobalLoads != 1 {
		t.Errorf("m0.UncoalescedGlobalLoads = %d, want 1", m0.UncoalescedGlobalLoads)
	}
	if m0.LocalMemInsts != 2 {
		t.Errorf("m0.LocalMemInsts = %d, want 2 (STL + LDL)", m0.LocalMemInsts)
	}
	if m0.TensorPipeInsts != 1 {
		t.Errorf("m0.TensorPipeInsts = %d, want 1", m0.TensorPipeInsts)
	}
	if m0.WaitStallWeight < 20 {
		t.Errorf("m0.WaitStallWeight = %d, want >= 20", m0.WaitStallWeight)
	}

	m1 := metrics[1]
	if m1.StageName != "tile_swizzled" {
		t.Errorf("m1.StageName = %q, want tile_swizzled", m1.StageName)
	}
	if m1.SharedBankConflictRisk {
		t.Errorf("m1.SharedBankConflictRisk = true, want false when LOP3 swizzle is present")
	}
	if m1.LocalMemInsts != 0 {
		t.Errorf("m1.LocalMemInsts = %d, want 0", m1.LocalMemInsts)
	}
	if m1.TensorPipeInsts != 1 {
		t.Errorf("m1.TensorPipeInsts = %d, want 1", m1.TensorPipeInsts)
	}
}

func TestSASSLoopBankConflictAndSpillsMutateTileGEMM(t *testing.T) {
	plan := &hopper.Plan{
		Name: "prefill_plan",
		Inputs: []hopper.Tensor{
			{Name: "x", DType: hopper.BF16, Shape: []int{128, 256}},
			{Name: "w", DType: hopper.BF16, Shape: []int{256, 256}},
		},
		Output: hopper.Tensor{Name: "out", DType: hopper.BF16, Shape: []int{128, 256}},
		Kernels: []hopper.PlanKernel{
			{
				Tile: &hopper.TileGEMM{
					Name:        "prefill_tile",
					N:           256,
					K:           256,
					BM:          128,
					BN:          128,
					Stages:      3,
					WGMMA:       true,
					Swizzle128B: false,
					Weight:      hopper.WeightBF16,
					Out:         hopper.BF16,
				},
				M:    128,
				Args: []string{"x", "w", "out"},
			},
		},
	}

	ptxCompiler := mockPTXASCompilerFunc(func(ctx context.Context, cudaSource, arch string) (ptxasinfo.Report, error) {
		return parseCleanPTXASReport(t, "prefill_tile"), nil
	})

	var iterCalls int
	sassAnalyzer := mockSASSAnalyzerFunc(func(sassText string) ([]StageSASSMetrics, error) {
		iterCalls++
		switch iterCalls {
		case 1:
			// Iteration 1: Unswizzled STS.128/LDS.128 bank conflict + STL/LDL local spills.
			return ParseSASSDisassembly(`
				Function : prefill_tile
				/*0000*/ LDG.E.128 R4, [R2] ;
				/*0010*/ STS.128 [R6], R4 ;
				/*0020*/ LDS.128 R8, [R6] ;
				/*0030*/ STL [R10], R8 ;
				/*0040*/ LDL R12, [R10] ;
				/*0050*/ WARPGROUP.DEPBAR.LE 0x0 ; {!12}
				/*0060*/ HGMMA.64x128x16.F32 R16, R8, R12, R16 ;
				/*0070*/ EXIT ;
			`)
		case 2:
			// Iteration 2: Swizzle128B is now enabled, but Stages=2 still has high wait stall weight.
			return ParseSASSDisassembly(`
				Function : prefill_tile
				/*0000*/ LDG.E.128 R4, [R2] ;
				/*0010*/ LOP3.LUT R6, R4, 0x3c, R5, 0x3c ; // swizzled
				/*0020*/ STS.128 [R6], R4 ;
				/*0030*/ LDS.128 R8, [R6] ;
				/*0040*/ WARPGROUP.DEPBAR.LE 0x0 ; {!16}
				/*0050*/ HGMMA.64x128x16.F32 R16, R8, R4, R16 ;
				/*0060*/ EXIT ;
			`)
		default:
			// Iteration 3: Swizzle128B=true and Stages=1 -> clean SASS.
			return ParseSASSDisassembly(`
				Function : prefill_tile
				/*0000*/ LDG.E.128 R4, [R2] ;
				/*0010*/ LOP3.LUT R6, R4, 0x3c, R5, 0x3c ; // swizzled
				/*0020*/ STS.128 [R6], R4 ;
				/*0030*/ LDS.128 R8, [R6] ;
				/*0040*/ HGMMA.64x128x16.F32 R16, R8, R4, R16 ;
				/*0050*/ EXIT ;
			`)
		}
	})

	gate := StaticGate{NoSpills: true, MaxRegisters: 255, MinCTAsPerSM: 1}
	res, finalMetrics, err := RefinePlanByPTXASAndSASSLoop(context.Background(), plan, ptxCompiler, sassAnalyzer, gate, 5)
	if err != nil {
		t.Fatalf("RefinePlanByPTXASAndSASSLoop: %v", err)
	}
	if !res.Converged {
		t.Fatalf("expected RefinePlanByPTXASAndSASSLoop to converge, got Converged=false after %d iterations", len(res.Iterations))
	}
	if len(res.Iterations) != 3 {
		t.Fatalf("expected 3 iterations to converge, got %d", len(res.Iterations))
	}

	gotTile := res.RefinedPlan.Kernels[0].Tile
	if gotTile == nil {
		t.Fatal("refined TileGEMM is nil")
	}
	if !gotTile.Swizzle128B {
		t.Errorf("refined TileGEMM.Swizzle128B = false, want true")
	}
	if gotTile.Stages != 2 {
		t.Errorf("refined TileGEMM.Stages = %d, want 2 (stepped down 3 -> 2)", gotTile.Stages)
	}
	if gotTile.BN != 64 {
		t.Errorf("refined TileGEMM.BN = %d, want 64 (stepped down 128 -> 64 on iteration 2)", gotTile.BN)
	}
	if len(finalMetrics) != 1 || finalMetrics[0].SharedBankConflictRisk || finalMetrics[0].LocalMemInsts != 0 {
		t.Errorf("unexpected final SASS metrics: %+v", finalMetrics)
	}
}

func TestSASSLoopSkinnyGEMMExcessiveWaitStallAdjustsSplitK(t *testing.T) {
	plan := &hopper.Plan{
		Name: "decode_plan",
		Inputs: []hopper.Tensor{
			{Name: "x", DType: hopper.BF16, Shape: []int{1, 1024}},
			{Name: "w", DType: hopper.BF16, Shape: []int{256, 1024}},
		},
		Output: hopper.Tensor{Name: "out", DType: hopper.BF16, Shape: []int{1, 256}},
		Kernels: []hopper.PlanKernel{
			{
				GEMM: &hopper.SkinnyGEMM{
					Name:         "decode_gemm",
					MaxM:         1,
					N:            256,
					K:            1024,
					RowsPerBlock: 4,
					Warps:        4,
					SplitK:       1,
					Prefetch:     2,
					X:            hopper.BF16,
					Weight:       hopper.WeightBF16,
					Out:          hopper.BF16,
				},
				M:    1,
				Args: []string{"x", "w", "out"},
			},
		},
	}

	ptxCompiler := mockPTXASCompilerFunc(func(ctx context.Context, cudaSource, arch string) (ptxasinfo.Report, error) {
		return parseCleanPTXASReport(t, "decode_gemm"), nil
	})

	var iterCalls int
	sassAnalyzer := mockSASSAnalyzerFunc(func(sassText string) ([]StageSASSMetrics, error) {
		iterCalls++
		if iterCalls == 1 {
			return ParseSASSDisassembly(`
				Function : decode_gemm
				/*0000*/ LDG.E.128 R4, [R2] ;
				/*0010*/ STL [R8], R4 ;
				/*0020*/ LDL R6, [R8] ;
				/*0030*/ DEPBAR.LE SB0, 0x0 ; {!14}
				/*0040*/ BAR.SYNC 0x0 ;
				/*0050*/ FFMA R10, R4, R6, R10 ;
				/*0060*/ EXIT ;
			`)
		}
		return ParseSASSDisassembly(`
			Function : decode_gemm
			/*0000*/ LDG.E.128 R4, [R2] ;
			/*0010*/ FFMA R10, R4, R6, R10 ;
			/*0020*/ EXIT ;
		`)
	})

	gate := StaticGate{NoSpills: true, MaxRegisters: 255, MinCTAsPerSM: 1}
	res, metrics, err := RefinePlanByPTXASAndSASSLoop(context.Background(), plan, ptxCompiler, sassAnalyzer, gate, 4)
	if err != nil {
		t.Fatalf("RefinePlanByPTXASAndSASSLoop: %v", err)
	}
	if !res.Converged {
		t.Fatalf("expected convergence, got Converged=false")
	}

	gotGEMM := res.RefinedPlan.Kernels[0].GEMM
	if gotGEMM == nil {
		t.Fatal("refined SkinnyGEMM is nil")
	}
	if gotGEMM.Prefetch != 0 {
		t.Errorf("refined SkinnyGEMM.Prefetch = %d, want 0 after SASS STL/LDL spill", gotGEMM.Prefetch)
	}
	if gotGEMM.SplitK <= 1 {
		t.Errorf("refined SkinnyGEMM.SplitK = %d, want > 1 after excessive SASS wait stall weight", gotGEMM.SplitK)
	}
	if len(metrics) != 1 || metrics[0].LocalMemInsts != 0 {
		t.Errorf("unexpected final SASS metrics: %+v", metrics)
	}
}
