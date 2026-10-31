package tilebridge

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func TestHopperStepToTileIR(t *testing.T) {
	tests := []struct {
		name     string
		step     TileIRStep
		wantOps  []string
		wantFunc string
	}{
		{
			name:     "RowKernel_RMSNorm",
			step:     NewHopperRMSNormKernel("rmsnorm_f32", 64, 1e-5),
			wantOps:  []string{"make_token", "get_tile_block_id", "iota", "load_ptr_tko", "ftof", "mulf", "subf", "store_ptr_tko", "return"},
			wantFunc: "rmsnorm_f32",
		},
		{
			name:     "RowKernel_ResidualRMSNorm",
			step:     NewHopperResidualRMSNormKernel("residual_rmsnorm_f32", 64, 1e-5),
			wantOps:  []string{"make_token", "load_ptr_tko", "addf", "ftof", "mulf", "store_ptr_tko", "return"},
			wantFunc: "residual_rmsnorm_f32",
		},
		{
			name:     "RowKernel_SwiGLU",
			step:     NewHopperSwiGLUKernel("swiglu_f32", 64),
			wantOps:  []string{"make_token", "load_ptr_tko", "negf", "divf", "mulf", "store_ptr_tko", "return"},
			wantFunc: "swiglu_f32",
		},
		{
			name: "TileGEMM_Identity",
			step: hopper.TileGEMM{
				Name:     "tile_gemm_f32",
				N:        64,
				K:        64,
				BM:       64,
				BN:       64,
				Stages:   2,
				Weight:   hopper.WeightBF16,
				Out:      hopper.F32,
				Epilogue: hopper.Identity,
			},
			wantOps:  []string{"make_token", "get_tile_block_id", "load_ptr_tko", "reshape", "mmaf", "store_ptr_tko", "return"},
			wantFunc: "tile_gemm_f32",
		},
		{
			name: "TileGEMM_GLU_WithResidual",
			step: hopper.TileGEMM{
				Name:     "tile_gemm_glu_res_f32",
				N:        128,
				K:        64,
				BM:       64,
				BN:       64,
				Stages:   2,
				Weight:   hopper.WeightBF16,
				Out:      hopper.F32,
				GLU:      true,
				Epilogue: hopper.MustLambda("(add (mul (silu $0) $1) $2)"),
				Extras:   []hopper.Operand{{Name: "residual", DType: hopper.F32, Broadcast: hopper.Full}},
			},
			wantOps:  []string{"make_token", "load_ptr_tko", "mmaf", "negf", "divf", "mulf", "addf", "store_ptr_tko", "return"},
			wantFunc: "tile_gemm_glu_res_f32",
		},
		{
			name: "SkinnyGEMM_SiLU",
			step: hopper.SkinnyGEMM{
				Name:     "skinny_gemm_silu_f32",
				N:        64,
				K:        256,
				MaxM:     4,
				Weight:   hopper.WeightBF16,
				Out:      hopper.F32,
				Epilogue: hopper.MustLambda("(silu $0)"),
			},
			wantOps:  []string{"make_token", "load_ptr_tko", "mmaf", "negf", "divf", "mulf", "store_ptr_tko", "return"},
			wantFunc: "skinny_gemm_silu_f32",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mod, err := StepHopperToTileIR(tc.step)
			if err != nil {
				t.Fatalf("StepHopperToTileIR failed: %v", err)
			}
			if len(mod.Functions) != 1 {
				t.Fatalf("got %d functions, want 1", len(mod.Functions))
			}
			fn := mod.Functions[0]
			if fn.Name != tc.wantFunc {
				t.Errorf("got func name %q, want %q", fn.Name, tc.wantFunc)
			}
			if !fn.Entry {
				t.Errorf("expected entry function")
			}

			text := FormatTileIR(mod)
			if !strings.Contains(text, "cuda_tile.module") {
				t.Fatalf("formatted TileIR missing cuda_tile.module:\n%s", text)
			}
			for _, op := range tc.wantOps {
				if !strings.Contains(text, "cuda_tile."+op) {
					t.Errorf("formatted TileIR missing op cuda_tile.%s:\n%s", op, text)
				}
			}

			// Verify round-trip Format -> Parse.
			parsed, err := ParseTileIR(text)
			if err != nil {
				t.Fatalf("ParseTileIR round-trip failed: %v\ntext:\n%s", err, text)
			}
			if len(parsed.Functions) != 1 || len(parsed.Functions[0].Operations) != len(fn.Operations) {
				t.Fatalf("parsed body op count mismatch: got %d, want %d", len(parsed.Functions[0].Operations), len(fn.Operations))
			}

			// Verify bytecode encoding on both original and round-tripped ASTs.
			bcOrig, err := EncodeTileIR(mod)
			if err != nil {
				t.Fatalf("EncodeTileIR(mod) failed: %v", err)
			}
			if !tileir.IsTileIRBytecode(bcOrig) {
				t.Fatalf("EncodeTileIR(mod) did not produce valid TileIR magic header")
			}
			bcParsed, err := EncodeTileIR(parsed)
			if err != nil {
				t.Fatalf("EncodeTileIR(parsed) failed: %v", err)
			}
			if !bytes.Equal(bcOrig, bcParsed) {
				t.Errorf("bytecode mismatch after Format -> Parse round-trip (%d bytes vs %d bytes)", len(bcOrig), len(bcParsed))
			}

			// Verify bytecode decodes cleanly back to a TileIR module.
			decodedBM, err := tileir.DecodeBytecodeBytes(bcParsed)
			if err != nil {
				t.Fatalf("DecodeBytecodeBytes failed: %v", err)
			}
			decodedMod := decodedBM.ToModule()
			if len(decodedMod.Functions) != 1 {
				t.Fatalf("decoded module has %d functions, want 1", len(decodedMod.Functions))
			}
		})
	}
}

func TestPlanToTileIR(t *testing.T) {
	// 1. Multi-step Plan compiled via e-graph Builder + Optimize + Compile (GEMM -> RMSNorm).
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.BF16, 4, 256)
	w := b.Input("w", hopper.BF16, 64, 256)
	nw := b.Input("norm_w", hopper.F32, 64)
	mm := b.MatMul(x, w)
	ss := b.ReduceSum(b.Map(hopper.MustLambda("(mul $0 $0)"), mm))
	normed := b.Map(hopper.MustLambda("(mul (mul $0 (rsqrt (add (div $1 64) 1e-5))) $2)"), mm, ss, nw)

	expr, err := hopper.Optimize(b.G, normed)
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}
	compiledPlan, err := hopper.Compile(expr, "gemm_norm", hopper.F32)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	compiledMods, err := PlanToTileIRModules(compiledPlan)
	if err != nil {
		t.Fatalf("PlanToTileIRModules(compiledPlan) failed: %v", err)
	}
	if len(compiledMods) != len(compiledPlan.Kernels) {
		t.Fatalf("got %d modules, want %d", len(compiledMods), len(compiledPlan.Kernels))
	}

	// 2. Explicit 3-step Plan (RMSNorm -> TileGEMM -> SwiGLU).
	rk0 := NewHopperRMSNormKernel("step0_rmsnorm", 64, 1e-5)
	tg1 := hopper.TileGEMM{
		Name:     "step1_tile_gemm",
		N:        64,
		K:        64,
		BM:       64,
		BN:       64,
		Stages:   2,
		Weight:   hopper.WeightBF16,
		Out:      hopper.F32,
		Epilogue: hopper.Identity,
	}
	rk2 := NewHopperSwiGLUKernel("step2_swiglu", 64)
	plan := &hopper.Plan{
		Name: "rmsnorm_gemm_swiglu",
		Kernels: []hopper.PlanKernel{
			{Row: &rk0},
			{Tile: &tg1},
			{Row: &rk2},
		},
	}

	mods, err := PlanToTileIRModules(plan)
	if err != nil {
		t.Fatalf("PlanToTileIRModules(plan) failed: %v", err)
	}
	if len(mods) != 3 {
		t.Fatalf("got %d modules, want 3", len(mods))
	}
	for i, mod := range mods {
		bc, err := EncodeTileIR(mod)
		if err != nil {
			t.Fatalf("step %d EncodeTileIR failed: %v", i, err)
		}
		if !tileir.IsTileIRBytecode(bc) {
			t.Fatalf("step %d invalid TileIR magic", i)
		}
	}
}

func TestVerifyHopperDualTarget(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name    string
		step    TileIRStep
		M, N, K int
		eps     float32
	}{
		{
			name: "RMSNorm",
			step: NewHopperRMSNormKernel("rmsnorm_f32", 64, 1e-5),
			M:    4,
			N:    64,
			K:    64,
			eps:  1e-5,
		},
		{
			name: "ResidualRMSNorm",
			step: NewHopperResidualRMSNormKernel("residual_rmsnorm_f32", 64, 1e-5),
			M:    4,
			N:    64,
			K:    64,
			eps:  1e-5,
		},
		{
			name: "SwiGLU_Row",
			step: NewHopperSwiGLUKernel("swiglu_f32", 64),
			M:    4,
			N:    64,
			K:    64,
			eps:  1e-5,
		},
		{
			name: "TileGEMM_Identity",
			step: hopper.TileGEMM{
				Name:     "tile_gemm_f32",
				N:        64,
				K:        64,
				BM:       64,
				BN:       64,
				Stages:   2,
				Weight:   hopper.WeightBF16,
				Out:      hopper.F32,
				Epilogue: hopper.Identity,
			},
			M:   16,
			N:   64,
			K:   64,
			eps: 1e-5,
		},
		{
			name: "TileGEMM_SiLU",
			step: hopper.TileGEMM{
				Name:     "tile_gemm_silu_f32",
				N:        64,
				K:        64,
				BM:       64,
				BN:       64,
				Stages:   2,
				Weight:   hopper.WeightBF16,
				Out:      hopper.F32,
				Epilogue: hopper.MustLambda("(silu $0)"),
			},
			M:   16,
			N:   64,
			K:   64,
			eps: 1e-5,
		},
		{
			name: "TileGEMM_GLU_ExtraAdd",
			step: hopper.TileGEMM{
				Name:     "tile_gemm_glu_add_f32",
				N:        128,
				K:        64,
				BM:       64,
				BN:       64,
				Stages:   2,
				Weight:   hopper.WeightBF16,
				Out:      hopper.F32,
				GLU:      true,
				Epilogue: hopper.MustLambda("(add (mul (silu $0) $1) $2)"),
				Extras:   []hopper.Operand{{Name: "residual", DType: hopper.F32, Broadcast: hopper.Full}},
			},
			M:   16,
			N:   128,
			K:   64,
			eps: 1e-5,
		},
		{
			name: "SkinnyGEMM_Identity",
			step: hopper.SkinnyGEMM{
				Name:     "skinny_gemm_f32",
				N:        64,
				K:        256,
				MaxM:     4,
				Weight:   hopper.WeightBF16,
				Out:      hopper.F32,
				Epilogue: hopper.Identity,
			},
			M:   4,
			N:   64,
			K:   256,
			eps: 1e-5,
		},
		{
			name: "SkinnyGEMM_SiLU",
			step: hopper.SkinnyGEMM{
				Name:     "skinny_gemm_silu_f32",
				N:        64,
				K:        256,
				MaxM:     4,
				Weight:   hopper.WeightBF16,
				Out:      hopper.F32,
				Epilogue: hopper.MustLambda("(silu $0)"),
			},
			M:   4,
			N:   64,
			K:   256,
			eps: 1e-5,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			report, err := VerifyHopperDualTargetAgreement(ctx, tc.step, tc.M, tc.N, tc.K, tc.eps)
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip("no host C++ compiler; skipping dual-target host emulation check")
			}
			if err != nil {
				t.Fatalf("VerifyHopperDualTargetAgreement(%s) failed: %v", tc.name, err)
			}
			t.Logf("%s: maxAbsDiff=%.3e (len=%d, bytecode=%d bytes)",
				report.KernelName, report.MaxAbsDiff, len(report.CUDAOutput), len(report.Bytecode))
		})
	}
}
