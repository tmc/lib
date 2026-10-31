package tilebridge

import (
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func buildTestCrossLayerMegakernel(t *testing.T, useSkinny bool) *hopper.Megakernel {
	t.Helper()
	const (
		M = 16
		D = 256
	)

	norm0 := NewHopperRMSNormKernel("l0_rmsnorm", D, 1e-5)
	var proj0 hopper.PlanKernel
	if useSkinny {
		sg0 := hopper.SkinnyGEMM{
			Name:         "l0_proj",
			N:            D,
			K:            D,
			MaxM:         M,
			Weight:       hopper.WeightBF16,
			X:            hopper.BF16,
			Out:          hopper.F32,
			RowsPerBlock: 16,
			Epilogue:     hopper.MustLambda("(silu $0)"),
		}
		proj0 = hopper.PlanKernel{
			GEMM: &sg0,
			M:    M,
			Args: []string{"l0_norm_bf16", "l0_w", "l0_out"},
		}
	} else {
		tg0 := hopper.TileGEMM{
			Name:     "l0_proj",
			N:        D,
			K:        D,
			BM:       64,
			BN:       64,
			Stages:   2,
			Weight:   hopper.WeightBF16,
			Out:      hopper.F32,
			Epilogue: hopper.MustLambda("(silu $0)"),
		}
		proj0 = hopper.PlanKernel{
			Tile: &tg0,
			M:    M,
			Args: []string{"l0_norm_bf16", "l0_w", "l0_out"},
		}
	}
	norm0.Outputs[0] = hopper.RowTensor{
		Name:  "l0_norm_bf16",
		DType: hopper.BF16,
		Kind:  hopper.RowData,
		Value: "y",
	}

	layer0 := &hopper.Plan{
		Name: "layer0",
		Inputs: []hopper.Tensor{
			{Name: "x_in", DType: hopper.F32, Shape: []int{M, D}},
			{Name: "l0_norm_w", DType: hopper.F32, Shape: []int{D}},
			{Name: "l0_w", DType: hopper.BF16, Shape: []int{D, D}},
		},
		Temps: []hopper.Tensor{
			{Name: "l0_norm_bf16", DType: hopper.BF16, Shape: []int{M, D}},
		},
		Output: hopper.Tensor{Name: "l0_out", DType: hopper.F32, Shape: []int{M, D}},
		Kernels: []hopper.PlanKernel{
			{
				Row:  &norm0,
				Rows: M,
				Args: []string{"x_in", "l0_norm_w", "l0_norm_bf16"},
			},
			proj0,
		},
	}

	resNorm1 := NewHopperResidualRMSNormKernel("l1_res_rmsnorm", D, 1e-5)
	resNorm1.Outputs[0] = hopper.RowTensor{
		Name:  "l1_norm_bf16",
		DType: hopper.BF16,
		Kind:  hopper.RowData,
		Value: "y",
	}
	tg1 := hopper.TileGEMM{
		Name:     "l1_proj",
		N:        D,
		K:        D,
		BM:       64,
		BN:       64,
		Stages:   2,
		Weight:   hopper.WeightBF16,
		Out:      hopper.F32,
		Epilogue: hopper.MustLambda("(add $0 $1)"),
		Extras: []hopper.Operand{
			{Name: "bias", DType: hopper.F32, Broadcast: hopper.PerCol},
		},
	}

	layer1 := &hopper.Plan{
		Name: "layer1",
		Inputs: []hopper.Tensor{
			{Name: "l1_x", DType: hopper.F32, Shape: []int{M, D}},
			{Name: "l1_res", DType: hopper.F32, Shape: []int{M, D}},
			{Name: "l1_norm_w", DType: hopper.F32, Shape: []int{D}},
			{Name: "l1_w", DType: hopper.BF16, Shape: []int{D, D}},
			{Name: "l1_bias", DType: hopper.F32, Shape: []int{D}},
		},
		Temps: []hopper.Tensor{
			{Name: "l1_norm_bf16", DType: hopper.BF16, Shape: []int{M, D}},
		},
		Output: hopper.Tensor{Name: "l1_out", DType: hopper.F32, Shape: []int{M, D}},
		Kernels: []hopper.PlanKernel{
			{
				Row:  &resNorm1,
				Rows: M,
				Args: []string{"l1_x", "l1_res", "l1_norm_w", "l1_norm_bf16"},
			},
			{
				Tile: &tg1,
				M:    M,
				Args: []string{"l1_norm_bf16", "l1_w", "l1_bias", "l1_out"},
			},
		},
	}

	fused, err := hopper.FuseCrossLayerPlans(layer0, layer1, hopper.CrossLayerGlue{
		Name:               "cross_l0_l1",
		LayerLPrefix:       "s0_",
		LayerNextPrefix:    "s1_",
		BridgeInput:        "l1_x",
		RetainBridgeOutput: false,
	})
	if err != nil {
		t.Fatalf("FuseCrossLayerPlans failed: %v", err)
	}
	mega, err := fused.Megakernel("cross_l0_l1_mega")
	if err != nil {
		t.Fatalf("fused.Megakernel failed: %v", err)
	}
	return &mega
}

func TestLowerAndLiftCrossLayerMegakernel(t *testing.T) {
	mega := buildTestCrossLayerMegakernel(t, false)

	mod, err := LowerCrossLayerMegakernelToTileIR(mega)
	if err != nil {
		t.Fatalf("LowerCrossLayerMegakernelToTileIR failed: %v", err)
	}
	if len(mod.Functions) != len(mega.Stages) {
		t.Fatalf("len(mod.Functions) = %d, want %d", len(mod.Functions), len(mega.Stages))
	}

	text := FormatTileIR(mod)
	for _, wantSub := range []string{
		"hopper.kind: megakernel",
		"hopper.mega.name: cross_l0_l1_mega",
		"hopper.mega.stage.0.wait_slots: none",
		"hopper.mega.stage.1.wait_slots: 0",
		"hopper.mega.stage.2.wait_slots: 1",
		"hopper.mega.stage.3.wait_slots: 2",
		"hopper.mega.stage.1.epilogue: (silu $0)",
		"hopper.mega.stage.3.epilogue: (add $0 $1)",
	} {
		if !strings.Contains(text, wantSub) {
			t.Errorf("formatted TileIR missing %q", wantSub)
		}
	}

	parsed, err := ParseTileIR(text)
	if err != nil {
		t.Fatalf("ParseTileIR failed: %v", err)
	}
	bc, err := EncodeTileIR(parsed)
	if err != nil {
		t.Fatalf("EncodeTileIR failed: %v", err)
	}
	if !tileir.IsTileIRBytecode(bc) {
		t.Fatalf("encoded bytes lack TileIR magic header")
	}

	archs := []hopper.Arch{
		hopper.TargetSM90A,
		hopper.TargetSM100A,
		hopper.TargetSM120A,
		hopper.TargetSM121A,
	}
	for _, arch := range archs {
		t.Run(string(arch), func(t *testing.T) {
			lifted, err := LiftTileIRToCrossLayerMegakernel(parsed, arch)
			if err != nil {
				t.Fatalf("LiftTileIRToCrossLayerMegakernel(%s) failed: %v", arch, err)
			}
			if lifted.Name != mega.Name {
				t.Errorf("lifted.Name = %q, want %q", lifted.Name, mega.Name)
			}
			if len(lifted.Stages) != len(mega.Stages) {
				t.Fatalf("len(lifted.Stages) = %d, want %d", len(lifted.Stages), len(mega.Stages))
			}
			for i := range mega.Stages {
				origSt := mega.Stages[i]
				liftSt := lifted.Stages[i]
				if !slices.Equal(origSt.WaitSlots, liftSt.WaitSlots) {
					t.Errorf("stage %d WaitSlots = %v, want %v", i, liftSt.WaitSlots, origSt.WaitSlots)
				}
				if origSt.SignalSlot != liftSt.SignalSlot {
					t.Errorf("stage %d SignalSlot = %d, want %d", i, liftSt.SignalSlot, origSt.SignalSlot)
				}
				if origSt.Grid != liftSt.Grid || origSt.BlockBase != liftSt.BlockBase || origSt.Blocks != liftSt.Blocks {
					t.Errorf("stage %d grid/CTA mismatch: got (%v, base=%d, blocks=%d), want (%v, base=%d, blocks=%d)",
						i, liftSt.Grid, liftSt.BlockBase, liftSt.Blocks, origSt.Grid, origSt.BlockBase, origSt.Blocks)
				}
			}
			origCUDA, err := mega.CUDA()
			if err != nil {
				t.Fatalf("mega.CUDA() failed: %v", err)
			}
			liftCUDA, err := lifted.CUDA()
			if err != nil {
				t.Fatalf("lifted.CUDA() failed: %v", err)
			}
			if origCUDA != liftCUDA {
				t.Errorf("lifted CUDA source differs from original megakernel CUDA source")
			}
		})
	}
}

func TestVerifyCrossLayerBridgeEquivalence(t *testing.T) {
	cases := []struct {
		name      string
		useSkinny bool
		seeds     []uint64
	}{
		{
			name:      "tile_gemm_cross_layer",
			useSkinny: false,
			seeds:     []uint64{1, 42, 20261031},
		},
		{
			name:      "skinny_gemm_cross_layer",
			useSkinny: true,
			seeds:     []uint64{7, 99},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mega := buildTestCrossLayerMegakernel(t, tc.useSkinny)
			for _, seed := range tc.seeds {
				if err := VerifyCrossLayerBridgeEquivalence(mega, seed); err != nil {
					t.Fatalf("VerifyCrossLayerBridgeEquivalence(seed=%d) failed: %v", seed, err)
				}
			}
		})
	}
}

func TestCrossLayerBridge_ValidationAndTamperDetection(t *testing.T) {
	if _, err := LowerCrossLayerMegakernelToTileIR(nil); err == nil {
		t.Errorf("expected error for nil megakernel in LowerCrossLayerMegakernelToTileIR")
	}
	if _, err := LiftTileIRToCrossLayerMegakernel(nil, hopper.TargetSM90A); err == nil {
		t.Errorf("expected error for nil TileModule in LiftTileIRToCrossLayerMegakernel")
	}
	if err := VerifyCrossLayerBridgeEquivalence(nil, 1); err == nil {
		t.Errorf("expected error for nil megakernel in VerifyCrossLayerBridgeEquivalence")
	}

	mega := buildTestCrossLayerMegakernel(t, false)
	mod, err := LowerCrossLayerMegakernelToTileIR(mega)
	if err != nil {
		t.Fatalf("LowerCrossLayerMegakernelToTileIR: %v", err)
	}

	// Unsupported architecture must fail cleanly.
	if _, err := LiftTileIRToCrossLayerMegakernel(mod, "sm_75"); err == nil {
		t.Errorf("expected error for unsupported architecture sm_75")
	}

	// Tamper with stage 0 wait_slots so it waits on a future stage (cycle/invalid).
	tamperedText := strings.Replace(
		FormatTileIR(mod),
		"hopper.mega.stage.0.wait_slots: none",
		"hopper.mega.stage.0.wait_slots: 2",
		1,
	)
	tamperedMod, err := ParseTileIR(tamperedText)
	if err != nil {
		t.Fatalf("ParseTileIR(tamperedText): %v", err)
	}
	if _, err := LiftTileIRToCrossLayerMegakernel(tamperedMod, hopper.TargetSM90A); err == nil {
		t.Errorf("expected error when lifting module with invalid stage 0 wait_slots")
	}
}
