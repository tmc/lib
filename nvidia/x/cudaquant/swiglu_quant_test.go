// SPDX-License-Identifier: Apache-2.0

package cudaquant

import (
	"math"
	"strings"
	"testing"
)

func TestSwiGLUPerTokenQuantCPU(t *testing.T) {
	tokens := 5
	interDim := 256
	blockSize := 128
	gateUp := make([]float32, tokens*2*interDim)
	for tIdx := 0; tIdx < tokens; tIdx++ {
		for c := 0; c < interDim; c++ {
			gateUp[tIdx*2*interDim+c] = float32((c%11)-5) * 0.4
			gateUp[tIdx*2*interDim+interDim+c] = float32(((c+tIdx)%9)-4) * 0.75
		}
	}

	for _, recipe := range []ScaleRecipe{ScaleRecipeFloat32, ScaleRecipeUE8M0} {
		t.Run(recipe.String(), func(t *testing.T) {
			cfg := SwiGLUQuantConfig{
				Tokens:              tokens,
				IntermediateDim:     interDim,
				BlockSize:           blockSize,
				Recipe:              recipe,
				OutputRowAlignBytes: 32,
				ScaleColAlignBytes:  16,
			}
			fp8Out, scales, packedColMajor, err := SwiGLUPerTokenQuantCPU(gateUp, cfg)
			if err != nil {
				t.Fatalf("SwiGLUPerTokenQuantCPU failed: %v", err)
			}
			outStride, err := AlignedOutputStride(interDim, 1, 32)
			if err != nil {
				t.Fatal(err)
			}
			if len(fp8Out) != tokens*outStride {
				t.Fatalf("len(fp8Out) = %d, want %d", len(fp8Out), tokens*outStride)
			}
			blocksPerRow := interDim / blockSize
			if len(scales) != tokens*blocksPerRow {
				t.Fatalf("len(scales) = %d, want %d", len(scales), tokens*blocksPerRow)
			}

			if recipe == ScaleRecipeUE8M0 {
				alignedRows, err := AlignedScaleRows(tokens, 16)
				if err != nil {
					t.Fatal(err)
				}
				scaleCols := (blocksPerRow + 3) / 4
				if len(packedColMajor) != scaleCols*alignedRows {
					t.Fatalf("len(packedColMajor) = %d, want %d", len(packedColMajor), scaleCols*alignedRows)
				}
				rowMajor, err := InverseTransformScaleLayoutColMajor(packedColMajor, tokens, scaleCols, alignedRows)
				if err != nil {
					t.Fatal(err)
				}
				for tok := 0; tok < tokens; tok++ {
					exps, err := UnpackUE8M0(rowMajor[tok*scaleCols:(tok+1)*scaleCols], blocksPerRow)
					if err != nil {
						t.Fatal(err)
					}
					for b := 0; b < blocksPerRow; b++ {
						gotScale := DequantizeScaleUE8M0(exps[b])
						if gotScale != scales[tok*blocksPerRow+b] {
							t.Fatalf("tok=%d b=%d: unpacked scale=%v, want %v", tok, b, gotScale, scales[tok*blocksPerRow+b])
						}
					}
				}
			} else if packedColMajor != nil {
				t.Fatalf("expected nil packedColMajor for float32 recipe, got len %d", len(packedColMajor))
			}

			// Independent FP64 SwiGLU oracle comparison.
			for tok := 0; tok < tokens; tok++ {
				for c := 0; c < interDim; c++ {
					g := float64(gateUp[tok*2*interDim+c])
					u := float64(gateUp[tok*2*interDim+interDim+c])
					want := g / (1.0 + math.Exp(-g)) * u

					qVal, ok := fp8E4M3ToFloat32(fp8Out[tok*outStride+c])
					if !ok {
						t.Fatalf("tok=%d col=%d produced NaN E4M3", tok, c)
					}
					b := c / blockSize
					got := float64(qVal * scales[tok*blocksPerRow+b])
					diff := math.Abs(got - want)
					tol := math.Max(0.15*math.Abs(want), 0.02)
					if diff > tol {
						t.Fatalf("tok=%d col=%d: got=%v, want=%v (diff=%v > tol=%v)", tok, c, got, want, diff, tol)
					}
				}
			}
		})
	}
}

func TestSwiGLUPerTokenQuantPaddingAndErrors(t *testing.T) {
	// Test non-32-aligned IntermediateDim (e.g. 36 with BlockSize=18 -> outStride=64) to verify zero padding.
	cfg := SwiGLUQuantConfig{
		Tokens:              3,
		IntermediateDim:     36,
		BlockSize:           18,
		Recipe:              ScaleRecipeUE8M0,
		OutputRowAlignBytes: 32,
		ScaleColAlignBytes:  16,
	}
	gateUp := make([]float32, cfg.Tokens*2*cfg.IntermediateDim)
	for i := range gateUp {
		gateUp[i] = 1.25
	}
	fp8Out, _, _, err := SwiGLUPerTokenQuantCPU(gateUp, cfg)
	if err != nil {
		t.Fatalf("SwiGLUPerTokenQuantCPU failed: %v", err)
	}
	outStride, err := AlignedOutputStride(cfg.IntermediateDim, 1, 32)
	if err != nil {
		t.Fatal(err)
	}
	if outStride != 64 {
		t.Fatalf("outStride = %d, want 64", outStride)
	}
	for tok := 0; tok < cfg.Tokens; tok++ {
		for c := cfg.IntermediateDim; c < outStride; c++ {
			if fp8Out[tok*outStride+c] != 0 {
				t.Fatalf("padding byte at tok=%d col=%d is %d, want 0", tok, c, fp8Out[tok*outStride+c])
			}
		}
	}

	if _, _, _, err := SwiGLUPerTokenQuantCPU(gateUp, SwiGLUQuantConfig{Tokens: 0, IntermediateDim: 128}); err == nil {
		t.Fatal("expected error for Tokens=0")
	}
	if _, _, _, err := SwiGLUPerTokenQuantCPU(gateUp, SwiGLUQuantConfig{Tokens: 2, IntermediateDim: 100, BlockSize: 128}); err == nil {
		t.Fatal("expected error when IntermediateDim not divisible by BlockSize")
	}
}

func TestFusedSwiGLUPerTokenCastSource(t *testing.T) {
	cfg := SwiGLUQuantConfig{
		Tokens:          7,
		IntermediateDim: 256,
		BlockSize:       128,
		Recipe:          ScaleRecipeUE8M0,
	}
	src, err := FusedSwiGLUPerTokenCastSource(cfg)
	if err != nil {
		t.Fatalf("FusedSwiGLUPerTokenCastSource failed: %v", err)
	}
	for _, needle := range []string{
		"extern \"C\" __global__ void fused_swiglu_per_token_fp8_quant",
		"__shfl_xor_sync",
		"#define SWIGLU_OUT_STRIDE 256",
		"#define SWIGLU_ALIGNED_ROWS 16",
		"#define SWIGLU_USE_UE8M0 1",
	} {
		if !strings.Contains(src, needle) {
			t.Fatalf("generated source missing %q", needle)
		}
	}

	if _, err := FusedSwiGLUPerTokenCastSource(SwiGLUQuantConfig{Tokens: -1, IntermediateDim: 128}); err == nil {
		t.Fatal("expected error for invalid config")
	}
}
