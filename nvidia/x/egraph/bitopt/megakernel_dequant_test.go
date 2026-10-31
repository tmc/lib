package bitopt

import (
	"strings"
	"testing"
)

func TestSynthesizeMegakernelDequantStageAllFormats(t *testing.T) {
	tests := []struct {
		format   string
		bitWidth int
		numCodes int
	}{
		{format: "INT2", bitWidth: 2, numCodes: 4},
		{format: "NF3", bitWidth: 3, numCodes: 8},
		{format: "NVFP4", bitWidth: 4, numCodes: 16},
		{format: "INT4", bitWidth: 4, numCodes: 16},
	}

	for _, tc := range tests {
		t.Run(tc.format, func(t *testing.T) {
			stage, err := SynthesizeMegakernelDequantStage(tc.format, tc.bitWidth)
			if err != nil {
				t.Fatalf("SynthesizeMegakernelDequantStage(%s, %d): %v", tc.format, tc.bitWidth, err)
			}
			if stage.Format != tc.format {
				t.Errorf("Format = %q, want %q", stage.Format, tc.format)
			}
			if stage.BitWidth != tc.bitWidth {
				t.Errorf("BitWidth = %d, want %d", stage.BitWidth, tc.bitWidth)
			}
			if len(stage.Codebook) != tc.numCodes {
				t.Fatalf("len(Codebook) = %d, want %d", len(stage.Codebook), tc.numCodes)
			}
			if stage.OptimizedInstrCount >= stage.NaiveInstrCount {
				t.Errorf("OptimizedInstrCount (%d) must be strictly less than NaiveInstrCount (%d)",
					stage.OptimizedInstrCount, stage.NaiveInstrCount)
			}
			if stage.InstructionSavings() <= 0 {
				t.Errorf("InstructionSavings() = %d, want > 0", stage.InstructionSavings())
			}
			if stage.RoutingSolution == nil {
				t.Fatal("expected non-nil RoutingSolution")
			}
			if err := stage.RoutingSolution.Check(); err != nil {
				t.Fatalf("RoutingSolution.Check(): %v", err)
			}
			if len(stage.SASSInsts) == 0 {
				t.Error("expected non-empty SASSInsts")
			}

			// Verify algebraic equivalence across all 2^b bit patterns and multiple scales.
			for _, scale := range []float32{0.25, 0.5, 1.0, 2.0} {
				for code := 0; code < tc.numCodes; code++ {
					want := stage.EvalNaiveCode(code, scale)
					got := stage.EvalSynthesizedCode(code, scale)
					if got != want {
						t.Errorf("%s code=%d (0b%0*b) scale=%v: synthesized=%v != naive=%v",
							tc.format, code, tc.bitWidth, code, scale, got, want)
					}
				}
			}

			// Verify packed 8-lane unpacking across all 2^b patterns.
			for base := 0; base < tc.numCodes; base++ {
				var codes [8]uint8
				for lane := 0; lane < 8; lane++ {
					codes[lane] = uint8((base + lane) % tc.numCodes)
				}
				want8 := stage.EvalNaivePacked8(codes, 0.5)
				got8 := stage.EvalSynthesizedPacked8(codes, 0.5)
				if got8 != want8 {
					t.Errorf("%s codes=%v: synthesized=%v != naive=%v",
						tc.format, codes, got8, want8)
				}
			}

			helper := stage.CUDAInlineHelper()
			for _, wantSub := range []string{
				"__device__ __forceinline__ void dequant_megakernel_stage_" + strings.ToLower(tc.format),
				"__device__ __forceinline__ void bitopt_route_" + strings.ToLower(tc.format),
				"lop3.b32",
				"prmt.b32",
				"fma.rn.f16x2",
			} {
				if !strings.Contains(helper, wantSub) {
					t.Errorf("%s CUDAInlineHelper missing %q:\n%s", tc.format, wantSub, helper)
				}
			}
		})
	}
}

func TestSynthesizeMegakernelDequantStageValidation(t *testing.T) {
	tests := []struct {
		name     string
		format   string
		bitWidth int
		wantSub  string
	}{
		{
			name:     "int2_wrong_bitwidth",
			format:   "INT2",
			bitWidth: 4,
			wantSub:  "requires bitWidth=2",
		},
		{
			name:     "nf3_wrong_bitwidth",
			format:   "NF3",
			bitWidth: 4,
			wantSub:  "requires bitWidth=3",
		},
		{
			name:     "nvfp4_wrong_bitwidth",
			format:   "NVFP4",
			bitWidth: 2,
			wantSub:  "requires bitWidth=4",
		},
		{
			name:     "int4_wrong_bitwidth",
			format:   "INT4",
			bitWidth: 3,
			wantSub:  "requires bitWidth=4",
		},
		{
			name:     "unsupported_format",
			format:   "FP8",
			bitWidth: 8,
			wantSub:  "unsupported format",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := SynthesizeMegakernelDequantStage(tc.format, tc.bitWidth)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantSub)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantSub)
			}
		})
	}
}

func TestRubinAndBlackwellMegakernelDequantSynthesis(t *testing.T) {
	TestSynthesizeMegakernelDequantStageAllFormats(t)
	TestSynthesizeMegakernelDequantStageValidation(t)
}
