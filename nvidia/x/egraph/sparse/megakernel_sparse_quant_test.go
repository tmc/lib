package sparse

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func runDualSparseQuantCompressionTests(t *testing.T) {
	t.Helper()
	const (
		rows = 16
		cols = 64
	)
	weights := make([]float32, rows*cols)
	for r := range rows {
		for c := range cols {
			// Construct each 4-element block with distinct magnitudes so top-2 is unambiguous.
			group := c / 4
			lane := c % 4
			var mag float32
			switch (r + group + lane) % 4 {
			case 0:
				mag = 0.25
			case 1:
				mag = 3.0
			case 2:
				mag = 0.5
			case 3:
				mag = 6.0
			}
			sign := float32(1.0)
			if (r+c)%2 == 1 {
				sign = -1.0
			}
			weights[r*cols+c] = sign * mag
		}
	}

	formats := []struct {
		name      string
		format    hopper.QuantFormat
		wantRatio float64
		wantBytes int
		maxRelErr float64
	}{
		{
			name:      "NVFP4_8xCompression",
			format:    hopper.FormatNVFP4,
			wantRatio: 8.0,
			wantBytes: (rows * cols) / 4,
			maxRelErr: 0.15,
		},
		{
			name:      "MXFP4_8xCompression",
			format:    hopper.FormatMXFP4,
			wantRatio: 8.0,
			wantBytes: (rows * cols) / 4,
			maxRelErr: 0.20,
		},
		{
			name:      "FP8E4M3_4xCompression",
			format:    hopper.FormatFP8E4M3,
			wantRatio: 4.0,
			wantBytes: (rows * cols) / 2,
			maxRelErr: 0.05,
		},
	}

	for _, tc := range formats {
		t.Run(tc.name, func(t *testing.T) {
			packed, metaE, scales, err := Compress2To4QuantizedWeights(weights, rows, cols, tc.format)
			if err != nil {
				t.Fatalf("Compress2To4QuantizedWeights(%s): %v", tc.name, err)
			}

			// Verify 8x (for NVFP4/MXFP4) and 4x (for FP8E4M3) footprint reduction vs FP16 (2 bytes/elem).
			fp16Bytes, spBytes, ratio := DualCompressionFootprint(rows, cols, tc.format)
			if fp16Bytes != rows*cols*2 {
				t.Fatalf("fp16Bytes = %d, want %d", fp16Bytes, rows*cols*2)
			}
			if len(packed) != tc.wantBytes || spBytes != tc.wantBytes {
				t.Fatalf("len(packed) = %d (spBytes=%d), want %d", len(packed), spBytes, tc.wantBytes)
			}
			if math.Abs(ratio-tc.wantRatio) > 1e-9 {
				t.Fatalf("compression ratio = %g, want %g", ratio, tc.wantRatio)
			}

			totalGroups := (rows * cols) / 4
			wantMetaWords := (totalGroups + 3) / 4
			if len(metaE) != wantMetaWords {
				t.Fatalf("len(metaE) = %d, want %d", len(metaE), wantMetaWords)
			}
			if len(scales) == 0 {
				t.Fatalf("expected non-empty scales")
			}

			recon, err := Decompress2To4QuantizedWeights(packed, metaE, scales, rows, cols, tc.format)
			if err != nil {
				t.Fatalf("Decompress2To4QuantizedWeights(%s): %v", tc.name, err)
			}
			if len(recon) != rows*cols {
				t.Fatalf("len(recon) = %d, want %d", len(recon), rows*cols)
			}

			// Verify every 4-element block has at most 2 non-zeros at the exact top-2 indices
			// encoded in the 2-bit selector metadata E, and accurate dequantized values.
			var dot, normRef, normRecon float64
			for g := range totalGroups {
				base := 4 * g
				blk := [4]float32{weights[base], weights[base+1], weights[base+2], weights[base+3]}
				wantIdx0, wantIdx1 := selectTop2Of4(blk)

				nibble := (metaE[g/4] >> (4 * (g % 4))) & 0x0f
				gotIdx0 := int(nibble & 0x03)
				gotIdx1 := int((nibble >> 2) & 0x03)
				if gotIdx0 != wantIdx0 || gotIdx1 != wantIdx1 {
					t.Fatalf("group %d metadata E indices = (%d, %d), want (%d, %d)",
						g, gotIdx0, gotIdx1, wantIdx0, wantIdx1)
				}

				nonZeroCount := 0
				for i := range 4 {
					v := recon[base+i]
					if v != 0 {
						nonZeroCount++
					}
					var refVal float32
					if i == wantIdx0 || i == wantIdx1 {
						refVal = blk[i]
					} else if v != 0 {
						t.Fatalf("group %d pruned index %d is non-zero (%g)", g, i, v)
					}
					dot += float64(refVal) * float64(v)
					normRef += float64(refVal) * float64(refVal)
					normRecon += float64(v) * float64(v)
				}
				if nonZeroCount > 2 {
					t.Fatalf("group %d has %d non-zeros, want <= 2", g, nonZeroCount)
				}
			}

			cosSim := dot / (math.Sqrt(normRef) * math.Sqrt(normRecon))
			if cosSim < 1.0-tc.maxRelErr {
				t.Fatalf("%s cosine similarity vs ideal 2:4 pruned weights = %g, want >= %g",
					tc.name, cosSim, 1.0-tc.maxRelErr)
			}
		})
	}
}

func runDualSparseQuantMegakernelTests(t *testing.T) {
	t.Helper()
	configs := []DualSparseQuantConfig{
		{
			NamePrefix:       "sparse_nvfp4_2l",
			BatchM:           2,
			HiddenDim:        128,
			IntermDim:        256,
			NumLayers:        2,
			Format:           hopper.FormatNVFP4,
			GroupSize:        16,
			ActivationSparse: true,
			Arch:             hopper.ArchSM90A,
		},
		{
			NamePrefix:       "sparse_fp8_2l",
			BatchM:           2,
			HiddenDim:        128,
			IntermDim:        256,
			NumLayers:        2,
			Format:           hopper.FormatFP8E4M3,
			GroupSize:        16,
			ActivationSparse: false,
			Arch:             hopper.ArchSM100A,
		},
	}

	for _, cfg := range configs {
		t.Run(cfg.NamePrefix, func(t *testing.T) {
			plan, mega, preview, err := BuildDualSparseQuantMegakernel(cfg)
			if err != nil {
				t.Fatalf("BuildDualSparseQuantMegakernel(%s): %v", cfg.NamePrefix, err)
			}
			if plan == nil || mega == nil {
				t.Fatalf("expected non-nil plan and megakernel")
			}
			if len(mega.Stages) < 8 {
				t.Fatalf("len(mega.Stages) = %d, want >= 8 across %d layers", len(mega.Stages), cfg.NumLayers)
			}

			wantTokens := []string{
				"mma.sp.sync.aligned",
				"eg_decode_2to4_meta_selector",
				"eg_mma_sp_2to4_tile",
				cfg.NamePrefix + "_mega",
				"eg_ready_wait",
				"eg_ready_signal",
			}
			if cfg.Format == hopper.FormatNVFP4 {
				wantTokens = append(wantTokens, "eg_sparse_dequant_nvfp4x4", "8x compression vs FP16")
			} else if cfg.Format == hopper.FormatFP8E4M3 {
				wantTokens = append(wantTokens, "eg_sparse_dequant_fp8e4m3x4", "4x compression vs FP16")
			}
			if cfg.ActivationSparse {
				wantTokens = append(wantTokens, "eg_ffn_relu2_topk_prune")
			}

			for _, tok := range wantTokens {
				if !strings.Contains(preview, tok) {
					t.Fatalf("preview for %s missing %q", cfg.NamePrefix, tok)
				}
			}
		})
	}
}

func TestDualSparseQuantCompression(t *testing.T) {
	runDualSparseQuantCompressionTests(t)
}

func TestDualSparseQuantMegakernel(t *testing.T) {
	runDualSparseQuantMegakernelTests(t)
}

func TestDeltaNetAndDualSparseQuantSuite(t *testing.T) {
	t.Run("Compression2To4And8x", runDualSparseQuantCompressionTests)
	t.Run("MegakernelCodegen", runDualSparseQuantMegakernelTests)
}
