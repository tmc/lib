package model

import (
	"fmt"
	"strings"
	"testing"
)

func TestDiffusionDenoisingMegakernelBlockSizes(t *testing.T) {
	for _, blockSize := range []int{16, 32, 64} {
		t.Run(fmt.Sprintf("B%d", blockSize), func(t *testing.T) {
			cfg := DiffusionBlockConfig{
				Name:                fmt.Sprintf("llada_mercury_b%d", blockSize),
				Target:              "sm_121a",
				BlockSize:           blockSize,
				NumLayers:           2,
				HiddenDim:           256,
				QKVDim:              256,
				InterDim:            512,
				VocabDim:            256,
				ConfidenceThreshold: 0.85,
			}

			blockAI, decodeM1AI := cfg.ArithmeticIntensityFlopsPerByte()
			if blockAI <= decodeM1AI*4.0 {
				t.Fatalf("B=%d arithmetic intensity = %.2f FLOPs/B, want > 4x M=1 decode (%.2f FLOPs/B)",
					blockSize, blockAI, decodeM1AI)
			}

			plan, mega, err := BuildDiffusionDenoisingMegakernel(cfg)
			if err != nil {
				t.Fatalf("BuildDiffusionDenoisingMegakernel(B=%d): %v", blockSize, err)
			}
			if plan == nil || mega == nil {
				t.Fatalf("expected non-nil plan and megakernel")
			}
			if got, want := plan.Output.Shape, []int{blockSize, 256}; len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
				t.Fatalf("plan.Output.Shape = %v, want %v", got, want)
			}
			if len(mega.Stages) < 5 {
				t.Fatalf("len(mega.Stages) = %d, want >= 5 stages across 2 bidirectional layers + confidence epilogue", len(mega.Stages))
			}

			src, spans, err := mega.LineInfoSource()
			if err != nil {
				t.Fatalf("mega.LineInfoSource: %v", err)
			}
			if len(spans) != len(mega.Stages) {
				t.Fatalf("len(spans) = %d, want %d", len(spans), len(mega.Stages))
			}
			for _, wantSub := range []string{
				"eg_ready_resolve_epoch",
				"eg_ready_wait",
				"eg_ready_signal",
				"denoise_tau_conf",
				"denoise_epilogue_norm",
			} {
				if !strings.Contains(src, wantSub) {
					t.Fatalf("megakernel source missing %q", wantSub)
				}
			}
		})
	}
}

func TestDiffusionConfidenceUnmaskingStepAndConvergence(t *testing.T) {
	const (
		blockSize = 16
		vocabSize = 8
		maskID    = -1
	)
	cfg := DiffusionBlockConfig{
		BlockSize:           blockSize,
		ConfidenceThreshold: 0.80,
		MinUnmaskPerStep:    2,
	}

	tokens := make([]int, blockSize)
	masked := make([]bool, blockSize)
	for i := range blockSize {
		tokens[i] = maskID
		masked[i] = true
	}

	// Step 1: 4 positions have high confidence (>= 0.80), 12 positions are flat.
	logits1 := make([][]float32, blockSize)
	for i := range blockSize {
		row := make([]float32, vocabSize)
		if i < 4 {
			row[i%vocabSize] = 10.0 // softmax confidence ~ 0.9997 >= 0.80
		} else {
			row[i%vocabSize] = 0.2 // low confidence < 0.80
		}
		logits1[i] = row
	}

	step1, err := SimulateConfidenceUnmaskingStep(cfg, logits1, tokens, masked)
	if err != nil {
		t.Fatalf("SimulateConfidenceUnmaskingStep step 1: %v", err)
	}
	if len(step1.UnmaskedIndices) != 4 {
		t.Fatalf("step 1 unmasked %v (len %d), want 4 high-confidence positions [0 1 2 3]",
			step1.UnmaskedIndices, len(step1.UnmaskedIndices))
	}
	if step1.RemainingMasked != 12 || step1.Converged {
		t.Fatalf("step 1 RemainingMasked=%d Converged=%v, want 12, false", step1.RemainingMasked, step1.Converged)
	}

	// Step 2: All remaining 12 positions have confidence below threshold (0.80),
	// so fallback progress unmasking must unmask top MinUnmaskPerStep = 2 positions.
	logits2 := make([][]float32, blockSize)
	for i := range blockSize {
		row := make([]float32, vocabSize)
		switch i {
		case 7:
			row[3] = 1.5 // highest among still-masked, but < 0.80
		case 11:
			row[5] = 1.2 // second highest among still-masked, but < 0.80
		default:
			row[1] = 0.1
		}
		logits2[i] = row
	}
	step2, err := SimulateConfidenceUnmaskingStep(cfg, logits2, step1.Tokens, step1.Masked)
	if err != nil {
		t.Fatalf("SimulateConfidenceUnmaskingStep step 2: %v", err)
	}
	if len(step2.UnmaskedIndices) != 2 || step2.UnmaskedIndices[0] != 7 || step2.UnmaskedIndices[1] != 11 {
		t.Fatalf("step 2 fallback unmasked = %v, want [7 11]", step2.UnmaskedIndices)
	}
	if step2.Tokens[7] != 3 || step2.Tokens[11] != 5 {
		t.Fatalf("step 2 unmasked tokens[7]=%d tokens[11]=%d, want 3, 5", step2.Tokens[7], step2.Tokens[11])
	}

	// Step 3: Remaining 10 positions all become confident -> full convergence.
	logits3 := make([][]float32, blockSize)
	for i := range blockSize {
		row := make([]float32, vocabSize)
		row[(i+2)%vocabSize] = 12.0
		logits3[i] = row
	}
	step3, err := SimulateConfidenceUnmaskingStep(cfg, logits3, step2.Tokens, step2.Masked)
	if err != nil {
		t.Fatalf("SimulateConfidenceUnmaskingStep step 3: %v", err)
	}
	if !step3.Converged || step3.RemainingMasked != 0 {
		t.Fatalf("step 3 Converged=%v RemainingMasked=%d, want true, 0", step3.Converged, step3.RemainingMasked)
	}
	for i, tok := range step3.Tokens {
		if tok < 0 || tok >= vocabSize {
			t.Fatalf("position %d still has invalid token %d after convergence", i, tok)
		}
	}
}

func TestDiffusionMultimodalEarlyFusionMegakernel(t *testing.T) {
	cfg := MultimodalEarlyFusionConfig{
		Name:               "omni_early_fusion",
		Target:             "sm_121a",
		BatchM:             16,
		HiddenDim:          256,
		VisionPatchDim:     256,
		VisionEmbedDim:     256,
		ViTLayers:          1,
		PixelShuffleFactor: 2,
		AudioRVQCodebooks:  8,
		AudioCodebookDim:   256,
		DecoderLayers:      2,
		QKVDim:             256,
		InterDim:           512,
	}

	plan, mega, err := BuildMultimodalEarlyFusionMegakernel(cfg)
	if err != nil {
		t.Fatalf("BuildMultimodalEarlyFusionMegakernel: %v", err)
	}
	if plan == nil || mega == nil {
		t.Fatalf("expected non-nil plan and megakernel")
	}
	if got, want := plan.Output.Shape, []int{16, 256}; len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("plan.Output.Shape = %v, want %v", got, want)
	}
	if len(mega.Stages) < 8 {
		t.Fatalf("len(mega.Stages) = %d, want >= 8 stages spanning vision patchify, 2D-RoPE ViT, pixel-shuffle connector, audio RVQ, and early-fusion decoder", len(mega.Stages))
	}

	src, spans, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("mega.LineInfoSource: %v", err)
	}
	if len(spans) != len(mega.Stages) {
		t.Fatalf("len(spans) = %d, want %d", len(spans), len(mega.Stages))
	}
	for _, wantSub := range []string{
		"vis_patch3d_proj",
		"vit0_rope2d_cos",
		"ps_mlp_up",
		"audio_rvq_proj",
		"mml0_qkv_proj",
		"mml1_mlp_down",
		"eg_ready_resolve_epoch",
	} {
		if !strings.Contains(src, wantSub) {
			t.Fatalf("multimodal early-fusion megakernel source missing %q", wantSub)
		}
	}
}
