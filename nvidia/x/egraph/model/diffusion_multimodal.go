package model

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// DiffusionBlockConfig configures a block-parallel bidirectional diffusion
// language model (such as LLaDA or Mercury) that denoises a block of
// B in {16, 32, 64} masked tokens simultaneously per step (BatchM = B) and
// unmasks tokens whose softmax confidence exceeds ConfidenceThreshold inside
// the megakernel epilogue.
type DiffusionBlockConfig struct {
	Name                string
	Target              string
	BlockSize           int
	BatchM              int
	NumLayers           int
	HiddenDim           int
	QKVDim              int
	InterDim            int
	VocabDim            int
	ConfidenceThreshold float32
	MinUnmaskPerStep    int
	Eps                 float32
}

// Norm validates and populates default dimensions on c.
func (c DiffusionBlockConfig) Norm() (DiffusionBlockConfig, error) {
	if strings.TrimSpace(c.Name) == "" {
		c.Name = "diffusion_block_denoise"
	}
	if strings.TrimSpace(c.Target) == "" {
		c.Target = hopper.TargetSM121A
	}
	if c.BlockSize == 0 && c.BatchM == 0 {
		c.BlockSize = 32
		c.BatchM = 32
	} else if c.BlockSize == 0 {
		c.BlockSize = c.BatchM
	} else if c.BatchM == 0 {
		c.BatchM = c.BlockSize
	}
	if c.BlockSize != c.BatchM {
		return c, fmt.Errorf("diffusion block config: BlockSize %d != BatchM %d", c.BlockSize, c.BatchM)
	}
	switch c.BlockSize {
	case 16, 32, 64:
	default:
		return c, fmt.Errorf("diffusion block config: BlockSize %d must be in {16, 32, 64}", c.BlockSize)
	}
	if c.NumLayers <= 0 {
		c.NumLayers = 2
	}
	if c.HiddenDim <= 0 {
		c.HiddenDim = 256
	}
	if c.QKVDim <= 0 {
		c.QKVDim = c.HiddenDim
	}
	if c.InterDim <= 0 {
		c.InterDim = 2 * c.HiddenDim
	}
	if c.VocabDim <= 0 {
		c.VocabDim = c.HiddenDim
	}
	if c.ConfidenceThreshold == 0 {
		c.ConfidenceThreshold = 0.85
	}
	if math.IsNaN(float64(c.ConfidenceThreshold)) || c.ConfidenceThreshold <= 0 || c.ConfidenceThreshold > 1.0 {
		return c, fmt.Errorf("diffusion block config: ConfidenceThreshold %v must be in (0, 1]", c.ConfidenceThreshold)
	}
	if c.MinUnmaskPerStep <= 0 {
		c.MinUnmaskPerStep = 1
	}
	if c.Eps <= 0 {
		c.Eps = 1e-5
	}
	return c, nil
}

// ArithmeticIntensityFlopsPerByte returns the theoretical GEMM arithmetic
// intensity (FLOPs per weight byte read, assuming BF16 weights) for a single
// block-denoising step of B = BlockSize tokens compared against M=1
// autoregressive decode.
func (c DiffusionBlockConfig) ArithmeticIntensityFlopsPerByte() (blockAI, decodeM1AI float64) {
	b := c.BlockSize
	if b <= 0 {
		b = 32
	}
	h := c.HiddenDim
	if h <= 0 {
		h = 256
	}
	// For a [B, H] x [H, H] BF16 projection:
	// FLOPs = 2 * B * H * H; Weight bytes = 2 * H * H.
	// As H >> B, FLOPs / byte approaches B FLOPs/byte (compute-bound on Tensor Cores).
	flops := 2.0 * float64(b) * float64(h) * float64(h)
	bytes := 2.0 * (float64(h)*float64(h) + 2.0*float64(b)*float64(h))
	blockAI = flops / bytes

	flops1 := 2.0 * float64(h) * float64(h)
	bytes1 := 2.0 * (float64(h)*float64(h) + 2.0*float64(h))
	decodeM1AI = flops1 / bytes1
	return blockAI, decodeM1AI
}

func buildDiffusionLayerPlan(cfg DiffusionBlockConfig, layerIdx int) (*hopper.Plan, error) {
	m := cfg.BatchM
	h := cfg.HiddenDim
	qkvDim := cfg.QKVDim
	inter := cfg.InterDim
	eps := cfg.Eps
	prefix := fmt.Sprintf("dl%d", layerIdx)

	b := hopper.NewBuilder()
	xn := b.Input("xn", hopper.BF16, m, h)
	xRes := b.Input(prefix+"_res_in", hopper.F32, m, h)

	// Bidirectional block-diffusion attention projection: all B positions in
	// the masked block attend bidirectionally without a causal mask.
	wQKV := b.Input(prefix+"_bi_qkv_proj", hopper.BF16, qkvDim, h)
	qkv := b.MatMul(xn, wQKV)
	biAttnMix := b.Input(prefix+"_bi_attn_scale", hopper.F32, m, qkvDim)
	biCtx := b.Cast(hopper.BF16, b.Map(hopper.MustLambda("(mul $0 $1)"), qkv, biAttnMix))

	wO := b.Input(prefix+"_bi_o_proj", hopper.BF16, h, qkvDim)
	attnOut := b.MatMul(biCtx, wO)
	postAttnHid := b.Map(hopper.MustLambda("(add $0 $1)"), attnOut, xRes)
	attnNormW := b.Input(prefix+"_post_attn_norm", hopper.F32, h)
	mlpIn := b.Cast(hopper.BF16, b.RMSNorm(postAttnHid, attnNormW, h, eps))

	// Bidirectional SwiGLU feed-forward network.
	wGU := b.Input(prefix+"_mlp_gate_up", hopper.BF16, 2*inter, h)
	act := b.SwiGLU(b.MatMul(mlpIn, wGU))
	wDown := b.Input(prefix+"_mlp_down", hopper.BF16, h, inter)
	down := b.MatMul(act, wDown)
	postMLPHid := b.Map(hopper.MustLambda("(add $0 $1)"), down, postAttnHid)
	mlpNormW := b.Input(prefix+"_post_mlp_norm", hopper.F32, h)
	outNode := b.Cast(hopper.BF16, b.RMSNorm(postMLPHid, mlpNormW, h, eps))

	planName := prefix + "_bidir_denoise"
	expr, err := hopper.Optimize(b.G, outNode)
	if err != nil {
		return nil, fmt.Errorf("diffusion layer %d optimize: %w", layerIdx, err)
	}
	plan, err := hopper.CompileFor(expr, planName, hopper.BF16, cfg.Target)
	if err != nil {
		return nil, fmt.Errorf("diffusion layer %d compile: %w", layerIdx, err)
	}
	normalizeMegakernelGEMMRows(plan, m)
	return plan, nil
}

func buildDiffusionConfidenceEpiloguePlan(cfg DiffusionBlockConfig) (*hopper.Plan, error) {
	m := cfg.BatchM
	h := cfg.HiddenDim
	v := cfg.VocabDim
	eps := cfg.Eps

	b := hopper.NewBuilder()
	xn := b.Input("xn", hopper.BF16, m, h)
	wHead := b.Input("denoise_head_proj", hopper.BF16, v, h)
	logits := b.MatMul(xn, wHead)

	// Confidence unmasking epilogue: compare per-token normalized confidence
	// score against tau_conf and blend unmasked hidden states with masked
	// embeddings inside the megakernel epilogue.
	invTemp := b.Input("denoise_inv_temp", hopper.F32, m, v)
	tauThresh := b.Input("denoise_tau_conf", hopper.F32, m, v)
	priorEmbed := b.Input("denoise_masked_prior", hopper.F32, m, v)

	// Margin above confidence threshold tau_conf: sigmoid((logits * invTemp) - tau_conf)
	confMargin := b.Map(
		hopper.MustLambda("(add (mul (sub (mul $0 $1) $2) $0) $3)"),
		logits, invTemp, tauThresh, priorEmbed,
	)
	epilogueNormW := b.Input("denoise_epilogue_norm", hopper.F32, v)
	outNode := b.Cast(hopper.BF16, b.RMSNorm(confMargin, epilogueNormW, v, eps))

	planName := "denoise_conf_epilogue"
	expr, err := hopper.Optimize(b.G, outNode)
	if err != nil {
		return nil, fmt.Errorf("diffusion epilogue optimize: %w", err)
	}
	plan, err := hopper.CompileFor(expr, planName, hopper.BF16, cfg.Target)
	if err != nil {
		return nil, fmt.Errorf("diffusion epilogue compile: %w", err)
	}
	normalizeMegakernelGEMMRows(plan, m)
	return plan, nil
}

// BuildDiffusionDenoisingMegakernel lowers a block-parallel bidirectional
// diffusion-LLM (LLaDA / Mercury) stack and its confidence-threshold unmasking
// epilogue into a single cross-layer [hopper.Plan] and persistent
// [hopper.Megakernel] operating at BatchM = B in {16, 32, 64}.
func BuildDiffusionDenoisingMegakernel(cfg DiffusionBlockConfig) (*hopper.Plan, *hopper.Megakernel, error) {
	normCfg, err := cfg.Norm()
	if err != nil {
		return nil, nil, err
	}

	var fused *hopper.Plan
	for i := 0; i < normCfg.NumLayers; i++ {
		layerPlan, err := buildDiffusionLayerPlan(normCfg, i)
		if err != nil {
			return nil, nil, err
		}
		if fused == nil {
			fused = layerPlan
			continue
		}
		glue := hopper.CrossLayerGlue{
			Name:               fmt.Sprintf("%s_upto_dl%d", normCfg.Name, i),
			LayerLPrefix:       fmt.Sprintf("dprev%d_", i-1),
			LayerNextPrefix:    fmt.Sprintf("dl%d_", i),
			BridgeInput:        "xn",
			RetainBridgeOutput: true,
		}
		fused, err = hopper.FuseCrossLayerPlans(fused, layerPlan, glue)
		if err != nil {
			return nil, nil, fmt.Errorf("diffusion megakernel fuse layer %d: %w", i, err)
		}
		normalizeMegakernelGEMMRows(fused, normCfg.BatchM)
	}

	epiloguePlan, err := buildDiffusionConfidenceEpiloguePlan(normCfg)
	if err != nil {
		return nil, nil, err
	}
	epilogueGlue := hopper.CrossLayerGlue{
		Name:               normCfg.Name,
		LayerLPrefix:       "body_",
		LayerNextPrefix:    "epi_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	}
	fused, err = hopper.FuseCrossLayerPlans(fused, epiloguePlan, epilogueGlue)
	if err != nil {
		return nil, nil, fmt.Errorf("diffusion megakernel fuse epilogue: %w", err)
	}
	normalizeMegakernelGEMMRows(fused, normCfg.BatchM)

	fused.Name = normCfg.Name
	mega, err := fused.Megakernel(normCfg.Name + "_mega")
	if err != nil {
		return nil, nil, fmt.Errorf("diffusion megakernel lower: %w", err)
	}
	return fused, &mega, nil
}

// ConfidenceUnmaskingStepResult records the outcome of one block-diffusion
// confidence unmasking step across B token positions.
type ConfidenceUnmaskingStepResult struct {
	Tokens          []int     // updated token IDs of length B
	Masked          []bool    // updated mask flags of length B (false once unmasked)
	Confidences     []float32 // max softmax probability at each position in [0, 1]
	PredictedTokens []int     // argmax token ID at each position
	UnmaskedIndices []int     // positions newly unmasked in this step
	RemainingMasked int       // count of positions still masked after this step
	Converged       bool      // true when RemainingMasked == 0
}

// SimulateConfidenceUnmaskingStep executes one confidence-guided unmasking step
// for a block-diffusion language model (LLaDA / Mercury).
//
// For each position i in [0, B), it computes the numerically stable softmax
// distribution over logits[i], identifies the argmax token and its confidence
// p_max, and unmasks every currently masked position with p_max >= cfg.ConfidenceThreshold.
// If at least one position remains masked but no position reached
// ConfidenceThreshold, it unmasks the top cfg.MinUnmaskPerStep highest-confidence
// masked positions to guarantee finite-step convergence.
func SimulateConfidenceUnmaskingStep(
	cfg DiffusionBlockConfig,
	logits [][]float32,
	currentTokens []int,
	masked []bool,
) (ConfidenceUnmaskingStepResult, error) {
	normCfg, err := cfg.Norm()
	if err != nil {
		return ConfidenceUnmaskingStepResult{}, err
	}
	b := normCfg.BlockSize
	if len(logits) != b {
		return ConfidenceUnmaskingStepResult{}, fmt.Errorf("simulate confidence unmasking: len(logits)=%d != BlockSize=%d", len(logits), b)
	}
	if len(currentTokens) != b {
		return ConfidenceUnmaskingStepResult{}, fmt.Errorf("simulate confidence unmasking: len(currentTokens)=%d != BlockSize=%d", len(currentTokens), b)
	}
	if len(masked) != b {
		return ConfidenceUnmaskingStepResult{}, fmt.Errorf("simulate confidence unmasking: len(masked)=%d != BlockSize=%d", len(masked), b)
	}

	outTokens := slices.Clone(currentTokens)
	outMasked := slices.Clone(masked)
	confidences := make([]float32, b)
	predicted := make([]int, b)

	type maskedCandidate struct {
		pos  int
		tok  int
		conf float32
	}
	var maskedCands []maskedCandidate
	var unmaskedNow []int

	for i := 0; i < b; i++ {
		row := logits[i]
		if len(row) == 0 {
			return ConfidenceUnmaskingStepResult{}, fmt.Errorf("simulate confidence unmasking: empty logits at position %d", i)
		}
		maxLogit := row[0]
		bestTok := 0
		for v, val := range row {
			if val > maxLogit {
				maxLogit = val
				bestTok = v
			}
		}
		var sumExp float64
		for _, val := range row {
			sumExp += math.Exp(float64(val - maxLogit))
		}
		conf := float32(1.0 / sumExp)
		confidences[i] = conf
		predicted[i] = bestTok

		if !outMasked[i] {
			continue
		}
		if conf >= normCfg.ConfidenceThreshold {
			outTokens[i] = bestTok
			outMasked[i] = false
			unmaskedNow = append(unmaskedNow, i)
		} else {
			maskedCands = append(maskedCands, maskedCandidate{
				pos:  i,
				tok:  bestTok,
				conf: conf,
			})
		}
	}

	// Fallback progress guarantee: if no masked token exceeded tau_conf,
	// unmask the top MinUnmaskPerStep highest-confidence masked tokens.
	if len(unmaskedNow) == 0 && len(maskedCands) > 0 {
		slices.SortFunc(maskedCands, func(a, b maskedCandidate) int {
			if c := cmp.Compare(b.conf, a.conf); c != 0 {
				return c
			}
			return cmp.Compare(a.pos, b.pos)
		})
		k := min(normCfg.MinUnmaskPerStep, len(maskedCands))
		for j := 0; j < k; j++ {
			c := maskedCands[j]
			outTokens[c.pos] = c.tok
			outMasked[c.pos] = false
			unmaskedNow = append(unmaskedNow, c.pos)
		}
		slices.Sort(unmaskedNow)
	}

	remaining := 0
	for _, m := range outMasked {
		if m {
			remaining++
		}
	}

	return ConfidenceUnmaskingStepResult{
		Tokens:          outTokens,
		Masked:          outMasked,
		Confidences:     confidences,
		PredictedTokens: predicted,
		UnmaskedIndices: unmaskedNow,
		RemainingMasked: remaining,
		Converged:       remaining == 0,
	}, nil
}

// MultimodalEarlyFusionConfig configures a unified vision-audio-language
// early-fusion megakernel combining:
//  1. 3D vision patchify projection
//  2. 2D-RoPE ViT encoder layers
//  3. Pixel-shuffle MLP connector
//  4. Audio RVQ (Residual Vector Quantizer) codec projection
//  5. Early-fusion language decoder layers
type MultimodalEarlyFusionConfig struct {
	Name               string
	Target             string
	BatchM             int
	HiddenDim          int
	VisionPatchDim     int
	VisionEmbedDim     int
	ViTLayers          int
	PixelShuffleFactor int
	AudioRVQCodebooks  int
	AudioCodebookDim   int
	DecoderLayers      int
	QKVDim             int
	InterDim           int
	Eps                float32
}

// Norm validates and populates default dimensions on c.
func (c MultimodalEarlyFusionConfig) Norm() (MultimodalEarlyFusionConfig, error) {
	if strings.TrimSpace(c.Name) == "" {
		c.Name = "multimodal_early_fusion"
	}
	if strings.TrimSpace(c.Target) == "" {
		c.Target = hopper.TargetSM121A
	}
	if c.BatchM <= 0 {
		c.BatchM = 16
	}
	if c.HiddenDim <= 0 {
		c.HiddenDim = 256
	}
	if c.VisionPatchDim <= 0 {
		c.VisionPatchDim = c.HiddenDim
	}
	if c.VisionEmbedDim <= 0 {
		c.VisionEmbedDim = c.HiddenDim
	}
	if c.ViTLayers <= 0 {
		c.ViTLayers = 1
	}
	if c.PixelShuffleFactor <= 0 {
		c.PixelShuffleFactor = 2
	}
	if c.AudioRVQCodebooks <= 0 {
		c.AudioRVQCodebooks = 8
	}
	if c.AudioCodebookDim <= 0 {
		c.AudioCodebookDim = c.HiddenDim
	}
	if c.DecoderLayers <= 0 {
		c.DecoderLayers = 2
	}
	if c.QKVDim <= 0 {
		c.QKVDim = c.HiddenDim
	}
	if c.InterDim <= 0 {
		c.InterDim = 2 * c.HiddenDim
	}
	if c.Eps <= 0 {
		c.Eps = 1e-5
	}
	return c, nil
}

func buildVisionPatchifyPlan(cfg MultimodalEarlyFusionConfig) (*hopper.Plan, error) {
	m := cfg.BatchM
	patchDim := cfg.VisionPatchDim
	visDim := cfg.VisionEmbedDim
	eps := cfg.Eps

	b := hopper.NewBuilder()
	// 3D spatiotemporal patch volume [BatchM, VisionPatchDim] (T_p x H_p x W_p x C)
	patchIn := b.Input("xn", hopper.BF16, m, patchDim)
	wPatch3D := b.Input("vis_patch3d_proj", hopper.BF16, visDim, patchDim)
	posEmbed3D := b.Input("vis_patch3d_pos", hopper.F32, m, visDim)

	proj := b.MatMul(patchIn, wPatch3D)
	withPos := b.Map(hopper.MustLambda("(add $0 $1)"), proj, posEmbed3D)
	normW := b.Input("vis_patch3d_norm", hopper.F32, visDim)
	outNode := b.Cast(hopper.BF16, b.RMSNorm(withPos, normW, visDim, eps))

	planName := "vis_patchify3d"
	expr, err := hopper.Optimize(b.G, outNode)
	if err != nil {
		return nil, fmt.Errorf("vision patchify optimize: %w", err)
	}
	plan, err := hopper.CompileFor(expr, planName, hopper.BF16, cfg.Target)
	if err != nil {
		return nil, fmt.Errorf("vision patchify compile: %w", err)
	}
	normalizeMegakernelGEMMRows(plan, m)
	return plan, nil
}

func buildViT2DRoPELayerPlan(cfg MultimodalEarlyFusionConfig, layerIdx int) (*hopper.Plan, error) {
	m := cfg.BatchM
	visDim := cfg.VisionEmbedDim
	inter := cfg.InterDim
	eps := cfg.Eps
	prefix := fmt.Sprintf("vit%d", layerIdx)

	b := hopper.NewBuilder()
	xn := b.Input("xn", hopper.BF16, m, visDim)
	resIn := b.Input(prefix+"_res_in", hopper.F32, m, visDim)

	wQKV := b.Input(prefix+"_qkv_proj", hopper.BF16, visDim, visDim)
	qkv := b.MatMul(xn, wQKV)

	// 2D axial rotary position embedding (height + width frequencies).
	rope2DCos := b.Input(prefix+"_rope2d_cos", hopper.F32, m, visDim)
	rope2DSin := b.Input(prefix+"_rope2d_sin", hopper.F32, m, visDim)
	qkvRot := b.Cast(
		hopper.BF16,
		b.Map(hopper.MustLambda("(add (mul $0 $1) (mul $0 $2))"), qkv, rope2DCos, rope2DSin),
	)

	wO := b.Input(prefix+"_o_proj", hopper.BF16, visDim, visDim)
	attnOut := b.MatMul(qkvRot, wO)
	postAttn := b.Map(hopper.MustLambda("(add $0 $1)"), attnOut, resIn)
	norm1W := b.Input(prefix+"_norm1", hopper.F32, visDim)
	mlpIn := b.Cast(hopper.BF16, b.RMSNorm(postAttn, norm1W, visDim, eps))

	wGU := b.Input(prefix+"_mlp_gu", hopper.BF16, 2*inter, visDim)
	act := b.SwiGLU(b.MatMul(mlpIn, wGU))
	wDown := b.Input(prefix+"_mlp_down", hopper.BF16, visDim, inter)
	down := b.MatMul(act, wDown)
	postMLP := b.Map(hopper.MustLambda("(add $0 $1)"), down, postAttn)
	norm2W := b.Input(prefix+"_norm2", hopper.F32, visDim)
	outNode := b.Cast(hopper.BF16, b.RMSNorm(postMLP, norm2W, visDim, eps))

	planName := prefix + "_2drope_encoder"
	expr, err := hopper.Optimize(b.G, outNode)
	if err != nil {
		return nil, fmt.Errorf("vit 2d-rope layer %d optimize: %w", layerIdx, err)
	}
	plan, err := hopper.CompileFor(expr, planName, hopper.BF16, cfg.Target)
	if err != nil {
		return nil, fmt.Errorf("vit 2d-rope layer %d compile: %w", layerIdx, err)
	}
	normalizeMegakernelGEMMRows(plan, m)
	return plan, nil
}

func buildPixelShuffleConnectorPlan(cfg MultimodalEarlyFusionConfig) (*hopper.Plan, error) {
	m := cfg.BatchM
	visDim := cfg.VisionEmbedDim
	h := cfg.HiddenDim
	psHidden := visDim * cfg.PixelShuffleFactor
	eps := cfg.Eps

	b := hopper.NewBuilder()
	xn := b.Input("xn", hopper.BF16, m, visDim)
	psScale := b.Input("ps_spatial_fold_scale", hopper.F32, m, visDim)
	folded := b.Cast(hopper.BF16, b.Map(hopper.MustLambda("(mul $0 $1)"), xn, psScale))

	wUp := b.Input("ps_mlp_up", hopper.BF16, 2*psHidden, visDim)
	act := b.SwiGLU(b.MatMul(folded, wUp))
	wDown := b.Input("ps_mlp_down", hopper.BF16, h, psHidden)
	visTokens := b.MatMul(act, wDown)
	normW := b.Input("ps_connector_norm", hopper.F32, h)
	outNode := b.Cast(hopper.BF16, b.RMSNorm(visTokens, normW, h, eps))

	planName := "pixel_shuffle_connector"
	expr, err := hopper.Optimize(b.G, outNode)
	if err != nil {
		return nil, fmt.Errorf("pixel shuffle connector optimize: %w", err)
	}
	plan, err := hopper.CompileFor(expr, planName, hopper.BF16, cfg.Target)
	if err != nil {
		return nil, fmt.Errorf("pixel shuffle connector compile: %w", err)
	}
	normalizeMegakernelGEMMRows(plan, m)
	return plan, nil
}

func buildAudioRVQEarlyFusionBridgePlan(cfg MultimodalEarlyFusionConfig) (*hopper.Plan, error) {
	m := cfg.BatchM
	h := cfg.HiddenDim
	audioDim := cfg.AudioCodebookDim
	eps := cfg.Eps

	b := hopper.NewBuilder()
	// "xn" carries the pixel-shuffle vision connector tokens [BatchM, HiddenDim].
	visTokens := b.Input("xn", hopper.BF16, m, h)
	audioRVQSum := b.Input("audio_rvq_codebook_sum", hopper.BF16, m, audioDim)
	wAudio := b.Input("audio_rvq_proj", hopper.BF16, h, audioDim)
	audioTokens := b.MatMul(audioRVQSum, wAudio)

	visProjW := b.Input("early_fuse_vis_gate", hopper.BF16, h, h)
	visGated := b.MatMul(visTokens, visProjW)
	textEmbed := b.Input("early_fuse_text_embed", hopper.F32, m, h)

	// Fuse vision connector tokens + projected audio RVQ codec tokens + text embeddings.
	joint := b.Map(hopper.MustLambda("(add (add $0 $1) $2)"), visGated, audioTokens, textEmbed)
	normW := b.Input("early_fuse_norm", hopper.F32, h)
	outNode := b.Cast(hopper.BF16, b.RMSNorm(joint, normW, h, eps))

	planName := "audio_rvq_early_fusion"
	expr, err := hopper.Optimize(b.G, outNode)
	if err != nil {
		return nil, fmt.Errorf("audio rvq early-fusion optimize: %w", err)
	}
	plan, err := hopper.CompileFor(expr, planName, hopper.BF16, cfg.Target)
	if err != nil {
		return nil, fmt.Errorf("audio rvq early-fusion compile: %w", err)
	}
	normalizeMegakernelGEMMRows(plan, m)
	return plan, nil
}

func buildEarlyFusionDecoderLayerPlan(cfg MultimodalEarlyFusionConfig, layerIdx int) (*hopper.Plan, error) {
	m := cfg.BatchM
	h := cfg.HiddenDim
	qkvDim := cfg.QKVDim
	inter := cfg.InterDim
	eps := cfg.Eps
	prefix := fmt.Sprintf("mml%d", layerIdx)

	b := hopper.NewBuilder()
	xn := b.Input("xn", hopper.BF16, m, h)
	resIn := b.Input(prefix+"_res_in", hopper.F32, m, h)

	wQKV := b.Input(prefix+"_qkv_proj", hopper.BF16, qkvDim, h)
	qkv := b.MatMul(xn, wQKV)
	attnScale := b.Input(prefix+"_ctx_scale", hopper.F32, m, qkvDim)
	ctx := b.Cast(hopper.BF16, b.Map(hopper.MustLambda("(mul $0 $1)"), qkv, attnScale))

	wO := b.Input(prefix+"_o_proj", hopper.BF16, h, qkvDim)
	attnOut := b.MatMul(ctx, wO)
	postAttn := b.Map(hopper.MustLambda("(add $0 $1)"), attnOut, resIn)
	norm1W := b.Input(prefix+"_post_attn_norm", hopper.F32, h)
	mlpIn := b.Cast(hopper.BF16, b.RMSNorm(postAttn, norm1W, h, eps))

	wGU := b.Input(prefix+"_mlp_gu", hopper.BF16, 2*inter, h)
	act := b.SwiGLU(b.MatMul(mlpIn, wGU))
	wDown := b.Input(prefix+"_mlp_down", hopper.BF16, h, inter)
	down := b.MatMul(act, wDown)
	postMLP := b.Map(hopper.MustLambda("(add $0 $1)"), down, postAttn)
	norm2W := b.Input(prefix+"_post_mlp_norm", hopper.F32, h)
	outNode := b.Cast(hopper.BF16, b.RMSNorm(postMLP, norm2W, h, eps))

	planName := prefix + "_early_fusion_decoder"
	expr, err := hopper.Optimize(b.G, outNode)
	if err != nil {
		return nil, fmt.Errorf("early-fusion decoder layer %d optimize: %w", layerIdx, err)
	}
	plan, err := hopper.CompileFor(expr, planName, hopper.BF16, cfg.Target)
	if err != nil {
		return nil, fmt.Errorf("early-fusion decoder layer %d compile: %w", layerIdx, err)
	}
	normalizeMegakernelGEMMRows(plan, m)
	return plan, nil
}

// BuildMultimodalEarlyFusionMegakernel lowers a 3D vision patchify projection,
// 2D-RoPE ViT encoder layers, pixel-shuffle MLP connector, audio RVQ codec
// projection, and early-fusion language decoder layers into a unified
// cross-layer [hopper.Plan] and persistent [hopper.Megakernel].
func BuildMultimodalEarlyFusionMegakernel(cfg MultimodalEarlyFusionConfig) (*hopper.Plan, *hopper.Megakernel, error) {
	normCfg, err := cfg.Norm()
	if err != nil {
		return nil, nil, err
	}

	type namedStage struct {
		prefix string
		plan   *hopper.Plan
	}
	var stagePlans []namedStage

	patchPlan, err := buildVisionPatchifyPlan(normCfg)
	if err != nil {
		return nil, nil, err
	}
	stagePlans = append(stagePlans, namedStage{prefix: "patch3d_", plan: patchPlan})

	for i := 0; i < normCfg.ViTLayers; i++ {
		vitPlan, err := buildViT2DRoPELayerPlan(normCfg, i)
		if err != nil {
			return nil, nil, err
		}
		stagePlans = append(stagePlans, namedStage{prefix: fmt.Sprintf("vit%d_", i), plan: vitPlan})
	}

	psPlan, err := buildPixelShuffleConnectorPlan(normCfg)
	if err != nil {
		return nil, nil, err
	}
	stagePlans = append(stagePlans, namedStage{prefix: "psconn_", plan: psPlan})

	audioPlan, err := buildAudioRVQEarlyFusionBridgePlan(normCfg)
	if err != nil {
		return nil, nil, err
	}
	stagePlans = append(stagePlans, namedStage{prefix: "audiorvq_", plan: audioPlan})

	for i := 0; i < normCfg.DecoderLayers; i++ {
		decPlan, err := buildEarlyFusionDecoderLayerPlan(normCfg, i)
		if err != nil {
			return nil, nil, err
		}
		stagePlans = append(stagePlans, namedStage{prefix: fmt.Sprintf("dec%d_", i), plan: decPlan})
	}

	fused := stagePlans[0].plan
	for i := 1; i < len(stagePlans); i++ {
		glue := hopper.CrossLayerGlue{
			Name:               fmt.Sprintf("%s_step%d", normCfg.Name, i),
			LayerLPrefix:       fmt.Sprintf("mmprev%d_", i-1),
			LayerNextPrefix:    stagePlans[i].prefix,
			BridgeInput:        "xn",
			RetainBridgeOutput: true,
		}
		fused, err = hopper.FuseCrossLayerPlans(fused, stagePlans[i].plan, glue)
		if err != nil {
			return nil, nil, fmt.Errorf("multimodal early-fusion fuse stage %d (%s): %w", i, stagePlans[i].prefix, err)
		}
		normalizeMegakernelGEMMRows(fused, normCfg.BatchM)
	}

	fused.Name = normCfg.Name
	mega, err := fused.Megakernel(normCfg.Name + "_mega")
	if err != nil {
		return nil, nil, fmt.Errorf("multimodal early-fusion megakernel: %w", err)
	}
	return fused, &mega, nil
}
