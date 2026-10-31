package main

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/model"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
)

// AutonomousDecodeConfig configures the end-to-end autonomous flight pipeline
// for a pair of adjacent LLaMA decoder layers in llama-infer.
type AutonomousDecodeConfig struct {
	Model            *ModelConfig
	QKVBias          bool
	LayerL           int
	LayerNext        int
	BatchM           int
	Target           string
	Device           string
	DeviceSMs        int
	DriverVersion    string
	CUDAVersion      string
	WeightDigest     string
	CacheDir         string
	Cache            *autotune.VerifiedPGOCache
	BeamWidth        int
	PTXASCompiler    qualify.PTXASCompiler
	PTXASGate        qualify.StaticGate
	MaxPTXASIters    int
	HotSwapService   *cudahotswap.Service
	BucketController *specdecode.AdaptiveBucketController
	BucketConfig     *specdecode.AdaptiveBucketConfig
	KernelName       string
}

// AutonomousDecodeEngine wires [model.RunAutonomousFlightPipeline],
// [buildCrossLayerDecodeMegakernel], [specdecode.AdaptiveBucketController],
// [autotune.VerifiedPGOCache], and [cudahotswap.Service] for a LLaMA decoder
// layer boundary.
type AutonomousDecodeEngine struct {
	Config               AutonomousDecodeConfig
	Cache                *autotune.VerifiedPGOCache
	BucketController     *specdecode.AdaptiveBucketController
	HotSwapService       *cudahotswap.Service
	PostGraph            *egraph.EGraph
	PostRoot             egraph.Id
	LayerLPostPlan       *hopper.Plan
	LayerNextAttnInPlan  *hopper.Plan
	CrossLayerPlan       *hopper.Plan
	CrossLayerMegakernel hopper.Megakernel
	Glue                 hopper.CrossLayerGlue
	KernelName           string
}

// NewAutonomousDecodeEngine constructs an [AutonomousDecodeEngine] for the
// inter-layer boundary (cfg.LayerL -> cfg.LayerNext).
func NewAutonomousDecodeEngine(cfg AutonomousDecodeConfig) (*AutonomousDecodeEngine, error) {
	if cfg.Model == nil {
		return nil, errors.New("autonomous decode engine: nil model config")
	}
	if cfg.LayerL == 0 && cfg.LayerNext == 0 {
		cfg.LayerNext = 1
	}
	if cfg.BatchM <= 0 {
		cfg.BatchM = 1
	}
	target := strings.TrimSpace(cfg.Target)
	if target == "" {
		target = "sm_121a"
	}
	cfg.Target = target
	if strings.TrimSpace(cfg.Device) == "" {
		cfg.Device = target
	}
	if strings.TrimSpace(cfg.DriverVersion) == "" {
		cfg.DriverVersion = "570.86.10"
	}
	if strings.TrimSpace(cfg.CUDAVersion) == "" {
		cfg.CUDAVersion = "12.8"
	}
	if strings.TrimSpace(cfg.WeightDigest) == "" {
		cfg.WeightDigest = fmt.Sprintf("llama_h%d_i%d_l%d",
			cfg.Model.HiddenSize, cfg.Model.IntermediateSize, cfg.Model.NumHiddenLayers)
	}
	if cfg.BeamWidth <= 0 {
		cfg.BeamWidth = 4
	}

	crossPlan, crossMega, err := buildCrossLayerDecodeMegakernel(
		cfg.Model, cfg.QKVBias, cfg.LayerL, cfg.LayerNext, cfg.BatchM, target,
	)
	if err != nil {
		return nil, fmt.Errorf("autonomous decode engine: %w", err)
	}

	prefixL := fmt.Sprintf("l%d", cfg.LayerL)
	prefixNext := fmt.Sprintf("l%d", cfg.LayerNext)
	_, postL, err := buildLayerPlans(cfg.Model, cfg.QKVBias, false, prefixL, cfg.BatchM, hopper.F32, target, true)
	if err != nil {
		return nil, fmt.Errorf("autonomous decode engine post plan: %w", err)
	}
	attnInNext, _, err := buildLayerPlans(cfg.Model, cfg.QKVBias, false, prefixNext, cfg.BatchM, hopper.F32, target, true)
	if err != nil {
		return nil, fmt.Errorf("autonomous decode engine next attnIn plan: %w", err)
	}

	postGraph, postRoot := buildLayerPostEGraph(cfg.Model, cfg.BatchM, hopper.F32, true)

	cache := cfg.Cache
	if cache == nil && strings.TrimSpace(cfg.CacheDir) != "" {
		cache = autotune.NewVerifiedPGOCache(cfg.CacheDir)
	}

	bucketCtrl := cfg.BucketController
	if bucketCtrl == nil {
		var bcfg specdecode.AdaptiveBucketConfig
		if cfg.BucketConfig != nil {
			bcfg = *cfg.BucketConfig
		} else {
			bcfg = specdecode.AdaptiveBucketConfig{
				HysteresisSteps: 2,
				InitialBucketM:  cfg.BatchM,
			}
		}
		bucketCtrl, err = specdecode.NewAdaptiveBucketController(bcfg)
		if err != nil {
			return nil, fmt.Errorf("autonomous decode engine bucket controller: %w", err)
		}
	}

	glue := hopper.CrossLayerGlue{
		Name:               crossMega.Name,
		LayerLPrefix:       prefixL + "_",
		LayerNextPrefix:    prefixNext + "_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	}
	kernelName := strings.TrimSpace(cfg.KernelName)
	if kernelName == "" {
		kernelName = crossMega.Name
	}

	return &AutonomousDecodeEngine{
		Config:               cfg,
		Cache:                cache,
		BucketController:     bucketCtrl,
		HotSwapService:       cfg.HotSwapService,
		PostGraph:            postGraph,
		PostRoot:             postRoot,
		LayerLPostPlan:       postL,
		LayerNextAttnInPlan:  attnInNext,
		CrossLayerPlan:       crossPlan,
		CrossLayerMegakernel: crossMega,
		Glue:                 glue,
		KernelName:           kernelName,
	}, nil
}

func buildLayerPostEGraph(cfg *ModelConfig, m int, attnDType hopper.DType, chain bool) (*egraph.EGraph, egraph.Id) {
	h := cfg.HiddenSize
	qDim := cfg.NumAttentionHeads * cfg.HeadDim()
	inter := cfg.IntermediateSize
	eps := cfg.RmsNormEps

	b := hopper.NewBuilder()
	ao := b.Input("attn", attnDType, m, qDim)
	if attnDType != hopper.BF16 {
		ao = b.Cast(hopper.BF16, ao)
	}
	x := b.Input("x", hopper.F32, m, h)
	hid := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(ao, b.Input("w_o", hopper.BF16, h, qDim)), x)
	norm := b.Cast(hopper.BF16, b.RMSNorm(hid, b.Input("ffn_norm", hopper.F32, h), h, eps))
	act := b.SwiGLU(b.MatMul(norm, b.Input("w_gu", hopper.BF16, 2*inter, h)))
	out := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", hopper.BF16, h, inter)), hid)
	if chain {
		out = b.Cast(hopper.BF16, b.RMSNorm(out, b.Input("next_norm", hopper.F32, h), h, eps))
	}
	return b.G, out
}

// CacheKey returns the hardware-and-model-bound [autotune.PGOCacheKey] for the
// engine's cross-layer decode plan at shape bucket bucketM.
func (e *AutonomousDecodeEngine) CacheKey(bucketM int) autotune.PGOCacheKey {
	if bucketM <= 0 {
		if e.BucketController != nil && e.BucketController.CurrentBucket().BucketM > 0 {
			bucketM = e.BucketController.CurrentBucket().BucketM
		} else if e.Config.BatchM > 0 {
			bucketM = e.Config.BatchM
		} else {
			bucketM = 1
		}
	}
	return autotune.PGOCacheKey{
		Arch:          e.Config.Target,
		DriverVersion: e.Config.DriverVersion,
		CUDAVersion:   e.Config.CUDAVersion,
		WeightDigest:  e.Config.WeightDigest,
		PlanKey:       e.KernelName,
		ShapeBucket:   bucketM,
	}
}

// Run executes [model.RunAutonomousFlightPipeline] for the engine's decoder
// layer boundary, wiring the verified PGO cache, structural beam extraction,
// cross-layer megakernel, ptxas loop refinement, PC stall attribution, holdout
// promotion gate, hot-swap service, and adaptive bucket controller.
func (e *AutonomousDecodeEngine) Run(ctx context.Context, req model.AutonomousEvalRequest) (*model.AutonomousPipelineResult, error) {
	if e == nil {
		return nil, errors.New("autonomous decode engine: nil engine")
	}
	if req.CrossLayerPost == nil {
		req.CrossLayerPost = e.LayerLPostPlan
	}
	if req.CrossLayerNextAttnIn == nil {
		req.CrossLayerNextAttnIn = e.LayerNextAttnInPlan
	}
	if req.CrossLayerGlue == (hopper.CrossLayerGlue{}) {
		req.CrossLayerGlue = e.Glue
	}
	if req.KernelName == "" {
		req.KernelName = e.KernelName
	}

	bucketM := e.Config.BatchM
	if e.BucketController != nil && e.BucketController.CurrentBucket().BucketM > 0 {
		bucketM = e.BucketController.CurrentBucket().BucketM
	}
	pCfg := model.AutonomousPipelineConfig{
		Cache:            e.Cache,
		CacheKey:         e.CacheKey(bucketM),
		Device:           e.Config.Device,
		Arch:             e.Config.Target,
		DeviceSMs:        e.Config.DeviceSMs,
		BeamWidth:        e.Config.BeamWidth,
		PTXASCompiler:    e.Config.PTXASCompiler,
		PTXASGate:        e.Config.PTXASGate,
		MaxPTXASIters:    e.Config.MaxPTXASIters,
		HotSwapService:   e.HotSwapService,
		BucketController: e.BucketController,
		KernelName:       e.KernelName,
	}
	return model.RunAutonomousFlightPipeline(
		ctx, e.PostGraph, e.PostRoot, e.KernelName, hopper.BF16, pCfg, req,
	)
}

// StepBuckets records a speculative verification step on the engine's
// [specdecode.AdaptiveBucketController] and returns the resulting transition.
func (e *AutonomousDecodeEngine) StepBuckets(ctx context.Context, batchSize, draftedTokens, acceptedTokens int) (specdecode.BucketTransition, error) {
	if e == nil || e.BucketController == nil {
		return specdecode.BucketTransition{}, errors.New("autonomous decode engine: nil bucket controller")
	}
	return e.BucketController.Step(ctx, batchSize, draftedTokens, acceptedTokens)
}

// RestorePrevious rolls the engine's hot-swap service and bucket controller
// back to the immediate prior retained version using lastResult.
func (e *AutonomousDecodeEngine) RestorePrevious(ctx context.Context, lastResult *model.AutonomousPipelineResult) (cudahotswap.State, error) {
	if lastResult != nil {
		return lastResult.RestorePrevious(ctx)
	}
	if e == nil || e.BucketController == nil {
		return cudahotswap.State{}, errors.New("autonomous decode engine: nil result and bucket controller")
	}
	return e.BucketController.RestorePrevious(ctx)
}

// runAutonomousDecodePipeline constructs an [AutonomousDecodeEngine] from cfg
// and runs [AutonomousDecodeEngine.Run].
func runAutonomousDecodePipeline(
	ctx context.Context,
	cfg AutonomousDecodeConfig,
	req model.AutonomousEvalRequest,
) (*model.AutonomousPipelineResult, error) {
	eng, err := NewAutonomousDecodeEngine(cfg)
	if err != nil {
		return nil, err
	}
	return eng.Run(ctx, req)
}
